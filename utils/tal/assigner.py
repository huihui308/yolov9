import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import bbox_iou


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchor center in gt

    Args:
        xy_centers (Tensor): shape(h*w, 4)
        gt_bboxes (Tensor): shape(b, n_boxes, 4)
    Return:
        (Tensor): shape(b, n_boxes, h*w)
    """
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
    # torch.Size([2, 125, 4]) torch.Size([250, 1, 2]) torch.Size([250, 1, 2]) torch.Size([1, 8400, 2])
    #print(gt_bboxes.shape, lt.shape, rb.shape, xy_centers[None].shape)
    # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
    # 第三个维度均大于0才说明在gt内部
    # M x 8400
    return bbox_deltas.amin(3).gt_(eps)


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        overlaps (Tensor): shape(b, n_max_boxes, h*w)
    Return:
        target_gt_idx (Tensor): shape(b, h*w)
        fg_mask (Tensor): shape(b, h*w)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
    """
    # 一个预测框匹配真实框的个数
    # (b, n_max_boxes, h*w) -> (b, h*w)
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])  # (b, n_max_boxes, h*w)
        # 与预测框IoU值最高的真实框的索引
        max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
        # 进行one-hot编码，与预测框IoU值最高的真实框的位置为 1 
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)  # (b, h*w, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)  # (b, n_max_boxes, h*w)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)  # (b, n_max_boxes, h*w)
        # 正样本的mask
        fg_mask = mask_pos.sum(-2)
    # 每个正样本与之匹配真实框的索引
    # find each grid serve which gt(index)
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos


    """
    https://blog.csdn.net/YXD0514/article/details/132116133?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170952091316800180674684%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170952091316800180674684&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-132116133-null-null.142^v99^pc_search_result_base3&utm_term=TaskAlignedAssigner&spm=1018.2226.3001.4187
    
    正负样本分配大体流程：
    (1) 网络输出的 pred_scores[bx8400xcls_num],进行sigmoid处理(每个类别按照2分类处理)。
    (2) 经过解码的 pred_bboxes[bx8400x4], 与 stride_tensor[8400x1]相乘,将bboxes转换到网络输入尺度[bx3x640x640];
    (3) 预处理得到的 anchors_points[8400x2],与stride_tensor[8400x1]相乘，将anchors的中心点坐标转换到输入尺度[bx3x640x640];
    (4) 将上述 pred_scores、pred_bboxes、anchors_points,还有标注数据预处理之后的gt_labels、gt_bboxes、gt_mask，相结合进行正样本的筛选工作。
    
    TaskAlignedAssigner 的匹配策略简单总结为：根据分类与回归的分数加权的分数选择正样本。
        t = s^α + u^β

        （1）计算真实框和预测框的匹配程度。
        其中，s是预测类别分值，u是预测框和真实框的ciou值，α 和β为权重超参数，两者相乘就可以衡量匹配程度，当分类的分值越高且ciou越高时，align_metric的值就越接近于1,此时预测框就与真实框越匹配，就越符合正样本的标准。
        （2）初筛，选取预测框中心点到真实框左上和右下两点距离都大于0的预测框。
        （3）对于每个真实框，直接对align_metric匹配程度排序，选取topK个预测框作为正样本。
        （4）对一个预测框与多个真实框匹配情况进行处理，保留ciou值最大的真实框。
    """
class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        # batch size 的大小
        self.bs = pd_scores.size(0)
        # 每个图片真实框个数不同，按图片中真实框最大的个数进行补零对齐。
        # n_max_boxes：最大真实框的个数
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                    torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        # 真实框的mask，正负样本的匹配程度，正负样本的IoU值
        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
                                                             mask_gt)

        # 对一个正样本匹配多个真实框的情况进行调整
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool()

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):

        # align_metric: 预测框和真实框的匹配程度、overlaps: 预测框和真实框的IoU值
        # get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        # 筛选锚点在真实框内的预测框, coarse select
        # get in_gts mask, (b, max_num_obj, h*w)
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # 由于为了使每张图片真实框的数量进行对齐，进行了补 0 操作，mask_gt 用于确定有效真实框
        # get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric * mask_in_gts,
                                                topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        # merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):

        gt_labels = gt_labels.to(torch.long)  # b, max_num_obj, 1
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # pd_scores[ind[0]] 将每个batch的生成的预测框的重复 max_num_obj 次 size 大小变为 b*max_num_obj*num_total_anchors*num_classes
        # bbox_scores 的 size 为 b*max_num_obj*num_total_anchors，ind[1] 对类别进行得分进行选取
        # get the scores of each grid for each gt cls
        bbox_scores = pd_scores[ind[0], :, ind[1]]  # b, max_num_obj, h*w

        # overlaps 的 size 为 b*max_num_obj*num_total_anchors
        # gt_bboxes.unsqueeze(2) 的 size 为 b*max_num_obj*1*4
        # pd_bboxes.unsqueeze(1) 的 size 为 b*1*num_total_anchors*4
        # bbox_iou 的计算结果 的 size 为 b*max_num_obj*num_total_anchors*1，所以进行维度的压缩
        overlaps = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False, CIoU=True).squeeze(3).clamp(0)
        # 预测框和真实框的匹配程度 = 预测类别分值**alpha × 预测框和真实框的ciou值**beta
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Args:
            metrics: (b, max_num_obj, h*w).
            topk_mask: (b, max_num_obj, topk) or None
        """

        num_anchors = metrics.shape[-1]  # h*w
        # 第一个值为排序的数组，第二个值为该数组中获取到的元素在原数组中的位置标号。
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        # 如果没有给出有效真实框的mask，通过真实框和预测框的匹配程度确定真实框的有效性
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        # (b, max_num_obj, topk)
        topk_idxs = torch.where(topk_mask, topk_idxs, 0)
        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)
        # filter invalid bboxes
        # assigned topk should be unique, this is for dealing with empty labels
        # since empty labels will generate index `0` through `F.one_hot`
        # NOTE: but what if the topk_idxs include `0`?
        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Args:
            gt_labels: (b, max_num_obj, 1)
            gt_bboxes: (b, max_num_obj, 4)
            target_gt_idx: (b, h*w)
            fg_mask: (b, h*w)
        """

        # assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # assigned target boxes, (b, max_num_obj, 4) -> (b, h*w)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        # assigned target scores
        target_labels.clamp(0)
        target_scores = F.one_hot(target_labels, self.num_classes)  # (b, h*w, 80)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores
