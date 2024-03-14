"""
    python3 yolo9c.py
"""

import torch
import sys
sys.path.append('./../../')

from models.yolo import Model


device = torch.device("cpu")
cfg = "./../models/detect/primary7-gelan-c.yaml"
#cfg = "./../../models/detect/yolov9-c.yaml"
model = Model(cfg, ch=3, nc=7, anchors=3)
#model = model.half()
model = model.to(device)
_ = model.eval()
ckpt = torch.load('./../../runs/train/primary7_yolov9_c_245epoch_scratch/weights/best.pt', map_location='cpu')
#ckpt = torch.load('./../../weights/yolov9-c.pt', map_location='cpu')
model.names = ckpt['model'].names
model.nc = ckpt['model'].nc


idx = 0
for k, v in model.state_dict().items():
    print(idx, k)
    if "model.{}.".format(idx) in k:
        if idx < 22:
            kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
        elif "model.{}.cv2.".format(idx) in k:
            kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
        elif "model.{}.cv3.".format(idx) in k:
            kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
        elif "model.{}.dfl.".format(idx) in k:
            kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
    else:
        while True:
            idx += 1
            if "model.{}.".format(idx) in k:
                break
        if idx < 22:
            kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
        elif "model.{}.cv2.".format(idx) in k:
            kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
        elif "model.{}.cv3.".format(idx) in k:
            kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
        elif "model.{}.dfl.".format(idx) in k:
            kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
_ = model.eval()


m_ckpt = {'model': model.half(),
          'optimizer': None,
          'best_fitness': None,
          'ema': None,
          'updates': None,
          'opt': None,
          'git': None,
          'date': None,
          'epoch': -1}
torch.save(m_ckpt, "./primary7-c-converted.pt")