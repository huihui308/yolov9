
# Inference

## gelan


## yolov9
```
$ python3 detect.py --weights weights/yolov9-c.pt --conf 0.1 --source david/test/cq.mp4 --device 0 --img 1280

$ !python detect.py \
--img 1280 --conf 0.1 --device 0 \
--weights {HOME}/yolov9/runs/train/exp/weights/best.pt \
--source {dataset.location}/valid/images


$ python3 detect.py --weights runs/train/primary7_yolov9_c_300epoch/weights/best_striped.pt --conf 0.1 --source david/test/cq.mp4 --device 0

```

# Train

## gelan
```
$ python3 train.py \
--batch 8 --epochs 300 --img 640 --device 0 --min-items 0 --close-mosaic 15 \
--data david/data/primary7.yaml \
--cfg david/models/detect/primary7-gelan-c.yaml \
--hyp hyp.scratch-high.yaml


# test
python3 train.py \
--batch 8 --epochs 20 --img 640 --device 0 --min-items 0 --close-mosaic 15 \
--data david/data/primary7.yaml \
--weights runs/train/primary7_gelan_c_300epoch_finetune/weights/best.pt \
--cfg david/models/detect/primary7-gelan-c.yaml \
--hyp hyp.scratch-high.yaml

```




## yolov9
```
$ python3 train_dual.py \
--batch 8 --epochs 300 --img 640 --device 0 --min-items 0 --close-mosaic 15 \
--data david/data/primary7.yaml \
--weights weights/yolov9-c.pt \
--cfg models/detect/primary7-yolov9-c.yaml \
--hyp hyp.scratch-high.yaml

$ python3 train_dual.py \
--batch 2 --epochs 500 --img 640 --device 0 --min-items 0 --close-mosaic 15 \
--data david/data/primary7.yaml \
--cfg david/models/detect/primary7-yolov9-c.yaml \
--hyp hyp.scratch-high.yaml
```




python3 train_dual.py \
--batch 2 --epochs 20 --img 640 --device 0 --min-items 0 --close-mosaic 15 \
--data david/data/primary7.yaml \
--cfg david/models/detect/primary7-yolov9-c.yaml \
--hyp hyp.scratch-high.yaml


python3 train_dual.py \
--batch 2 --epochs 20 --img 640 --device 0 --min-items 0 --close-mosaic 15 \
--data david/data/primary7.yaml \
--weights runs/train/primary7_yolov9_c_245epoch_scratch/weights/best.pt \
--cfg david/models/detect/primary7-yolov9-c.yaml \
--hyp hyp.scratch-high.yaml