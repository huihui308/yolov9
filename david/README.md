
# Inference

```
$ python3 detect.py --weights weights/yolov9-c.pt --conf 0.1 --source data/cq.mp4 --device 0 --img 1280

$ !python detect.py \
--img 1280 --conf 0.1 --device 0 \
--weights {HOME}/yolov9/runs/train/exp/weights/best.pt \
--source {dataset.location}/valid/images
```

# Train
```
$ python3 train.py \
--batch 8 --epochs 300 --img 640 --device 0 --min-items 0 --close-mosaic 15 \
--data david/data/primary7.yaml \
--weights weights/yolov9-c.pt \
--cfg models/detect/gelan-c.yaml \
--hyp hyp.scratch-high.yaml
```



