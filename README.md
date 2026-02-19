# CSNet
This is repository for end-to-end object detection by a capsule neural symobilic network.

## How to run


Run YOLO26
```
python train_caps.py --model yolo26n.pt --data configs/data/nematode.yaml --pretrained yolo26n.pt --epochs 1
```

Run YOLO26_capshead
```
python train_caps.py --model configs/model/yolo26_capshead.yaml --data configs/data/nematode.yaml --pretrained yolo26n.pt --epochs 1
```

Run YOLO26_capsneck
```
python train_caps.py --model configs/model/yolo26_capsneck.yaml --data configs/data/nematode.yaml --epochs 100 --imgsz 640
```

```
python train_caps.py --model configs/model/yolo26_capsneck.yaml --data configs/data/coco.yaml --epochs 50 --imgsz 640 --batch 64 --pretrained runs\detect\runs\train\CapsNeck_CoCo2\weights\best.pt --optimizer AdamW --lr0 5e-4 --lrf 0.001 --warmup_epochs 3 --warmup_bias_lr 0.0 --weight_decay 0.0003 --name CapsNeck_CoCo
```

```
python train_caps.py --model configs/model/yolo26_capsneck.yaml --data configs/data/coco.yaml --epochs 50 --imgsz 640 --batch 64 --pretrained runs\detect\runs\train\CapsNeck_CoCo2\weights\best.pt --optimizer AdamW --lr0 5e-4 --lrf 0.001 --warmup_epochs 5 --name CapsNeck_CoCo
```

```
python train_caps.py --model configs/model/yolo26_capsneck.yaml --data configs/data/coco.yaml --epochs 50 --imgsz 640 --batch 64 --pretrained runs\detect\runs\train\CapsNeck_CoCo3\weights\best.pt --optimizer AdamW --lr0 5e-4 --lrf 0.001 --warmup_epochs 5 --name CapsNeck_CoCo
```