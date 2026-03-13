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

```
python train_caps.py --model configs/model/yolo26_capsneckhead_v2.yaml --data configs/data/coco.yaml --epochs 100 --imgsz 640 --batch 64 --pretrained runs\detect\runs\train\CapsNeckHead_CoCo\weights\best.pt --optimizer AdamW --warmup_epochs 5 --name CapsNeckHeadv1_CoCo
```


next step
```
python train_caps.py --model configs/det_model/yolo26_capsneckhead_v6.yaml --data configs/data/coco.yaml --epochs 30 --imgsz 640 --batch 32 --workers 8 --pretrained runs\detect\runs\train\CapsNeckHeadv6_Seg_CoCo_Pretrained\weights\best.pt --optimizer AdamW --lr0 2e-4 --lrf 0.01 --warmup_epochs 1 --weight_decay 3e-4 --project runs\detect\runs\train --name CapsNeckHeadv6_DetFromSeg --exist-ok
```