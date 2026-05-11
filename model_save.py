from ultralytics import YOLO
model = YOLO("runs\CapsNeckHeadv6_SegV2_AuxDet_Mild\weights\\best.pt")
model.save("symbolic_capsule_network_segmentation.pt")