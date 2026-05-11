from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import CapsuleOVD


def main() -> None:
    model = CapsuleOVD("runs/detect/caps_ovd_debug_smoke/weights/best.pt", verbose=False)
    model.train(
        data="configs/data/ovd_coco_grounding_lvis_local.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        patience=20,
        optimizer="AdamW",
        lr0=5e-5,
        lrf=0.1,
        warmup_epochs=5,
        warmup_bias_lr=0.0,
        weight_decay=5e-4,
        name="caps_ovd_coco_grounding_lvis_formal_b16",
    )


if __name__ == "__main__":
    main()
