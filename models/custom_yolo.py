from __future__ import annotations

from .layers import CustomC2f, CustomSPPF, CustomDetect
from modules import DeformableCapsBlock


CUSTOM_MODULES = {
    "CustomC2f": CustomC2f,
    "CustomSPPF": CustomSPPF,
    "CustomDetect": CustomDetect,
    "DeformableCapsBlock": DeformableCapsBlock,
}


def register_ultralytics_modules() -> None:
    """Register custom modules so Ultralytics can resolve them in YAML."""
    import ultralytics.nn.tasks as nn_tasks
    import ultralytics.nn.modules as nn_modules

    for name, cls in CUSTOM_MODULES.items():
        setattr(nn_tasks, name, cls)
        setattr(nn_modules, name, cls)
