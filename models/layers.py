from __future__ import annotations

from ultralytics.nn.modules import C2f, SPPF, Detect


class CustomC2f(C2f):
    """C2f block with extension points for custom logic."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        # TODO: add custom layers or parameters here.

    def forward(self, x):
        # TODO: customize forward if needed.
        return super().forward(x)


class CustomSPPF(SPPF):
    """SPPF block with extension points for custom logic."""

    def __init__(self, c1, c2, k=5):
        super().__init__(c1, c2, k=k)
        # TODO: add custom layers or parameters here.

    def forward(self, x):
        # TODO: customize forward if needed.
        return super().forward(x)


class CustomDetect(Detect):
    """Detection head with extension points for custom logic."""

    def __init__(self, nc=80, ch=(), **kwargs):
        super().__init__(nc=nc, ch=ch, **kwargs)
        # TODO: add custom layers or parameters here.

    def forward(self, x):
        # TODO: customize forward if needed.
        return super().forward(x)
