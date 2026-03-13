from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .run_audit import main
except ImportError:
    from tools.audit_caps.unsupervised.run_audit import main


if __name__ == "__main__":
    main()
