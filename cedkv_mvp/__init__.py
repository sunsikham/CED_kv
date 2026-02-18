from __future__ import annotations

import pkgutil
from pathlib import Path

__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]
_src_pkg = Path(__file__).resolve().parents[1] / "src" / "cedkv_mvp"
if _src_pkg.exists():
    __path__.append(str(_src_pkg))  # type: ignore[attr-defined]

