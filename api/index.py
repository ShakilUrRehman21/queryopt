import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_packages_dir = _repo_root / "packages"
for _p in (_repo_root, _packages_dir):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

from apps.api.main import app
