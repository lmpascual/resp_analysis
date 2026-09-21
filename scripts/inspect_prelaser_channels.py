"""
Backwards-compatibility shim.
Canonical location: scripts/experiments\e097\inspect_prelaser_channels.py
"""
import sys
import runpy
from pathlib import Path

# Ensure repository root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Target script path
_TARGET = Path(__file__).resolve().parent / "experiments\e097\inspect_prelaser_channels.py"

if __name__ == "__main__":
    runpy.run_path(str(_TARGET), run_name="__main__")
else:
    # Forward module attributes for cross-script imports
    import importlib
    _mod = importlib.import_module("scripts.experiments.e097.inspect_prelaser_channels")
    for _k, _v in _mod.__dict__.items():
        if not _k.startswith("__"):
            globals()[_k] = _v
