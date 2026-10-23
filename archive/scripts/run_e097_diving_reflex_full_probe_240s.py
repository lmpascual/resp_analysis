"""
Backwards-compatibility shim.
Canonical location: scripts/experiments/e097/run_e097_diving_reflex_full_probe_240s.py
"""
import sys
import runpy
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TARGET = Path(__file__).resolve().parent / "experiments" / "e097" / "run_e097_diving_reflex_full_probe_240s.py"

if __name__ == "__main__":
    runpy.run_path(str(_TARGET), run_name="__main__")
