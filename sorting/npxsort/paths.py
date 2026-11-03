"""Where the data live on this computer, from configs/local_paths.yaml at the repository root.

Every config and script names its folders relative to four roots, so the same files work on any computer:

    raw_data   SpikeGLX run folders as recorded (e.g. <raw_data>/e097_g2/e097_g2_t0.nidq.bin)
    catgt      CatGT output folders (the sorting input)
    scratch    sorting scratch space: preprocessed binaries and sorter folders (~10 GB per window)
    outputs    everything the pipeline and the analyses write

The file may define more roots for data kept elsewhere (e.g. dynaresp: the processed folder of the Bush & Ramirez
2024 recordings); only configs that use them need them. Configs write roots as ${raw_data}, ${outputs}, ...;
scripts call path("outputs") etc. Copy configs/local_paths.example.yaml to configs/local_paths.yaml and edit it (that
file isn't in git). The RESP_ANALYSIS_PATHS environment variable can point to a different file.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
KEYS = ("raw_data", "catgt", "scratch", "outputs")     # required in every local_paths.yaml
_VAR = re.compile(r"\$\{(\w+)\}")


def paths_file() -> Path:
    return Path(os.environ.get("RESP_ANALYSIS_PATHS", REPO / "configs" / "local_paths.yaml"))


@lru_cache(maxsize=1)
def local_paths() -> dict[str, Path]:
    f = paths_file()
    if not f.exists():
        raise FileNotFoundError(
            f"{f} not found. Copy configs/local_paths.example.yaml to configs/local_paths.yaml and set the "
            "folders for this computer.")
    with open(f, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    missing = [k for k in KEYS if not raw.get(k)]
    if missing:
        raise ValueError(f"{f}: set {', '.join(missing)}")
    bad = sorted(k for k, v in raw.items() if not isinstance(v, str) or not re.fullmatch(r"\w+", k))
    if bad:
        raise ValueError(f"{f}: {bad} must be names (letters, digits, _) set to a folder")
    return {k: Path(v) for k, v in raw.items()}


def path(name: str) -> Path:
    """One root folder: path("outputs"), path("raw_data"), or an extra root defined in local_paths.yaml."""
    roots = local_paths()
    if name not in roots:
        raise KeyError(f"{name!r} isn't set in {paths_file()}; it has {sorted(roots)}")
    return roots[name]


def relocated(recorded, root: str = "scratch") -> Path:
    """A folder a run recorded as an absolute path (config_used.yaml's scratch_dir), possibly on another computer:
    the recorded path if it exists here, else the folder of the same name under this computer's root."""
    p = Path(recorded)
    if p.exists():
        return p
    try:
        alt = path(root) / p.name
    except (FileNotFoundError, ValueError):
        return p
    return alt if alt.exists() else p


def expand(value):
    """Replace ${raw_data}, ${outputs}, ... (any root in local_paths.yaml) in a string, or in every string of a
    dict / list."""
    if isinstance(value, str):
        if "${" not in value:
            return value
        return _VAR.sub(lambda m: path(m.group(1)).as_posix(), value)
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    return value
