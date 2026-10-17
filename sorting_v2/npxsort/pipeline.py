"""Run the whole pipeline with resumable stages.

Stages (in order): preprocess -> sort -> postprocess -> compare -> export.
Each finished stage writes a marker in <output_dir>/.stages/. Re-running skips finished stages;
`redo` re-runs a stage and everything after it.
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import spikeinterface
import spikeinterface.core as si

from .config import save_config
from .preprocess import preprocess, data_is_referenced
from .sorters import run_sorter
from .postprocess import postprocess_sorter
from .compare import compare_sorters
from .export import export_sorter

log = logging.getLogger("npxsort")

STAGES = ["preprocess", "sort", "postprocess", "compare", "export"]


def _setup_logging(out_dir: Path) -> Path:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    logging.captureWarnings(True)
    return log_file


def _environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "spikeinterface": spikeinterface.__version__,
    }
    for pkg in ("numpy", "scipy", "sklearn", "numba", "torch", "kilosort", "probeinterface"):
        try:
            mod = __import__(pkg)
            env[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env[pkg] = None
    try:
        import torch

        env["cuda_available"] = bool(torch.cuda.is_available())
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        env["cuda_available"] = False
    try:
        repo = Path(__file__).resolve().parents[2]
        env["git_commit"] = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        env["git_commit"] = None
    return env


class Stages:
    def __init__(self, out_dir: Path):
        self.dir = out_dir / ".stages"
        self.dir.mkdir(parents=True, exist_ok=True)

    def done(self, key: str) -> bool:
        return (self.dir / f"{key}.done").exists()

    def mark(self, key: str) -> None:
        (self.dir / f"{key}.done").write_text(datetime.now().isoformat())

    def clear_from(self, stage: str) -> None:
        for later in STAGES[STAGES.index(stage):]:
            for f in self.dir.glob(f"{later}*.done"):
                f.unlink()


def run_pipeline(cfg: dict, redo: str | None = None, stop_after: str | None = None) -> dict:
    out_dir = Path(cfg["paths"]["output_dir"])
    scratch = Path(cfg["paths"]["scratch_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    log_file = _setup_logging(out_dir)

    job_kwargs = dict(cfg["job_kwargs"])
    si.set_global_job_kwargs(**job_kwargs)
    job_kwargs = si.get_global_job_kwargs()

    stages = Stages(out_dir)
    if redo:
        stages.clear_from(redo)
    save_config(cfg, out_dir / "config_used.yaml")

    sorters = [name for name, s in cfg["sorters"].items() if s["enabled"]]
    manifest_path = out_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({
        "experiment": cfg["experiment_name"],
        "config_file": cfg["_config_file"],
        "log_file": str(log_file),
        "environment": _environment(),
        "job_kwargs": {k: str(v) for k, v in job_kwargs.items()},
        "sorters": sorters,
    })
    manifest.setdefault("timings_s", {})

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    def timed(key, fn, *a, **k):
        t = time.time()
        result = fn(*a, **k)
        manifest["timings_s"][key] = round(time.time() - t, 1)
        save_manifest()
        return result

    log.info("Experiment %s | sorters %s | output %s", cfg["experiment_name"], sorters, out_dir)
    save_manifest()

    # 1. preprocess
    if stages.done("preprocess"):
        rec = si.load(scratch / "preprocessed")
        log.info("preprocess: already done, loaded %s", scratch / "preprocessed")
    else:
        rec, info = timed("preprocess", preprocess, cfg, job_kwargs)
        manifest["preprocessing"] = info
        stages.mark("preprocess")
    if stop_after == "preprocess":
        return manifest

    referenced = data_is_referenced(cfg)

    # 2. sort
    sortings = {}
    for name in sorters:
        key = f"sort_{name}"
        if stages.done(key):
            sortings[name] = si.load(out_dir / name / "sorting_raw")
            log.info("%s: already sorted", name)
        else:
            sortings[name] = timed(key, run_sorter, name, rec, cfg, job_kwargs, referenced)
            stages.mark(key)
    if stop_after == "sort":
        return manifest

    # 3. postprocess + label
    analyzers, tables = {}, {}
    for name in sorters:
        key = f"postprocess_{name}"
        if stages.done(key):
            analyzers[name] = si.load_sorting_analyzer(out_dir / name / "analyzer")
            tables[name] = pd.read_csv(out_dir / name / "units.tsv", sep="\t", index_col="unit_id")
            tables[name].index = analyzers[name].unit_ids
            log.info("%s: already postprocessed", name)
        else:
            analyzers[name], tables[name] = timed(key, postprocess_sorter, name, sortings[name], rec, cfg, job_kwargs)
            stages.mark(key)
        manifest.setdefault("label_counts", {})[name] = {k: int(v) for k, v in tables[name]["label"].value_counts().items()}
    save_manifest()
    if stop_after == "postprocess":
        return manifest

    # 4. compare
    if cfg["comparison"]["enabled"] and len(sorters) >= 2:
        result = timed("compare", compare_sorters, analyzers, tables, cfg)
        for name, df in result["per_unit"].items():
            tables[name] = tables[name].drop(columns=[c for c in df.columns if c in tables[name].columns]).join(df, how="left")
        manifest["comparison"] = result["summary"]
        stages.mark("compare")
    if stop_after == "compare":
        save_manifest()
        return manifest

    # 5. export
    t0_s = float(cfg["input"]["start_s"] or 0.0)
    for name in sorters:
        key = f"export_{name}"
        if stages.done(key):
            continue
        timed(key, export_sorter, name, analyzers[name], tables[name], cfg, t0_s, job_kwargs)
        stages.mark(key)

    save_manifest()
    log.info("Done. Summary: labels %s | comparison %s", manifest.get("label_counts"), manifest.get("comparison"))
    return manifest
