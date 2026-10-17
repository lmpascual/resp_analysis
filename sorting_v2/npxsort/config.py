"""Load, validate and save pipeline configuration.

The experiment YAML is merged over ``defaults.yaml``. Any key that does not exist in the defaults
raises an error, except inside the free-form sections listed in ``FREEFORM`` (sorter parameters,
job kwargs, motion/bombcell overrides), which are passed through to SpikeInterface as-is.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS_PATH = Path(__file__).with_name("defaults.yaml")

FREEFORM = {
    "job_kwargs",
    "sorters.kilosort4.params",
    "sorters.lupin.params",
    "motion.detect_kwargs",
    "motion.estimate_motion_kwargs",
    "motion.interpolate_motion_kwargs",
    "curation.threshold_overrides",
}

CHOICES = {
    "input.source": {"catgt", "spikeglx_raw", "si_folder"},
    "preprocessing.common_reference": {"global", "local", "none"},
    "preprocessing.bad_channels.action": {"remove", "interpolate", "keep"},
    "curation.method": {"bombcell"},
    "job_kwargs.pool_engine": {"process", "thread"},
}


class ConfigError(ValueError):
    pass


def _merge(base: dict, override: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        key_path = f"{path}.{key}" if path else key
        if path in FREEFORM:
            out[key] = copy.deepcopy(value)
            continue
        if key not in base:
            raise ConfigError(f"Unknown config key '{key_path}'. Check spelling against npxsort/defaults.yaml.")
        if isinstance(base[key], dict) and key_path not in FREEFORM:
            if not isinstance(value, dict):
                raise ConfigError(f"Config key '{key_path}' must be a mapping.")
            out[key] = _merge(base[key], value, key_path)
        elif isinstance(base[key], dict):  # free-form section: take the user's mapping wholesale
            if not isinstance(value, dict):
                raise ConfigError(f"Config key '{key_path}' must be a mapping.")
            out[key] = copy.deepcopy(value) if not base[key] else {**base[key], **value}
        else:
            out[key] = copy.deepcopy(value)
    return out


def _get(cfg: dict, dotted: str) -> Any:
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def _validate(cfg: dict) -> None:
    for dotted, allowed in CHOICES.items():
        value = _get(cfg, dotted)
        if value not in allowed:
            raise ConfigError(f"'{dotted}' is '{value}'; allowed: {sorted(allowed)}")
    for dotted in ("paths.output_dir", "paths.scratch_dir", "input.folder"):
        if not _get(cfg, dotted):
            raise ConfigError(f"'{dotted}' must be set in the experiment config.")
    if not any(s["enabled"] for s in cfg["sorters"].values()):
        raise ConfigError("Enable at least one sorter.")

    # Prevent double drift correction: if SpikeInterface corrects motion, the sorters must not.
    if cfg["motion"]["enabled"]:
        if cfg["sorters"]["kilosort4"]["params"].get("do_correction", False):
            raise ConfigError("motion.enabled is true, so sorters.kilosort4.params.do_correction must not be true.")
        if cfg["sorters"]["lupin"]["params"].get("apply_motion_correction", False):
            raise ConfigError("motion.enabled is true, so sorters.lupin.params.apply_motion_correction must not be true.")

    win = cfg["input"]
    if win["start_s"] is not None and win["start_s"] < 0:
        raise ConfigError("input.start_s must be >= 0.")
    if win["duration_s"] is not None and win["duration_s"] <= 0:
        raise ConfigError("input.duration_s must be > 0.")


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
        defaults = yaml.safe_load(f)
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    cfg = _merge(defaults, user)
    _validate(cfg)
    cfg["paths"]["output_dir"] = str(Path(cfg["paths"]["output_dir"]))
    cfg["paths"]["scratch_dir"] = str(Path(cfg["paths"]["scratch_dir"]))
    cfg["input"]["folder"] = str(Path(cfg["input"]["folder"]))
    cfg["_config_file"] = str(path.resolve())
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
