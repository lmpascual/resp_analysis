"""Run Kilosort 4 and Lupin on the shared preprocessed recording.

The pipeline sets the handful of parameters that must agree with the preprocessing
(referencing, drift correction, Lupin's internal filtering/whitening). Everything else is the
sorter's own default unless the experiment config overrides it. No sorter internals are patched.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import spikeinterface.core as si
import spikeinterface.sorters as ss

from .config import ConfigError

log = logging.getLogger(__name__)


def kilosort4_params(cfg: dict, referenced: bool) -> dict:
    user = dict(cfg["sorters"]["kilosort4"]["params"])
    params = {
        # CatGT/our CMR already referenced the data; a second CAR would only remove signal.
        "do_CAR": not referenced,
        # If SpikeInterface corrected drift, Kilosort must not correct it again.
        "do_correction": not cfg["motion"]["enabled"],
        "torch_device": "auto",
        "save_extra_vars": False,
    }
    params.update(user)
    if params["torch_device"] == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "sorters.kilosort4.params.torch_device is 'cuda' but torch cannot see a CUDA GPU. "
                "Install the CUDA build of torch (see README) or run check_env.py."
            )
    return params


def lupin_params(cfg: dict, job_kwargs: dict) -> dict:
    user = dict(cfg["sorters"]["lupin"]["params"])
    if "preprocessing_dict" in user and user["preprocessing_dict"] is None:
        raise ConfigError(
            "sorters.lupin.params.preprocessing_dict = null would make Lupin re-filter and re-reference "
            "the already preprocessed data. Leave it unset."
        )
    params = {
        # Keep Lupin's whitening (it needs it) but replace its own filter + CMR with a gentle
        # 300-7000 Hz band-pass, because referencing was already done upstream.
        "apply_preprocessing": True,
        "preprocessing_dict": [
            {
                "name": "bandpass_filter",
                "kwargs": {"freq_min": 300.0, "freq_max": 7000.0, "ftype": "bessel", "filter_order": 2, "dtype": "float32"},
            }
        ],
        "apply_motion_correction": False if cfg["motion"]["enabled"] else True,
        "seed": cfg["postprocessing"]["seed"],
        "job_kwargs": dict(job_kwargs),
    }
    params.update(user)
    return params


def run_sorter(name: str, rec: si.BaseRecording, cfg: dict, job_kwargs: dict, referenced: bool) -> si.BaseSorting:
    scratch = Path(cfg["paths"]["scratch_dir"]) / "sorters" / name
    out = Path(cfg["paths"]["output_dir"]) / name
    out.mkdir(parents=True, exist_ok=True)

    if name == "kilosort4":
        params = kilosort4_params(cfg, referenced)
    elif name == "lupin":
        params = lupin_params(cfg, job_kwargs)
    else:
        raise ConfigError(f"Unknown sorter {name}")

    with open(out / "sorter_params_used.json", "w") as f:
        json.dump(params, f, indent=2, default=str)
    log.info("Running %s with %s", name, {k: v for k, v in params.items() if k != "job_kwargs"})

    sorting = ss.run_sorter(
        sorter_name=name,
        recording=rec,
        folder=scratch,
        remove_existing_folder=True,
        verbose=True,
        raise_error=True,
        **params,
    )
    sorting = sorting.save(folder=out / "sorting_raw", overwrite=True)
    log.info("%s found %d units", name, len(sorting.unit_ids))
    return sorting
