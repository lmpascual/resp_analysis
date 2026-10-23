"""Load the recording, remove bad channels, correct drift, and save one preprocessed binary.

Both sorters and all quality metrics read this same binary, so every downstream number refers
to exactly the data the sorters saw.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import spikeinterface.core as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

log = logging.getLogger(__name__)


def load_input(cfg: dict) -> si.BaseRecording:
    inp = cfg["input"]
    folder = Path(inp["folder"])
    if not folder.exists():
        raise FileNotFoundError(f"input.folder does not exist: {folder}")

    if inp["source"] in ("catgt", "spikeglx_raw"):
        rec = se.read_spikeglx(folder, stream_id=inp["stream_id"])
    else:  # si_folder
        rec = si.load(folder)

    if rec.get_probe() is None:
        raise RuntimeError("Recording has no probe geometry attached; cannot sort.")

    start_s, duration_s = inp["start_s"], inp["duration_s"]
    if start_s is not None or duration_s is not None:
        fs = rec.get_sampling_frequency()
        start = int(round((start_s or 0.0) * fs))
        end = rec.get_num_samples() if duration_s is None else start + int(round(duration_s * fs))
        if end > rec.get_num_samples():
            raise ValueError(
                f"Requested window ends at {end / fs:.2f} s but the recording is only "
                f"{rec.get_total_duration():.2f} s long."
            )
        rec = rec.frame_slice(start_frame=start, end_frame=end)

    log.info(
        "Loaded %s: %d channels, %.1f Hz, %.2f s",
        folder, rec.get_num_channels(), rec.get_sampling_frequency(), rec.get_total_duration(),
    )
    return rec


def _bad_channels(rec: si.BaseRecording, cfg: dict, out_dir: Path, job_kwargs: dict, already_referenced: bool):
    bc = cfg["preprocessing"]["bad_channels"]
    if not bc["detect"]:
        return rec, pd.DataFrame()

    seed = cfg["postprocessing"]["seed"]
    noise = si.get_noise_levels(rec, return_in_uV=False, method="mad", random_slices_kwargs={"seed": seed}, **job_kwargs)
    ratio = noise / np.median(noise)

    if already_referenced:
        # The IBL coherence method compares each channel to the probe-wide median signal, which a
        # common/local reference (e.g. CatGT -loccar) has already removed; on such data it labels
        # active channels "dead". Use noise level instead: flat channels are dead, very loud ones noisy.
        method = "noise_level_ratio"
        labels = np.full(rec.get_num_channels(), "good", dtype="U5")
        labels[ratio < bc["flat_ratio"]] = "dead"
        labels[ratio > bc["noisy_ratio"]] = "noise"
    else:
        method = "coherence+psd"
        _, labels = spre.detect_bad_channels(rec, method="coherence+psd", seed=seed, job_kwargs=job_kwargs)

    locs = rec.get_channel_locations()
    table = pd.DataFrame(
        {
            "channel_id": rec.channel_ids,
            "label": labels,
            "noise_mad": noise,
            "noise_ratio_to_median": ratio,
            "x_um": locs[:, 0],
            "y_um": locs[:, 1],
        }
    )
    table.to_csv(out_dir / "bad_channels.csv", index=False)
    counts = table["label"].value_counts().to_dict()
    log.info("Bad-channel detection (%s): %s", method, counts)

    if bc["action"] == "keep" or (table["label"] == "good").all():
        return rec, table

    out_ids = table.loc[table["label"] == "out", "channel_id"].to_numpy()
    dead_noisy = table.loc[table["label"].isin(["dead", "noise"]), "channel_id"].to_numpy()

    if bc["action"] == "interpolate" and len(dead_noisy) > 0:
        rec = spre.interpolate_bad_channels(rec, bad_channel_ids=dead_noisy)
        to_remove = out_ids
    else:
        to_remove = np.concatenate([out_ids, dead_noisy])

    if len(to_remove) > 0:
        rec = rec.remove_channels(to_remove)
    log.info("Channels after bad-channel handling: %d", rec.get_num_channels())
    return rec, table


def _estimate_motion_kwargs(m: dict) -> dict:
    """User estimate_motion_kwargs, plus a fix for a SpikeInterface preset bug.

    In SpikeInterface 0.105.x the 'dredge' preset passes batching_mode='offline', which the DREDGE
    estimator rejects (it accepts 'full' or 'online'). 'full' is DREDGE's own default.
    """
    from spikeinterface.preprocessing.motion import motion_options_preset

    kwargs = dict(m["estimate_motion_kwargs"])
    preset_kwargs = motion_options_preset.get(m["preset"], {}).get("estimate_motion_kwargs", {})
    if preset_kwargs.get("batching_mode") == "offline" and "batching_mode" not in kwargs:
        kwargs["batching_mode"] = "full"
        log.warning("Motion preset '%s' sets batching_mode='offline' (invalid for DREDGE); using 'full'.", m["preset"])
    return kwargs


def _motion(rec: si.BaseRecording, cfg: dict, out_dir: Path, job_kwargs: dict):
    m = cfg["motion"]
    motion_folder = out_dir / "motion"
    rec_corr, motion_info = spre.correct_motion(
        rec,
        preset=m["preset"],
        folder=motion_folder,
        output_motion_info=True,
        overwrite=True,
        detect_kwargs=m["detect_kwargs"],
        estimate_motion_kwargs=_estimate_motion_kwargs(m),
        interpolate_motion_kwargs=m["interpolate_motion_kwargs"],
        **job_kwargs,
    )
    motion = motion_info["motion"]
    disp = np.concatenate([np.asarray(d).ravel() for d in motion.displacement])
    summary = {
        "preset": m["preset"],
        "max_abs_displacement_um": float(np.max(np.abs(disp))),
        "median_abs_displacement_um": float(np.median(np.abs(disp))),
        "displacement_range_um": float(np.ptp(disp)),
    }
    with open(out_dir / "motion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Motion: max |displacement| %.1f um, range %.1f um", summary["max_abs_displacement_um"], summary["displacement_range_um"])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import spikeinterface.widgets as sw

        w = sw.plot_motion_info(motion_info, rec, color_amplitude=True, scatter_decimate=5, backend="matplotlib")
        w.figure.set_size_inches(16, 9)
        w.figure.savefig(out_dir / "motion.png", dpi=110)
        plt.close(w.figure)
    except Exception:  # a figure failing must not stop sorting, but it must be visible
        log.exception("Could not draw the motion figure (sorting continues).")

    return rec_corr, summary


def preprocess(cfg: dict, job_kwargs: dict) -> tuple[si.BaseRecording, dict]:
    """Run preprocessing and save the result. Returns (saved recording, info dict)."""
    out_dir = Path(cfg["paths"]["output_dir"]) / "preprocessing"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_folder = Path(cfg["paths"]["scratch_dir"]) / "preprocessed"
    pp = cfg["preprocessing"]
    source = cfg["input"]["source"]

    rec = load_input(cfg)
    info = {
        "source": source,
        "n_channels_in": int(rec.get_num_channels()),
        "duration_s": float(rec.get_total_duration()),
        "sampling_frequency": float(rec.get_sampling_frequency()),
        "steps": [],
    }

    if source == "spikeglx_raw":
        if pp["phase_shift"]:
            if "inter_sample_shift" not in rec.get_property_keys():
                raise RuntimeError("phase_shift requested but the recording has no 'inter_sample_shift' property.")
            rec = spre.phase_shift(rec)
            info["steps"].append("phase_shift")
        rec = spre.highpass_filter(rec, freq_min=pp["highpass_hz"], dtype="float32")
        info["steps"].append(f"highpass_{pp['highpass_hz']}Hz")

    rec, bad_table = _bad_channels(rec, cfg, out_dir, job_kwargs, already_referenced=source != "spikeglx_raw")
    if len(bad_table):
        info["steps"].append(f"bad_channels_{pp['bad_channels']['action']}")
        info["bad_channel_counts"] = {k: int(v) for k, v in bad_table["label"].value_counts().items()}

    if source == "spikeglx_raw" and pp["common_reference"] != "none":
        if pp["common_reference"] == "global":
            rec = spre.common_reference(rec, reference="global", operator="median")
        else:
            rec = spre.common_reference(rec, reference="local", operator="median", local_radius=tuple(pp["local_radius_um"]))
        info["steps"].append(f"common_reference_{pp['common_reference']}")

    if cfg["motion"]["enabled"]:
        rec, motion_summary = _motion(rec, cfg, out_dir, job_kwargs)
        info["steps"].append(f"motion_{cfg['motion']['preset']}")
        info["motion"] = motion_summary

    log.info("Saving preprocessed binary (%s) to %s", pp["save_dtype"], save_folder)
    rec_saved = rec.save(folder=save_folder, format="binary", dtype=pp["save_dtype"], overwrite=True, **job_kwargs)
    info["n_channels_out"] = int(rec_saved.get_num_channels())
    info["preprocessed_folder"] = str(save_folder)

    with open(out_dir / "preprocessing_info.json", "w") as f:
        json.dump(info, f, indent=2)
    return rec_saved, info


def data_is_referenced(cfg: dict) -> bool:
    """Whether the data the sorters receive already has a common reference applied."""
    source = cfg["input"]["source"]
    if source == "catgt":
        return True  # CatGT -loccar / -gblcar
    if source == "si_folder":
        return True  # documented assumption: si_folder input is already preprocessed
    return cfg["preprocessing"]["common_reference"] != "none"
