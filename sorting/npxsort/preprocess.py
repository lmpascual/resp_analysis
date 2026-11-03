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


BRIDGE_EDGE_S = 5.0     # seconds of estimate averaged just before and just after each motion.bridge_s window


def _bridge_motion(rec: si.BaseRecording, motion_info: dict, m: dict, out_dir: Path):
    """Motion estimate that skips the motion.bridge_s windows (file time).

    Registration by spike density fails when the spike pattern changes abruptly, as during gasping: fewer spikes, and
    different neurons firing (e097 D2; m2021-31_g4, m2020-19). So the peaks inside each window are left out, the motion
    is estimated again from the rest, and inside each window it is replaced by a straight line between its means over
    the BRIDGE_EDGE_S seconds before and after. Returns the bridged Motion and, per window, the two edge values.
    """
    from spikeinterface.core.motion import Motion
    from spikeinterface.preprocessing.motion import motion_options_preset
    from spikeinterface.sortingcomponents.motion import estimate_motion

    t_start = rec.get_start_time()
    windows = [(t_start + a, t_start + b) for a, b in m["bridge_s"]]
    peaks, locations = motion_info["peaks"], motion_info["peak_locations"]
    t = rec.sample_index_to_time(peaks["sample_index"])
    keep = np.ones(len(peaks), dtype=bool)
    for a, b in windows:
        keep &= ~((t >= a) & (t < b))
    kwargs = dict(motion_options_preset[m["preset"]]["estimate_motion_kwargs"], **_estimate_motion_kwargs(m))
    est = estimate_motion(rec, peaks[keep], locations[keep], **kwargs)
    tb = np.asarray(est.temporal_bins_s[0])
    raw = np.asarray(est.displacement[0], dtype=float)
    d = raw.copy()
    edges = []
    for (a, b), (fa, fb) in zip(windows, m["bridge_s"]):
        before = (tb >= a - BRIDGE_EDGE_S) & (tb < a)
        after = (tb >= b) & (tb < b + BRIDGE_EDGE_S)
        left = raw[before].mean(axis=0) if before.any() else raw[after].mean(axis=0)
        right = raw[after].mean(axis=0) if after.any() else left
        inside = (tb >= a) & (tb < b)
        frac = ((tb[inside] - a) / (b - a))[:, None]
        d[inside] = left + frac * (right - left)
        edges.append({"start_s": fa, "end_s": fb, "before_um": np.round(left, 1).tolist(),
                      "after_um": np.round(right, 1).tolist()})
        log.info("Motion bridged over %.1f-%.1f s: %s -> %s um", fa, fb, np.round(left, 1), np.round(right, 1))
    np.savez(out_dir / "motion_bridge.npz", temporal_bins_s=tb - t_start, spatial_bins_um=est.spatial_bins_um,
             estimate_without_bridge_peaks=raw, bridged=d, bridge_s=np.asarray(m["bridge_s"], dtype=float))
    return Motion(d, tb, est.spatial_bins_um, direction=est.direction), edges


def motion_info_all_peaks(folder: Path, t_start: float):
    """(time bins in file time, displacement) of the estimate saved by compute_motion in folder, or None."""
    try:
        from spikeinterface.core.motion import Motion

        mo = Motion.load(folder / "motion")
        return np.asarray(mo.temporal_bins_s[0]) - t_start, np.asarray(mo.displacement[0])
    except Exception:
        return None


def _motion(rec: si.BaseRecording, cfg: dict, out_dir: Path, job_kwargs: dict):
    m = cfg["motion"]
    motion_folder = out_dir / "motion"
    bridges = []
    if not m["bridge_s"]:
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
    else:
        from spikeinterface.preprocessing.motion import motion_options_preset
        from spikeinterface.sortingcomponents.motion import interpolate_motion

        # motion_folder keeps the estimate from all peaks, for comparison with the bridged one
        _, motion_info = spre.compute_motion(
            rec,
            preset=m["preset"],
            folder=motion_folder,
            output_motion_info=True,
            overwrite=True,
            detect_kwargs=m["detect_kwargs"],
            estimate_motion_kwargs=_estimate_motion_kwargs(m),
            **job_kwargs,
        )
        motion_info["motion"], bridges = _bridge_motion(rec, motion_info, m, out_dir)
        interp = dict(motion_options_preset[m["preset"]]["interpolate_motion_kwargs"], **m["interpolate_motion_kwargs"])
        rec_float = rec if rec.get_dtype().kind == "f" else rec.astype("float32")
        rec_corr = interpolate_motion(rec_float, motion_info["motion"], **interp)
    motion = motion_info["motion"]
    disp = np.concatenate([np.asarray(d).ravel() for d in motion.displacement])
    summary = {
        "preset": m["preset"],
        "max_abs_displacement_um": float(np.max(np.abs(disp))),
        "median_abs_displacement_um": float(np.median(np.abs(disp))),
        "displacement_range_um": float(np.ptp(disp)),
    }
    if bridges:
        summary["bridges"] = bridges
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
        if bridges:
            z = np.load(out_dir / "motion_bridge.npz")
            all_peaks = motion_info_all_peaks(out_dir / "motion", rec.get_start_time())
            fig, ax = plt.subplots(figsize=(14, 4))
            if all_peaks is not None:
                ax.plot(all_peaks[0], np.median(all_peaks[1], axis=1), color="0.7", lw=0.8, label="all spikes (DREDGE as is)")
            ax.plot(z["temporal_bins_s"], np.median(z["estimate_without_bridge_peaks"], axis=1), color="0.3", lw=0.8,
                    label="without the bridge windows' spikes")
            ax.plot(z["temporal_bins_s"], np.median(z["bridged"], axis=1), color="#d93636", lw=1.2, label="applied (bridged)")
            for a, b in z["bridge_s"]:
                ax.axvspan(a, b, color="#d93636", alpha=0.08, lw=0)
            ax.set_xlabel("file time (s)")
            ax.set_ylabel("displacement (um; median over depth)")
            ax.legend(loc="upper left", fontsize=8, frameon=False)
            fig.tight_layout()
            fig.savefig(out_dir / "motion_bridge.png", dpi=110)
            plt.close(fig)
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
