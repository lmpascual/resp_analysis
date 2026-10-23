"""Optotagging without re-sorting: find the sorted units' spikes in the laser window and test their light responses.

Only the intact window of e097 was sorted; the laser trains fall outside it. Sorting a longer window would
let drift build up and merge or lose brief units, so instead this script keeps the sorted units and
looks for them in a separate CatGT extract of the laser window (same CatGT settings):

  pulses   laser pulse times from the NIDQ file, moved onto the imec clock of the laser extract with the
           shared 1 Hz sync pulse.
  drift    motion estimate only (no correction) over the laser window and the sorted window together:
           did the probe move between them?
  validate whitening and templates from the sorted window; template matching checked against the
           sorter's own spikes in part of the sorted window.
  match    template matching of every sorted unit in the laser window (no clustering, so unit ids stay those
           of the sort and the review). With matching.correct_motion the laser window is first interpolated
           into the sorted window's frame using the drift estimate.
  metrics  per unit: pulse-locked response, first-spike latency and jitter, SALT test, evoked vs
           spontaneous waveform, photoelectric-artefact checks, and a class (direct / slow / ...).
  report   figures and report.md.

    python optotag.py configs/e097_optotag.yaml
    python optotag.py configs/e097_optotag.yaml --redo metrics
    python optotag.py configs/e097_optotag.yaml --stop-after drift

Times in the outputs are seconds from the start of the laser extract unless a column says "raw".
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sorting"))   # npxsort
from npxsort.paths import expand  # noqa: E402

log = logging.getLogger("optotag")

STAGES = ["pulses", "drift", "validate", "match", "metrics", "report"]

DEFAULTS = {
    "paths": {
        "sorted_output_dir": None,  # run_sorting.py output_dir of the sorted window
        "sorted_catgt": None,  # CatGT folder the sorted window was read from
        "laser_catgt": None,  # CatGT extract of the laser window, same CatGT settings
        "nidq_bin": None,  # SpikeGLX NIDQ .bin of the raw run (its .meta must sit next to it)
        "session_json": None,  # optional: laser_trains from the session dashboard, only compared against
        "output_dir": None,
        "scratch_dir": None,  # motion peaks etc.
    },
    "sorter": "kilosort4",
    "stream_id": "imec0.ap",
    "nidq": {
        "laser_channel": 6,
        "laser_threshold_v": 0.5,
        "sync_column": 8,
        "sync_bit": 7,
        "train_gap_s": 1.0,  # pulses further apart than this start a new train
    },
    "drift": {
        "preset": "dredge",
        "estimate_motion_kwargs": {},
    },
    "matching": {
        "method": "wobble",  # wobble (Lupin's default) | circus-omp | any spikeinterface matching method
        "method_kwargs": {},
        "engine": "torch",
        "torch_device": "cuda",
        "whitening_radius_um": 100.0,  # same local whitening as Lupin
        "ms_before": 1.0,  # same window as the analyzer templates
        "ms_after": 2.0,
        "max_spikes_per_template": 5000,
        "validation_window_s": [0.0, 140.0],  # part of the sorted window to compare against the sorter
        "tolerance_ms": 0.4,
        "job_kwargs": {"n_jobs": 4, "chunk_duration": "1s", "progress_bar": True},
        # Interpolate the laser window into the sorted window's frame with the drift-stage estimate first.
        "correct_motion": False,
        "interpolate_motion_kwargs": {"border_mode": "force_extrapolate", "spatial_interpolation_method": "kriging", "sigma_um": 20.0, "p": 2},
    },
    "response": {
        "trains": None,  # list of train numbers to analyse; None = all
        # Trains the pooled metrics and the class come from (None = all analysed). The other analysed trains are
        # reported per train and as "sec_*" columns, e.g. trains during which the tissue moved.
        "primary_trains": None,
        "window_ms": 10.0,  # response window after pulse onset (SALT test window)
        "latency_bin_ms": 1.0,  # SALT latency histogram bin
        "psth_bin_ms": 0.5,
        "psth_range_ms": [-10.0, 50.0],
        "control_window_ms": [35.0, 45.0],  # same pulses, late in the 50 ms cycle: pulse-locked or not
        "baseline_s": 20.0,  # spontaneous period before each train
        "baseline_gap_s": 0.5,  # ... ending this long before the first pulse
        "sham_phases": 5,  # sham pulse grids per 50 ms period (window_ms apart, disjoint)
        "n_waveforms": 300,
        "n_top_channels": 8,
    },
    "classify": {
        "alpha": 0.01,
        "min_prob_increase": 0.05,
        "max_latency_ms": 6.0,
        "max_jitter_ms": 1.5,
        "min_waveform_r": 0.9,
        "min_evoked_spikes": 10,
        # A pulse-locked class needs the unit's projection to have reproduced the sorter's spikes
        # (validate stage); otherwise the class is locked_poor_projection.
        "min_val_recall": 0.8,
        "min_val_precision": 0.8,
        # During the light-evoked volley, template matching can split a burst of overlapping spikes among templates.
        # A unit whose matched spikes 0-rp_window_ms after a pulse contain two spikes closer than refractory_ms on more
        # than this fraction of responding pulses is locked_volley_split: its evoked spike train is not one neuron's.
        "refractory_ms": 1.5,
        "rp_window_ms": 15.0,
        "max_evoked_rp_violation_frac": 0.1,
        "artefact_min_latency_ms": 1.0,  # median latency below this: not a ChR2 spike
        "artefact_offset_window_ms": [4.7, 5.5],  # locked to pulse offset (5 ms pulses)...
        "artefact_max_jitter_ms": 0.2,  # ... with almost no jitter (SD, ms)
    },
    "report": {
        "max_unit_figures": 80,
        "max_trace_figures": 40,
        "compare_with": None,  # output_dir of another optotag run (e.g. without motion correction) to compare against
    },
    "job_kwargs": {"n_jobs": 12, "chunk_duration": "1s", "progress_bar": True},
}
FREEFORM = {"drift.estimate_motion_kwargs", "matching.method_kwargs", "matching.job_kwargs", "matching.interpolate_motion_kwargs", "job_kwargs"}


# ----------------------------------------------------------------------------------------------- config


def _merge(base: dict, override: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        key_path = f"{path}.{key}" if path else key
        if path in FREEFORM:
            out[key] = copy.deepcopy(value)
            continue
        if key not in base:
            raise ValueError(f"Unknown config key '{key_path}'. Check spelling against DEFAULTS in optotag.py.")
        if isinstance(base[key], dict) and key_path not in FREEFORM:
            out[key] = _merge(base[key], value, key_path)
        elif isinstance(base[key], dict):
            out[key] = {**base[key], **value}
        else:
            out[key] = value
    return out


def load_config(path) -> dict:
    with open(path) as f:
        cfg = _merge(DEFAULTS, expand(yaml.safe_load(f) or {}))   # ${outputs} etc. from configs/local_paths.yaml
    missing = [k for k in ("sorted_output_dir", "sorted_catgt", "laser_catgt", "nidq_bin", "output_dir", "scratch_dir") if not cfg["paths"][k]]
    if missing:
        raise ValueError(f"paths.{missing[0]} must be set")
    return cfg


# ---------------------------------------------------------------------------------------------- helpers


def read_meta(path: Path) -> dict:
    meta = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                meta[k.lstrip("~")] = v
    return meta


def catgt_files(folder) -> tuple[Path, dict]:
    bins = sorted(Path(folder).glob("*.ap.bin"))
    if len(bins) != 1:
        raise FileNotFoundError(f"Expected one *.ap.bin in {folder}, found {len(bins)}")
    return bins[0], read_meta(bins[0].with_suffix(".meta"))


def catgt_start_s(folder) -> float:
    """Raw-run time (imec clock) of the extract's first sample, from CatGT's ct_offsets file."""
    for p in Path(folder).glob("*_ct_offsets.txt"):
        for line in open(p):
            if line.startswith("sec_imap0:"):
                return -float(line.split()[1])
    raise FileNotFoundError(f"No *_ct_offsets.txt with sec_imap0 in {folder}")


def read_catgt(folder, stream_id):
    import spikeinterface.extractors as se

    return se.read_spikeglx(folder, stream_id=stream_id)


def sorted_unit_tables(cfg) -> pd.DataFrame:
    """units.tsv of the sorted window, with the manual review verdict if there is one."""
    base = Path(cfg["paths"]["sorted_output_dir"])
    units = pd.read_csv(base / cfg["sorter"] / "units.tsv", sep="\t")
    keep = ["unit_id", "label", "peak_channel_id", "peak_channel_index", "x_um", "depth_um", "num_spikes", "firing_rate", "snr", "found_by_both"]
    units = units[[c for c in keep if c in units.columns]]
    review = base / "review" / f"{cfg['sorter']}_unit_review.tsv"
    if review.exists():
        r = pd.read_csv(review, sep="\t")[["unit_id", "verdict", "note"]].rename(columns={"verdict": "review_verdict", "note": "review_note"})
        units = units.merge(r, on="unit_id", how="left")
    return units


def stage_done(out_dir: Path, stage: str) -> Path:
    return out_dir / stage / "done.json"


def mark_done(out_dir: Path, stage: str, info: dict):
    with open(stage_done(out_dir, stage), "w") as f:
        json.dump({"finished": time.strftime("%Y-%m-%dT%H:%M:%S"), **info}, f, indent=2, default=float)


def first_after(times: np.ndarray, events: np.ndarray, window: float) -> np.ndarray:
    """Latency of the first spike in [event, event + window) for each event; NaN if none."""
    lat = np.full(len(events), np.nan)
    if len(times) == 0:
        return lat
    idx = np.searchsorted(times, events, side="left")
    ok = idx < len(times)
    d = np.full(len(events), np.inf)
    d[ok] = times[idx[ok]] - events[ok]
    hit = d < window
    lat[hit] = d[hit]
    return lat


def aligned(times: np.ndarray, events: np.ndarray, lo: float, hi: float):
    """Spike times relative to each event within [lo, hi), and the event index of each."""
    i0, i1 = np.searchsorted(times, events + lo), np.searchsorted(times, events + hi)
    n = i1 - i0
    ev = np.repeat(np.arange(len(events)), n)
    idx = np.repeat(i0 - np.concatenate([[0], np.cumsum(n)[:-1]]), n) + np.arange(n.sum())
    return times[idx] - events[ev], ev


def md_table(df: pd.DataFrame, fmt: str = "{:.3g}") -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else fmt.format(v)
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)

    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def rp_violation_frac(times: np.ndarray, events: np.ndarray, window: float, rp: float) -> tuple[float, int]:
    """Fraction of events with spikes in [event, event + window) that include two spikes closer than rp."""
    i0, i1 = np.searchsorted(times, events), np.searchsorted(times, events + window)
    hit = np.flatnonzero(i1 - i0 > 0)
    bad = sum(1 for k in hit if i1[k] - i0[k] > 1 and np.min(np.diff(times[i0[k]:i1[k]])) < rp)
    return (bad / len(hit) if len(hit) else np.nan), len(hit)


def any_in(times: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    return np.searchsorted(times, stops, side="left") > np.searchsorted(times, starts, side="left")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.grid": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    })
    return plt


# Reference categorical palette (dataviz skill), fixed order; light-blue band marks the 5 ms light pulse.
C_BLUE, C_ORANGE, C_AQUA, C_GRAY, C_INK, C_LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#cde2fb"


# ----------------------------------------------------------------------------------------------- pulses


def stage_pulses(cfg: dict, out_dir: Path) -> dict:
    d = out_dir / "pulses"
    d.mkdir(parents=True, exist_ok=True)
    nc = cfg["nidq"]
    nidq_bin = Path(cfg["paths"]["nidq_bin"])
    meta = read_meta(nidq_bin.with_suffix(".meta"))
    fs_ni = float(meta["niSampRate"])
    nch = int(meta["nSavedChans"])
    v_per_count = float(meta["niAiRangeMax"]) / float(meta["niMaxInt"])

    laser_folder = Path(cfg["paths"]["laser_catgt"])
    _, laser_meta = catgt_files(laser_folder)
    start_raw = catgt_start_s(laser_folder)
    duration = float(laser_meta["fileTimeSecs"])

    t0 = max(0.0, start_raw - 5.0)
    s0 = int(t0 * fs_ni)
    n = int((duration + 10.0) * fs_ni)
    with open(nidq_bin, "rb") as f:
        f.seek(s0 * nch * 2)
        raw = np.fromfile(f, dtype=np.int16, count=n * nch)
    raw = raw[: (raw.size // nch) * nch].reshape(-1, nch)
    log.info("Read NIDQ %.1f-%.1f s raw (%d samples, %.0f Hz)", t0, t0 + raw.shape[0] / fs_ni, raw.shape[0], fs_ni)

    v = raw[:, nc["laser_channel"]].astype(np.float64) * v_per_count
    thr = nc["laser_threshold_v"]
    above = v >= thr
    on = np.flatnonzero(~above[:-1] & above[1:]) + 1
    off = np.flatnonzero(above[:-1] & ~above[1:]) + 1
    off = off[off > on[0]] if len(on) else off
    n_p = min(len(on), len(off))
    on, off = on[:n_p], off[:n_p]
    if np.any(off <= on):
        raise RuntimeError("Laser rising and falling edges do not alternate")

    def crossing(idx):  # linear interpolation of the threshold crossing between samples idx-1 and idx
        frac = (thr - v[idx - 1]) / (v[idx] - v[idx - 1])
        return (s0 + idx - 1 + frac) / fs_ni

    on_ni, off_ni = crossing(on), crossing(off)
    peak_v = np.array([v[a:b].max() for a, b in zip(on, off)])

    sync = (raw[:, nc["sync_column"]].astype(np.uint16) >> nc["sync_bit"]) & 1
    sync_ni = (s0 + np.flatnonzero((sync[:-1] == 0) & (sync[1:] == 1)) + 1) / fs_ni

    edge_files = sorted(laser_folder.glob("*.ap.xd_384_6_500.txt"))
    if not edge_files or edge_files[0].stat().st_size == 0:
        raise FileNotFoundError(f"No CatGT imec sync-edge file (*.ap.xd_384_6_500.txt) in {laser_folder}")
    sync_im = np.loadtxt(edge_files[0])

    pred = sync_ni - start_raw
    j = np.clip(np.searchsorted(sync_im, pred), 1, len(sync_im) - 1)
    j = np.where(np.abs(sync_im[j - 1] - pred) < np.abs(sync_im[j] - pred), j - 1, j)
    keep = np.abs(sync_im[j] - pred) < 0.25
    a, b = np.polyfit(sync_ni[keep], sync_im[j[keep]], 1)
    resid = sync_im[j[keep]] - (a * sync_ni[keep] + b)
    fit = {
        "n_sync_pairs": int(keep.sum()),
        "slope": float(a),
        "intercept_s": float(b),
        "residual_sd_us": float(resid.std() * 1e6),
        "residual_max_abs_us": float(np.abs(resid).max() * 1e6),
        "extract_start_raw_s": start_raw,
        "nidq_minus_imec_at_start_ms": float((-b / a - start_raw) * 1e3),
        "nidq_sample_period_ms": 1e3 / fs_ni,
    }
    log.info("Sync fit: %d pairs, residual SD %.1f us, max %.1f us", fit["n_sync_pairs"], fit["residual_sd_us"], fit["residual_max_abs_us"])

    on_s, off_s = a * on_ni + b, a * off_ni + b
    inside = (on_s > 0) & (off_s < duration)
    on_s, off_s, on_ni, peak_v = on_s[inside], off_s[inside], on_ni[inside], peak_v[inside]
    train = np.concatenate([[0], np.cumsum(np.diff(on_s) > nc["train_gap_s"])]) + 1
    pulse_in_train = np.concatenate([np.arange((train == t).sum()) for t in np.unique(train)])
    pulses = pd.DataFrame({
        "train": train, "pulse": pulse_in_train, "onset_s": on_s, "offset_s": off_s,
        "width_ms": (off_s - on_s) * 1e3, "onset_raw_nidq_s": on_ni, "peak_v": peak_v,
    })
    pulses.to_csv(d / "pulses.tsv", sep="\t", index=False, float_format="%.6f")

    trains = pulses.groupby("train").agg(
        onset_s=("onset_s", "first"), last_offset_s=("offset_s", "last"), n_pulses=("onset_s", "size"),
        onset_raw_nidq_s=("onset_raw_nidq_s", "first"), width_ms=("width_ms", "mean"),
    ).reset_index()
    trains["period_ms"] = pulses.groupby("train")["onset_s"].apply(lambda x: np.diff(x).mean() * 1e3).values
    trains["onset_raw_imec_s"] = trains["onset_s"] + start_raw
    js = cfg["paths"].get("session_json")
    if js and Path(js).exists():
        jt = {t["id"]: t["t_start"] for t in json.load(open(js))["laser_trains"]}
        trains["json_t_start"] = trains["train"].map(jt)
        trains["nidq_minus_json_ms"] = (trains["onset_raw_nidq_s"] - trains["json_t_start"]) * 1e3
    trains.to_csv(d / "trains.tsv", sep="\t", index=False, float_format="%.6f")
    with open(d / "sync_fit.json", "w") as f:
        json.dump(fit, f, indent=2)
    log.info("%d pulses in %d trains, width %.2f-%.2f ms", len(pulses), len(trains), pulses.width_ms.min(), pulses.width_ms.max())
    return {"n_pulses": len(pulses), "n_trains": len(trains), **fit}


# ------------------------------------------------------------------------------------------------ drift


def stage_drift(cfg: dict, out_dir: Path) -> dict:
    import spikeinterface.core as si
    import spikeinterface.preprocessing as spre
    from npxsort.preprocess import _estimate_motion_kwargs

    d = out_dir / "drift"
    d.mkdir(parents=True, exist_ok=True)
    rec_l = read_catgt(cfg["paths"]["laser_catgt"], cfg["stream_id"])
    rec_s = read_catgt(cfg["paths"]["sorted_catgt"], cfg["stream_id"])
    rec = si.concatenate_recordings([rec_l, rec_s])
    dur_l = rec_l.get_total_duration()
    start_l, start_s = catgt_start_s(cfg["paths"]["laser_catgt"]), catgt_start_s(cfg["paths"]["sorted_catgt"])
    m = {"preset": cfg["drift"]["preset"], "estimate_motion_kwargs": cfg["drift"]["estimate_motion_kwargs"]}
    motion, info = spre.compute_motion(
        rec, preset=m["preset"], folder=Path(cfg["paths"]["scratch_dir"]) / "motion", overwrite=True,
        output_motion_info=True, estimate_motion_kwargs=_estimate_motion_kwargs(m), **cfg["job_kwargs"],
    )
    disp = np.asarray(motion.displacement[0])
    tb = np.asarray(motion.temporal_bins_s[0])
    sb = np.asarray(motion.spatial_bins_um)
    in_l, in_s = tb < dur_l, tb >= dur_l
    med_l, med_s = np.median(disp[in_l], axis=0), np.median(disp[in_s], axis=0)
    step = med_s - med_l
    summary = {
        "preset": m["preset"],
        "spatial_bins_um": sb.tolist(),
        "median_displacement_laser_um": med_l.tolist(),
        "median_displacement_sorted_um": med_s.tolist(),
        "step_sorted_minus_laser_um": step.tolist(),
        "median_abs_step_um": float(np.median(np.abs(step))),
        "max_abs_step_um": float(np.max(np.abs(step))),
        "laser_window_p5_p95_um": [float(np.percentile(disp[in_l], 5)), float(np.percentile(disp[in_l], 95))],
        "sorted_window_p5_p95_um": [float(np.percentile(disp[in_s], 5)), float(np.percentile(disp[in_s], 95))],
        "n_peaks": int(len(info["peaks"])),
    }
    with open(d / "drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(d / "motion.npz", displacement=disp, temporal_bins_s=tb, spatial_bins_um=sb, laser_duration_s=dur_l)

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]
    peaks, locs = info["peaks"], info["peak_locations"]
    sel = np.random.default_rng(0).choice(len(peaks), size=min(len(peaks), 200_000), replace=False)
    t_pk = peaks["sample_index"][sel] / rec.get_sampling_frequency()
    raw_t = np.where(t_pk < dur_l, t_pk + start_l, t_pk - dur_l + start_s)
    ax.scatter(raw_t, locs["y"][sel], s=0.3, c=C_GRAY, alpha=0.3, rasterized=True, linewidths=0)
    raw_tb = np.where(tb < dur_l, tb + start_l, tb - dur_l + start_s)
    for k in range(disp.shape[1]):
        for mask in (in_l, in_s):
            ax.plot(raw_tb[mask], sb[k] + disp[mask, k], color=C_BLUE, lw=1.2)
    ax.set_xlabel("raw-run time (s)")
    ax.set_ylabel("depth (um); blue = spatial bin + estimated displacement")
    ax.set_title(f"Peaks and DREDGE motion estimate ({m['preset']}); laser window and sorted window estimated together")
    ax2 = axes[1]
    ax2.barh(sb, step, height=(sb[1] - sb[0]) * 0.8 if len(sb) > 1 else 50, color=C_BLUE)
    ax2.axvline(0, color=C_INK, lw=0.8)
    ax2.set_xlabel("sorted minus laser window\nmedian displacement (um)")
    ax2.set_title(f"median |step| {summary['median_abs_step_um']:.1f} um")
    fig.tight_layout()
    fig.savefig(d / "drift.png", dpi=110)
    plt.close(fig)
    log.info("Drift between windows: median |step| %.1f um, max %.1f um", summary["median_abs_step_um"], summary["max_abs_step_um"])
    drift_tables(out_dir)
    return {k: summary[k] for k in ("median_abs_step_um", "max_abs_step_um")}


def drift_tables(out_dir: Path):
    """Laser-window displacement relative to the sorted window: per segment between trains, and around each train."""
    z = np.load(out_dir / "drift" / "motion.npz")
    disp, tb, sb, dur_l = z["displacement"], z["temporal_bins_s"], z["spatial_bins_um"], float(z["laser_duration_s"])
    ref = np.median(disp[tb >= dur_l], axis=0)
    trains = pd.read_csv(out_dir / "pulses" / "trains.tsv", sep="\t")
    edges = [0.0] + list(trains.onset_s) + [dur_l]
    rows = []
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (tb >= a + 1) & (tb < b - 0.5)
        rel = np.median(disp[sel], axis=0) - ref
        rows.append({"segment": "before train 1" if k == 0 else f"after train {k}", "start_s": a, "end_s": b,
                     **{f"y{int(round(y))}_um": v for y, v in zip(sb, rel)}})
    pd.DataFrame(rows).to_csv(out_dir / "drift" / "segments_vs_sorted_window.tsv", sep="\t", index=False, float_format="%.1f")
    rel_t = np.arange(-5, 26)
    rows = []
    for _, t in trains.iterrows():
        for y_i, y in enumerate(sb):
            v = np.interp(t.onset_s + rel_t, tb[tb < dur_l], disp[tb < dur_l, y_i] - ref[y_i])
            rows.append({"train": int(t.train), "y_um": int(round(y)), **{f"t{r:+d}s": x for r, x in zip(rel_t, v)}})
    pd.DataFrame(rows).to_csv(out_dir / "drift" / "around_trains_vs_sorted_window.tsv", sep="\t", index=False, float_format="%.1f")


# ------------------------------------------------------------------------------------------------ match


def _load_sorted(cfg):
    import spikeinterface.core as si

    ana = si.load_sorting_analyzer(Path(cfg["paths"]["sorted_output_dir"]) / cfg["sorter"] / "analyzer", load_extensions=False)
    out = {
        "unit_ids": np.asarray(ana.unit_ids),
        "mask": np.asarray(ana.sparsity.mask).copy(),
        "spikes": ana.sorting.to_spike_vector().copy(),
        "fs": float(ana.sampling_frequency),
        "channel_ids": np.asarray(ana.channel_ids),
        "num_samples": int(ana.get_num_samples()),
    }
    del ana
    return out


def _sorted_whitened(cfg):
    import spikeinterface.preprocessing as spre

    mc = cfg["matching"]
    rec_s = spre.scale_to_uV(read_catgt(cfg["paths"]["sorted_catgt"], cfg["stream_id"]))
    return spre.whiten(rec_s, dtype="float32", mode="local", radius_um=mc["whitening_radius_um"], seed=0)


def _laser_motion(out_dir: Path, t_start: float):
    """Laser-window displacement relative to the sorted window's median position, as a Motion object.

    The drift stage's time bins count from the start of the laser extract; SpikeGLX recordings carry their own
    start time (first sample of the run), so the bins are shifted by `t_start` onto the recording's clock.
    """
    from spikeinterface.core.motion import Motion

    z = np.load(out_dir / "drift" / "motion.npz")
    disp, tb, sb, dur_l = z["displacement"], z["temporal_bins_s"], z["spatial_bins_um"], float(z["laser_duration_s"])
    ref = np.median(disp[tb >= dur_l], axis=0)
    sel = tb < dur_l
    return Motion(displacement=[disp[sel] - ref], temporal_bins_s=[tb[sel] + t_start], spatial_bins_um=sb, direction="y")


def laser_recording(cfg: dict, out_dir: Path):
    """The laser window in uV, motion-corrected into the sorted window's frame if matching.correct_motion.

    The corrected recording is saved once to scratch_dir (int16, like the CatGT file), so matching and the
    waveform metrics read the same data.
    """
    import spikeinterface.core as si
    import spikeinterface.preprocessing as spre
    from spikeinterface.sortingcomponents.motion import interpolate_motion

    rec = read_catgt(cfg["paths"]["laser_catgt"], cfg["stream_id"])
    if not cfg["matching"]["correct_motion"]:
        return spre.scale_to_uV(rec)
    folder = Path(cfg["paths"]["scratch_dir"]) / "laser_motion_corrected"
    if not (folder / "optotag_complete.json").exists():
        log.info("Interpolating the laser window into the sorted window's frame, saving to %s", folder)
        t_start = rec.get_start_time()
        motion = _laser_motion(out_dir, t_start)
        tb = motion.temporal_bins_s[0]
        if tb[0] > t_start + 5 or tb[-1] < rec.get_end_time() - 5:
            raise RuntimeError(f"Motion bins {tb[0]:.1f}-{tb[-1]:.1f} s do not cover the recording {t_start:.1f}-{rec.get_end_time():.1f} s")
        corr = interpolate_motion(spre.astype(rec, "float32"), motion, **cfg["matching"]["interpolate_motion_kwargs"])
        corr.save(folder=folder, format="binary", dtype="int16", overwrite=True, **cfg["job_kwargs"])
        with open(folder / "optotag_complete.json", "w") as f:
            json.dump({"interpolate_motion_kwargs": cfg["matching"]["interpolate_motion_kwargs"], "motion_t_start_s": t_start,
                       "made": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
    return spre.scale_to_uV(si.load(folder))


def laser_data(cfg: dict):
    """int16 memmap (samples x channels) of the data the matching ran on, and gains to uV for the 384 channels."""
    rec = read_catgt(cfg["paths"]["laser_catgt"], cfg["stream_id"])
    gains = rec.get_channel_gains().astype(np.float32)
    if cfg["matching"]["correct_motion"]:
        f = Path(cfg["paths"]["scratch_dir"]) / "laser_motion_corrected" / "traces_cached_seg0.raw"
        return np.memmap(f, dtype=np.int16, mode="r").reshape(-1, rec.get_num_channels()), gains
    laser_bin, meta = catgt_files(cfg["paths"]["laser_catgt"])
    return np.memmap(laser_bin, dtype=np.int16, mode="r").reshape(-1, int(meta["nSavedChans"])), gains


def _hits(ref: np.ndarray, test: np.ndarray, tol: float):
    """Number of ref spikes with a test spike within tol, and the (test - ref) lags of those pairs."""
    if len(ref) == 0 or len(test) == 0:
        return 0, np.zeros(0)
    i = np.searchsorted(ref, test)
    lo, hi = np.clip(i - 1, 0, len(ref) - 1), np.clip(i, 0, len(ref) - 1)
    near = np.where(np.abs(test - ref[lo]) <= np.abs(ref[hi] - test), lo, hi)
    lag = test - ref[near]
    ok = np.abs(lag) <= tol
    return len(np.unique(near[ok])), lag[ok]


def _templates(out_dir: Path):
    from spikeinterface.core import Templates

    z = np.load(out_dir / "validate" / "templates_whitened.npz", allow_pickle=True)
    return Templates(
        templates_array=z["templates_array"], sampling_frequency=float(z["fs"]), nbefore=int(z["nbefore"]),
        channel_ids=z["channel_ids"], unit_ids=z["unit_ids"], sparsity_mask=z["mask"], probe=None, is_in_uV=False,
    ), z


def _match(cfg, rec, templates, name):
    from spikeinterface.sortingcomponents.matching import find_spikes_from_templates

    mc = cfg["matching"]
    mk = dict(mc["method_kwargs"])
    mk.setdefault("engine", mc["engine"])
    mk.setdefault("torch_device", mc["torch_device"])
    t = time.time()
    sp = find_spikes_from_templates(rec, templates, method=mc["method"], method_kwargs=mk,
                                    pipeline_kwargs={"gather_mode": "memory"}, job_kwargs=mc["job_kwargs"])
    log.info("Matching (%s) on %s: %d spikes in %.0f s", mc["method"], name, len(sp), time.time() - t)
    return np.sort(sp, order="sample_index"), time.time() - t, mk


def stage_validate(cfg: dict, out_dir: Path) -> dict:
    """Whitening matrix and templates from the sorted window; matching checked against the sorter there."""
    from spikeinterface.core.waveform_tools import estimate_templates_with_accumulator

    d = out_dir / "validate"
    mc = cfg["matching"]
    s = _load_sorted(cfg)
    w_s = _sorted_whitened(cfg)
    fs = s["fs"]
    if not np.array_equal(np.asarray(w_s.channel_ids), s["channel_ids"]):
        raise RuntimeError("Channel ids of the analyzer and the sorted CatGT recording differ")
    if w_s.get_num_samples() != s["num_samples"]:
        raise RuntimeError("Sorted CatGT recording does not match the analyzer (length)")

    spikes = s["spikes"]
    rng = np.random.default_rng(0)
    pick = []
    for ui in range(len(s["unit_ids"])):
        idx = np.flatnonzero(spikes["unit_index"] == ui)
        if len(idx) > mc["max_spikes_per_template"]:
            idx = np.sort(rng.choice(idx, mc["max_spikes_per_template"], replace=False))
        pick.append(idx)
    sub = spikes[np.sort(np.concatenate(pick))]
    nbefore, nafter = int(round(mc["ms_before"] * fs / 1e3)), int(round(mc["ms_after"] * fs / 1e3))
    t0 = time.time()
    tarr = estimate_templates_with_accumulator(
        w_s, sub, s["unit_ids"], nbefore, nafter, return_in_uV=False, sparsity_mask=s["mask"],
        job_name="templates on whitened sorted window", **cfg["job_kwargs"],
    )
    np.savez(d / "templates_whitened.npz", templates_array=tarr, fs=fs, nbefore=nbefore, channel_ids=np.asarray(w_s.channel_ids),
             unit_ids=s["unit_ids"], mask=s["mask"])
    np.save(d / "whitening_W.npy", np.asarray(w_s._kwargs["W"], dtype=np.float32))
    log.info("Whitened templates for %d units in %.0f s", len(s["unit_ids"]), time.time() - t0)
    templates, _ = _templates(out_dir)

    v0, v1 = (int(round(x * fs)) for x in mc["validation_window_s"])
    v1 = min(v1, w_s.get_num_samples())
    sp_v, dt_v, mk = _match(cfg, w_s.frame_slice(v0, v1), templates, f"sorted window {mc['validation_window_s']} s")
    sp_v_samples = sp_v["sample_index"].astype(np.int64) + v0

    # Same-unit agreement with the sorter, away from the window edges.
    margin = 3 * (nbefore + nafter)
    ref_in = (spikes["sample_index"] >= v0 + margin) & (spikes["sample_index"] < v1 - margin)
    test_in = (sp_v_samples >= v0 + margin) & (sp_v_samples < v1 - margin)
    wide = int(round(1.0 * fs / 1e3))
    lags = []
    for ui in range(len(s["unit_ids"])):
        ref = spikes["sample_index"][ref_in & (spikes["unit_index"] == ui)].astype(np.int64)
        test = sp_v_samples[test_in & (sp_v["cluster_index"] == ui)]
        lags.append(_hits(ref, test, wide)[1])
    lags = np.concatenate(lags)
    lag = int(np.round(np.median(lags))) if len(lags) else 0
    tol = int(round(mc["tolerance_ms"] * fs / 1e3))
    rows = []
    for ui, u in enumerate(s["unit_ids"]):
        ref = spikes["sample_index"][ref_in & (spikes["unit_index"] == ui)].astype(np.int64)
        test = sp_v_samples[test_in & (sp_v["cluster_index"] == ui)] - lag
        h, _ = _hits(ref, test, tol)
        other = sp_v_samples[test_in & (sp_v["cluster_index"] != ui)] - lag
        h_any, _ = _hits(ref, np.sort(np.concatenate([test, other])), tol)
        rows.append({
            "unit_id": u, "val_n_sorter": len(ref), "val_n_matched": len(test), "val_hits": h,
            "val_recall": h / len(ref) if len(ref) else np.nan,
            "val_precision": h / len(test) if len(test) else np.nan,
            "val_recall_any_unit": h_any / len(ref) if len(ref) else np.nan,
        })
    val = pd.DataFrame(rows)
    val.to_csv(d / "validation.tsv", sep="\t", index=False, float_format="%.4f")
    big = val.val_n_sorter >= 50
    log.info("Validation: lag %d samples; median recall %.2f, precision %.2f (%d units with >= 50 sorter spikes)",
             lag, val.loc[big, "val_recall"].median(), val.loc[big, "val_precision"].median(), int(big.sum()))
    np.savez(d / "validation_spikes.npz", sample_index=sp_v_samples - lag, unit_index=sp_v["cluster_index"],
             amplitude=sp_v["amplitude"], unit_ids=s["unit_ids"], sampling_frequency=fs)
    info = {
        "method": mc["method"], "method_kwargs": mk, "n_templates": len(s["unit_ids"]), "lag_samples": lag,
        "validation_window_s": [v0 / fs, v1 / fs], "validation_seconds": dt_v,
        "whitening": f"local, radius {mc['whitening_radius_um']} um, matrix from the sorted window, applied to both windows",
        "templates": f"mean of up to {mc['max_spikes_per_template']} sorter spikes per unit on the whitened sorted window, analyzer sparsity",
    }
    with open(d / "validate_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    return {"lag_samples": lag, "median_recall": float(val.loc[big, "val_recall"].median()), "median_precision": float(val.loc[big, "val_precision"].median())}


def stage_match(cfg: dict, out_dir: Path) -> dict:
    """Find every sorted unit's spikes in the laser window (no clustering: unit ids are the sort's)."""
    import spikeinterface.preprocessing as spre

    d = out_dir / "match"
    mc = cfg["matching"]
    templates, z = _templates(out_dir)
    vinfo = json.load(open(out_dir / "validate" / "validate_info.json"))
    W = np.load(out_dir / "validate" / "whitening_W.npy")
    rec_l = laser_recording(cfg, out_dir)
    if not np.array_equal(np.asarray(rec_l.channel_ids), z["channel_ids"]) or abs(rec_l.get_sampling_frequency() - float(z["fs"])) > 1e-6:
        raise RuntimeError("Laser recording channels or sampling rate differ from the sorted window's")
    w_l = spre.whiten(rec_l, dtype="float32", mode="local", radius_um=mc["whitening_radius_um"], W=W, M=None)
    sp_l, dt_l, mk = _match(cfg, w_l, templates, "laser window" + (" (motion-corrected)" if mc["correct_motion"] else ""))
    lag = vinfo["lag_samples"]
    np.savez(d / "laser_spikes.npz", sample_index=sp_l["sample_index"].astype(np.int64) - lag, unit_index=sp_l["cluster_index"],
             amplitude=sp_l["amplitude"], unit_ids=z["unit_ids"], sampling_frequency=float(z["fs"]))
    info = {"correct_motion": mc["correct_motion"], "interpolate_motion_kwargs": mc["interpolate_motion_kwargs"] if mc["correct_motion"] else None,
            "lag_samples_removed": lag, "seconds": dt_l, "n_spikes": int(len(sp_l))}
    with open(d / "match_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    return {"n_spikes_laser": int(len(sp_l)), "correct_motion": mc["correct_motion"]}


# ---------------------------------------------------------------------------------------------- metrics


def sham_sets(pulses: pd.DataFrame, rc: dict) -> np.ndarray:
    """Sham pulse times (n_sets, n_pulses) on the stimulus grid, in the spontaneous period before each train.

    For every train the spontaneous period is cut into blocks as long as the train; within a block the
    train's own pulse offsets are laid down at `sham_phases` phases window_ms apart, so all sham windows
    of all sets are disjoint. Every set has one sham per real pulse, as SALT needs.
    """
    wn = rc["window_ms"] / 1e3
    period = np.median(np.diff(pulses.onset_s.values)[np.diff(pulses.train.values) == 0])
    if rc["sham_phases"] * wn > period + 1e-9:
        raise ValueError("sham_phases * window_ms is longer than the pulse period; sham windows would overlap")
    per_train, n_blocks = [], []
    for _, p in pulses.groupby("train"):
        onset = p.onset_s.values
        off = onset - onset[0]
        length = off[-1] + period
        b1 = onset[0] - rc["baseline_gap_s"]
        b0 = max(0.0, b1 - rc["baseline_s"])
        nb = int((b1 - b0) // length)
        per_train.append((b0, length, off))
        n_blocks.append(nb)
    nb = min(n_blocks)
    if nb < 1:
        raise ValueError("Spontaneous period before a train is shorter than the train")
    sets = []
    for b in range(nb):
        for ph in range(rc["sham_phases"]):
            sets.append(np.concatenate([b0 + b * length + ph * wn + off for b0, length, off in per_train]))
    return np.array(sets)


def _js_distance(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return np.sum(a[nz] * np.log2(a[nz] / b[nz]))

    return np.sqrt(2 * (0.5 * kl(p, m) + 0.5 * kl(q, m)))


def salt(test_lat: np.ndarray, base_lat: np.ndarray, wn: float, dt: float):
    """SALT (Kvitsiani et al. 2013, salt.m): first-spike latency distributions, test vs baseline sets.

    test_lat: (n_trials,) latencies (NaN = no spike in window). base_lat: (n_sets, n_trials).
    Returns (p, I): fraction of baseline-baseline distances >= the median test-baseline distance, and the
    difference between that median and the median baseline-baseline distance.
    """
    nb = int(round(wn / dt))

    def hist(lat):
        b = np.where(np.isnan(lat), nb, np.minimum(np.floor(np.nan_to_num(lat) / dt), nb - 1)).astype(int)
        h = np.bincount(b, minlength=nb + 1).astype(float)
        return h / h.sum()

    hb = [hist(x) for x in base_lat]
    ht = hist(test_lat)
    null = np.array([_js_distance(hb[i], hb[j]) for i in range(len(hb)) for j in range(i + 1, len(hb))])
    test = np.median([_js_distance(h, ht) for h in hb])
    return float(np.mean(null >= test)), float(test - np.median(null))


def _snippets(data, gains, samples, chans, nb, na):
    samples = samples[(samples - nb >= 0) & (samples + na < data.shape[0])]
    if len(samples) == 0:
        return np.zeros((0, nb + na, len(chans)), np.float32)
    rows = samples[:, None] + np.arange(-nb, na)[None, :]
    return data[rows.ravel()][:, chans].reshape(len(samples), nb + na, len(chans)).astype(np.float32) * gains[chans]


def _corr(a, b):
    if a is None or b is None or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return np.nan
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def classify(r: dict, c: dict) -> str:
    if r["n_spikes_laser"] == 0:
        return "no_spikes"
    if r["artefact_flag"]:
        return "artefact"
    up = r["p_locked_up"] < c["alpha"] and r["salt_p"] < c["alpha"] and r["resp_prob"] - r["base_prob"] >= c["min_prob_increase"]
    if up and not r["projection_ok"]:
        return "locked_poor_projection"
    if up and r["evoked_rp_violation_frac"] > c["max_evoked_rp_violation_frac"]:
        return "locked_volley_split"
    if up:
        fast = r["latency_median_ms"] <= c["max_latency_ms"] and r["jitter_sd_ms"] <= c["max_jitter_ms"]
        if not fast:
            return "locked_slow"
        if r["n_evoked_wf"] < c["min_evoked_spikes"]:
            return "fast_few_spikes"
        if not r["wf_r_evoked_spont"] >= c["min_waveform_r"]:
            return "fast_waveform_mismatch"
        return "direct"
    if r["p_locked_down"] < c["alpha"]:
        return "pulse_suppressed"
    if r["train_rate_p"] < c["alpha"]:
        return "train_rate_up" if r["train_rate_ratio"] > 1 else "train_rate_down"
    return "none"


def stage_metrics(cfg: dict, out_dir: Path) -> dict:
    from scipy.stats import binomtest

    d = out_dir / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    rc, cc = cfg["response"], cfg["classify"]
    pulses_all = pd.read_csv(out_dir / "pulses" / "pulses.tsv", sep="\t")
    trains_all = pd.read_csv(out_dir / "pulses" / "trains.tsv", sep="\t")
    pulses, trains = pulses_all, trains_all
    if rc["trains"]:
        pulses = pulses_all[pulses_all.train.isin(rc["trains"])].reset_index(drop=True)
        trains = trains_all[trains_all.train.isin(rc["trains"])].reset_index(drop=True)
        log.info("Analysing trains %s only (%d pulses)", rc["trains"], len(pulses))
    pulses_an, trains_an = pulses, trains
    prim = list(rc["primary_trains"] or trains_an.train)
    pulses = pulses_an[pulses_an.train.isin(prim)].reset_index(drop=True)
    trains = trains_an[trains_an.train.isin(prim)].reset_index(drop=True)
    sec_mask = ~pulses_an.train.isin(prim).values
    onsets_an = pulses_an.onset_s.values
    log.info("Pooled metrics and classes from trains %s (%d pulses); %d pulses in other trains reported as secondary",
             list(trains.train), len(pulses), int(sec_mask.sum()))
    m = np.load(out_dir / "match" / "laser_spikes.npz")
    fs = float(m["sampling_frequency"])
    unit_ids = m["unit_ids"]
    order = np.argsort(m["sample_index"], kind="stable")
    samp, uidx = m["sample_index"][order], m["unit_index"][order]

    # Waveforms come from the data the matching ran on (motion-corrected or not); the artefact check
    # always uses the CatGT data as recorded.
    data, gains = laser_data(cfg)
    laser_bin, laser_meta = catgt_files(cfg["paths"]["laser_catgt"])
    raw = np.memmap(laser_bin, dtype=np.int16, mode="r").reshape(-1, int(laser_meta["nSavedChans"]))
    rec = read_catgt(cfg["paths"]["laser_catgt"], cfg["stream_id"])
    ypos = rec.get_channel_locations()[:, 1]
    duration = raw.shape[0] / fs

    ana_dir = Path(cfg["paths"]["sorted_output_dir"]) / cfg["sorter"] / "analyzer"
    tmpl = np.load(ana_dir / "extensions" / "templates" / "average.npy", mmap_mode="r")
    mask = np.load(ana_dir / "sparsity_mask.npy")
    tparams = json.load(open(ana_dir / "extensions" / "templates" / "params.json"))
    nb_t = int(round(tparams["ms_before"] * fs / 1e3))
    na_t = tmpl.shape[1] - nb_t

    onsets = pulses.onset_s.values
    wn = rc["window_ms"] / 1e3
    cw = np.array(rc["control_window_ms"]) / 1e3
    sham = sham_sets(pulses, rc)
    n_sets = len(sham)
    log.info("SALT baseline: %d disjoint sham sets of %d pulses (%d null distances)", n_sets, len(onsets), n_sets * (n_sets - 1) // 2)

    base_iv, train_iv = [], []
    for _, t in trains.iterrows():
        b1 = t.onset_s - rc["baseline_gap_s"]
        base_iv.append((max(0.0, b1 - rc["baseline_s"]), b1))
        train_iv.append((t.onset_s, t.last_offset_s + (t.period_ms / 1e3 - t.width_ms / 1e3)))
    base_iv, train_iv = np.array(base_iv), np.array(train_iv)
    t_base, t_train = np.sum(np.diff(base_iv)), np.sum(np.diff(train_iv))

    # Pulse-triggered average of the voltage, all channels: photoelectric artefacts at onset / offset.
    pre, post = int(round(0.005 * fs)), int(round(0.015 * fs))
    onsets_all = pulses_all.onset_s.values
    on_samp = np.round(onsets_all * fs).astype(np.int64)
    pta_raw = np.stack([np.asarray(raw[s - pre: s + post, :384]) for s in on_samp]).astype(np.float32)  # (pulses, time, ch)
    pta = pta_raw.mean(axis=0) * gains
    pta -= pta[:pre].mean(axis=0)
    noise_sd = np.median(np.abs(pta_raw[:, :pre, :] * gains), axis=(0, 1)) / 0.6745
    onset_win = slice(pre, pre + int(round(0.002 * fs)))
    offset_win = slice(pre + int(round(0.005 * fs)), pre + int(round(0.007 * fs)))
    art = pd.DataFrame({
        "channel_index": np.arange(384), "y_um": ypos, "x_um": rec.get_channel_locations()[:, 0],
        "onset_pp_uV": np.ptp(pta[onset_win], axis=0), "offset_pp_uV": np.ptp(pta[offset_win], axis=0),
        "baseline_pp_uV": np.ptp(pta[:pre], axis=0), "noise_sd_uV": noise_sd,
        # Photoelectric transients are sub-millisecond and start exactly at the light edges; spikes in the volley come later.
        "onset_fast_pp_uV": np.ptp(pta[pre - int(0.0001 * fs): pre + int(0.0004 * fs)], axis=0),
        "offset_fast_pp_uV": np.ptp(pta[pre + int(0.0049 * fs): pre + int(0.0054 * fs)], axis=0),
    })
    peak_ch = np.array([np.argmax(np.ptp(tmpl[i], axis=0)) for i in range(len(unit_ids))])
    art["dist_to_nearest_unit_um"] = [np.min(np.abs(ypos[peak_ch] - y)) for y in ypos]
    art.to_csv(d / "artefact_channels.tsv", sep="\t", index=False, float_format="%.3f")
    np.savez(d / "pulse_triggered_average.npz", pta_uV=pta, t_ms=(np.arange(-pre, post) / fs) * 1e3)

    psth_edges = np.arange(rc["psth_range_ms"][0], rc["psth_range_ms"][1] + 1e-9, rc["psth_bin_ms"]) / 1e3
    units = sorted_unit_tables(cfg)
    val = pd.read_csv(out_dir / "validate" / "validation.tsv", sep="\t")
    proj_ok = set(val.loc[(val.val_recall >= cc["min_val_recall"]) & (val.val_precision >= cc["min_val_precision"]), "unit_id"])
    # Pulse-triggered average of the data the waveforms come from (primary pulses), all channels. The light drives a
    # large synchronous volley; subtracting this average at each evoked spike's latency removes the time-locked part
    # of other cells' activity from the evoked waveform (it also removes a smeared fraction of the unit's own spikes).
    on_p = np.round(onsets * fs).astype(np.int64)
    pta_data = np.zeros((pre + post, 384), np.float64)
    for s0 in on_p:
        pta_data += data[s0 - pre: s0 + post, :384]
    pta_data = (pta_data / len(on_p) * gains).astype(np.float32)
    pta_data -= pta_data[:pre].mean(axis=0)
    width_ms = float(pulses.width_ms.median())
    rng = np.random.default_rng(0)
    rows, wf_store, psth_store, by_train = [], {}, {}, []
    by_unit = {ui: np.sort(samp[uidx == ui]) for ui in range(len(unit_ids))}
    for ui, u in enumerate(unit_ids):
        s_u = by_unit[ui]
        t = s_u / fs
        r = {"unit_id": u, "n_spikes_laser": len(t), "rate_laser_window_hz": len(t) / duration}
        n_base = int(np.sum(np.searchsorted(t, base_iv[:, 1]) - np.searchsorted(t, base_iv[:, 0])))
        n_train = int(np.sum(np.searchsorted(t, train_iv[:, 1]) - np.searchsorted(t, train_iv[:, 0])))
        r["rate_pre_train_hz"] = n_base / t_base
        r["rate_in_train_hz"] = n_train / t_train
        r["train_rate_ratio"] = r["rate_in_train_hz"] / r["rate_pre_train_hz"] if n_base else np.nan
        r["train_rate_p"] = binomtest(n_train, n_base + n_train, t_train / (t_base + t_train)).pvalue if n_base + n_train else 1.0

        lat = first_after(t, onsets, wn)
        resp = ~np.isnan(lat)
        ctrl = any_in(t, onsets + cw[0], onsets + cw[1])
        sham_lat = np.stack([first_after(t, sh, wn) for sh in sham])
        r["resp_prob"] = resp.mean()
        r["base_prob"] = float(np.mean(~np.isnan(sham_lat)))
        r["control_prob"] = ctrl.mean()
        n10, n01 = int(np.sum(resp & ~ctrl)), int(np.sum(~resp & ctrl))
        r["p_locked_up"] = binomtest(n10, n10 + n01, 0.5, alternative="greater").pvalue if n10 + n01 else 1.0
        r["p_locked_down"] = binomtest(n01, n10 + n01, 0.5, alternative="greater").pvalue if n10 + n01 else 1.0
        r["salt_p"], r["salt_I"] = salt(lat, sham_lat, wn, rc["latency_bin_ms"] / 1e3) if len(t) else (1.0, 0.0)
        first = pulses.pulse.values == 0
        r["resp_prob_first_pulse"] = resp[first].mean()
        r["resp_prob_last10"] = resp[pulses.groupby("train").pulse.transform("max").values - pulses.pulse.values < 10].mean()
        lr = lat[resp] * 1e3
        r["n_responses"] = int(resp.sum())
        r["latency_median_ms"] = float(np.median(lr)) if len(lr) else np.nan
        r["latency_mean_ms"] = float(np.mean(lr)) if len(lr) else np.nan
        r["jitter_sd_ms"] = float(np.std(lr, ddof=1)) if len(lr) > 1 else np.nan
        r["jitter_iqr_ms"] = float(np.subtract(*np.percentile(lr, [75, 25]))) if len(lr) > 1 else np.nan
        r["frac_first_spike_during_pulse"] = float(np.mean(lr < width_ms)) if len(lr) else np.nan
        rpw, rpt = cc["rp_window_ms"] / 1e3, cc["refractory_ms"] / 1e3
        r["evoked_rp_violation_frac"], _ = rp_violation_frac(t, onsets, rpw, rpt)
        r["sham_rp_violation_frac"], _ = rp_violation_frac(t, sham.ravel(), rpw, rpt)

        counts = np.histogram(aligned(t, onsets, psth_edges[0], psth_edges[-1])[0], psth_edges)[0]
        lam = r["rate_pre_train_hz"] * len(onsets) * rc["psth_bin_ms"] / 1e3
        z = (counts - lam) / np.sqrt(lam) if lam > 0 else np.full(len(counts), np.nan)
        centers = (psth_edges[:-1] + psth_edges[1:]) / 2
        w = (centers >= 0) & (centers < wn)
        if np.any(np.isfinite(z[w])):
            k = np.nanargmax(z[w])
            r["psth_peak_z"], r["psth_peak_latency_ms"] = float(z[w][k]), float(centers[w][k] * 1e3)
        else:
            r["psth_peak_z"], r["psth_peak_latency_ms"] = np.nan, np.nan
        psth_store[u] = counts

        # Waveforms on the unit's largest template channels: evoked (first spike 0-window after a pulse),
        # spontaneous (pre-train periods), and the sorted-window template.
        chans = np.argsort(np.ptp(tmpl[ui], axis=0))[::-1][: rc["n_top_channels"]]
        idx = np.flatnonzero(resp)
        if len(idx) > rc["n_waveforms"]:
            idx = np.sort(rng.choice(idx, rc["n_waveforms"], replace=False))
        ev = np.round((onsets[idx] + lat[idx]) * fs).astype(np.int64)
        ev_lag = ev - np.round(onsets[idx] * fs).astype(np.int64)
        in_base = np.zeros(len(t), bool)
        for b0, b1 in base_iv:
            in_base |= (t >= b0) & (t < b1)
        sp = s_u[in_base]
        if len(sp) > rc["n_waveforms"]:
            sp = np.sort(rng.choice(sp, rc["n_waveforms"], replace=False))
        w_ev = _snippets(data, gains, ev, chans, nb_t, na_t)
        w_sp = _snippets(data, gains, sp, chans, nb_t, na_t)
        m_ev_raw = w_ev.mean(axis=0) if len(w_ev) else None
        m_ev = None
        if len(w_ev) and len(w_ev) == len(ev_lag) and pre + ev_lag.max() + na_t <= pta_data.shape[0]:
            bg = np.stack([pta_data[pre + L - nb_t: pre + L + na_t][:, chans] for L in ev_lag])
            m_ev = (w_ev - bg).mean(axis=0)
        m_sp = w_sp.mean(axis=0) if len(w_sp) else None
        m_t = np.asarray(tmpl[ui][:, chans], dtype=np.float32)
        r["n_evoked_wf"], r["n_spont_wf"] = len(w_ev), len(w_sp)
        r["wf_r_evoked_spont"] = _corr(m_ev, m_sp) if m_ev is not None and m_sp is not None else np.nan
        r["wf_r_evoked_spont_raw"] = _corr(m_ev_raw, m_sp) if m_ev_raw is not None and m_sp is not None else np.nan
        r["wf_r_evoked_template"] = _corr(m_ev, m_t) if m_ev is not None else np.nan
        r["wf_r_spont_template"] = _corr(m_sp, m_t) if m_sp is not None else np.nan
        r["amp_ratio_evoked_spont"] = float(np.ptp(m_ev[:, 0]) / np.ptp(m_sp[:, 0])) if m_ev is not None and m_sp is not None else np.nan
        r["amp_template_uV"] = float(np.ptp(m_t[:, 0]))
        r["projection_ok"] = bool(u in proj_ok)
        # Artefact on the unit's own top channel, from pulses where the unit did not fire near the pulse.
        quiet = ~any_in(t, onsets_all - 0.002, onsets_all + 0.012)
        if quiet.sum() >= 20:
            a = pta_raw[quiet][:, :, chans[0]].mean(axis=0) * gains[chans[0]]
            a -= a[:pre].mean()
            r["artefact_own_ch_pp_uV"] = float(max(np.ptp(a[onset_win]), np.ptp(a[offset_win])))
        else:
            r["artefact_own_ch_pp_uV"] = np.nan
        lm = r["latency_median_ms"]
        r["artefact_flag"] = bool(
            r["n_responses"] >= cc["min_evoked_spikes"] and r["p_locked_up"] < cc["alpha"] and (
                lm < cc["artefact_min_latency_ms"]
                or (cc["artefact_offset_window_ms"][0] <= lm <= cc["artefact_offset_window_ms"][1] and r["jitter_sd_ms"] < cc["artefact_max_jitter_ms"])
            )
        )
        r["class"] = classify(r, cc)
        lat_an = first_after(t, onsets_an, wn)
        resp_an = ~np.isnan(lat_an)
        ctrl_an = any_in(t, onsets_an + cw[0], onsets_an + cw[1])
        if sec_mask.any():
            n10, n01 = int(np.sum(resp_an[sec_mask] & ~ctrl_an[sec_mask])), int(np.sum(~resp_an[sec_mask] & ctrl_an[sec_mask]))
            ls = lat_an[sec_mask & resp_an] * 1e3
            r["sec_n_pulses"] = int(sec_mask.sum())
            r["sec_resp_prob"] = resp_an[sec_mask].mean()
            r["sec_control_prob"] = ctrl_an[sec_mask].mean()
            r["sec_p_locked_up"] = binomtest(n10, n10 + n01, 0.5, alternative="greater").pvalue if n10 + n01 else 1.0
            r["sec_latency_median_ms"] = float(np.median(ls)) if len(ls) else np.nan
            r["sec_jitter_sd_ms"] = float(np.std(ls, ddof=1)) if len(ls) > 1 else np.nan
            # Do the secondary trains show the same pulse-locked response (same latency within 1 ms)?
            r["secondary_agrees"] = bool(r["sec_p_locked_up"] < cc["alpha"] and abs(r["sec_latency_median_ms"] - r["latency_median_ms"]) <= 1.0)
        rows.append(r)
        for _, tr in trains_an.iterrows():
            sel = pulses_an.train.values == tr.train
            lt = lat_an[sel & resp_an] * 1e3
            b0, b1 = max(0.0, tr.onset_s - rc["baseline_gap_s"] - rc["baseline_s"]), tr.onset_s - rc["baseline_gap_s"]
            by_train.append({
                "unit_id": u, "train": int(tr.train), "primary": bool(tr.train in prim), "n_pulses": int(sel.sum()),
                "resp_prob": resp_an[sel].mean(), "control_prob": ctrl_an[sel].mean(), "latency_median_ms": float(np.median(lt)) if len(lt) else np.nan,
                "jitter_sd_ms": float(np.std(lt, ddof=1)) if len(lt) > 1 else np.nan,
                "rate_pre_train_hz": (np.searchsorted(t, b1) - np.searchsorted(t, b0)) / (b1 - b0),
                "rate_in_train_hz": (np.searchsorted(t, tr.last_offset_s) - np.searchsorted(t, tr.onset_s)) / (tr.last_offset_s - tr.onset_s),
            })
        wf_store[u] = {"chans": chans, "evoked": m_ev, "evoked_raw": m_ev_raw, "spont": m_sp, "template": m_t}

    res = pd.DataFrame(rows)
    res = units.merge(val, on="unit_id", how="left").merge(res, on="unit_id", how="right")
    lead = ["unit_id", "class", "label", "review_verdict", "peak_channel_index", "depth_um", "val_recall", "val_precision"]
    res = res[lead + [c for c in res.columns if c not in lead]]
    res.to_csv(out_dir / "light_response.tsv", sep="\t", index=False, float_format="%.4g")
    bt = pd.DataFrame(by_train).merge(res[["unit_id", "class"]], on="unit_id")
    bt.to_csv(out_dir / "light_response_by_train.tsv", sep="\t", index=False, float_format="%.4g")
    np.savez_compressed(
        d / "unit_waveforms.npz",
        **{f"{k}_{u}": v[k] for u, v in wf_store.items() for k in ("chans", "evoked", "evoked_raw", "spont", "template") if v[k] is not None},
    )
    np.save(d / "pulse_triggered_average_matched_data.npy", pta_data)
    np.savez_compressed(d / "psth.npz", edges_s=psth_edges, unit_ids=np.array(list(psth_store)), counts=np.array(list(psth_store.values())), n_pulses=len(onsets))
    counts = res["class"].value_counts().to_dict()
    log.info("Classes: %s", counts)
    return {"classes": counts, "n_sham_sets": n_sets}


# ----------------------------------------------------------------------------------------------- report


def _unit_figure(u, row, t, pulses, trains, wf, fs, path, rc, prim):
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    ax = axes[0, 0]
    rel_r, ys = aligned(t, pulses.onset_s.values, rc["psth_range_ms"][0] / 1e3, rc["psth_range_ms"][1] / 1e3)
    ax.axvspan(0, 5, color=C_LIGHT, zorder=0)
    for tr_id in np.unique(pulses.train):
        rows_tr = np.flatnonzero(pulses.train.values == tr_id)
        if tr_id not in prim:
            ax.axhspan(rows_tr[0] - 0.5, rows_tr[-1] + 0.5, color="#ecebe8", zorder=0)
        ax.text(rc["psth_range_ms"][1], rows_tr.mean(), f" {int(tr_id)}", va="center", fontsize=7, color="#52514e")
    ax.scatter(rel_r * 1e3, ys, s=2, c=C_INK, linewidths=0)
    for b in np.flatnonzero(np.diff(pulses.train.values)) + 0.5:
        ax.axhline(b, color=C_GRAY, lw=0.5)
    ax.set_xlim(*rc["psth_range_ms"])
    ax.set_ylim(len(pulses), -1)
    ax.set_xlabel("time from pulse onset (ms)")
    ax.set_ylabel("pulse, by train (gray rows: secondary trains)")
    ax.set_title(f"KS4 #{u}: {row['class']}")
    onsets = pulses.onset_s.values[pulses.train.isin(prim).values]
    rel_all = aligned(t, onsets, rc["psth_range_ms"][0] / 1e3, rc["psth_range_ms"][1] / 1e3)[0] * 1e3

    ax = axes[0, 1]
    edges = np.arange(rc["psth_range_ms"][0], rc["psth_range_ms"][1] + 1e-9, rc["psth_bin_ms"])
    c = np.histogram(rel_all, edges)[0] / (len(onsets) * rc["psth_bin_ms"] / 1e3)
    ax.axvspan(0, 5, color=C_LIGHT, zorder=0)
    ax.bar(edges[:-1], c, width=rc["psth_bin_ms"] * 0.9, align="edge", color=C_BLUE)
    ax.axhline(row["rate_pre_train_hz"], color=C_ORANGE, lw=2, label="pre-train rate")
    ax.set_xlim(*rc["psth_range_ms"])
    ax.set_xlabel("time from pulse onset (ms)")
    ax.set_ylabel("rate (spikes/s)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(f"PSTH, trains {', '.join(str(int(x)) for x in prim)}, {rc['psth_bin_ms']} ms bins")

    ax = axes[0, 2]
    lat = first_after(t, onsets, rc["window_ms"] / 1e3) * 1e3
    lat = lat[~np.isnan(lat)]
    ax.axvspan(0, 5, color=C_LIGHT, zorder=0)
    ax.hist(lat, bins=np.arange(0, rc["window_ms"] + 1e-9, 0.25), color=C_BLUE, rwidth=0.9)
    ax.set_xlabel(f"first-spike latency, 0-{rc['window_ms']:.0f} ms (ms)")
    ax.set_ylabel("pulses")
    ax.set_title(f"median {row['latency_median_ms']:.2f} ms, SD {row['jitter_sd_ms']:.2f} ms")

    ax = axes[1, 0]
    if wf is not None:
        n_show = min(4, wf["template"].shape[1])
        tt = (np.arange(wf["template"].shape[0]) - int(round(1.0 * fs / 1e3))) / fs * 1e3
        scale = np.ptp(wf["template"][:, 0]) * 1.2 or 1
        for k in range(n_show):
            off = -k * scale
            ax.plot(tt, wf["template"][:, k] + off, color=C_INK, lw=1.5, ls="--", label="sorted-window template" if k == 0 else None)
            if wf.get("spont") is not None:
                ax.plot(tt, wf["spont"][:, k] + off, color=C_GRAY, lw=2, label="spontaneous (pre-train)" if k == 0 else None)
            if wf.get("evoked_raw") is not None:
                ax.plot(tt, wf["evoked_raw"][:, k] + off, color=C_BLUE, lw=1, alpha=0.45, label="light-evoked, as recorded" if k == 0 else None)
            if wf.get("evoked") is not None:
                ax.plot(tt, wf["evoked"][:, k] + off, color=C_BLUE, lw=2, label="light-evoked minus pulse average" if k == 0 else None)
            ax.text(tt[-1], off, f" ch{wf['chans'][k]}", va="center", fontsize=8, color="#52514e")
        ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("uV (channels offset)")
    ax.set_title(f"waveforms: r evoked/spont {row['wf_r_evoked_spont']:.2f}")

    ax = axes[1, 1]
    pre_s, post_s = 10.0, 20.0
    allrel = []
    for k, tr in trains.iterrows():
        i0, i1 = np.searchsorted(t, [tr.onset_s - pre_s, tr.onset_s + post_s])
        rel = t[i0:i1] - tr.onset_s
        allrel.append(rel)
        if tr.train not in prim:
            ax.axhspan(k + 0.6, k + 1.4, color="#ecebe8", zorder=0)
        ax.scatter(rel, np.full(len(rel), k + 1), s=1, c=C_INK, linewidths=0)
        ax.plot([0, tr.last_offset_s - tr.onset_s], [k + 1, k + 1], color=C_BLUE, lw=6, alpha=0.35, solid_capstyle="butt")
    ax.set_ylim(len(trains) + 0.7, 0.3)
    ax.set_xlim(-pre_s, post_s)
    ax.set_xlabel("time from train onset (s)")
    ax.set_yticks(range(1, len(trains) + 1), [str(int(x)) for x in trains.train])
    ax.set_ylabel("train")
    ax.set_title(f"train rate x{row['train_rate_ratio']:.2f} (p={row['train_rate_p']:.2g})")

    ax = axes[1, 2]
    ax.axis("off")
    fields = [
        ("bombcell label", row.get("label")), ("review verdict", row.get("review_verdict")),
        ("depth (um)", row.get("depth_um")), ("validation recall / precision", f"{row['val_recall']:.2f} / {row['val_precision']:.2f}"),
        ("spikes in laser window", row["n_spikes_laser"]), ("pre-train rate (Hz)", f"{row['rate_pre_train_hz']:.2f}"),
        ("P(spike 0-10 ms) pulse / sham / 35-45 ms", f"{row['resp_prob']:.3f} / {row['base_prob']:.3f} / {row['control_prob']:.3f}"),
        ("first pulse / last 10 pulses", f"{row['resp_prob_first_pulse']:.2f} / {row['resp_prob_last10']:.2f}"),
        ("SALT p / I", f"{row['salt_p']:.3g} / {row['salt_I']:.3f}"),
        ("secondary trains: P(spike) / latency / agrees", f"{row.get('sec_resp_prob', np.nan):.3f} / {row.get('sec_latency_median_ms', np.nan):.2f} / {row.get('secondary_agrees', '')}"), ("pulse-locked p (up / down)", f"{row['p_locked_up']:.2g} / {row['p_locked_down']:.2g}"),
        ("latency median / mean (ms)", f"{row['latency_median_ms']:.2f} / {row['latency_mean_ms']:.2f}"),
        ("jitter SD / IQR (ms)", f"{row['jitter_sd_ms']:.2f} / {row['jitter_iqr_ms']:.2f}"),
        ("waveform r evoked-spont / evoked-template", f"{row['wf_r_evoked_spont']:.3f} / {row['wf_r_evoked_template']:.3f}"),
        ("evoked / spont spikes averaged", f"{row['n_evoked_wf']} / {row['n_spont_wf']}"),
        ("artefact on own channel (uV pp)", f"{row['artefact_own_ch_pp_uV']:.1f}"),
        ("first spikes during the pulse", f"{row['frac_first_spike_during_pulse']:.2f}"),
        ("pulses with 2 spikes < refractory, evoked / sham", f"{row['evoked_rp_violation_frac']:.2f} / {row['sham_rp_violation_frac']:.2f}"),
        ("waveform r evoked-spont as recorded", f"{row['wf_r_evoked_spont_raw']:.3f}"),
    ]
    note = row.get("review_note")
    txt = "\n".join(f"{k}: {v}" for k, v in fields)
    if isinstance(note, str) and note:
        txt += f"\nreview note: {note[:120]}"
    ax.text(0, 1, txt, va="top", fontsize=8.5, family="monospace")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _trace_figure(u, row, t, pulses, prim, data, gains, chans, fs, path):
    """Voltage on the unit's largest channels around example pulses, with this unit's matched spikes marked."""
    plt = _plt()
    picks = []
    for tr_id in prim:
        p = pulses[pulses.train == tr_id]
        picks += [p.iloc[0], p.iloc[len(p) // 2]] if len(p) else []
    picks = picks[:6]
    fig, axes = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 6), sharey=True, squeeze=False)
    pre, post = int(0.005 * fs), int(0.020 * fs)
    chans = chans[:4]
    for ax, pr in zip(axes[0], picks):
        s0 = int(round(pr.onset_s * fs))
        tr = np.asarray(data[s0 - pre: s0 + post, :384][:, chans], dtype=np.float32) * gains[chans]
        tt = (np.arange(-pre, post)) / fs * 1e3
        step = max(float(np.ptp(tr)) * 0.45, 30.0)
        ax.axvspan(0, pr.offset_s * 1e3 - pr.onset_s * 1e3, color=C_LIGHT, zorder=0)
        for k in range(len(chans)):
            ax.plot(tt, tr[:, k] - k * step, color=C_INK, lw=0.8)
        sp = t[(t >= pr.onset_s - pre / fs) & (t < pr.onset_s + post / fs)]
        for x in (sp - pr.onset_s) * 1e3:
            ax.axvline(x, color=C_ORANGE, lw=1.2, alpha=0.8)
        ax.set_title(f"train {int(pr.train)}, pulse {int(pr.pulse) + 1}", fontsize=9)
        ax.set_xlabel("ms from pulse onset")
    axes[0][0].set_ylabel("uV, channels " + ", ".join(f"ch{c}" for c in chans) + " (top to bottom)")
    fig.suptitle(f"KS4 #{u} ({row['class']}): orange = this unit's matched spikes; blue band = light pulse", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def compare_runs(out_dir: Path, other_dir: Path, trains: pd.DataFrame, min_val: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """This run vs another (e.g. without motion correction): matched spike rate and amplitude per period between trains for
    units whose projection validated well, and the per-unit response in the trains both runs analysed."""
    val = pd.read_csv(out_dir / "validate" / "validation.tsv", sep="\t")
    ok = set(val.loc[(val.val_recall >= min_val) & (val.val_precision >= min_val), "unit_id"])
    edges = np.r_[0.0, trains.onset_s.values, np.inf]
    names = ["before train 1"] + [f"after train {int(k)}" for k in trains.train]
    excl = np.c_[trains.onset_s.values, trains.last_offset_s.values + 1.0]
    out = {}
    for name, folder in (("this", out_dir), ("other", other_dir)):
        m = np.load(folder / "match" / "laser_spikes.npz")
        t = m["sample_index"] / float(m["sampling_frequency"])
        keep = np.isin(m["unit_ids"][m["unit_index"]], list(ok))
        for a, b in excl:
            keep &= ~((t >= a) & (t < b))
        seg = np.searchsorted(edges, t, side="right") - 1
        out[name] = (seg[keep], m["amplitude"][keep], t.max())
    rows = []
    for k, nm in enumerate(names):
        dur = min(edges[k + 1], out["this"][2]) - edges[k] - (excl[k - 1, 1] - excl[k - 1, 0] if k else 0)
        r = {"period": nm, "n_units": len(ok)}
        for name, (seg, amp, _) in out.items():
            r[f"{name}_spikes_per_s"] = np.sum(seg == k) / dur
            r[f"{name}_median_amplitude"] = float(np.median(amp[seg == k])) if np.any(seg == k) else np.nan
        rows.append(r)
    seg_tab = pd.DataFrame(rows)
    a = pd.read_csv(out_dir / "light_response_by_train.tsv", sep="\t")
    b = pd.read_csv(other_dir / "light_response_by_train.tsv", sep="\t")
    j = a.merge(b, on=["unit_id", "train"], suffixes=("_this", "_other"))
    rows = []
    for tr_id, g in j.groupby("train"):
        g = g[g.unit_id.isin(ok)]
        rows.append({"train": int(tr_id), "n_units": len(g), "r_resp_prob": g.resp_prob_this.corr(g.resp_prob_other),
                     "median_abs_diff_resp_prob": float((g.resp_prob_this - g.resp_prob_other).abs().median()),
                     "median_abs_diff_latency_ms": float((g.latency_median_ms_this - g.latency_median_ms_other).abs().median())})
    return seg_tab, pd.DataFrame(rows)


def stage_report(cfg: dict, out_dir: Path) -> dict:
    plt = _plt()
    rc, cc = cfg["response"], cfg["classify"]
    fig_dir = out_dir / "figures"
    for sub in ("units", "traces"):  # file names carry the class, so clear figures of an earlier classification
        (fig_dir / sub).mkdir(parents=True, exist_ok=True)
        for f in (fig_dir / sub).glob("*.png"):
            f.unlink()
    res = pd.read_csv(out_dir / "light_response.tsv", sep="\t")
    pulses_all = pd.read_csv(out_dir / "pulses" / "pulses.tsv", sep="\t")
    trains_all = pd.read_csv(out_dir / "pulses" / "trains.tsv", sep="\t")
    pulses, trains = pulses_all, trains_all
    if rc["trains"]:
        pulses = pulses_all[pulses_all.train.isin(rc["trains"])].reset_index(drop=True)
        trains = trains_all[trains_all.train.isin(rc["trains"])].reset_index(drop=True)
    prim = list(rc["primary_trains"] or trains.train)
    fit = json.load(open(out_dir / "pulses" / "sync_fit.json"))
    minfo = json.load(open(out_dir / "validate" / "validate_info.json"))
    match_info = json.load(open(out_dir / "match" / "match_info.json"))
    drift = json.load(open(out_dir / "drift" / "drift_summary.json")) if (out_dir / "drift" / "drift_summary.json").exists() else None
    segs = pd.read_csv(out_dir / "drift" / "segments_vs_sorted_window.tsv", sep="\t") if (out_dir / "drift" / "segments_vs_sorted_window.tsv").exists() else None
    bt = pd.read_csv(out_dir / "light_response_by_train.tsv", sep="\t")
    m = np.load(out_dir / "match" / "laser_spikes.npz")
    fs = float(m["sampling_frequency"])
    wfz = np.load(out_dir / "metrics" / "unit_waveforms.npz")
    art = pd.read_csv(out_dir / "metrics" / "artefact_channels.tsv", sep="\t")
    pz = np.load(out_dir / "metrics" / "pulse_triggered_average.npz")

    # Summary: latency vs jitter for pulse-locked units.
    locked = res[(res.p_locked_up < cc["alpha"]) & (res.salt_p < cc["alpha"])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    groups = [("direct", C_BLUE), ("locked_slow", C_ORANGE), ("fast_waveform_mismatch", C_AQUA)]
    other = locked[~locked["class"].isin([g for g, _ in groups])]
    ax.scatter(other.latency_median_ms, other.jitter_sd_ms, s=30, c=C_GRAY, label=f"other locked ({len(other)})", edgecolors="#fcfcfb", linewidths=1.5)
    for g, col in groups:
        x = locked[locked["class"] == g]
        ax.scatter(x.latency_median_ms, x.jitter_sd_ms, s=40, c=col, label=f"{g} ({len(x)})", edgecolors="#fcfcfb", linewidths=1.5)
        for _, r in x.iterrows():
            if g == "direct":
                ax.annotate(f"#{r.unit_id}", (r.latency_median_ms, r.jitter_sd_ms), xytext=(4, 2), textcoords="offset points", fontsize=8, color="#52514e")
    ax.axvline(cc["max_latency_ms"], color=C_GRAY, lw=0.8, ls=":")
    ax.axhline(cc["max_jitter_ms"], color=C_GRAY, lw=0.8, ls=":")
    ax.set_xlabel("median first-spike latency (ms)")
    ax.set_ylabel("jitter, SD of first-spike latency (ms)")
    ax.set_title("Pulse-locked units (SALT and late-window test p < %.2g)" % cc["alpha"])
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    x = res[res.n_spikes_laser > 0]
    ax.scatter(x.base_prob, x.resp_prob, s=12, c=C_GRAY, linewidths=0)
    ax.scatter(locked.base_prob, locked.resp_prob, s=20, c=C_BLUE, linewidths=0, label="pulse-locked")
    lim = max(0.05, float(np.nanmax(x[["base_prob", "resp_prob"]].values)) * 1.05)
    ax.plot([0, lim], [0, lim], color=C_INK, lw=0.8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("P(spike in 10 ms), sham pulses before trains")
    ax.set_ylabel("P(spike in 0-10 ms after a pulse)")
    ax.set_title("Response probability, all units")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    v = res[res.val_n_sorter >= 50]
    ax.scatter(v.val_recall, v.val_precision, s=12, c=C_BLUE, linewidths=0)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall of the sorter's spikes")
    ax.set_ylabel("precision (matched spikes the sorter also has)")
    ax.set_title(f"Matching vs sorter, {minfo['validation_window_s'][0]:.0f}-{minfo['validation_window_s'][1]:.0f} s of the sorted window\n(units with >= 50 sorter spikes, n={len(v)})")
    fig.tight_layout()
    fig.savefig(fig_dir / "summary.png", dpi=110)
    plt.close(fig)

    # Artefact figure: pulse-triggered average, channels by depth, and the channels farthest from units.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pta, tms = pz["pta_uV"], pz["t_ms"]
    order = np.argsort(art.y_um.values)
    lim = np.percentile(np.abs(pta), 99.5)
    ax = axes[0]
    im = ax.imshow(pta[:, order].T, aspect="auto", origin="lower", extent=[tms[0], tms[-1], 0, len(order)], cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.axvline(0, color=C_INK, lw=0.8)
    ax.axvline(5, color=C_INK, lw=0.8, ls="--")
    ax.set_xlabel("time from pulse onset (ms)")
    ax.set_ylabel("channel, ordered by depth")
    ax.set_title(f"Pulse-triggered average, all {len(pulses)} pulses (uV)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax = axes[1]
    far = art.sort_values("dist_to_nearest_unit_um", ascending=False).head(6)
    for k, (_, r) in enumerate(far.iterrows()):
        ax.plot(tms, pta[:, int(r.channel_index)] - k * 10, color=C_INK, lw=1)
        ax.text(tms[-1], -k * 10, f" ch{int(r.channel_index)} ({r.dist_to_nearest_unit_um:.0f} um from a unit)", va="center", fontsize=8, color="#52514e")
    ax.axvspan(0, 5, color=C_LIGHT, zorder=0)
    ax.set_xlabel("time from pulse onset (ms)")
    ax.set_ylabel("uV (10 uV between channels)")
    ax.set_title("Channels farthest from any unit")
    fig.tight_layout()
    fig.savefig(fig_dir / "artefact.png", dpi=110)
    plt.close(fig)

    order = np.argsort(m["sample_index"], kind="stable")
    samp, uidx = m["sample_index"][order], m["unit_index"][order]
    unit_ids = list(m["unit_ids"])
    rank = {"direct": 0, "fast_waveform_mismatch": 1, "fast_few_spikes": 2, "locked_slow": 3, "artefact": 4, "pulse_suppressed": 5,
            "locked_volley_split": 6}
    show = res[res["class"].isin(list(rank))].copy()
    show["rank"] = show["class"].map(rank)
    show = show.sort_values(["rank", "salt_I"], ascending=[True, False]).head(cfg["report"]["max_unit_figures"])
    for _, row in show.iterrows():
        u = row.unit_id
        ui = unit_ids.index(u)
        t = np.sort(samp[uidx == ui]) / fs
        wf = None
        if f"chans_{u}" in wfz:
            wf = {k: (wfz[f"{k}_{u}"] if f"{k}_{u}" in wfz else None) for k in ("chans", "evoked", "evoked_raw", "spont", "template")}
        _unit_figure(u, row, t, pulses, trains, wf, fs, fig_dir / "units" / f"{row['class']}_unit{u}.png", rc, prim)

    # Voltage traces around example pulses for the fast pulse-locked units (the check by eye).
    (fig_dir / "traces").mkdir(exist_ok=True)
    data, gains = laser_data(cfg)
    fast = show[show["class"].isin(["direct", "fast_waveform_mismatch", "fast_few_spikes", "locked_volley_split"])].head(cfg["report"]["max_trace_figures"])
    for _, row in fast.iterrows():
        u = row.unit_id
        if f"chans_{u}" not in wfz:
            continue
        t = np.sort(samp[uidx == unit_ids.index(u)]) / fs
        _trace_figure(u, row, t, pulses, prim, data, gains, wfz[f"chans_{u}"], fs, fig_dir / "traces" / f"{row['class']}_unit{u}.png")

    counts = res["class"].value_counts()
    val_ok = res[res.val_n_sorter >= 50]
    lines = [
        "# e097 optotagging: sorted units projected into the laser window",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `analysis/units/optotag.py` from `{cfg['paths']['sorted_output_dir']}` ({cfg['sorter']}).",
        "Per-unit table: `light_response.tsv`. Times are seconds from the start of the laser extract "
        f"(raw run {fit['extract_start_raw_s']:.3f} s).",
        "",
        "## Laser pulses",
        "",
        f"{len(pulses_all)} pulses in {len(trains_all)} trains read from NIDQ XA6, {pulses.width_ms.mean():.2f} ms wide, "
        f"{trains.period_ms.mean():.2f} ms period. Moved onto the imec clock with {fit['n_sync_pairs']} sync edges "
        f"(residual SD {fit['residual_sd_us']:.0f} us, max {fit['residual_max_abs_us']:.0f} us); onset precision is "
        f"limited by the NIDQ sample period ({fit['nidq_sample_period_ms']:.2f} ms).",
        "",
        md_table(trains_all, "{:.3f}"),
        "",
        "## Drift between the laser window and the sorted window",
        "",
    ]
    if drift:
        lines += [
            f"DREDGE ({drift['preset']}, estimate only) on both windows concatenated: median |step| between the windows' "
            f"median displacement {drift['median_abs_step_um']:.1f} um (max over depth bins {drift['max_abs_step_um']:.1f} um). "
            f"Within-window 5-95% range: laser {drift['laser_window_p5_p95_um'][0]:.1f} to {drift['laser_window_p5_p95_um'][1]:.1f} um, "
            f"sorted {drift['sorted_window_p5_p95_um'][0]:.1f} to {drift['sorted_window_p5_p95_um'][1]:.1f} um.",
            "",
        ]
    if segs is not None:
        lines += ["Laser-window displacement relative to the sorted window, per period between trains (um; column = probe y of the "
                  "DREDGE spatial bin, 0 = tip). The tissue moves during trains 3-8 and stays put between trains:", "",
                  md_table(segs, "{:.1f}"), ""]
    lines += [
        "![drift](drift/drift.png)",
        "",
        "## Template matching",
        "",
        ("The laser window was first interpolated into the sorted window's frame with the drift estimate "
         f"({match_info['interpolate_motion_kwargs']}); waveforms below come from that corrected data, the artefact check from the data as recorded."
         if match_info["correct_motion"] else "The laser window was matched as recorded (no motion correction)."),
        "",
        f"{minfo['method']} on locally whitened data ({minfo['whitening']}); templates: {minfo['templates']}. "
        f"All {minfo['n_templates']} sorted units were matched together (noise and non-soma units too, so they absorb "
        f"what they explained in the sort). Unit ids are those of the sort and of `review/{cfg['sorter']}_unit_review.tsv`.",
        "",
        f"Validation against the sorter's own spikes ({minfo['validation_window_s'][0]:.0f}-{minfo['validation_window_s'][1]:.0f} s of the sorted window, "
        f"+-{cfg['matching']['tolerance_ms']} ms, same unit), {len(val_ok)} units with >= 50 sorter spikes: median recall "
        f"{val_ok.val_recall.median():.2f}, median precision {val_ok.val_precision.median():.2f}; "
        f"{int((val_ok.val_recall >= 0.8).sum())} units have recall >= 0.8 and {int(((val_ok.val_recall >= 0.8) & (val_ok.val_precision >= 0.8)).sum())} both >= 0.8. "
        "Read a unit's light response together with its `val_recall` / `val_precision`.",
        "",
        "## Photoelectric artefact",
        "",
        f"Pulse-triggered average over all pulses: largest peak-to-peak at pulse onset (0-2 ms) {art.onset_pp_uV.max():.1f} uV "
        f"(ch {int(art.loc[art.onset_pp_uV.idxmax(), 'channel_index'])}), median over channels {art.onset_pp_uV.median():.1f} uV; "
        f"at offset (5-7 ms) max {art.offset_pp_uV.max():.1f} uV, median {art.offset_pp_uV.median():.1f} uV; "
        f"pre-pulse baseline median {art.baseline_pp_uV.median():.1f} uV. The average includes time-locked spikes, so channels "
        "near responding units show those too; the panel on the right shows the channels farthest from any unit.",
        "",
        "Fast transients right at the light edges (-0.1 to 0.4 ms after onset, 4.9 to 5.4 ms; `onset_fast_pp_uV`, `offset_fast_pp_uV` "
        "in `metrics/artefact_channels.tsv`), channels above 3x the median:",
        "",
        md_table(art.loc[art.onset_fast_pp_uV > 3 * art.onset_fast_pp_uV.median(),
                         ["channel_index", "y_um", "x_um", "onset_fast_pp_uV", "offset_fast_pp_uV", "noise_sd_uV"]]
                 .sort_values("onset_fast_pp_uV", ascending=False).head(15), "{:.1f}"),
        "",
        "![artefact](figures/artefact.png)",
        "",
        "## Light responses",
        "",
        "Criteria (config `classify`): pulse-locked = SALT p < alpha (first-spike latency in 0-10 ms after pulses vs disjoint sham-pulse sets "
        f"on the same grid in the {rc['baseline_s']:.0f} s before each train) and more spikes 0-10 ms than "
        f"{rc['control_window_ms'][0]:.0f}-{rc['control_window_ms'][1]:.0f} ms after the same pulses (one-sided binomial on discordant pulses), "
        f"with P(spike) at least {cc['min_prob_increase']} above sham. A pulse-locked unit whose projection did not reproduce the sorter "
        f"(validation recall < {cc['min_val_recall']} or precision < {cc['min_val_precision']}) is `locked_poor_projection`; one whose matched "
        f"spikes in the {cc['rp_window_ms']:.0f} ms after a pulse include two closer than {cc['refractory_ms']} ms on more than "
        f"{cc['max_evoked_rp_violation_frac']:.0%} of responding pulses is `locked_volley_split` (template matching split a burst of "
        "overlapping spikes; that spike train is not one neuron's, see `evoked_rp_violation_frac` vs `sham_rp_violation_frac`). `direct` "
        f"additionally needs median latency <= {cc['max_latency_ms']} ms, jitter SD <= {cc['max_jitter_ms']} ms and evoked-vs-spontaneous "
        f"waveform r >= {cc['min_waveform_r']} (>= {cc['min_evoked_spikes']} evoked spikes); the evoked waveform has the pulse-triggered "
        f"average (same latency) subtracted first, because the light drives a large synchronous volley (`wf_r_evoked_spont_raw` is without "
        f"that subtraction). `artefact` = locked with median latency < {cc['artefact_min_latency_ms']} ms, or locked to pulse offset "
        f"({cc['artefact_offset_window_ms'][0]}-{cc['artefact_offset_window_ms'][1]} ms) with SD < {cc['artefact_max_jitter_ms']} ms. "
        f"alpha = {cc['alpha']}.",
        "",
        f"Trains analysed: {', '.join(str(int(x)) for x in trains.train)} ({len(pulses)} pulses). Pooled metrics and classes use "
        f"trains {', '.join(str(int(x)) for x in prim)} ({int(pulses.train.isin(prim).sum())} pulses)"
        + ("; the other trains are secondary: `sec_*` columns, and `secondary_agrees` = pulse-locked there too with median latency "
           "within 1 ms." if len(prim) < len(trains) else "."),
        "",
        "| class | units |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in counts.items()],
        "",
    ]
    cols = ["unit_id", "class", "label", "review_verdict", "depth_um", "val_recall", "val_precision", "rate_pre_train_hz", "resp_prob", "base_prob",
            "salt_p", "latency_median_ms", "jitter_sd_ms", "wf_r_evoked_spont", "evoked_rp_violation_frac", "train_rate_ratio"]
    cols += [c for c in ("sec_resp_prob", "sec_latency_median_ms", "secondary_agrees") if c in res.columns]
    tab = res[res["class"].isin(["direct", "fast_waveform_mismatch", "fast_few_spikes", "locked_slow", "artefact"])][cols].sort_values(["class", "latency_median_ms"])
    if len(tab):
        lines += ["Pulse-locked units:", "", md_table(tab), ""]
        per = bt[bt.unit_id.isin(tab.unit_id)].pivot(index="unit_id", columns="train", values="resp_prob")
        per.columns = [f"train {c}" for c in per.columns]
        lines += ["P(spike 0-10 ms after a pulse) per train for those units (`light_response_by_train.tsv` has latency, "
                  "jitter and rates per train):", "", md_table(per.reset_index(), "{:.2f}"), ""]
    figs = sorted((fig_dir / "units").glob("*.png"))
    lines += ["![summary](figures/summary.png)", "", "Examples (all in `figures/units/`, voltage traces in `figures/traces/`):", ""]
    for p in [f for f in figs if f.name.startswith("direct")][:4] + [f for f in figs if not f.name.startswith("direct")][:2]:
        lines += [f"![{p.stem}](figures/units/{p.name})", ""]
        tp = fig_dir / "traces" / p.name
        if tp.exists():
            lines += [f"![traces {p.stem}](figures/traces/{p.name})", ""]
    other = cfg["report"]["compare_with"]
    if other and (Path(other) / "light_response_by_train.tsv").exists():
        seg_tab, tr_tab = compare_runs(out_dir, Path(other), trains_all)
        seg_tab.to_csv(out_dir / "report" / "compare_runs_periods.tsv", sep="\t", index=False, float_format="%.4g")
        tr_tab.to_csv(out_dir / "report" / "compare_runs_trains.tsv", sep="\t", index=False, float_format="%.4g")
        lines += ["## Comparison with the run in `" + str(other) + "`", "",
                  "Matched spikes outside the trains, units with validation recall and precision >= 0.9. Where the drift was large "
                  "(before train 7), a correction that works finds more spikes, with match amplitudes like those after train 7, "
                  "where both runs agree:", "",
                  md_table(seg_tab, "{:.3g}"), "", "Per-unit response in the trains both runs analysed:", "", md_table(tr_tab, "{:.3g}"), ""]
    lines += [
        "## Caveats",
        "",
        "- The light caused apneas, so firing rates change during trains for network reasons; `train_rate_*` columns report that "
        "separately from pulse-locked responses.",
        "- Spontaneous spikes that fall 0-10 ms after a pulse add to the latency spread, so jitter is overestimated for fast-firing units "
        "(compare `resp_prob` with `base_prob`).",
        "- A unit is only as good as its projection: low `val_recall` / `val_precision` means the laser-window spike train is incomplete or "
        "contaminated.",
        "- Units that fire only in the laser window (not in the sorted window) cannot be found this way.",
        "- Each pulse evokes a large synchronous volley (see the pulse-triggered average); template matching fits overlapping spikes "
        "during it less reliably than in spontaneous activity, so check fast units on the voltage traces (`figures/traces/`).",
        "",
        "## Reproduce",
        "",
        "```",
        "python optotag.py configs/e097_optotag.yaml",
        "```",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"n_unit_figures": len(show)}


# ------------------------------------------------------------------------------------------------- main


RUNNERS = {"pulses": stage_pulses, "drift": stage_drift, "validate": stage_validate, "match": stage_match, "metrics": stage_metrics, "report": stage_report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--redo", choices=STAGES, help="re-run this stage and every stage after it")
    parser.add_argument("--only", choices=STAGES, help="run just this stage (re-runs it if done)")
    parser.add_argument("--stop-after", choices=STAGES)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "logs" / f"optotag_{time.strftime('%Y%m%d_%H%M%S')}.log")],
    )
    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    stages = [args.only] if args.only else STAGES
    redo_from = STAGES.index(args.redo) if args.redo else len(STAGES)
    for stage in stages:
        (out_dir / stage).mkdir(exist_ok=True)
        if not args.only and stage_done(out_dir, stage).exists() and STAGES.index(stage) < redo_from:
            log.info("Stage %s already done, skipping", stage)
        else:
            log.info("=== %s ===", stage)
            t = time.time()
            info = RUNNERS[stage](cfg, out_dir)
            mark_done(out_dir, stage, {"seconds": time.time() - t, **(info or {})})
        if args.stop_after == stage:
            break
    return 0


if __name__ == "__main__":  # required on Windows: worker processes re-import this file
    sys.exit(main())
