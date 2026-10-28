"""Motion map from the units an uncorrected sort split across a tissue move, and the recording corrected with it.

When the tissue jumps relative to the probe (m2020-19: during gasping), an uncorrected sort ends each neuron's cluster
at the jump and starts a new one at the shifted position; the two clusters have the same waveform, shifted along the
probe. `build` finds those pairs (units firing before and silent after, matched to units silent before and firing
after, by template cosine over whole-row shifts), takes each pair's displacement (difference of the two units'
monopolar positions) as an anchor at that depth, smooths the anchors into a displacement-by-depth profile, and writes a
SpikeInterface Motion: 0 before the move, the profile after it, and a straight line across --ramp (where spikes are too
sparse to time the move). With several moves (--states, --ramps), each later state gets its own profile against the
first state, joined by straight lines across the moves. Spike-density registration (DREDGE) can't do this for m2020-19: it fails during gasping and
gets the upper half of the probe wrong (drift_check_2026-10-26 in the uncorrected run).

    python anchor_motion.py build <uncorrected run> --before 20-90 --after 145-275 --ramp 98-117 ^
        --ramp-top 100-140 --ramp-blend 2000-2900 --out configs/motion/m2020-19_g4_anchor_map
    python anchor_motion.py build <uncorrected run> --states 20-90 140-190 245-315 382-400 ^
        --ramps 104-114 190-235 318-381 --out configs/motion/m2020-19_g3_anchor_map
    python anchor_motion.py apply configs/motion/m2020-19_g4_anchor_map <uncorrected scratch>/preprocessed <out folder>

build writes the Motion folder <out> (temporal_bins_s in seconds from the start of the sorted window, i.e. file time),
<out>/anchors.tsv (every candidate pair; `used` marks the anchors in the profile) and <out>/anchor_map.png. apply shifts
the preprocessed recording by the map (kriging, as SpikeInterface's DREDGE correction) and saves it as a binary
folder: the input (`source: si_folder`) of the corrected sort.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("anchor_motion")

LABELS = ("good", "mua", "non_soma")   # units that can anchor (not noise)
MIN_RATE_HZ = 2.0                       # firing in the window it belongs to
SILENT_FRAC = 0.1                       # ... and below this fraction of that rate in the other window
N_CH = 12                               # template channels compared (the unit's largest)
MAX_ROWS = 10                           # shifts tried: +-10 rows of 20 um
ROW_UM = 20.0                           # Neuropixels 1.0: 2 sites per 20 um row, so one row = 2 channels
MIN_COS = 0.88                          # m2020-19_g4: 0.95 kept only 277-1818 um; 0.88-0.95 pairs agree with them
                                        # and fill the upper probe (wrong-row matches fail the neighbour check)
MAX_DX_UM = 20.0                        # same column position (monopolar x) within this
MAX_ROW_MISMATCH_UM = 15.0              # the position difference agrees with the matched row shift within this
NEIGHBOUR_UM = 400.0                    # outliers: > OUTLIER_UM from the median of the anchors within this distance
OUTLIER_UM = 20.0
PROFILE_STEP_UM = 50.0
PROFILE_HALF_WIDTH_UM = 250.0
INTERPOLATION = dict(border_mode="force_extrapolate", spatial_interpolation_method="kriging", sigma_um=20.0, p=2)


def window(s: str) -> tuple[float, float]:
    a, b = (float(x) for x in s.split("-"))
    return a, b


def rates(spike_times, spike_clusters, ids, w):
    m = (spike_times >= w[0]) & (spike_times < w[1])
    counts = np.bincount(spike_clusters[m], minlength=int(spike_clusters.max()) + 1)
    return pd.Series(counts[ids] / (w[1] - w[0]), index=ids)


def cosine(a, b):
    n = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / n) if n > 0 else 0.0


def match_pairs(run: Path, sorter: str, before, after) -> pd.DataFrame:
    """Every unit that stops firing across the move, with its best shifted partner among the units that start."""
    import spikeinterface.core as si

    sdir = run / sorter
    units = pd.read_csv(sdir / "units.tsv", sep="\t").set_index("unit_id")
    st = np.load(sdir / "alf" / "spike_times.npy")
    sc = np.load(sdir / "alf" / "spike_clusters.npy").astype(np.int64)
    ids = units.index[units["label"].isin(LABELS)].to_numpy()
    rb, ra = rates(st, sc, ids, before), rates(st, sc, ids, after)
    gone = ids[(rb >= MIN_RATE_HZ) & (ra < SILENT_FRAC * rb)]
    new = ids[(ra >= MIN_RATE_HZ) & (rb < SILENT_FRAC * ra)]
    log.info("%d units stop firing across the move, %d start", len(gone), len(new))

    an = si.load_sorting_analyzer(sdir / "analyzer", load_extensions=False)
    templates = an.get_extension("templates").get_data(operator="average")
    index = {u: i for i, u in enumerate(an.unit_ids)}
    n_chan = templates.shape[2]
    cands = []
    for u in gone:
        tu = templates[index[u]]
        ch = np.sort(np.argsort(np.abs(tu).max(axis=0))[::-1][:N_CH])
        for v in new:
            tv = templates[index[v]]
            for k in range(-MAX_ROWS, MAX_ROWS + 1):
                c2 = ch + 2 * k
                if c2.min() < 0 or c2.max() >= n_chan:
                    continue
                cands.append((cosine(tu[:, ch], tv[:, c2]), u, v, k * ROW_UM))
    # one-to-one, best cosine first
    rows, used_u, used_v = [], set(), set()
    for c, u, v, s in sorted(cands, reverse=True):
        if u in used_u or v in used_v:
            continue
        used_u.add(u)
        used_v.add(v)
        rows.append({"unit_before": int(u), "unit_after": int(v), "cosine": round(c, 4), "row_shift_um": s,
                     "depth_before_um": units.loc[u, "depth_um"], "depth_after_um": units.loc[v, "depth_um"],
                     "dx_um": units.loc[v, "x_um"] - units.loc[u, "x_um"],
                     "label_before": units.loc[u, "label"], "label_after": units.loc[v, "label"],
                     "rate_before_hz": round(rb[u], 2), "rate_after_hz": round(ra[v], 2)})
    df = pd.DataFrame(rows)
    df["displacement_um"] = df["depth_after_um"] - df["depth_before_um"]
    df["depth_um"] = df["depth_before_um"]
    return df.sort_values("depth_um").reset_index(drop=True)


def select_anchors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["reason"] = ""
    df.loc[df["cosine"] < MIN_COS, "reason"] = f"cosine < {MIN_COS}"
    df.loc[(df["reason"] == "") & (df["dx_um"].abs() > MAX_DX_UM), "reason"] = f"|dx| > {MAX_DX_UM:.0f} um"
    df.loc[(df["reason"] == "") & ((df["displacement_um"] - df["row_shift_um"]).abs() > MAX_ROW_MISMATCH_UM),
           "reason"] = "position and row shift disagree"
    for _ in range(2):                               # drop anchors far from their neighbours, then once more
        ok = df["reason"] == ""
        for i in df.index[ok]:
            near = ok & ((df["depth_um"] - df.loc[i, "depth_um"]).abs() <= NEIGHBOUR_UM) & (df.index != i)
            if near.sum() >= 2:
                ref = df.loc[near, "displacement_um"].median()
                if abs(df.loc[i, "displacement_um"] - ref) > OUTLIER_UM:
                    df.loc[i, "reason"] = f"> {OUTLIER_UM:.0f} um from neighbours ({ref:+.0f})"
    df["used"] = df["reason"] == ""
    return df


def profile(anchors: pd.DataFrame, top_um: float):
    """Displacement on a depth grid: median of the anchors within PROFILE_HALF_WIDTH_UM, linear between, flat beyond."""
    grid = np.arange(0.0, top_um + PROFILE_STEP_UM, PROFILE_STEP_UM)
    y, d = anchors["depth_um"].to_numpy(), anchors["displacement_um"].to_numpy()
    val = np.full(grid.size, np.nan)
    for i, g in enumerate(grid):
        near = np.abs(y - g) <= PROFILE_HALF_WIDTH_UM
        if near.any():
            val[i] = np.median(d[near])
    have = np.isfinite(val)
    val = np.interp(grid, grid[have], val[have])      # np.interp holds the end values beyond the outer anchors
    return grid, val


def time_course(tb, ramps, profiles):
    """Displacement over time: profile k between moves, a straight line from profile k to k+1 across ramp k."""
    disp = np.empty((tb.size, profiles[0].size))
    for j, t in enumerate(tb):
        inside = [i for i, (r0, r1) in enumerate(ramps) if r0 <= t < r1]
        if inside:
            i = inside[0]
            w = (t - ramps[i][0]) / (ramps[i][1] - ramps[i][0])
            disp[j] = (1 - w) * profiles[i] + w * profiles[i + 1]
        else:
            disp[j] = profiles[sum(t >= r1 for _, r1 in ramps)]
    return disp


def build(a):
    run = Path(a.run)
    cfg = yaml.safe_load((run / "config_used.yaml").read_text())
    duration = float(cfg["input"]["duration_s"])
    if a.states:
        states, ramps = [window(s) for s in a.states], [window(r) for r in (a.ramps or [])]
        if len(states) < 2 or len(ramps) != len(states) - 1:
            raise SystemExit("--states needs the reference window and at least one more; --ramps one window per move")
        if a.ramp_top:
            raise SystemExit("--ramp-top only works with --before/--after/--ramp (one move)")
    else:
        if not (a.before and a.after and a.ramp):
            raise SystemExit("give --before, --after and --ramp, or --states and --ramps")
        states, ramps = [window(a.before), window(a.after)], [window(a.ramp)]
    import spikeinterface.core as si
    from spikeinterface.core.motion import Motion

    an = si.load_sorting_analyzer(run / a.sorter / "analyzer", load_extensions=False)
    top = float(an.get_channel_locations()[:, 1].max())
    profiles, tables = [], []
    for st in states[1:]:                             # each later state against the reference (first) state
        pairs = select_anchors(match_pairs(run, a.sorter, states[0], st))
        pairs.insert(0, "state_s", f"{st[0]:.0f}-{st[1]:.0f}")
        anchors = pairs[pairs["used"]]
        if len(anchors) < 3:
            raise SystemExit(f"state {st}: only {len(anchors)} anchors; not enough for a map")
        grid, prof = profile(anchors, top)
        profiles.append(prof)
        tables.append(pairs)
        log.info("state %.0f-%.0f s: %d of %d pairs used as anchors; profile %+.0f um (tip) .. %+.0f um (top)",
                 st[0], st[1], len(anchors), len(pairs), prof[0], prof[-1])
    profiles = [np.zeros_like(profiles[0])] + profiles
    tb = np.arange(0.5, duration, 1.0)                # 1 s bins, file time
    if a.ramp_top:
        # the move's timing may depend on depth (m2020-19_g4: deep tissue 98-117 s, upper tissue ~100-140 s): --ramp
        # applies at and below --ramp-blend's lower depth, --ramp-top at and above its upper depth, blended in between
        ramp, ramp_top = ramps[0], window(a.ramp_top)
        blend = window(a.ramp_blend) if a.ramp_blend else (0.0, 1.0)
        f = np.clip((grid - blend[0]) / (blend[1] - blend[0]), 0.0, 1.0)
        start = (1 - f) * ramp[0] + f * ramp_top[0]
        end = (1 - f) * ramp[1] + f * ramp_top[1]
        disp = np.clip((tb[:, None] - start[None, :]) / (end - start)[None, :], 0.0, 1.0) * profiles[1][None, :]
    else:
        ramp_top, blend = None, None
        disp = time_course(tb, ramps, profiles)
    out = Path(a.out)
    Motion(disp.astype("float32"), tb, grid, direction="y").save(out, overwrite=True)
    pairs = pd.concat(tables, ignore_index=True)
    pairs.to_csv(out / "anchors.tsv", sep="\t", index=False, float_format="%.2f")
    meta = {"source_run": str(run), "sorter": a.sorter, "states_s": states, "ramps_s": ramps,
            "ramp_top_s": ramp_top, "ramp_blend_um": blend,
            "n_anchors": {t["state_s"].iloc[0]: int(t["used"].sum()) for t in tables},
            "time_base": "temporal_bins_s are seconds from the start of the sorted window (file time)",
            "profiles_um": {f"{s[0]:.0f}-{s[1]:.0f}": {f"{g:.0f}": round(float(v), 1) for g, v in zip(grid, p)}
                            for s, p in zip(states[1:], profiles[1:])}}
    (out / "anchor_map.json").write_text(json.dumps(meta, indent=1))
    plot(pairs, grid, profiles[1:], states[1:], tb, disp, ramps, out / "anchor_map.png", run.name)
    print(pairs[pairs["used"]].round(2).to_string(index=False))


def plot(pairs, grid, profiles, states, tb, disp, ramps, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#2a78d6", "#d93636", "#1baf7a", "#eda100", "#6250d6"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw=dict(width_ratios=[1, 1.4]))
    u = pairs["used"]
    ax[0].scatter(pairs.loc[~u, "displacement_um"], pairs.loc[~u, "depth_um"], s=14, facecolors="none",
                  edgecolors="#9a9a96", label="not used")
    for k, (st, prof) in enumerate(zip(states, profiles)):
        name = f"{st[0]:.0f}-{st[1]:.0f}"
        m = u & (pairs["state_s"] == name)
        c = colours[k % len(colours)]
        ax[0].scatter(pairs.loc[m, "displacement_um"], pairs.loc[m, "depth_um"], s=24, color=c)
        ax[0].plot(prof, grid, color=c, lw=1.5, label=f"profile, {name} s")
    ax[0].set_xlabel("displacement vs the reference window (um, + = away from the tip)")
    ax[0].set_ylabel("depth in the reference window (um from the tip)")
    ax[0].legend(loc="lower right", fontsize=8)
    im = ax[1].imshow(disp.T, aspect="auto", origin="lower", extent=[0, tb[-1] + 0.5, grid[0], grid[-1]],
                      cmap="magma")
    for r in ramps:
        ax[1].axvspan(*r, color="w", alpha=0.15, lw=0)
    ax[1].set_xlabel("file time (s)")
    ax[1].set_ylabel("depth (um)")
    fig.colorbar(im, ax=ax[1], label="displacement (um)")
    fig.suptitle(f"{title}: motion map from units split across the moves (shaded: moves)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def apply(a):
    import spikeinterface.core as si
    from spikeinterface.core.motion import Motion
    from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording

    rec = si.load(a.preprocessed)
    mo = Motion.load(a.map)
    t0 = rec.get_start_time()
    mo = Motion([np.asarray(d) for d in mo.displacement], [np.asarray(t) + t0 for t in mo.temporal_bins_s],
                np.asarray(mo.spatial_bins_um), direction=mo.direction)
    rec_float = rec if rec.get_dtype().kind == "f" else rec.astype("float32")
    corrected = InterpolateMotionRecording(rec_float, mo, **INTERPOLATION)
    job = dict(n_jobs=a.n_jobs, chunk_duration="1s", progress_bar=False, mp_context="spawn")
    corrected.save(folder=a.out, format="binary", dtype=str(rec.get_dtype()), overwrite=True, **job)
    (Path(a.out) / "anchor_motion_applied.json").write_text(json.dumps(
        {"map": str(Path(a.map).resolve()), "preprocessed": str(Path(a.preprocessed).resolve()),
         "interpolation": INTERPOLATION, "recording_start_s": t0}, indent=1))
    log.info("corrected recording saved to %s", a.out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="motion map from an uncorrected sort")
    b.add_argument("run", help="output folder of the uncorrected sort")
    b.add_argument("--sorter", default="kilosort4")
    b.add_argument("--before", help="file-time window before the move, e.g. 20-90")
    b.add_argument("--after", help="file-time window after the move, e.g. 145-275")
    b.add_argument("--ramp", help="file-time window over which the move happens, e.g. 98-117")
    b.add_argument("--states", nargs="+", help="several moves: windows where the tissue holds still, the first being "
                                               "the reference (e.g. 20-90 140-190 245-315 382-400)")
    b.add_argument("--ramps", nargs="+", help="with --states: the window of each move (e.g. 104-114 190-235 318-381)")
    b.add_argument("--ramp-top", default=None, help="a different window for the upper probe, e.g. 100-140")
    b.add_argument("--ramp-blend", default=None,
                   help="depths (um) between which --ramp turns into --ramp-top, e.g. 2000-2900")
    b.add_argument("--out", required=True, help="Motion folder to write")
    ap = sub.add_parser("apply", help="correct a preprocessed recording with a map")
    ap.add_argument("map", help="Motion folder from build")
    ap.add_argument("preprocessed", help="the uncorrected run's scratch preprocessed folder")
    ap.add_argument("out", help="folder for the corrected recording")
    ap.add_argument("--n-jobs", type=int, default=12)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build(a) if a.cmd == "build" else apply(a)
    return 0


if __name__ == "__main__":  # required on Windows: worker processes re-import this file
    sys.exit(main())
