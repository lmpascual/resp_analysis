"""Split the recovery after anoxia into epochs at change points in breathing and population state.

Reads the outputs of population_dynamics.py (breaths.tsv, hmm_fits.npz) and finds, between the end of gasping and the
end of the window, the boundaries that best split the per-breath features into n segments (exact least-squares
segmentation by dynamic programming, minimum segment length MIN_SEG breaths). Features per breath cycle, each
smoothed with a 5-breath centred median/mean and z-scored over the span:
  breathing   log breath interval, inspiratory duration
  population  fraction of the cycle spent in each state of the reference HMM (Viterbi path)
The two groups get equal weight. Change points are reported for breathing alone, population alone and both; the
combined set is written as the epochs file that population_dynamics.py --conditions-file reads. The first segment
(the ~15 breaths right after the last gasp, still in the gasping state) is kept out of every condition.
e097 anoxia #1 (2026-10-21): breathing alone 1876 / 1916 / 2028 / 2072 s, both 1873 / 1913 / 2029 / 2072 s. Names:
early recovery, mid recovery, slow breathing (107/min, 3 of its 78 breaths are sighs by cibrrig's rule), later recovery.

    C:/Users/lpascu/AppData/Local/miniforge3/envs/cibrrig/python.exe epoch_change_points.py ^
        C:/Users/lpascu/resp_analysis/outputs/e097/population_dynamics_v2 --ref-k 6 --n-segments 5 ^
        --out C:/Users/lpascu/resp_analysis/outputs/e097/population_dynamics_v2/epochs_anoxia1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MIN_SEG = 15            # breaths
SMOOTH = 5              # breaths
OFFSET_S = 1740.0       # file time -> raw-run time
NAMES = ["post-gasp transition", "early recovery", "mid recovery", "slow breathing", "later recovery"]
UNASSIGNED = {"post-gasp transition"}   # segments kept out of every condition (like the old transition breaths)
# condition colours: gray reference, reds for the hypoxic challenge, greens for recovery, yellow for the slow-breathing
# stretch. Every time-adjacent pair passes validate_palette.js (CVD and normal vision); the gray has no chroma on
# purpose. In all-condition overlays red vs dark green merge for protanopes, so those panels rely on labels.
COLOURS = {"baseline": "#6a6a66", "anoxia": "#f2918a", "gasping": "#d93636", "early recovery": "#77c06a",
           "mid recovery": "#1c9157", "slow breathing": "#eda100", "later recovery": "#0f6b3a"}
FIXED = [  # conditions before the recovery, raw-run time; gasping is the Bush & Ramirez 2024 rule (gasp flag)
    {"name": "baseline", "start_raw_s": 1740.0, "end_raw_s": 1800.0, "source": "60 s before the logged N2 switch (30:00)"},
    {"name": "anoxia", "start_raw_s": 1800.0, "end_raw_s": 1825.0, "source": "logged 15 s N2 (30:00-30:15) + 10 s"},
    {"name": "gasping", "rule": "gasp_flag", "source": "Bush & Ramirez 2024: 7-breath median IBI > 1 s until < 0.85 s"},
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("analysis_dir", help="population_dynamics.py output folder (breaths.tsv, hmm_fits.npz)")
    p.add_argument("--ref-k", type=int, default=6)
    p.add_argument("--n-segments", type=int, default=5)
    p.add_argument("--names", default=",".join(NAMES))
    p.add_argument("--out", required=True)
    return p.parse_args()


def segment(Z, n_seg, min_len):
    """Exact least-squares segmentation of rows of Z into n_seg contiguous segments; returns start indices."""
    n = Z.shape[0]
    cs = np.vstack([np.zeros(Z.shape[1]), np.cumsum(Z, axis=0)])
    cs2 = np.r_[0.0, np.cumsum((Z ** 2).sum(axis=1))]

    def cost(i, j):          # rows i..j-1
        s = cs[j] - cs[i]
        return cs2[j] - cs2[i] - (s @ s) / (j - i)
    C = np.full((n_seg + 1, n + 1), np.inf)
    arg = np.zeros((n_seg + 1, n + 1), int)
    C[0, 0] = 0.0
    for m in range(1, n_seg + 1):
        for j in range(m * min_len, n + 1):
            best, bi = np.inf, -1
            for i in range((m - 1) * min_len, j - min_len + 1):
                v = C[m - 1, i] + cost(i, j)
                if v < best:
                    best, bi = v, i
            C[m, j], arg[m, j] = best, bi
    starts, j = [], n
    for m in range(n_seg, 0, -1):
        i = arg[m, j]
        starts.append(i)
        j = i
    return sorted(starts), float(C[n_seg, n])


def main():
    a = parse_args()
    d_in = Path(a.analysis_dir)
    b = pd.read_csv(d_in / "breaths.tsv", sep="\t")
    fits = np.load(d_in / "hmm_fits.npz")
    k, dt = a.ref_k, float(fits["bin_s"])
    seq = fits[f"k{k}_states"]
    ons = b["on_sec"].to_numpy()
    gasp = b["gasp"].to_numpy().astype(bool)
    first = int(np.flatnonzero(gasp).max()) + 1                     # first breath after the last gasp
    last = len(b) - 2                                               # last breath with a complete cycle
    idx = np.arange(first, last + 1)
    mix = np.stack([np.bincount(seq[int(ons[i] / dt):int(ons[i + 1] / dt)], minlength=k) for i in idx]).astype(float)
    mix /= np.maximum(mix.sum(axis=1, keepdims=True), 1)
    roll = lambda x, f: pd.Series(x).rolling(SMOOTH, center=True, min_periods=1).agg(f).to_numpy()
    breath = np.column_stack([roll(np.log(b["ibi_s"].to_numpy()[idx]), "median"),
                              roll(b["duration_sec"].to_numpy()[idx], "median")])
    pop = np.column_stack([roll(mix[:, s], "mean") for s in range(k) if mix[:, s].max() > 0.02])
    z = lambda X: (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9) / np.sqrt(X.shape[1])
    feats = {"breathing": z(breath), "population": z(pop), "both": np.hstack([z(breath), z(pop)])}
    names = a.names.split(",")
    result = {}
    for key, Z in feats.items():
        starts, cost = segment(Z, a.n_segments, MIN_SEG)
        result[key] = {"start_breath": [int(idx[s]) for s in starts],
                       "start_raw_s": [round(float(ons[idx[s]] + OFFSET_S), 2) for s in starts], "cost": round(cost, 2)}
        print(f"{key:<11} starts (raw s): {result[key]['start_raw_s']}")
    starts = result["both"]["start_breath"] + [last + 1]
    rec = []
    for m, nm in enumerate(names):
        i0, i1 = starts[m], starts[m + 1]
        end = ons[i1] if i1 < len(b) else ons[-1]
        rec.append({"name": nm, "start_raw_s": round(float(ons[i0] + OFFSET_S), 3),
                    "end_raw_s": round(float(end + OFFSET_S), 3), "n_breaths": int(i1 - i0),
                    "median_rate_per_min": round(float(60 / np.median(b["ibi_s"].to_numpy()[i0:i1])), 1),
                    "source": "change point in breathing + population state (epoch_change_points.py)",
                    **({"include": False} if nm in UNASSIGNED else {})})
    conditions = [{**c, **({"colour": COLOURS[c["name"]]} if c["name"] in COLOURS else {})} for c in FIXED + rec]
    out = {"recording": "e097_g2 intact window (raw-run time)", "ref_k": k, "min_segment_breaths": MIN_SEG,
           "smooth_breaths": SMOOTH, "conditions": conditions, "change_points": result}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for r in rec:
        print(f"{r['name']:<18} {r['start_raw_s']:8.1f}-{r['end_raw_s']:8.1f} s  {r['n_breaths']:4d} breaths  "
              f"{r['median_rate_per_min']:.0f}/min")
    plot(b, idx, mix, fits, k, result, rec, Path(a.out).with_suffix(".png"))


def sighs(b):
    """cibrrig's sigh rule (physiology.compute_sighs): breath AUC above the rolling 20 s median by > 7 rolling MADs."""
    t = pd.to_datetime(b["on_sec"].to_numpy(), unit="s")
    x = pd.Series(b["auc"].to_numpy(), index=t)
    mad = x.rolling("20s", center=True).apply(lambda v: np.nanmedian(np.abs(v - np.nanmedian(v))))
    return ((x - x.rolling("20s").median()) > 7 * mad).to_numpy()


def plot(b, idx, mix, fits, k, result, rec, path, offset_s=OFFSET_S, trial="anoxia #1"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    raw = b["on_sec"].to_numpy()[idx] + offset_s
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
           "#9a9893", "#52514e", "#c3c2b7"]       # HMM state colours (population_dynamics.PALETTE)
    cols = fits[f"k{k}_colours"]
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1.6]})
    axes[0].plot(raw, 60 / b["ibi_s"].to_numpy()[idx], ".", ms=2.5, color="#52514e")
    axes[0].set_ylabel("breaths/min")
    axes[1].plot(raw, 1e3 * b["duration_sec"].to_numpy()[idx], ".", ms=2.5, color="#52514e")
    axes[1].set_ylabel("Ti (ms)")
    amp = b["amp_z"].to_numpy()[idx]
    axes[2].plot(raw, amp, ".", ms=2.5, color="#52514e")
    big = sighs(b)[idx]
    axes[2].plot(raw[big], amp[big], "v", ms=6, color="#0b0b0b", label="sigh (cibrrig: AUC > 7x rolling 20 s MAD)")
    axes[2].set_ylabel("\u222bdia amplitude (z)")
    axes[2].legend(frameon=False, fontsize=8)
    order = np.argsort(cols)
    axes[3].stackplot(raw, mix[:, order].T, colors=[pal[cols[s] % len(pal)] for s in order], lw=0)
    axes[3].set_ylim(0, 1)
    axes[3].set_ylabel(f"fraction of cycle\nin each K = {k} state")
    axes[3].set_xlabel("raw-run time (s)")
    styles = {"breathing": (":", "breathing only"), "population": ("--", "population only"), "both": ("-", "both (used)")}
    for key, (ls, lab) in styles.items():
        for j, t in enumerate(result[key]["start_raw_s"][1:]):
            for ax in axes:
                ax.axvline(t, color="#0b0b0b", ls=ls, lw=1.2 if key == "both" else 0.9,
                           label=lab if (ax is axes[0] and j == 0) else None)
    for j, r in enumerate(rec):
        axes[0].text((r["start_raw_s"] + r["end_raw_s"]) / 2, 1.02 + 0.13 * (j % 2), r["name"],
                     transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Recovery after {trial}: change points in breathing and population state (5-breath smoothing)",
                 x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
