"""Recovery after gasping, compared across animals and trials on measures that don't need the same neurons.

Each trial is a population_dynamics.py run (its cache/ctx.pkl) plus the gasping event to start from. Time zero is the
end of the event's last Bush-rule gasp cycle; the recovery is cut into bins of time since then and every measure is
relative to the same run's baseline condition, so animals with different units and breathing rates can be laid side
by side (e097 and the Bush & Ramirez recordings vs the Grp KO mice).

Per time bin (non-gasp breaths and spikes inside it):
- breathing: rate and inspiratory duration (Ti) relative to baseline (median IBI / Ti);
- population rate: summed rate of the units active at baseline (>= --min-rate Hz) relative to baseline, and the
  fraction of those units firing at >= 20 % of their baseline rate;
- phase tuning: for units phase-locked at baseline (vector-mean L >= 0.2 in the run's tuning table), the correlation
  of their spike-rate-by-phase curve (18 phase bins) with the baseline curve; median over units;
- correlation structure: pairwise correlations of 25 ms spike counts (sqrt), r of the upper triangle with baseline's;
- subspace: mean cosine of the 3 principal angles between the bin's and baseline's top-3 PCs (5 ms bins, Gaussian
  10 ms, sqrt; as population_dynamics.py).
The correlation measures are noisier in short bins; the baseline's own first-half vs second-half value (same bin
length) is written as the ceiling for each run.

    python recovery_compare.py --trial "e097 anoxia #1|<outputs>/e097/population_dynamics_v2|1800|1900|2160" ^
        --trial "..." --out <folder>

--trial "label|run folder|gasp search start (run s)|gasp search end (run s)|recovery end (run s)[|baseline name]"
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import population_dynamics as pdm  # noqa: E402

EDGES = [0, 15, 30, 45, 60, 90, 120, 150, 180, 240, 300, 360, 480, 600, 900, 1200, 1500, 1800]
PHASE_BINS = 18
CORR_BIN_S = 0.025


def load_context(run: Path) -> dict:
    import __main__
    __main__.RatePoissonHMM = pdm.RatePoissonHMM          # the pickle was written by population_dynamics as __main__
    with open(run / "cache" / "ctx.pkl", "rb") as f:
        return pickle.load(f)


def counts(spikes, ids, t0, t1, dt):
    n = max(int((t1 - t0) / dt), 1)
    out = np.zeros((n, len(ids)))
    for j, u in enumerate(ids):
        k = np.floor((spikes[u][(spikes[u] >= t0) & (spikes[u] < t1)] - t0) / dt).astype(int)
        out[:, j] = np.bincount(k[k < n], minlength=n)
    return out


def corr_upper(spikes, ids, t0, t1):
    c = np.corrcoef(np.sqrt(counts(spikes, ids, t0, t1, CORR_BIN_S)).T)
    return c[np.triu_indices(len(ids), 1)]


def top_pcs(spikes, ids, t0, t1, k=3):
    z = np.sqrt(gaussian_filter1d(counts(spikes, ids, t0, t1, pdm.PCA_BIN_S), pdm.PCA_SIGMA_S / pdm.PCA_BIN_S, axis=0))
    z = z - z.mean(axis=0)
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    return vt[:k].T


def nan_r(x, y):
    """Pearson r over the pairs both have (pairs with a unit silent in the bin are NaN)."""
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 3 else np.nan


def subspace_cos(a, b):
    return float(np.mean(np.linalg.svd(a.T @ b, compute_uv=False)))


def phase_curves(spikes, ids, phi_t, phi, spans):
    """Rate by phase (spikes / s in each of PHASE_BINS bins) over a list of (t0, t1) spans."""
    edges = np.linspace(-np.pi, np.pi, PHASE_BINS + 1)
    occ = np.zeros(PHASE_BINS)
    cnt = np.zeros((PHASE_BINS, len(ids)))
    dt = float(np.median(np.diff(phi_t)))
    for t0, t1 in spans:
        m = (phi_t >= t0) & (phi_t < t1) & np.isfinite(phi)
        occ += np.histogram(phi[m], edges)[0] * dt
        for j, u in enumerate(ids):
            s = spikes[u][(spikes[u] >= t0) & (spikes[u] < t1)]
            p = phi[np.clip(np.searchsorted(phi_t, s), 0, phi.size - 1)]
            cnt[:, j] += np.histogram(p[np.isfinite(p)], edges)[0]
    return cnt / np.maximum(occ, 1e-9)[:, None]


def cycle_spans(d, mask):
    """(onset, next onset) of the breaths selected by mask."""
    ons = d["on_sec"].to_numpy()
    idx = np.flatnonzero(mask[:-1])
    return list(zip(ons[idx], ons[idx + 1]))


def trial_table(label, run, g0_raw, g1_raw, end_raw, base_name, min_rate):
    c = load_context(run)
    off, d, spikes = c["time_offset_s"], c["dia_df"].copy(), c["spikes"]
    ons, gasp = d["on_sec"].to_numpy(), d["gasp"].to_numpy().astype(bool)
    g = np.flatnonzero(gasp & (ons + off >= g0_raw) & (ons + off < g1_raw))
    if not g.size:
        raise ValueError(f"{label}: no gasps between {g0_raw} and {g1_raw} s")
    t_zero = ons[g[-1] + 1]                                          # end of the last gasp cycle (file time)
    first_gasp, n_gasps = ons[g[0]], g.size
    end = min(end_raw - off, ons[-1])
    base = d["condition"].to_numpy() == base_name
    b_spans = cycle_spans(d, base)
    b0, b1 = b_spans[0][0], b_spans[-1][1]
    b_dur = sum(t1 - t0 for t0, t1 in b_spans)
    ids_all = c["ids"]
    rate_b = np.array([sum(((spikes[u] >= t0) & (spikes[u] < t1)).sum() for t0, t1 in b_spans) / b_dur
                       for u in ids_all])
    act = rate_b >= min_rate
    ids = [u for u, a in zip(ids_all, act) if a]
    tun = c["tuning"]
    tb = tun[tun["condition"] == base_name].set_index("unit_id")
    locked = [u for u in ids if tb.loc[u, "modulation_L"] >= pdm.TONIC_MAX_L]
    ibi_b = np.median(d.loc[base, "ibi_s"])
    ti_b = np.median(d.loc[base, "duration_sec"])
    pc_b = phase_curves(spikes, locked, c["phi_t"], c["phi"], b_spans)
    corr_b = corr_upper(spikes, ids, b0, b1)
    sub_b = top_pcs(spikes, ids, b0, b1)
    rows = []
    for e0, e1 in zip(EDGES[:-1], EDGES[1:]):
        t0, t1 = t_zero + e0, t_zero + e1
        if t0 >= end:
            break
        t1 = min(t1, end)
        sel = (ons >= t0) & (ons < t1) & ~gasp
        sel[-1] = False
        spans = cycle_spans(d, sel)
        if len(spans) < 3:
            continue
        dur = sum(s1 - s0 for s0, s1 in spans)
        r = np.array([sum(((spikes[u] >= s0) & (spikes[u] < s1)).sum() for s0, s1 in spans) / dur for u in ids])
        rb = rate_b[act]
        pcr = phase_curves(spikes, locked, c["phi_t"], c["phi"], spans)
        pr = [np.corrcoef(pcr[:, j], pc_b[:, j])[0, 1] for j in range(len(locked)) if pcr[:, j].std() > 0]
        rows.append({"trial": label, "bin_start_s": e0, "bin_end_s": t1 - t_zero, "n_breaths": len(spans),
                     "breath_rate_ratio": ibi_b / np.median(d.loc[sel, "ibi_s"]),
                     "ti_ratio": np.median(d.loc[sel, "duration_sec"]) / ti_b,
                     "pop_rate_ratio": r.sum() / rb.sum(),
                     "median_unit_rate_ratio": float(np.median(r / rb)),
                     "frac_units_active": float(np.mean(r >= 0.2 * rb)),
                     "phase_curve_r": float(np.nanmedian(pr)) if pr else np.nan,
                     "corr_structure_r": nan_r(corr_upper(spikes, ids, t0, t1), corr_b),
                     "subspace_cos": subspace_cos(top_pcs(spikes, ids, t0, t1), sub_b)})
    # ceilings: the baseline's halves against each other, at the median recovery-bin length
    half = (b0 + b1) / 2
    ceil = {"trial": label, "corr_structure_r": nan_r(corr_upper(spikes, ids, b0, half), corr_upper(spikes, ids, half, b1)),
            "subspace_cos": subspace_cos(top_pcs(spikes, ids, b0, half), top_pcs(spikes, ids, half, b1)),
            "baseline_s": b1 - b0, "n_units": len(ids), "n_phase_locked": len(locked), "n_gasps": int(n_gasps),
            "gasping_s": float(t_zero - first_gasp), "first_gasp_raw_s": float(first_gasp + off),
            "recovery_start_raw_s": float(t_zero + off), "recovery_end_raw_s": float(end + off),
            "baseline_breaths_per_min": 60 / ibi_b, "baseline_ti_ms": ti_b * 1e3, "units": c.get("unit_desc", "")}
    return pd.DataFrame(rows), ceil


MEASURES = [("breath_rate_ratio", "breathing rate / baseline"), ("ti_ratio", "Ti / baseline"),
            ("pop_rate_ratio", "summed unit rate / baseline"), ("frac_units_active", "units at >= 20 % of baseline"),
            ("phase_curve_r", "phase curve r with baseline\n(median, phase-locked units)"),
            ("corr_structure_r", "pairwise-correlation r\nwith baseline"),
            ("subspace_cos", "subspace similarity\n(mean cos, top 3 PCs)")]


def figure(tab, ceil, out, styles):
    fig, axes = plt.subplots(len(MEASURES), 1, figsize=(9, 2.1 * len(MEASURES)), sharex=True)
    for ax, (m, lab) in zip(axes, MEASURES):
        for tr, g in tab.groupby("trial", sort=False):
            x = (g["bin_start_s"] + g["bin_end_s"]) / 2
            st = styles.get(tr, {})
            ax.plot(x, g[m], marker="o", ms=3, lw=1.4, label=tr, **st)
            if m in ("corr_structure_r", "subspace_cos"):
                ax.axhline(ceil.set_index("trial").loc[tr, m], lw=0.6, ls=":", color=st.get("color", "k"))
        if m.endswith("ratio") or m == "frac_units_active":
            ax.axhline(1, color="0.6", lw=0.6)
        ax.set_ylabel(lab, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    axes[-1].set_xscale("symlog", linthresh=60)
    axes[-1].set_xlabel("time since the end of gasping (s; linear to 60 s, log after)")
    fig.suptitle("Recovery after gasping, relative to each trial's own baseline\n"
                 "(dotted: the baseline's first vs second half, the ceiling for the correlation measures)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "recovery_compare.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trial", action="append", required=True)
    p.add_argument("--style", action="append", default=[], help="label|colour|linestyle")
    p.add_argument("--min-rate", type=float, default=0.5, help="units included: baseline rate >= this (Hz)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tabs, ceils = [], []
    for spec in a.trial:
        f = spec.split("|")
        t, cl = trial_table(f[0], Path(f[1]), float(f[2]), float(f[3]), float(f[4]),
                            f[5] if len(f) > 5 else "baseline", a.min_rate)
        tabs.append(t)
        ceils.append(cl)
        print(f"{f[0]}: {cl['n_units']} units ({cl['n_phase_locked']} phase-locked), {cl['n_gasps']} gasps, "
              f"recovery {cl['recovery_start_raw_s']:.1f}-{cl['recovery_end_raw_s']:.1f} s")
    tab, ceil = pd.concat(tabs, ignore_index=True), pd.DataFrame(ceils)
    tab.to_csv(out / "recovery_compare.tsv", sep="\t", index=False, float_format="%.4f")
    ceil.to_csv(out / "recovery_compare_trials.tsv", sep="\t", index=False, float_format="%.4f")
    styles = {}
    for s in a.style:
        lab, col, ls = s.split("|")
        styles[lab] = {"color": col, "ls": ls}
    figure(tab, ceil, out, styles)


if __name__ == "__main__":
    main()
