"""Soft-normalised PCA next to the Bush & Ramirez PCA, for both e097 anoxia trials.

The Bush & Ramirez PCA (population_dynamics.py: 5 ms counts, 10 ms Gaussian, square root, no per-unit scaling, fit
on the baseline bins) gives PC2 almost wholly to one tonic unit, #89 (loading 0.997), and PC2's steps are single spikes
of that unit (analysis record F18). This script fits a second PCA with every unit soft-normalised as in Churchland et
al. 2012 (Nature 487:51): firing rate in Hz (the same 5 ms bins and 10 ms Gaussian) divided by the unit's range + 5
sp/s. Soft normalisation is defined on trial-averaged responses, so the range and the axes come from anoxia #1's
baseline breath-averaged cycle (rates averaged in 50 phase bins), and every 5 ms bin of both trials is projected on
those axes. Applied to single 5 ms bins instead, it hands PC1, PC2 and PC3 each to one fast, weakly modulated unit
(record D13). The current PCA, the HMM and the runs' caches are untouched.

Writes into each run's figures/:
  hmm_states_pca_softnorm_K6_by_epoch.png (PC1-PC3), ..._PC1-PC2.png, ..._PC2-PC3.png
into <anoxia #1 run>/figures/:
  pca_pc2_single_unit_steps.png   the current PCA's PC2 in every 5 ms bin of each trial, split by how many spikes its
                                  largest unit has within the 10 ms Gaussian's reach, with the PC2 values one spike
                                  at each offset predicts (F18)
and <anoxia #1 run>/cache/pca_softnorm.npz: unit ids, weights, axes, both trials' projections and I-off points.

  cibrrig python pca_softnorm.py [--runs <anoxia #1 run> <anoxia #2 run>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import population_dynamics as pdm  # noqa: E402
import figures_population as fp  # noqa: E402
import transition_windows as tw  # noqa: E402
from figures_population import INK2, GRID, save_fig  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

RUNS = [pdm.OUT / "population_dynamics_v2", pdm.OUT / "population_dynamics_anoxia2"]
SPIKE_CLASSES = [("no spike within the kernel's reach", "#2a78d6"), ("exactly one spike", "#eb6834"),
                 ("two or more spikes", "#1baf7a")]
SOFT_HZ = 5.0               # Churchland et al. 2012
CYCLE_BINS = 50             # phase bins of the baseline mean cycle the axes are fit on
N_COMP = 10
PLANES = ((0, 2), (0, 1), (1, 2))
FIT_LABEL = "anoxia #1's baseline mean cycle, units soft-normalised"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", nargs="+", default=[str(r) for r in RUNS],
                   help="population_dynamics.py output folders; the first is the reference the axes are fit on")
    return p.parse_args()


def rates_hz(c, n_bins):
    counts = pdm.bin_counts(c["spikes"], c["ids"], pdm.PCA_BIN_S, n_bins)
    return gaussian_filter1d(counts, pdm.PCA_SIGMA_S / pdm.PCA_BIN_S, axis=0) / pdm.PCA_BIN_S


def baseline_bins(c):
    """population_dynamics.py's PCA fit bins: the baseline epoch of the reference trial."""
    e = c["epochs"]["baseline"]
    return (c["centres"] >= e[0]) & (c["centres"] < e[1])


def fit(c):
    """Weights 1 / (range + SOFT_HZ) and PCA axes, both from the reference trial's baseline breath-averaged cycle."""
    _, cyc = fp.phase_binned(rates_hz(c, len(c["P"])), c["ph"], baseline_bins(c), nbins=CYCLE_BINS)
    assert not np.isnan(cyc).any(), "a phase bin of the baseline cycle is empty"
    w = 1 / (cyc.max(axis=0) - cyc.min(axis=0) + SOFT_HZ)
    return w, PCA(N_COMP).fit(cyc * w)


def project(c, ids_ref, w, pca):
    """Every 5 ms bin on the soft-normalised axes (a unit missing from this trial sits at the axes' mean and adds 0),
    and the inspiration-off point as population_dynamics.py makes it: mean position in the bin after each baseline
    inspiration offset."""
    n = len(c["P"])
    X = np.tile(pca.mean_, (n, 1))
    cols = [ids_ref.index(u) for u in c["ids"]]
    X[:, cols] = rates_hz(c, n) * w[cols]
    P = pca.transform(X)
    off = np.floor(c["dia_df"]["off_sec"].to_numpy() / pdm.PCA_BIN_S).astype(int) + 1
    return P, P[off[(np.asarray(c["cond"]) == "baseline") & (off < n)], :3].mean(axis=0)


def limits(Ps):
    """Per PC, the 0.2-99.8 percentile span over every trial, padded by 10% and rounded out to 0.1."""
    allp = np.vstack([P[:, :3] for P in Ps])
    lo, hi = np.percentile(allp, 0.2, axis=0), np.percentile(allp, 99.8, axis=0)
    pad = 0.1 * (hi - lo)
    return [(np.floor((l - p) * 10) / 10, np.ceil((h + p) * 10) / 10) for l, h, p in zip(lo, hi, pad)]


def top_unit_shares(pca, ids, label):
    for d in range(3):
        L = pca.components_[d]
        o = np.argsort(-np.abs(L))[:4]
        print(f"{label} PC{d + 1}: {pca.explained_variance_ratio_[d] * 100:.1f}% of its fit data; largest unit "
              f"{ids[o[0]]} carries {L[o[0]] ** 2 * 100:.0f}% (" + ", ".join(f"{ids[j]} {L[j]:+.2f}" for j in o) + ")")


def fig_pc2_steps(cs, ids_ref, path):
    """PC2 of the current PCA (in the runs' P) in every 5 ms bin, split by how many spikes PC2's largest unit fires
    within the Gaussian's reach, over the PC2 that one spike at each offset from the bin gives (sqrt of the kernel
    weight times the loading, the other units at their means)."""
    pca = cs[0]["pca"]
    L = pca.components_[1]
    j = int(np.argmax(np.abs(L)))
    u = ids_ref[j]
    imp = np.zeros(81)
    imp[40] = 1
    kern = gaussian_filter1d(imp, pdm.PCA_SIGMA_S / pdm.PCA_BIN_S)
    half = int(np.flatnonzero(kern > 0).max()) - 40
    levels = L[j] * (np.sqrt(np.r_[0, kern[40:41 + half]]) - pca.mean_[j])
    lo = levels.min() - 0.06
    edges = np.arange(lo, max(np.percentile(c["P"][:, 1], 99.9) for c in cs), 0.004)
    fig, axes = plt.subplots(len(cs), 1, figsize=(12, 3.3 * len(cs) + 1.0), sharex=True, squeeze=False)
    for ax, c in zip(axes[:, 0], cs):
        n = len(c["P"])
        cnt = pdm.bin_counts(c["spikes"], [u], pdm.PCA_BIN_S, n)[:, 0] if u in c["ids"] else np.zeros(n)
        near = np.convolve(cnt, np.ones(2 * half + 1), mode="same")
        pc2 = c["P"][:, 1]
        groups = [pc2[near == 0], pc2[near == 1], pc2[near >= 2]]
        ax.hist(groups, bins=edges, stacked=True, histtype="stepfilled", color=[col for _, col in SPIKE_CLASSES],
                lw=0, label=[f"{name} ({g.size / n * 100:.0f}% of bins)" for (name, _), g in zip(SPIKE_CLASSES, groups)])
        ax.set_yscale("log")
        for x in levels:
            ax.axvline(x, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=0)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_title(c.get("trial", "anoxia #1"), loc="left", fontsize=10.5)
        ax.set_ylabel("5 ms bins (log)")
        ax.legend(loc="upper right", frameon=False, fontsize=8.5, title=f"unit {u} spikes within ±{half * 5} ms",
                  title_fontsize=8.5)
    axes[-1, 0].set_xlabel("PC2 of the Bush & Ramirez PCA")
    fig.suptitle(f"PC2 is unit {u} (loading {L[j]:+.3f}): its steps are single spikes of that unit.\nDashed, left to "
                 f"right: PC2 with unit {u} silent, then one spike {', '.join(f'{k * 5}' for k in range(half, -1, -1))} "
                 "ms from the bin (other units at their mean)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, path, dpi=120)
    plt.close(fig)
    return path


def main():
    a = parse_args()
    runs = [Path(r) for r in a.runs]
    cs = [tw.load_context(r) for r in runs]
    ids_ref = list(cs[0]["ids"])
    top_unit_shares(cs[0]["pca"], ids_ref, "Bush & Ramirez")
    print(fig_pc2_steps(cs, ids_ref, runs[0] / "figures" / "pca_pc2_single_unit_steps.png"))
    w, pca = fit(cs[0])
    top_unit_shares(pca, ids_ref, "soft-normalised")
    proj = [project(c, ids_ref, w, pca) for c in cs]
    base = baseline_bins(cs[0])
    X0 = rates_hz(cs[0], len(proj[0][0]))[base] * w
    share = proj[0][0][base, :3].var(axis=0) / X0.var(axis=0).sum() * 100
    print("soft-normalised PC1-PC3 carry " + ", ".join(f"{s:.1f}" for s in share) +
          "% of the variance of anoxia #1's baseline 5 ms bins (the rest is mostly spike noise)")
    lims = limits([P for P, _ in proj])
    for run, c, (P, ioff) in zip(runs, cs, proj):
        cond = tw.label_transitions(c["cond"])[0]
        cz = {**c, "P": P, "ioff_point": ioff, "pca_fit": FIT_LABEL}
        for d in PLANES:
            print(tw.fig_states_pca_epochs(cz, cond, run / "figures", dims=d, xlim=lims[d[0]], ylim=lims[d[1]],
                                           tag="softnorm"))
    np.savez(runs[0] / "cache" / "pca_softnorm.npz", unit_ids=np.array(ids_ref), weights=w,
             components=pca.components_, mean=pca.mean_, cycle_variance_ratio=pca.explained_variance_ratio_,
             trials=np.array([c.get("trial", "anoxia #1") for c in cs]), limits=np.array(lims),
             **{f"P_{i}": P.astype(np.float32) for i, (P, _) in enumerate(proj)},
             **{f"ioff_{i}": ioff for i, (_, ioff) in enumerate(proj)})


if __name__ == "__main__":
    main()
