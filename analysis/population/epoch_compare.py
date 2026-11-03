"""Is the network in one epoch different from another? Same units, same recording (one long sort).

Reads a population_dynamics.py run (cache/ctx.pkl) and compares pairs of its conditions, e.g. the pre-trial baseline
against the post-trial period, and, as the yardstick for change that needs no trial, an earlier trial-free control
against the baseline (a similar time apart). Units: those firing >= --min-rate Hz in either condition of a pair.

Per pair (epoch_compare.tsv, one row per pair):
- breathing: rate, Ti and amplitude per condition;
- single units: log2 rate ratio per unit with a 95 % block-bootstrap interval over breath cycles (blocks of
  --block breaths); units whose interval excludes 0 are "changed" (unit_changes.tsv);
- phase tuning: correlation of each phase-locked unit's rate-by-phase curve between the two conditions (median);
- decoding: logistic regression (L2) on per-breath population vectors (sqrt spikes per unit in 4 phase quarters),
  cross-validated over --folds contiguous blocks of each condition, so neighbouring breaths never sit in both train
  and test. Chance comes from shuffling the condition labels of whole blocks (--n-perm);
- distance: Euclidean distance between the two conditions' mean breath vectors, divided by the mean distance between
  the halves of each condition (1 = no more different than the condition is from itself);
- correlation structure and subspace similarity (as recovery_compare.py), with each condition's split-half value.
These measure difference, not cause: a difference between baseline and post-trial that is no larger than control vs
baseline cannot be attributed to the trials.

    python epoch_compare.py --run <out>/grp/e038/population_dynamics_long --pair "control|baseline" ^
        --pair "baseline|post-trial" --out <out>/grp/e038/epoch_compare
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler as _Std  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recovery_compare import corr_upper, load_context, nan_r, subspace_cos, top_pcs  # noqa: E402

N_QUARTERS = 4


def cycle_vectors(c, cond_name, ids):
    """Per breath cycle of a condition: (onset, next onset), spikes per unit per phase quarter."""
    d = c["dia_df"]
    ons = d["on_sec"].to_numpy()
    idx = np.flatnonzero(d["condition"].to_numpy()[:-1] == cond_name)
    phi_t, phi = c["phi_t"], c["phi"]
    edges = np.linspace(-np.pi, np.pi, N_QUARTERS + 1)
    V = np.zeros((idx.size, len(ids), N_QUARTERS))
    for j, u in enumerate(ids):
        s = c["spikes"][u]
        cyc = np.searchsorted(ons, s, side="right") - 1
        p = phi[np.clip(np.searchsorted(phi_t, s), 0, phi.size - 1)]
        pos = np.searchsorted(idx, cyc)
        ok = (pos < idx.size) & (idx[np.clip(pos, 0, idx.size - 1)] == cyc) & np.isfinite(p)
        q = np.clip(np.digitize(p[ok], edges) - 1, 0, N_QUARTERS - 1)
        np.add.at(V, (pos[ok], j, q), 1)
    dur = ons[idx + 1] - ons[idx]
    return idx, dur, V


def block_ids(n, block):
    return np.arange(n) // block


def boot_log2_ratio(ca, da, cb, db, block, rng, n_boot=2000):
    """95 % interval of log2(rate B / rate A) for one unit, resampling blocks of breaths in each condition."""
    ba, bb = block_ids(ca.size, block), block_ids(cb.size, block)
    ua, ub = np.unique(ba), np.unique(bb)
    sa = np.array([[ca[ba == k].sum(), da[ba == k].sum()] for k in ua])
    sb = np.array([[cb[bb == k].sum(), db[bb == k].sum()] for k in ub])
    ia = rng.integers(0, ua.size, (n_boot, ua.size))
    ib = rng.integers(0, ub.size, (n_boot, ub.size))
    ra = (sa[ia, 0].sum(1) + 0.5) / sa[ia, 1].sum(1)
    rb = (sb[ib, 0].sum(1) + 0.5) / sb[ib, 1].sum(1)
    return np.percentile(np.log2(rb / ra), [2.5, 97.5])


def decode(Xa, Xb, folds, n_perm, block, rng):
    X = np.sqrt(np.vstack([Xa, Xb]))
    y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    fa = (np.arange(len(Xa)) * folds) // len(Xa)
    fb = (np.arange(len(Xb)) * folds) // len(Xb)
    fold = np.r_[fa, fb]

    def cv_acc(yy):
        acc = []
        for f in range(folds):
            tr, te = fold != f, fold == f
            if len(np.unique(yy[tr])) < 2:
                continue
            sc = _Std().fit(X[tr])
            m = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X[tr]), yy[tr])
            pr = m.predict(sc.transform(X[te]))
            acc.append(0.5 * ((pr[yy[te] == 0] == 0).mean() if (yy[te] == 0).any() else np.nan)
                       + 0.5 * ((pr[yy[te] == 1] == 1).mean() if (yy[te] == 1).any() else np.nan))
        return float(np.nanmean(acc))

    obs = cv_acc(y)
    # null: shuffle labels between blocks of breaths (keeps the autocorrelation inside blocks)
    blocks = np.r_[block_ids(len(Xa), block), block_ids(len(Xb), block) + 10 ** 6]
    ub, inv = np.unique(blocks, return_inverse=True)
    blab = np.array([y[blocks == b][0] for b in ub])
    null = np.array([cv_acc(rng.permutation(blab)[inv]) for _ in range(n_perm)])
    return obs, float(np.mean(null)), float((np.sum(null >= obs) + 1) / (n_perm + 1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--pair", action="append", required=True, help="conditionA|conditionB")
    p.add_argument("--min-rate", type=float, default=0.5)
    p.add_argument("--block", type=int, default=10, help="breaths per block (bootstrap and permutation)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    c = load_context(Path(a.run))
    d = c["dia_df"]
    rng = np.random.default_rng(0)
    rows, unit_rows = [], []
    for pair in a.pair:
        A, B = pair.split("|")
        ids_all = c["ids"]
        ia, da, Va = cycle_vectors(c, A, ids_all)
        ib, db, Vb = cycle_vectors(c, B, ids_all)
        ra, rb = Va.sum(2).sum(0) / da.sum(), Vb.sum(2).sum(0) / db.sum()
        keep = (ra >= a.min_rate) | (rb >= a.min_rate)
        ids = [u for u, k in zip(ids_all, keep) if k]
        Va, Vb, ra, rb = Va[:, keep], Vb[:, keep], ra[keep], rb[keep]
        changed_up = changed_down = 0
        for j, u in enumerate(ids):
            lo, hi = boot_log2_ratio(Va[:, j].sum(1), da, Vb[:, j].sum(1), db, a.block, rng)
            ch = "up" if lo > 0 else ("down" if hi < 0 else "")
            changed_up += ch == "up"
            changed_down += ch == "down"
            unit_rows.append({"pair": pair, "unit_id": u, f"rate_A_hz": ra[j], f"rate_B_hz": rb[j],
                              "log2_ratio": np.log2((rb[j] + 0.01) / (ra[j] + 0.01)), "ci_lo": lo, "ci_hi": hi,
                              "changed": ch})
        # phase curves per unit (rate per quarter)
        qa, qb = Va.sum(0) / da.sum(), Vb.sum(0) / db.sum()
        pr = [np.corrcoef(qa[j], qb[j])[0, 1] for j in range(len(ids)) if qa[j].std() > 0 and qb[j].std() > 0]
        acc, chance, pval = decode(Va.reshape(len(Va), -1), Vb.reshape(len(Vb), -1), a.folds, a.n_perm, a.block, rng)
        ma, mb = np.sqrt(Va / da[:, None, None]).reshape(len(Va), -1), np.sqrt(Vb / db[:, None, None]).reshape(len(Vb), -1)
        between = np.linalg.norm(ma.mean(0) - mb.mean(0))
        within = np.mean([np.linalg.norm(m[: len(m) // 2].mean(0) - m[len(m) // 2:].mean(0)) for m in (ma, mb)])
        spans = {}
        for nm, idx in ((A, ia), (B, ib)):
            ons = d["on_sec"].to_numpy()
            spans[nm] = (ons[idx[0]], ons[idx[-1] + 1])
        (a0, a1), (b0, b1) = spans[A], spans[B]
        ah, bh = (a0 + a1) / 2, (b0 + b1) / 2
        corr_ab = nan_r(corr_upper(c["spikes"], ids, a0, a1), corr_upper(c["spikes"], ids, b0, b1))
        corr_aa = nan_r(corr_upper(c["spikes"], ids, a0, ah), corr_upper(c["spikes"], ids, ah, a1))
        corr_bb = nan_r(corr_upper(c["spikes"], ids, b0, bh), corr_upper(c["spikes"], ids, bh, b1))
        sub_ab = subspace_cos(top_pcs(c["spikes"], ids, a0, a1), top_pcs(c["spikes"], ids, b0, b1))
        sub_aa = subspace_cos(top_pcs(c["spikes"], ids, a0, ah), top_pcs(c["spikes"], ids, ah, a1))
        sub_bb = subspace_cos(top_pcs(c["spikes"], ids, b0, bh), top_pcs(c["spikes"], ids, bh, b1))
        br = {}
        for nm in (A, B):
            m = d["condition"] == nm
            br[nm] = (60 / d.loc[m, "ibi_s"].median(), d.loc[m, "duration_sec"].median() * 1e3, d.loc[m, "amp_z"].median())
        rows.append({"pair": pair, "A": A, "B": B, "A_raw_s": f"{a0 + c['time_offset_s']:.0f}-{a1 + c['time_offset_s']:.0f}",
                     "B_raw_s": f"{b0 + c['time_offset_s']:.0f}-{b1 + c['time_offset_s']:.0f}",
                     "gap_s": b0 - a1, "n_breaths_A": len(ia), "n_breaths_B": len(ib), "n_units": len(ids),
                     "breaths_per_min_A": br[A][0], "breaths_per_min_B": br[B][0], "ti_ms_A": br[A][1],
                     "ti_ms_B": br[B][1], "amp_z_A": br[A][2], "amp_z_B": br[B][2],
                     "summed_rate_ratio_B_over_A": rb.sum() / ra.sum(),
                     "units_up": changed_up, "units_down": changed_down,
                     "median_abs_log2_ratio": float(np.median(np.abs(np.log2((rb + 0.01) / (ra + 0.01))))),
                     "phase_curve_r_median": float(np.median(pr)) if pr else np.nan,
                     "decode_balanced_acc": acc, "decode_chance": chance, "decode_p": pval,
                     "distance_between_over_within": between / within,
                     "corr_structure_r_AB": corr_ab, "corr_structure_r_A_halves": corr_aa,
                     "corr_structure_r_B_halves": corr_bb, "subspace_cos_AB": sub_ab, "subspace_cos_A_halves": sub_aa,
                     "subspace_cos_B_halves": sub_bb})
        print(json.dumps(rows[-1], indent=1, default=float))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "epoch_compare.tsv", sep="\t", index=False, float_format="%.4f")
    units = pd.DataFrame(unit_rows)
    units.to_csv(out / "unit_changes.tsv", sep="\t", index=False, float_format="%.4f")

    # figure: per-unit rate changes for each pair, side by side
    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 4.2), squeeze=False)
    for ax, r in zip(axes[0], rows):
        u = units[units["pair"] == r["pair"]]
        colr = np.where(u["changed"] == "up", "#d93636", np.where(u["changed"] == "down", "#2a78d6", "0.6"))
        ax.scatter(np.log10(u["rate_A_hz"] + 0.05), np.log10(u["rate_B_hz"] + 0.05), c=colr, s=12)
        lim = [np.log10(0.05), np.log10(max(u["rate_A_hz"].max(), u["rate_B_hz"].max()) + 1)]
        ax.plot(lim, lim, color="0.4", lw=0.6)
        ax.set_xlabel(f"log10 rate, {r['A']} (Hz)")
        ax.set_ylabel(f"log10 rate, {r['B']} (Hz)")
        ax.set_title(f"{r['A']} -> {r['B']}\n{r['units_up']} up, {r['units_down']} down of {r['n_units']}; "
                     f"decoding {r['decode_balanced_acc']:.2f} (chance {r['decode_chance']:.2f})\n"
                     f"distance {r['distance_between_over_within']:.1f}x within; corr r {r['corr_structure_r_AB']:.2f}",
                     fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "epoch_compare.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
