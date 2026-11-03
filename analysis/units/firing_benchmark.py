"""How much does a known neuron's firing drift, and can firing alone tell it from its neighbours? (e097 benchmark)

The e097 anoxia #1 and #2 windows (sorted separately, no tissue movement between them) give neurons whose identity
is settled by their templates (match_windows.py pairs, verdict "match", cosine >= --min-cos; the 27 reviewed singles
are among them). For each comparison below, every same-neuron pair's firing similarity (firing_features.py) is set
against the same measure for the units within --radius-um of it that are other neurons (the realistic wrong
partners when firing is used to propose matches):

  across windows (identity from the templates)
    later recovery #1 -> baseline #2     ~150 s apart, similar breathing
    baseline #1 -> baseline #2           ~510 s, across anoxia #1 and its recovery
    baseline #1 -> recovery 5 of #2      ~800 s, across both anoxias
  within one window (same unit on both sides)
    baseline 1st half -> 2nd half        30 s, the noise floor
    baseline -> early recovery           across gasping, ~100 s
    baseline -> later recovery           across anoxia #1, ~330 s

Writes benchmark_pairs.tsv (every pair and measure), benchmark_summary.tsv (per comparison and measure: same-neuron
10th / 50th / 90th percentiles, other-neuron median, AUC of same vs other) and benchmark.png. Units: not noise, with
>= --min-spikes spikes in both conditions.

    python firing_benchmark.py --pairs <match_windows out>/matched_pairs.tsv --out <outputs>/e097/firing_benchmark
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "population"))
from firing_features import cycle_spans, features, similarity  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "outputs" / "e097"
SORT_A, SORT_B = OUT / "intact_window_v2", OUT / "anoxia2_window_v2"
RUN_A, RUN_B = OUT / "population_dynamics_v2", OUT / "population_dynamics_anoxia2"
MEASURES = [("curve_r", "phase-curve r", +1), ("peth_r", "onset PETH r", +1), ("acg_r", "ACG r", +1),
            ("log2_rate_ratio", "|log2 rate ratio|", -1), ("pref_diff_rad", "preferred-phase difference (rad)", -1),
            ("L_diff", "|L difference|", -1), ("cv2_diff", "|CV2 difference|", -1)]


def load_ctx(run):
    import population_dynamics as pdm  # noqa: F401
    import __main__
    __main__.RatePoissonHMM = pdm.RatePoissonHMM
    with open(run / "cache" / "ctx.pkl", "rb") as f:
        return pickle.load(f)


class Side:
    def __init__(self, sort, run):
        alf = sort / "kilosort4" / "alf"
        st, sc = np.load(alf / "spike_times.npy"), np.load(alf / "spike_clusters.npy")
        self.units = pd.read_csv(sort / "kilosort4" / "units.tsv", sep="\t").set_index("unit_id")
        self.units = self.units[self.units["label"] != "noise"]
        order = np.argsort(sc, kind="stable")
        b = np.searchsorted(sc[order], self.units.index.to_numpy())
        e = np.searchsorted(sc[order], self.units.index.to_numpy(), side="right")
        self.spk = {u: np.sort(st[order[i:j]]) for u, i, j in zip(self.units.index, b, e)}
        c = load_ctx(run)
        self.d, self.phi_t, self.phi = c["dia_df"], c["phi_t"], c["phi"]
        self.ons = self.d["on_sec"].to_numpy()

    def spans(self, cond, half=None):
        sp = cycle_spans(self.ons, self.d["condition"].to_numpy() == cond)
        if half is not None:
            n = len(sp) // 2
            sp = sp[:n] if half == 0 else sp[n:]
        return sp

    def feats(self, spans):
        return {u: features(self.spk[u], spans, self.ons, self.phi_t, self.phi) for u in self.units.index}


def auc(same, other, sign):
    s, o = np.asarray(same, float), np.asarray(other, float)
    s, o = s[np.isfinite(s)], o[np.isfinite(o)]
    if not s.size or not o.size:
        return np.nan
    if sign < 0:
        s, o = -np.abs(s), -np.abs(o)
    return float(np.mean(s[:, None] > o[None, :]) + 0.5 * np.mean(s[:, None] == o[None, :]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True, help="match_windows.py matched_pairs.tsv for the two e097 windows")
    p.add_argument("--min-cos", type=float, default=0.98)
    p.add_argument("--radius-um", type=float, default=150.0)
    p.add_argument("--min-spikes", type=int, default=50)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A, B = Side(SORT_A, RUN_A), Side(SORT_B, RUN_B)
    pairs = pd.read_csv(a.pairs, sep="\t")
    pairs = pairs[(pairs["verdict"] == "match") & (pairs["cosine"] >= a.min_cos)]
    pa = dict(zip(pairs["unit_a"], pairs["unit_b"]))
    comps = [("later rec #1 -> baseline #2 (~150 s)", A, "later recovery", None, B, "baseline", None, True),
             ("baseline #1 -> baseline #2 (~510 s)", A, "baseline", None, B, "baseline", None, True),
             ("baseline #1 -> recovery 5 #2 (~800 s)", A, "baseline", None, B, "recovery 5", None, True),
             ("baseline halves (30 s)", A, "baseline", 0, A, "baseline", 1, False),
             ("baseline -> early recovery #1 (~100 s)", A, "baseline", None, A, "early recovery", None, False),
             ("baseline -> later recovery #1 (~330 s)", A, "baseline", None, A, "later recovery", None, False)]
    rows = []
    for name, S1, c1, h1, S2, c2, h2, cross in comps:
        F1, F2 = S1.feats(S1.spans(c1, h1)), S2.feats(S2.spans(c2, h2))
        for u in S1.units.index:
            v_true = pa.get(u) if cross else u
            if v_true is None or v_true not in F2 or F1[u]["n_spikes"] < a.min_spikes:
                continue
            if F2[v_true]["n_spikes"] < a.min_spikes:
                continue
            y = S1.units.loc[u, "depth_um"]
            near = S2.units.index[(np.abs(S2.units["depth_um"] - y) <= a.radius_um)]
            for v in near:
                if F2[v]["n_spikes"] < a.min_spikes:
                    continue
                same = v == v_true
                r = similarity(F1[u], F2[v])
                rows.append({"comparison": name, "unit_1": u, "unit_2": v, "same_neuron": same,
                             "dist_um": abs(S2.units.loc[v, "depth_um"] - y), "rate_1": F1[u]["rate_hz"],
                             "rate_2": F2[v]["rate_hz"], "L_1": F1[u]["L"], "L_2": F2[v]["L"], **r})
    T = pd.DataFrame(rows)
    T.to_csv(out / "benchmark_pairs.tsv", sep="\t", index=False, float_format="%.4f")
    summ = []
    for name, g in T.groupby("comparison", sort=False):
        s, o = g[g["same_neuron"]], g[~g["same_neuron"]]
        for m, lab, sign in MEASURES:
            sv = s[m] if sign > 0 else s[m].abs()
            ov = o[m] if sign > 0 else o[m].abs()
            summ.append({"comparison": name, "measure": m, "n_same": int(sv.notna().sum()), "n_other": int(ov.notna().sum()),
                         "same_p10": sv.quantile(0.1), "same_median": sv.median(), "same_p90": sv.quantile(0.9),
                         "other_median": ov.median(), "auc_same_vs_other": auc(s[m], o[m], sign)})
    S = pd.DataFrame(summ)
    S.to_csv(out / "benchmark_summary.tsv", sep="\t", index=False, float_format="%.3f")
    names = list(dict.fromkeys(T["comparison"]))
    fig, axes = plt.subplots(len(MEASURES), 1, figsize=(12, 2.3 * len(MEASURES)), sharex=True)
    for ax, (m, lab, sign) in zip(axes, MEASURES):
        for i, nm in enumerate(names):
            g = T[T["comparison"] == nm]
            for same, col, off in ((True, "#2a78d6", -0.17), (False, "0.6", 0.17)):
                v = g.loc[g["same_neuron"] == same, m]
                v = (v if sign > 0 else v.abs()).dropna()
                if len(v):
                    ax.boxplot(v, positions=[i + off], widths=0.28, showfliers=False, patch_artist=True,
                               boxprops=dict(facecolor=col, alpha=0.6), medianprops=dict(color="k"))
            au = S[(S["comparison"] == nm) & (S["measure"] == m)]["auc_same_vs_other"].iloc[0]
            ax.text(i, 1.02, f"AUC {au:.2f}", transform=ax.get_xaxis_transform(), ha="center", fontsize=7)
        ax.set_ylabel(lab, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(range(len(names)))
    axes[-1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    fig.suptitle("e097: firing similarity of the same neuron (blue) vs other units within "
                 f"{a.radius_um:.0f} um (grey)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "benchmark.png", dpi=130)
    with pd.option_context("display.width", 200):
        print(S.pivot(index="measure", columns="comparison", values="auc_same_vs_other").round(2).to_string())
        print(S[S["measure"].isin(["curve_r", "acg_r", "log2_rate_ratio"])][
            ["comparison", "measure", "n_same", "same_p10", "same_median", "same_p90", "other_median"]].round(2).to_string())


if __name__ == "__main__":
    main()
