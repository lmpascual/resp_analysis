"""Generates publication-quality figure comparing Lupin vs Kilosort 4 6-state canonical respiratory HMMs.
Outputs:
  outputs/e097/intact_window/intact_window_cross_sorter_6state_concordance.png
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_DIR = REPO_ROOT / "outputs" / "e097" / "intact_window"
ARTIFACT_DIR = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\2f5e4e38-0e84-4946-871a-26957179bf52")

STATE_NAMES_SHORT = [
    "S1: Insp Burst",
    "S2: Post-I Decel",
    "S3: I_off Quench",
    "S4: Late Exp E2",
    "S5: Pre-I Priming",
    "S6: Pause"
]

STATE_COLORS = ["#10b981", "#06b6d4", "#f97316", "#38bdf8", "#a855f7", "#64748b"]


def main():
    with open(OUT_DIR / "cross_sorter_6state_agreement.json", "r") as f:
        agr = json.load(f)
    with open(OUT_DIR / "kilosort4_intact_6state_analysis_results.json", "r") as f:
        ks4 = json.load(f)
    with open(OUT_DIR / "lupin_intact_6state_analysis_results.json", "r") as f:
        lupin = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    plt.subplots_adjust(hspace=0.35, wspace=0.30)

    # ----------------------------------------------------
    # Panel A: Normalized Confusion Matrix
    # ----------------------------------------------------
    ax = axes[0, 0]
    conf_norm = np.array(agr["normalized_confusion_matrix"]) * 100.0  # as percentage

    sns.heatmap(
        conf_norm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar_kws={"label": "Transition Concordance (%)"},
        xticklabels=STATE_NAMES_SHORT,
        yticklabels=STATE_NAMES_SHORT,
        ax=ax,
        vmin=0,
        vmax=100
    )
    ax.set_title(f"A. Cross-Sorter State Concordance (KS4 vs Lupin)\nOverall Agreement = {agr['overall_concordance_pct']:.1f}% | ARI = {agr['ari']:.3f} | Macro F1 = {agr['macro_f1']:.3f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Lupin 6-State Assignment", fontsize=10, fontweight="bold")
    ax.set_ylabel("Kilosort 4 6-State Assignment", fontsize=10, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    # ----------------------------------------------------
    # Panel B: Per-State F1 Concordance Score
    # ----------------------------------------------------
    ax = axes[0, 1]
    f1_scores = agr["f1_per_state"]
    x = np.arange(len(STATE_NAMES_SHORT))

    bars = ax.bar(x, f1_scores, color=STATE_COLORS, edgecolor="black", linewidth=1.2, width=0.6, alpha=0.85)
    for bar, score in zip(bars, f1_scores):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{score:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(agr["macro_f1"], color="#ef4444", linestyle="--", linewidth=1.5, label=f"Macro F1 ({agr['macro_f1']:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_NAMES_SHORT, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 Alignment Score", fontsize=10, fontweight="bold")
    ax.set_title("B. Cross-Sorter Harmonization by State (F1 Score)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # ----------------------------------------------------
    # Panel C: Subsampling Dropout Robustness Comparison
    # ----------------------------------------------------
    ax = axes[1, 0]
    ks4_sizes = [r["N"] for r in ks4["subsampling"]["results_by_size"]]
    ks4_aris = [r["ari_mean"] for r in ks4["subsampling"]["results_by_size"]]
    ks4_sems = [r["ari_sem"] for r in ks4["subsampling"]["results_by_size"]]

    lup_sizes = [r["N"] for r in lupin["subsampling"]["results_by_size"]]
    lup_aris = [r["ari_mean"] for r in lupin["subsampling"]["results_by_size"]]
    lup_sems = [r["ari_sem"] for r in lupin["subsampling"]["results_by_size"]]

    ax.errorbar(ks4_sizes, ks4_aris, yerr=ks4_sems, fmt="o-", color="#2563eb", linewidth=2.0, capsize=4, label="Kilosort 4 (120 units)", alpha=0.9)
    ax.errorbar(lup_sizes, lup_aris, yerr=lup_sems, fmt="s-", color="#d97706", linewidth=2.0, capsize=4, label="Lupin (120 units)", alpha=0.9)

    ax.axhline(0.80, color="#10b981", linestyle=":", linewidth=1.2, label="High Fidelity Threshold (ARI=0.80)")
    ax.axhline(0.50, color="#64748b", linestyle=":", linewidth=1.2, label="Acceptable Threshold (ARI=0.50)")

    ax.set_xlabel("Sampled Population Size (N units)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Reconstruction Accuracy (ARI vs N=120)", fontsize=10, fontweight="bold")
    ax.set_title("C. Population Subsampling Robustness (KS4 vs Lupin)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 125)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)

    # ----------------------------------------------------
    # Panel D: Driver Attribution Lorenz Inequality Curves
    # ----------------------------------------------------
    ax = axes[1, 1]
    
    # Lorenz curve from Lupin
    mi_lup = np.sort([u["mi_bits"] for u in lupin["attribution"]["unit_stats"]])
    n_lup = len(mi_lup)
    cum_units_lup = np.linspace(0, 1, n_lup + 1)
    cum_info_lup = np.concatenate(([0.0], np.cumsum(mi_lup))) / (np.sum(mi_lup) + 1e-12)
    gini_lup = (np.sum((2 * np.arange(1, n_lup + 1) - n_lup - 1) * mi_lup)) / (n_lup * np.sum(mi_lup) + 1e-12)

    # Lorenz curve from KS4
    mi_ks4 = np.sort([u["mi_bits"] for u in ks4["attribution"]["unit_stats"]])
    n_ks4 = len(mi_ks4)
    cum_units_ks4 = np.linspace(0, 1, n_ks4 + 1)
    cum_info_ks4 = np.concatenate(([0.0], np.cumsum(mi_ks4))) / (np.sum(mi_ks4) + 1e-12)
    gini_ks4 = (np.sum((2 * np.arange(1, n_ks4 + 1) - n_ks4 - 1) * mi_ks4)) / (n_ks4 * np.sum(mi_ks4) + 1e-12)

    ax.plot(cum_units_ks4 * 100, cum_info_ks4 * 100, color="#2563eb", linewidth=2.2, label=f"Kilosort 4 (Gini = {gini_ks4:.3f})")
    ax.plot(cum_units_lup * 100, cum_info_lup * 100, color="#d97706", linewidth=2.2, label=f"Lupin (Gini = {gini_lup:.3f})")
    ax.plot([0, 100], [0, 100], "k--", linewidth=1.2, label="Line of Perfect Equality (Gini = 0)")

    ax.set_xlabel("Cumulative Percentage of Population (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Cumulative Information / Attribution (%)", fontsize=10, fontweight="bold")
    ax.set_title("D. Lorenz Driver Inequality (Gini Invariance)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)

    out_fig = OUT_DIR / "intact_window_cross_sorter_6state_concordance.png"
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close()
    print(f"Successfully saved cross-sorter figure to {out_fig}")


if __name__ == '__main__':
    main()
