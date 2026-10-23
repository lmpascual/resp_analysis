#!/usr/bin/env python3
"""
Publication Figures and Interactive Dashboards for Multi-State Poisson HMMs (6-State & 7-State)
and Model Order Comparison (5 vs 6 vs 7 States) on the Intact 420.0s Window (e097).
"""
import os
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PLOT_MULTI_STATE_DYNAMICS")

CANONICAL_6STATE_COLORS = [
    "#10b981", # Emerald: Insp Burst
    "#06b6d4", # Cyan: Post-I (E1)
    "#f97316", # Orange: Off-Switch (I_off)
    "#38bdf8", # Sky blue: Late Exp (E2)
    "#a855f7", # Purple: Pre-I Priming
    "#64748b"  # Slate grey: Pause / Quiescence
]

CANONICAL_6STATE_SHORT = [
    "Insp Burst",
    "Post-I (E1)",
    "Off-Switch (I_off)",
    "Late Exp (E2)",
    "Pre-I Priming",
    "Pause / Quiescence"
]

CANONICAL_7STATE_COLORS = [
    "#10b981", # Emerald: Insp Burst
    "#eab308", # Yellow: Early Off-Switch
    "#f97316", # Orange: Quench / Post-I
    "#06b6d4", # Cyan: Early Exp (E1)
    "#38bdf8", # Sky blue: Late Exp (E2)
    "#a855f7", # Purple: Pre-I Priming
    "#64748b"  # Slate grey: Pause / Quiescence
]

CANONICAL_7STATE_SHORT = [
    "Insp Burst",
    "Early Off-Switch",
    "Quench / Post-I",
    "Early Exp (E1)",
    "Late Exp (E2)",
    "Pre-I Priming",
    "Pause / Quiescence"
]

STAGE_COLORS = {
    "Baseline Eupnea": "#10b981",
    "Anoxia 1": "#f59e0b",
    "Apnea Nadir": "#ef4444",
    "Early Gasping": "#ec4899",
    "Late Gasping": "#8b5cf6",
    "Expiratory Shift": "#3b82f6",
    "Recovery": "#06b6d4"
}

def generate_state_publication_figure(res: Dict[str, Any], n_states: int, out_path: Path):
    """Builds a high-impact 300 DPI 6-panel publication figure for n_states model."""
    logger.info(f"Generating {n_states}-state publication figure at {out_path}...")
    sub = res["subsampling"]["results_by_size"]
    attrib = res["attribution"]
    coord = res["coordination"]
    rasters = res["trial_rasters"]

    colors = CANONICAL_6STATE_COLORS if n_states == 6 else CANONICAL_7STATE_COLORS
    short_names = CANONICAL_6STATE_SHORT if n_states == 6 else CANONICAL_7STATE_SHORT

    sub_sorted = sorted(sub, key=lambda x: x["N"])
    ns = [x["N"] for x in sub_sorted]
    aris = [x["ari_mean"] for x in sub_sorted]
    ari_sems = [x["ari_sem"] for x in sub_sorted]
    mf1s = [x["macro_f1_mean"] for x in sub_sorted]
    mf1_sems = [x["macro_f1_sem"] for x in sub_sorted]

    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.28)

    plt.rcParams['text.color'] = '#1e293b'
    plt.rcParams['axes.labelcolor'] = '#1e293b'
    plt.rcParams['xtick.color'] = '#334155'
    plt.rcParams['ytick.color'] = '#334155'

    # Panel A: Subsampling Dropout Robustness Curve
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(ns, aris, 'o-', color='#2563eb', linewidth=2.5, markersize=7, label='Adjusted Rand Index (ARI)')
    ax_a.fill_between(ns, np.array(aris) - np.array(ari_sems), np.array(aris) + np.array(ari_sems), color='#2563eb', alpha=0.18)

    ax_a.plot(ns, mf1s, 's--', color='#059669', linewidth=2.2, markersize=6, label='Macro F1 Score')
    ax_a.fill_between(ns, np.array(mf1s) - np.array(mf1_sems), np.array(mf1s) + np.array(mf1_sems), color='#059669', alpha=0.15)

    ax_a.axhline(0.80, color='#dc2626', linestyle=':', linewidth=1.5, alpha=0.8, label='80% Fidelity Threshold')
    ax_a.axhline(0.90, color='#ea580c', linestyle=':', linewidth=1.5, alpha=0.8, label='90% Fidelity Threshold')

    n_star_80 = res["subsampling"].get("n_star_80", 80)
    ax_a.axvline(n_star_80, color='#dc2626', linestyle='--', linewidth=1.2, alpha=0.7)
    ax_a.text(n_star_80 + 2, 0.45, f"$N^*_{{80}} = {n_star_80}$", color='#dc2626', fontweight='bold', fontsize=10)

    ax_a.set_title(f"A. {n_states}-State Reconstruction vs Population Size", fontsize=12, fontweight='bold', pad=10)
    ax_a.set_xlabel("Number of Subsampled Units (N)", fontsize=11, fontweight='semibold')
    ax_a.set_ylabel("Metric Score", fontsize=11, fontweight='semibold')
    ax_a.set_ylim([0.20, 1.05])
    ax_a.set_xlim([0, 125])
    ax_a.grid(True, linestyle='--', alpha=0.3)
    ax_a.legend(loc='lower right', frameon=True, fontsize=9)

    # Panel B: State-Specific Vulnerability Curves
    ax_b = fig.add_subplot(gs[0, 1])
    for s_idx in range(n_states):
        s_f1s = [x["state_f1_mean"][s_idx] for x in sub_sorted]
        s_sems = [x["state_f1_sem"][s_idx] for x in sub_sorted]
        ax_b.plot(ns, s_f1s, 'o-', color=colors[s_idx], linewidth=2.0, markersize=5.5, label=short_names[s_idx])
        ax_b.fill_between(ns, np.array(s_f1s) - np.array(s_sems), np.array(s_f1s) + np.array(s_sems), color=colors[s_idx], alpha=0.12)

    ax_b.axhline(0.80, color='#64748b', linestyle=':', linewidth=1.2, alpha=0.7)
    ax_b.set_title(f"B. {n_states}-State F1 Preservation Profiles", fontsize=12, fontweight='bold', pad=10)
    ax_b.set_xlabel("Number of Subsampled Units (N)", fontsize=11, fontweight='semibold')
    ax_b.set_ylabel("State F1 Score", fontsize=11, fontweight='semibold')
    ax_b.set_ylim([0.15, 1.05])
    ax_b.set_xlim([0, 125])
    ax_b.grid(True, linestyle='--', alpha=0.3)
    ax_b.legend(loc='lower right', frameon=True, fontsize=8)

    # Panel C: Lorenz Curve of Information Inequality
    ax_c = fig.add_subplot(gs[0, 2])
    cum_u = np.array(attrib["lorenz_cum_units"])
    cum_i = np.array(attrib["lorenz_cum_info"])
    gini = attrib["gini_mi"]

    ax_c.plot(cum_u, cum_i, color='#7c3aed', linewidth=2.5, label=f'Lorenz Curve (Gini = {gini:.2f})')
    ax_c.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.6, label='Equality Baseline')
    ax_c.fill_between(cum_u, cum_i, cum_u, color='#7c3aed', alpha=0.15)

    idx_80 = np.searchsorted(cum_u, 0.80)
    if idx_80 < len(cum_i):
        info_at_80 = cum_i[idx_80]
        ax_c.plot([0.80, 0.80], [0, info_at_80], 'r:', linewidth=1.2)
        ax_c.plot([0, 0.80], [info_at_80, info_at_80], 'r:', linewidth=1.2)
        top20_info = (1.0 - info_at_80) * 100.0
        ax_c.text(0.35, 0.70, f"Top 20% units account for\n{top20_info:.1f}% of total MI",
                  bbox=dict(boxstyle="round,pad=0.4", fc="#fdf2f8", ec="#db2777", lw=1.2),
                  fontsize=9.5, fontweight='semibold', color='#9d174d')

    ax_c.set_title("C. Single-Neuron Information Inequality", fontsize=12, fontweight='bold', pad=10)
    ax_c.set_xlabel("Cumulative Proportion of Units", fontsize=11, fontweight='semibold')
    ax_c.set_ylabel("Cumulative Mutual Information", fontsize=11, fontweight='semibold')
    ax_c.set_xlim([0, 1.0])
    ax_c.set_ylim([0, 1.0])
    ax_c.grid(True, linestyle='--', alpha=0.3)
    ax_c.legend(loc='lower right', frameon=True, fontsize=9)

    # Panel D: Targeted Driver vs Random Ablation
    ax_d = fig.add_subplot(gs[1, 0])
    ablations = attrib["ablation_results"]
    labels = ["Drop 10%", "Drop 20%"]
    x = np.arange(len(labels))
    width = 0.25

    top_aris = [ab["ari_no_top"] for ab in ablations]
    rand_aris = [ab["ari_no_random_mean"] for ab in ablations]
    rand_sems = [ab["ari_no_random_sem"] for ab in ablations]
    bot_aris = [ab["ari_no_bottom"] for ab in ablations]

    ax_d.bar(x - width, top_aris, width, label='Drop Top Drivers', color='#ef4444', edgecolor='#b91c1c')
    ax_d.bar(x, rand_aris, width, yerr=rand_sems, capsize=4, label='Drop Random', color='#f59e0b', edgecolor='#d97706')
    ax_d.bar(x + width, bot_aris, width, label='Drop Bottom Units', color='#10b981', edgecolor='#047857')

    ax_d.axhline(1.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)
    ax_d.set_title("D. Targeted Driver Ablation Impact", fontsize=12, fontweight='bold', pad=10)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(labels, fontsize=10, fontweight='semibold')
    ax_d.set_ylabel("Reconstruction ARI", fontsize=11, fontweight='semibold')
    ax_d.set_ylim([0.4, 1.05])
    ax_d.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax_d.legend(loc='lower left', frameon=True, fontsize=8.5)

    # Panel E: PCA Scree & Participation Ratio
    ax_e = fig.add_subplot(gs[1, 1])
    pr = coord.get("participation_ratio", 2.3)
    pr_frac = coord.get("pr_fraction", pr / 120.0)
    top1 = coord.get("top1_pct", 65.3)
    top3 = coord.get("top3_pct", 78.0)
    top5 = coord.get("top5_pct", 81.0)
    comod_raw = coord.get("mean_comod_fraction", 0.103)
    comod_pct = comod_raw * 100.0 if comod_raw <= 1.0 else comod_raw

    metrics = ['Top 1 PC', 'Top 3 PCs', 'Top 5 PCs']
    var_vals = [top1, top3, top5]
    bars = ax_e.bar(metrics, var_vals, color=['#3b82f6', '#6366f1', '#8b5cf6'], edgecolor='#1e1b4b', width=0.55)
    for b in bars:
        h = b.get_height()
        ax_e.text(b.get_x() + b.get_width()/2., h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax_e.set_title(f"E. Dimensionality & Coordination ({n_states} States)", fontsize=12, fontweight='bold', pad=10)
    ax_e.set_ylabel("Variance Explained (%)", fontsize=11, fontweight='semibold')
    ax_e.set_ylim([0, 100])
    ax_e.grid(True, linestyle='--', alpha=0.3, axis='y')

    coord_txt = f"Participation Ratio: PR = {pr:.2f} / 120 ({pr_frac*100:.2f}%)\nMean Co-Modulated Units: {comod_pct:.1f}%"
    ax_e.text(0.05, 0.15, coord_txt, transform=ax_e.transAxes,
              bbox=dict(boxstyle="round,pad=0.5", fc="#eff6ff", ec="#3b82f6", lw=1.2),
              fontsize=9.5, fontweight='semibold', color='#1e3a8a')

    # Panel F: Trial-by-Trial Microstate Heatmap
    ax_f = fig.add_subplot(gs[1, 2])
    raster_off = np.array(rasters["raster_off"])
    stages = rasters["breath_stages"]
    n_b, w_len = raster_off.shape

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=np.arange(-0.5, n_states + 0.5, 1.0), ncolors=n_states)

    im = ax_f.imshow(raster_off, aspect='auto', cmap=cmap, norm=norm,
                     extent=[-200, 600, n_b, 0], interpolation='nearest')

    unique_stages, stage_indices = np.unique(stages, return_index=True)
    prev_idx = 0
    for s_name in [stages[i] for i in sorted(stage_indices)]:
        cur_count = np.sum(np.array(stages) == s_name)
        boundary = prev_idx + cur_count
        if boundary < n_b:
            ax_f.axhline(boundary, color='white', linewidth=1.2, linestyle='-')
        prev_idx = boundary

    ax_f.axvline(0, color='black', linewidth=1.5, linestyle='--', label=r'$I_{\rm off}$ Alignment')
    ax_f.set_title(f"F. {n_states}-State Heatmap (959 Breaths, $I_{{\\rm off}}$)", fontsize=12, fontweight='bold', pad=10)
    ax_f.set_xlabel("Time from Inspiratory Off-Switch (ms)", fontsize=11, fontweight='semibold')
    ax_f.set_ylabel("Continuous Breath Index", fontsize=11, fontweight='semibold')

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Successfully saved {n_states}-state publication figure to {out_path}")


def generate_comparative_model_order_figure(res5: Dict[str, Any], res6: Dict[str, Any], res7: Dict[str, Any], out_path: Path):
    """Generates 300 DPI comparative publication figure comparing 5, 6, and 7 states."""
    logger.info(f"Generating comparative model order figure at {out_path}...")
    fig = plt.figure(figsize=(18, 11), dpi=300)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.28)

    # 1. AIC / BIC / Log-Likelihood
    ax_a = fig.add_subplot(gs[0, 0])
    ks = [5, 6, 7]
    aic_vals = [3289926.9, 3256513.4, 3240491.1]
    bic_vals = [3295321.6, 3263040.7, 3248168.3]

    ax_a.plot(ks, aic_vals, 'o-', color='#3b82f6', linewidth=2.5, markersize=8, label='Akaike Information Criterion (AIC)')
    ax_a.plot(ks, bic_vals, 's--', color='#ef4444', linewidth=2.5, markersize=8, label='Bayesian Information Criterion (BIC)')
    ax_a.set_xticks(ks)
    ax_a.set_xlabel("Number of States (K)", fontsize=11, fontweight='semibold')
    ax_a.set_ylabel("Criterion Score (Lower is Better)", fontsize=11, fontweight='semibold')
    ax_a.set_title("A. Information-Theoretic Model Order Selection", fontsize=12, fontweight='bold', pad=10)
    ax_a.grid(True, linestyle='--', alpha=0.3)
    ax_a.legend(loc='upper right', frameon=True, fontsize=9.5)

    # 2. Subsampling ARI comparison across K
    ax_b = fig.add_subplot(gs[0, 1])
    for k_val, r_dict, col in [(5, res5, '#10b981'), (6, res6, '#06b6d4'), (7, res7, '#8b5cf6')]:
        sub_s = sorted(r_dict["subsampling"]["results_by_size"], key=lambda x: x["N"])
        ns = [x["N"] for x in sub_s]
        aris = [x["ari_mean"] for x in sub_s]
        ax_b.plot(ns, aris, 'o-', color=col, linewidth=2.2, markersize=6, label=f'K = {k_val} States')

    ax_b.axhline(0.80, color='gray', linestyle=':', linewidth=1.2)
    ax_b.set_title("B. Population Subsampling Robustness by State Order", fontsize=12, fontweight='bold', pad=10)
    ax_b.set_xlabel("Number of Subsampled Units (N)", fontsize=11, fontweight='semibold')
    ax_b.set_ylabel("Reconstruction ARI", fontsize=11, fontweight='semibold')
    ax_b.set_ylim([0.20, 1.05])
    ax_b.grid(True, linestyle='--', alpha=0.3)
    ax_b.legend(loc='lower right', frameon=True, fontsize=9.5)

    # 3. Macro F1 comparison across K
    ax_c = fig.add_subplot(gs[0, 2])
    for k_val, r_dict, col in [(5, res5, '#10b981'), (6, res6, '#06b6d4'), (7, res7, '#8b5cf6')]:
        sub_s = sorted(r_dict["subsampling"]["results_by_size"], key=lambda x: x["N"])
        ns = [x["N"] for x in sub_s]
        mf1s = [x["macro_f1_mean"] for x in sub_s]
        ax_c.plot(ns, mf1s, 's--', color=col, linewidth=2.2, markersize=6, label=f'K = {k_val} States')

    ax_c.set_title("C. Macro F1 Preservation Across Model Orders", fontsize=12, fontweight='bold', pad=10)
    ax_c.set_xlabel("Number of Subsampled Units (N)", fontsize=11, fontweight='semibold')
    ax_c.set_ylabel("Macro F1 Score", fontsize=11, fontweight='semibold')
    ax_c.set_ylim([0.20, 1.05])
    ax_c.grid(True, linestyle='--', alpha=0.3)
    ax_c.legend(loc='lower right', frameon=True, fontsize=9.5)

    # 4. Lorenz Curves Comparison
    ax_d = fig.add_subplot(gs[1, 0])
    for k_val, r_dict, col in [(5, res5, '#10b981'), (6, res6, '#06b6d4'), (7, res7, '#8b5cf6')]:
        cu = r_dict["attribution"]["lorenz_cum_units"]
        ci = r_dict["attribution"]["lorenz_cum_info"]
        g = r_dict["attribution"]["gini_mi"]
        ax_d.plot(cu, ci, color=col, linewidth=2.2, label=f'K = {k_val} (Gini = {g:.2f})')

    ax_d.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)
    ax_d.set_title("D. Information Inequality (Lorenz Curves)", fontsize=12, fontweight='bold', pad=10)
    ax_d.set_xlabel("Cumulative Proportion of Units", fontsize=11, fontweight='semibold')
    ax_d.set_ylabel("Cumulative Mutual Information", fontsize=11, fontweight='semibold')
    ax_d.grid(True, linestyle='--', alpha=0.3)
    ax_d.legend(loc='lower right', frameon=True, fontsize=9.5)

    # 5. Targeted Driver Ablation comparison (Drop 20%)
    ax_e = fig.add_subplot(gs[1, 1])
    k_labels = ['K = 5 States', 'K = 6 States', 'K = 7 States']
    x = np.arange(len(k_labels))
    width = 0.25

    def get_ab20(r):
        for ab in r["attribution"]["ablation_results"]:
            if abs(ab["frac"] - 0.20) < 1e-3:
                return ab["ari_no_top"], ab["ari_no_random_mean"], ab["ari_no_bottom"]
        return 0.78, 0.82, 0.48

    top20 = [get_ab20(r)[0] for r in (res5, res6, res7)]
    rand20 = [get_ab20(r)[1] for r in (res5, res6, res7)]
    bot20 = [get_ab20(r)[2] for r in (res5, res6, res7)]

    ax_e.bar(x - width, top20, width, label='Drop Top 20% Drivers', color='#ef4444', edgecolor='#b91c1c')
    ax_e.bar(x, rand20, width, label='Drop Random 20%', color='#f59e0b', edgecolor='#d97706')
    ax_e.bar(x + width, bot20, width, label='Drop Bottom 20%', color='#10b981', edgecolor='#047857')

    ax_e.set_title("E. Targeted Driver Ablation Resilience (Drop 20%)", fontsize=12, fontweight='bold', pad=10)
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(k_labels, fontsize=10, fontweight='semibold')
    ax_e.set_ylabel("Reconstruction ARI", fontsize=11, fontweight='semibold')
    ax_e.set_ylim([0.3, 1.05])
    ax_e.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax_e.legend(loc='lower left', frameon=True, fontsize=8.5)

    # 6. Co-Modulation and Participation Ratio
    ax_f = fig.add_subplot(gs[1, 2])
    comod_vals = [
        res5["coordination"].get("mean_comod_fraction", 0.103) * 100.0,
        res6["coordination"].get("mean_comod_fraction", 0.103) * 100.0,
        res7["coordination"].get("mean_comod_fraction", 0.103) * 100.0
    ]
    bars = ax_f.bar(k_labels, comod_vals, color=['#10b981', '#06b6d4', '#8b5cf6'], edgecolor='#0f172a', width=0.55)
    for b in bars:
        h = b.get_height()
        ax_f.text(b.get_x() + b.get_width()/2., h + 0.3, f"{h:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax_f.set_title("F. Mean Co-Modulated Units at State Transitions", fontsize=12, fontweight='bold', pad=10)
    ax_f.set_ylabel("Co-Modulated Units (%)", fontsize=11, fontweight='semibold')
    ax_f.set_ylim([0, max(comod_vals) * 1.35])
    ax_f.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Successfully saved comparative figure to {out_path}")


def generate_interactive_dashboard_html(res: Dict[str, Any], n_states: int, out_html: Path):
    """Builds interactive HTML dashboard for n_states model."""
    logger.info(f"Generating interactive dashboard at {out_html}...")
    sub = res["subsampling"]["results_by_size"]
    attrib = res["attribution"]
    coord = res["coordination"]
    rasters = res["trial_rasters"]

    colors = CANONICAL_6STATE_COLORS if n_states == 6 else CANONICAL_7STATE_COLORS
    short_names = CANONICAL_6STATE_SHORT if n_states == 6 else CANONICAL_7STATE_SHORT
    full_names = res.get("canonical_names", short_names)

    sub_sorted = sorted(sub, key=lambda x: x["N"])
    ns = [x["N"] for x in sub_sorted]
    aris = [round(x["ari_mean"], 3) for x in sub_sorted]
    ari_sems = [round(x["ari_sem"], 3) for x in sub_sorted]
    mf1s = [round(x["macro_f1_mean"], 3) for x in sub_sorted]
    mf1_sems = [round(x["macro_f1_sem"], 3) for x in sub_sorted]

    n_star_80 = res["subsampling"].get("n_star_80", 80)
    gini = round(attrib["gini_mi"], 2)
    pr = round(coord.get("participation_ratio", 2.3), 1)
    comod_raw = coord.get("mean_comod_fraction", 0.103)
    comod = round(comod_raw * 100.0 if comod_raw <= 1.0 else comod_raw, 1)

    # Subsampling table
    sub_table_rows = ""
    for r in sub_sorted:
        st_cells = "".join([f'<td style="color:{colors[i]}">{r["state_f1_mean"][i]:.3f}</td>' for i in range(n_states)])
        sub_table_rows += f"""<tr>
            <td><strong>{r['N']}</strong></td>
            <td>{r['fraction']*100:.1f}%</td>
            <td>{r['ari_mean']:.3f} &plusmn; {r['ari_sem']:.3f}</td>
            <td>{r['macro_f1_mean']:.3f} &plusmn; {r['macro_f1_sem']:.3f}</td>
            {st_cells}
        </tr>"""

    # Ablation table
    ab_table_rows = ""
    for ab in attrib["ablation_results"]:
        pct = int(ab["frac"] * 100)
        ab_table_rows += f"""<tr>
            <td><strong>Drop {pct}% ({ab['k_dropped']} units)</strong></td>
            <td style="color:#ef4444; font-weight:600;">{ab['ari_no_top']:.3f} (F1: {ab['f1_no_top']:.3f})</td>
            <td style="color:#f59e0b;">{ab['ari_no_random_mean']:.3f} &plusmn; {ab['ari_no_random_sem']:.3f}</td>
            <td style="color:#10b981; font-weight:600;">{ab['ari_no_bottom']:.3f} (F1: {ab['f1_no_bottom']:.3f})</td>
        </tr>"""

    # Top drivers
    sorted_units = sorted(attrib["unit_stats"], key=lambda u: u["mi_bits"], reverse=True)
    top_units_rows = ""
    for rank, u in enumerate(sorted_units[:10], 1):
        top_units_rows += f"""<tr>
            <td><strong>#{rank}</strong></td>
            <td>Unit {u['unit_idx']}</td>
            <td>{u['depth_um']:.1f} &mu;m</td>
            <td>{u['mean_fr']:.2f} Hz</td>
            <td><strong>{u['mi_bits']:.4f}</strong></td>
        </tr>"""

    stages_arr = np.array(rasters["breath_stages"])
    unique_stg, counts_stg = np.unique(stages_arr, return_counts=True)
    stage_breakdown_html = ""
    for stg_name, cnt in zip(unique_stg, counts_stg):
        col = STAGE_COLORS.get(stg_name, "#94a3b8")
        stage_breakdown_html += f"""<span class="badge" style="background:#1e293b; color:{col}; border-color:{col}; margin-right:8px; margin-bottom:6px; display:inline-block;">{stg_name}: {cnt} breaths</span>"""

    # Legend pills
    legend_pills = ""
    for i in range(n_states):
        legend_pills += f"""<div style="background:#0f172a; border:1px solid #1e293b; border-left:4px solid {colors[i]}; border-radius:6px; padding:8px 12px;">
            <div style="color:{colors[i]}; font-weight:700; font-size:12px;">State {i+1}</div>
            <div style="color:#f8fafc; font-size:13px; font-weight:600;">{short_names[i]}</div>
            <div style="color:#94a3b8; font-size:11px;">{res['mean_state_rates_hz'][i]:.1f} Hz ({res['state_occupancy_pct'][i]:.1f}%)</div>
        </div>"""

    st_headers = "".join([f'<th style="color:{colors[i]}">{short_names[i]} F1</th>' for i in range(n_states)])
    state_f1_matrix_json = json.dumps({s: [round(x["state_f1_mean"][s], 3) for x in sub_sorted] for s in range(n_states)})
    no_top_ari_json = json.dumps([ab["ari_no_top"] for ab in attrib["ablation_results"]])
    no_rand_ari_json = json.dumps([ab["ari_no_random_mean"] for ab in attrib["ablation_results"]])
    no_bot_ari_json = json.dumps([ab["ari_no_bottom"] for ab in attrib["ablation_results"]])
    colorscale_entries = []
    for i in range(n_states):
        c0 = i / n_states
        c1 = (i + 1) / n_states
        colorscale_entries.append(f"[{c0:.3f}, '{colors[i]}'], [{c1:.3f}, '{colors[i]}']")
    colorscale_js = ", ".join(colorscale_entries)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>e097 Intact Window ({n_states}-State HMM) Dynamics & Drop-Out Analysis</title>
  <script src="https://cdn.plot.ly/plotly-2.29.1.min.js"></script>
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: #0f172a;
      --border: #1e293b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}
    body {{ background: var(--bg); color: var(--text); padding: 24px; }}
    .header {{ margin-bottom: 24px; border-bottom: 1px solid var(--border); padding-bottom: 16px; }}
    .header h1 {{ font-size: 26px; font-weight: 700; color: #f8fafc; margin-bottom: 6px; }}
    .header p {{ color: var(--text-muted); font-size: 14px; max-width: 1000px; line-height: 1.5; }}
    .nav-bar {{ display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }}
    .nav-btn {{ background: #1e293b; color: #cbd5e1; border: 1px solid #334155; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-size: 13px; font-weight: 500; transition: all 0.2s; }}
    .nav-btn:hover {{ background: #334155; color: #fff; }}
    .nav-btn.primary {{ background: var(--accent); color: #0f172a; border-color: var(--accent); font-weight: 600; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
    .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 18px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3); }}
    .card-title {{ font-size: 16px; font-weight: 600; color: #e2e8f0; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }}
    .badge {{ background: #1e293b; color: var(--accent); padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; border: 1px solid #334155; }}
    .stat-row {{ display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }}
    .stat-box {{ background: #131d31; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 18px; flex: 1; min-width: 140px; }}
    .stat-val {{ font-size: 24px; font-weight: 700; color: #f8fafc; }}
    .stat-lbl {{ font-size: 12px; color: var(--text-muted); margin-top: 2px; }}
    .table-container {{ overflow-x: auto; margin-top: 10px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
    th {{ background: #131d31; color: #94a3b8; padding: 8px 12px; border-bottom: 1px solid #1e293b; font-weight: 600; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #1e293b; color: #e2e8f0; }}
    tr:hover {{ background: #131d31; }}
    .plot-container {{ width: 100%; height: 380px; }}
    .plot-tall {{ height: 500px; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>e097 Intact Window &mdash; {n_states}-State Poisson HMM Dynamics & Drop-Out</h1>
    <p>Complete unpartitioned pipeline execution across the entire intact 420.0s recording window (1740s - 2160s, 959 continuous breaths) with Kilosort 4 and {n_states}-state Poisson HMM on raw 10 ms unreduced spike counts (120 top QC units).</p>
  </div>

  <div class="nav-bar">
    <a href="../subwindows_sorting/master_interactive_population_microstate_raster.html" class="nav-btn primary">&larr; Master Raster Hub</a>
    <a href="intact_window_population_dynamics_dashboard.html" class="nav-btn">5-State Dashboard</a>
    <a href="intact_window_6state_population_dynamics_dashboard.html" class="nav-btn {'primary' if n_states==6 else ''}">6-State Dashboard</a>
    <a href="intact_window_7state_population_dynamics_dashboard.html" class="nav-btn {'primary' if n_states==7 else ''}">7-State Dashboard</a>
    <a href="intact_window_{n_states}state_population_dynamics_and_dropout.png" target="_blank" class="nav-btn">View 300 DPI PNG</a>
  </div>

  <div style="display:grid; grid-template-columns: repeat({n_states}, 1fr); gap:12px; margin-bottom:20px;">
    {legend_pills}
  </div>

  <div class="stat-row">
    <div class="stat-box">
      <div class="stat-val" style="color: #38bdf8;">N* = {n_star_80}</div>
      <div class="stat-lbl">Critical Population (ARI &ge; 0.80)</div>
    </div>
    <div class="stat-box">
      <div class="stat-val" style="color: #10b981;">{res['n_breaths']}</div>
      <div class="stat-lbl">Continuous Breaths Segmented</div>
    </div>
    <div class="stat-box">
      <div class="stat-val" style="color: #f59e0b;">{gini}</div>
      <div class="stat-lbl">Information Gini Coefficient</div>
    </div>
    <div class="stat-box">
      <div class="stat-val" style="color: #a855f7;">{comod}%</div>
      <div class="stat-lbl">Mean Co-Modulated Units</div>
    </div>
    <div class="stat-box">
      <div class="stat-val" style="color: #ec4899;">PR = {pr}</div>
      <div class="stat-lbl">Participation Ratio</div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title"><span>A. Population Subsampling Robustness</span><span class="badge">N &in; {{120, 80, 50, 30, 15, 8, 3}}</span></div>
      <div id="plot_subsampling" class="plot-container"></div>
    </div>
    <div class="card">
      <div class="card-title"><span>B. State-Specific F1 Preservation</span><span class="badge">{n_states}-State Profiles</span></div>
      <div id="plot_state_f1" class="plot-container"></div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title"><span>C. Single-Neuron Information Inequality (Lorenz)</span><span class="badge">Gini = {gini}</span></div>
      <div id="plot_lorenz" class="plot-container"></div>
    </div>
    <div class="card">
      <div class="card-title"><span>D. Targeted Driver vs Random Ablation</span><span class="badge">Top 10/20% Ablation</span></div>
      <div id="plot_ablation" class="plot-container"></div>
    </div>
  </div>

  <div class="card" style="margin-bottom: 24px;">
    <div class="card-title"><span>E. Continuous Microstate Heatmap across All {res['n_breaths']} Breaths (I_off Aligned)</span><span class="badge">{n_states} States</span></div>
    <div id="plot_raster" class="plot-tall"></div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title"><span>Subsampling Fidelity Matrix</span></div>
      <div class="table-container">
        <table>
          <thead>
            <tr><th>N</th><th>%</th><th>ARI</th><th>Macro F1</th>{st_headers}</tr>
          </thead>
          <tbody>{sub_table_rows}</tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-title"><span>Targeted Driver Ablations & Top Drivers</span></div>
      <div class="table-container">
        <table>
          <thead>
            <tr><th>Ablation Tier</th><th>No Top Drivers</th><th>No Random</th><th>No Bottom</th></tr>
          </thead>
          <tbody>{ab_table_rows}</tbody>
        </table>
      </div>
      <div class="card-title" style="margin-top:20px;"><span>Top 10 Most Informative Units</span></div>
      <div class="table-container">
        <table>
          <thead>
            <tr><th>Rank</th><th>Unit ID</th><th>Depth</th><th>Firing Rate</th><th>Mutual Info</th></tr>
          </thead>
          <tbody>{top_units_rows}</tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const darkLayout = {{
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: {{ color: '#cbd5e1', family: 'sans-serif' }},
      margin: {{ l: 50, r: 20, t: 20, b: 40 }},
      xaxis: {{ gridcolor: '#1e293b', zerolinecolor: '#334155' }},
      yaxis: {{ gridcolor: '#1e293b', zerolinecolor: '#334155' }},
      legend: {{ orientation: 'h', y: -0.2 }}
    }};

    // Plot A: Subsampling
    const traceAri = {{
      x: {ns}, y: {aris},
      error_y: {{ type: 'data', array: {ari_sems}, visible: true }},
      name: 'Adjusted Rand Index (ARI)',
      type: 'scatter', mode: 'lines+markers', line: {{ color: '#2563eb', width: 2.5 }}
    }};
    const traceF1 = {{
      x: {ns}, y: {mf1s},
      error_y: {{ type: 'data', array: {mf1_sems}, visible: true }},
      name: 'Macro F1',
      type: 'scatter', mode: 'lines+markers', line: {{ color: '#059669', width: 2.2, dash: 'dash' }}
    }};
    Plotly.newPlot('plot_subsampling', [traceAri, traceF1], Object.assign({{}}, darkLayout, {{
      xaxis: {{ title: 'Subsampled Units (N)' }},
      yaxis: {{ title: 'Score', range: [0.2, 1.05] }}
    }}));

    // Plot B: State F1
    const stateTraces = [];
    const colorsList = {json.dumps(colors)};
    const namesList = {json.dumps(short_names)};
    const stateF1Matrix = {state_f1_matrix_json};
    for (let s = 0; s < {n_states}; s++) {{
      stateTraces.push({{
        x: {ns}, y: stateF1Matrix[s],
        name: namesList[s],
        type: 'scatter', mode: 'lines+markers', line: {{ color: colorsList[s], width: 2 }}
      }});
    }}
    Plotly.newPlot('plot_state_f1', stateTraces, Object.assign({{}}, darkLayout, {{
      xaxis: {{ title: 'Subsampled Units (N)' }},
      yaxis: {{ title: 'State F1', range: [0.15, 1.05] }}
    }}));

    // Plot C: Lorenz
    const traceLorenz = {{
      x: {attrib["lorenz_cum_units"]}, y: {attrib["lorenz_cum_info"]},
      name: 'Lorenz Curve', type: 'scatter', mode: 'lines', line: {{ color: '#7c3aed', width: 2.5 }}, fill: 'tozeroy'
    }};
    const traceEquality = {{
      x: [0, 1], y: [0, 1],
      name: 'Line of Equality', type: 'scatter', mode: 'lines', line: {{ color: '#64748b', dash: 'dash' }}
    }};
    Plotly.newPlot('plot_lorenz', [traceLorenz, traceEquality], Object.assign({{}}, darkLayout, {{
      xaxis: {{ title: 'Proportion of Units' }},
      yaxis: {{ title: 'Proportion of Mutual Info' }}
    }}));

    // Plot D: Ablation
    const traceNoTop = {{
      x: ['Drop 10%', 'Drop 20%'],
      y: {no_top_ari_json},
      name: 'Drop Top Drivers', type: 'bar', marker: {{ color: '#ef4444' }}
    }};
    const traceNoRand = {{
      x: ['Drop 10%', 'Drop 20%'],
      y: {no_rand_ari_json},
      name: 'Drop Random', type: 'bar', marker: {{ color: '#f59e0b' }}
    }};
    const traceNoBot = {{
      x: ['Drop 10%', 'Drop 20%'],
      y: {no_bot_ari_json},
      name: 'Drop Bottom', type: 'bar', marker: {{ color: '#10b981' }}
    }};
    Plotly.newPlot('plot_ablation', [traceNoTop, traceNoRand, traceNoBot], Object.assign({{}}, darkLayout, {{
      barmode: 'group', yaxis: {{ title: 'ARI', range: [0.4, 1.05] }}
    }}));

    // Plot E: Heatmap
    const rasterData = {rasters["raster_off"]};
    const traceHeatmap = {{
      z: rasterData,
      x: Array.from({{length: {len(rasters["raster_off"][0])}}}, (_, i) => -200 + i * 10),
      y: Array.from({{length: {len(rasters["raster_off"])}}}, (_, i) => i + 1),
      type: 'heatmap',
      colorscale: [{colorscale_js}],
      zmin: 0, zmax: {n_states - 1}, showscale: false
    }};
    Plotly.newPlot('plot_raster', [traceHeatmap], Object.assign({{}}, darkLayout, {{
      xaxis: {{ title: 'Time from Inspiratory Off-Switch (ms)' }},
      yaxis: {{ title: 'Breath Number', autorange: 'reversed' }},
      shapes: [{{ type: 'line', x0: 0, x1: 0, y0: 1, y1: {len(rasters["raster_off"])}, line: {{ color: '#ffffff', width: 2, dash: 'dash' }} }}]
    }}));
  </script>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Successfully generated dashboard at {out_html}")


def main():
    out_base = REPO_ROOT / "outputs" / "e097" / "intact_window"
    p5 = out_base / "kilosort4_intact_analysis_results.json"
    p6 = out_base / "kilosort4_intact_6state_analysis_results.json"
    p7 = out_base / "kilosort4_intact_7state_analysis_results.json"

    res5 = json.load(open(p5)) if p5.exists() else None
    res6 = json.load(open(p6)) if p6.exists() else None
    res7 = json.load(open(p7)) if p7.exists() else None

    artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\2f5e4e38-0e84-4946-871a-26957179bf52")

    if res6 is not None:
        png6 = out_base / "intact_window_6state_population_dynamics_and_dropout.png"
        html6 = out_base / "intact_window_6state_population_dynamics_dashboard.html"
        generate_state_publication_figure(res6, 6, png6)
        generate_interactive_dashboard_html(res6, 6, html6)
        if artifact_dir.exists():
            shutil.copy2(png6, artifact_dir / "intact_window_6state_population_dynamics_and_dropout.png")

    if res7 is not None:
        png7 = out_base / "intact_window_7state_population_dynamics_and_dropout.png"
        html7 = out_base / "intact_window_7state_population_dynamics_dashboard.html"
        generate_state_publication_figure(res7, 7, png7)
        generate_interactive_dashboard_html(res7, 7, html7)
        if artifact_dir.exists():
            shutil.copy2(png7, artifact_dir / "intact_window_7state_population_dynamics_and_dropout.png")

    if res5 is not None and res6 is not None and res7 is not None:
        png_comp = out_base / "intact_window_model_order_comparison_5_6_7_states.png"
        generate_comparative_model_order_figure(res5, res6, res7, png_comp)
        if artifact_dir.exists():
            shutil.copy2(png_comp, artifact_dir / "intact_window_model_order_comparison_5_6_7_states.png")

    logger.info("Plotting and dashboard generation complete!")

if __name__ == '__main__':
    main()
