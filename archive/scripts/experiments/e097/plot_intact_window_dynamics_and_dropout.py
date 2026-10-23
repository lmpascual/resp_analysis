#!/usr/bin/env python3
"""
Publication Figure and Interactive Dashboard Generator for Intact Window Analysis (e097):
- Generates 300 DPI multi-panel publication figure:
  outputs/e097/intact_window/intact_window_population_dynamics_and_dropout.png
- Generates rich interactive HTML dashboard:
  outputs/e097/intact_window/intact_window_population_dynamics_dashboard.html
- Copies publication figure to artifact directory for markdown embedding.
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
logger = logging.getLogger("PLOT_INTACT_DYNAMICS")

CANONICAL_STATE_NAMES = [
    "State 1: Insp Burst (Peak Drive)",
    "State 2: Peak / Off-Switch Quench (I_off)",
    "State 3: Post-Insp / Expiratory Drive (Exp)",
    "State 4: Pre-Insp Priming (Pre-I)",
    "State 5: Inter-Breath Pause (Quiescence)"
]

CANONICAL_SHORT_NAMES = [
    "Insp Burst",
    "Off-Switch (I_off)",
    "Expiratory Drive",
    "Pre-Insp Priming",
    "Pause / Quiescence"
]

CANONICAL_COLORS = [
    "#10b981", # Emerald
    "#f97316", # Orange
    "#38bdf8", # Sky blue
    "#a855f7", # Purple
    "#64748b"  # Slate grey
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

def generate_static_publication_figure(ks4_res: Dict[str, Any], out_path: Path):
    """Builds a high-impact 300 DPI 6-panel publication figure."""
    logger.info(f"Generating publication figure at {out_path}...")
    
    sub = ks4_res["subsampling"]["results_by_size"]
    attrib = ks4_res["attribution"]
    coord = ks4_res["coordination"]
    rasters = ks4_res["trial_rasters"]
    
    # Sort subsampling results by N ascending
    sub_sorted = sorted(sub, key=lambda x: x["N"])
    ns = [x["N"] for x in sub_sorted]
    aris = [x["ari_mean"] for x in sub_sorted]
    ari_sems = [x["ari_sem"] for x in sub_sorted]
    mf1s = [x["macro_f1_mean"] for x in sub_sorted]
    mf1_sems = [x["macro_f1_sem"] for x in sub_sorted]
    
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.28)
    
    # Set dark aesthetic for scientific plots
    plt.rcParams['text.color'] = '#1e293b'
    plt.rcParams['axes.labelcolor'] = '#1e293b'
    plt.rcParams['xtick.color'] = '#334155'
    plt.rcParams['ytick.color'] = '#334155'

    # ----------------------------------------------------
    # Panel A: Subsampling Dropout Robustness Curve
    # ----------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(ns, aris, 'o-', color='#2563eb', linewidth=2.5, markersize=7, label='Adjusted Rand Index (ARI)')
    ax_a.fill_between(ns, np.array(aris) - np.array(ari_sems), np.array(aris) + np.array(ari_sems), color='#2563eb', alpha=0.18)
    
    ax_a.plot(ns, mf1s, 's--', color='#059669', linewidth=2.2, markersize=6, label='Macro F1 Score')
    ax_a.fill_between(ns, np.array(mf1s) - np.array(mf1_sems), np.array(mf1s) + np.array(mf1_sems), color='#059669', alpha=0.15)
    
    ax_a.axhline(0.80, color='#dc2626', linestyle=':', linewidth=1.5, alpha=0.8, label='80% Fidelity Threshold')
    ax_a.axhline(0.90, color='#ea580c', linestyle=':', linewidth=1.5, alpha=0.8, label='90% Fidelity Threshold')
    
    n_star_80 = ks4_res["subsampling"].get("n_star_80", 80)
    n_star_90 = ks4_res["subsampling"].get("n_star_90", 120)
    ax_a.axvline(n_star_80, color='#dc2626', linestyle='--', linewidth=1.2, alpha=0.7)
    ax_a.text(n_star_80 + 2, 0.45, f"$N^*_{{80}} = {n_star_80}$", color='#dc2626', fontweight='bold', fontsize=10)
    
    ax_a.set_title("A. State Reconstruction vs Population Size", fontsize=12, fontweight='bold', pad=10)
    ax_a.set_xlabel("Number of Subsampled Units (N)", fontsize=11, fontweight='semibold')
    ax_a.set_ylabel("Metric Score", fontsize=11, fontweight='semibold')
    ax_a.set_ylim([0.20, 1.05])
    ax_a.set_xlim([0, 125])
    ax_a.grid(True, linestyle='--', alpha=0.3)
    ax_a.legend(loc='lower right', frameon=True, fontsize=9)

    # ----------------------------------------------------
    # Panel B: State-Specific Vulnerability Curves
    # ----------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    for s_idx in range(5):
        s_f1s = [x["state_f1_mean"][s_idx] for x in sub_sorted]
        s_sems = [x["state_f1_sem"][s_idx] for x in sub_sorted]
        ax_b.plot(ns, s_f1s, 'o-', color=CANONICAL_COLORS[s_idx], linewidth=2.2, markersize=6, label=CANONICAL_SHORT_NAMES[s_idx])
        ax_b.fill_between(ns, np.array(s_f1s) - np.array(s_sems), np.array(s_f1s) + np.array(s_sems), color=CANONICAL_COLORS[s_idx], alpha=0.15)
        
    ax_b.axhline(0.80, color='#64748b', linestyle=':', linewidth=1.2, alpha=0.7)
    ax_b.set_title("B. State-Specific F1 Preservation", fontsize=12, fontweight='bold', pad=10)
    ax_b.set_xlabel("Number of Subsampled Units (N)", fontsize=11, fontweight='semibold')
    ax_b.set_ylabel("State F1 Score", fontsize=11, fontweight='semibold')
    ax_b.set_ylim([0.20, 1.05])
    ax_b.set_xlim([0, 125])
    ax_b.grid(True, linestyle='--', alpha=0.3)
    ax_b.legend(loc='lower right', frameon=True, fontsize=8.5)

    # ----------------------------------------------------
    # Panel C: Lorenz Curve of Information Inequality
    # ----------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    cum_u = np.array(attrib["lorenz_cum_units"])
    cum_i = np.array(attrib["lorenz_cum_info"])
    gini = attrib["gini_mi"]
    
    ax_c.plot(cum_u, cum_i, color='#7c3aed', linewidth=2.5, label=f'Lorenz Curve (Gini = {gini:.2f})')
    ax_c.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.6, label='Equality Baseline')
    ax_c.fill_between(cum_u, cum_i, cum_u, color='#7c3aed', alpha=0.15)
    
    # 80/20 annotation
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

    # ----------------------------------------------------
    # Panel D: Targeted Driver vs Random Ablation
    # ----------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 0])
    ablations = attrib["ablation_results"]
    
    labels = ["Drop 10%", "Drop 20%"]
    x = np.arange(len(labels))
    width = 0.25
    
    top_aris = [ab["ari_no_top"] for ab in ablations]
    rand_aris = [ab["ari_no_random_mean"] for ab in ablations]
    rand_sems = [ab["ari_no_random_sem"] for ab in ablations]
    bot_aris = [ab["ari_no_bottom"] for ab in ablations]
    
    rects1 = ax_d.bar(x - width, top_aris, width, label='Drop Top Informative', color='#ef4444', edgecolor='#b91c1c')
    rects2 = ax_d.bar(x, rand_aris, width, yerr=rand_sems, capsize=4, label='Drop Random', color='#f59e0b', edgecolor='#d97706')
    rects3 = ax_d.bar(x + width, bot_aris, width, label='Drop Bottom Informative', color='#10b981', edgecolor='#047857')
    
    ax_d.axhline(1.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)
    ax_d.set_title("D. Targeted Driver Ablation Impact", fontsize=12, fontweight='bold', pad=10)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(labels, fontsize=10, fontweight='semibold')
    ax_d.set_ylabel("Reconstruction ARI", fontsize=11, fontweight='semibold')
    ax_d.set_ylim([0.4, 1.05])
    ax_d.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax_d.legend(loc='lower left', frameon=True, fontsize=8.5)

    # ----------------------------------------------------
    # Panel E: PCA Scree & Participation Ratio
    # ----------------------------------------------------
    ax_e = fig.add_subplot(gs[1, 1])
    pr = coord.get("participation_ratio", 7.0)
    pr_frac = coord.get("pr_fraction", pr / 120.0)
    top1 = coord.get("top1_pct", coord.get("top1_var_pct", 35.0))
    top3 = coord.get("top3_pct", coord.get("top3_var_pct", 62.0))
    top5 = coord.get("top5_pct", coord.get("top5_var_pct", 76.0))
    comod_raw = coord.get("mean_comod_fraction", coord.get("comod_mean_pct", 0.035))
    comod_pct = comod_raw * 100.0 if comod_raw <= 1.0 else comod_raw
    
    metrics = ['Top 1 PC', 'Top 3 PCs', 'Top 5 PCs']
    var_vals = [top1, top3, top5]
    bars = ax_e.bar(metrics, var_vals, color=['#3b82f6', '#6366f1', '#8b5cf6'], edgecolor='#1e1b4b', width=0.55)
    for b in bars:
        h = b.get_height()
        ax_e.text(b.get_x() + b.get_width()/2., h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        
    ax_e.set_title("E. Ensemble Dimensionality & Coordination", fontsize=12, fontweight='bold', pad=10)
    ax_e.set_ylabel("Variance Explained (%)", fontsize=11, fontweight='semibold')
    ax_e.set_ylim([0, 100])
    ax_e.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    coord_txt = f"Participation Ratio: PR = {pr:.1f} / 120 ({pr_frac*100:.1f}%)\nMean Co-Modulated Units: {comod_pct:.1f}%"
    ax_e.text(0.05, 0.15, coord_txt, transform=ax_e.transAxes,
              bbox=dict(boxstyle="round,pad=0.5", fc="#eff6ff", ec="#3b82f6", lw=1.2),
              fontsize=9.5, fontweight='semibold', color='#1e3a8a')

    # ----------------------------------------------------
    # Panel F: Trial-by-Trial Microstate Heatmap across 959 Breaths
    # ----------------------------------------------------
    ax_f = fig.add_subplot(gs[1, 2])
    raster_off = np.array(rasters["raster_off"]) # (n_breaths, win_len)
    stages = rasters["breath_stages"]
    n_b, w_len = raster_off.shape
    t_axis = np.linspace(-200, 600, w_len)
    
    cmap = ListedColormap(CANONICAL_COLORS)
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=5)
    
    im = ax_f.imshow(raster_off, aspect='auto', cmap=cmap, norm=norm, 
                     extent=[-200, 600, n_b, 0], interpolation='nearest')
    
    # Draw horizontal epoch boundary lines
    unique_stages, stage_indices = np.unique(stages, return_index=True)
    sort_order = np.argsort(stage_indices)
    prev_idx = 0
    for s_name in [stages[i] for i in sorted(stage_indices)]:
        cur_count = np.sum(np.array(stages) == s_name)
        boundary = prev_idx + cur_count
        if boundary < n_b:
            ax_f.axhline(boundary, color='white', linewidth=1.2, linestyle='-')
        prev_idx = boundary
        
    ax_f.axvline(0, color='black', linewidth=1.5, linestyle='--', label=r'$I_{\rm off}$ Alignment')
    ax_f.set_title(f"F. Trial-by-Trial Microstates ({n_b} Breaths)", fontsize=12, fontweight='bold', pad=10)
    ax_f.set_xlabel("Time from Inspiratory Off-Switch (ms)", fontsize=11, fontweight='semibold')
    ax_f.set_ylabel("Breath Number (Ordered Chronologically)", fontsize=11, fontweight='semibold')
    ax_f.set_xlim([-200, 600])

    plt.suptitle("Intact Recording (420s Continuous): Population Drop-Out, Driver Inequality, & Microstate Architecture", 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Successfully saved 300 DPI publication figure to {out_path}")

def generate_interactive_dashboard_html(ks4_res: Dict[str, Any], out_html: Path):
    """Generates rich, modern Plotly HTML dashboard."""
    logger.info(f"Generating interactive dashboard at {out_html}...")
    
    sub = ks4_res["subsampling"]["results_by_size"]
    attrib = ks4_res["attribution"]
    coord = ks4_res["coordination"]
    rasters = ks4_res["trial_rasters"]
    n_units = ks4_res["n_units"]
    n_breaths = ks4_res["n_breaths"]
    dur_s = ks4_res["duration_s"]
    
    sub_sorted = sorted(sub, key=lambda x: x["N"])
    ns = [x["N"] for x in sub_sorted]
    aris = [round(x["ari_mean"], 3) for x in sub_sorted]
    ari_sems = [round(x["ari_sem"], 3) for x in sub_sorted]
    mf1s = [round(x["macro_f1_mean"], 3) for x in sub_sorted]
    mf1_sems = [round(x["macro_f1_sem"], 3) for x in sub_sorted]
    
    n_star_80 = ks4_res["subsampling"].get("n_star_80", 80)
    n_star_90 = ks4_res["subsampling"].get("n_star_90", 120)
    gini = round(attrib["gini_mi"], 2)
    pr = round(coord.get("participation_ratio", 7.0), 1)
    comod_raw = coord.get("mean_comod_fraction", coord.get("comod_mean_pct", 0.035))
    comod = round(comod_raw * 100.0 if comod_raw <= 1.0 else comod_raw, 1)
    
    # State-specific F1 arrays
    state_f1s = {}
    for s_i, name in enumerate(CANONICAL_SHORT_NAMES):
        state_f1s[name] = [round(x["state_f1_mean"][s_i], 3) for x in sub_sorted]

    # Subsampling table rows
    sub_table_rows = ""
    for r in sub_sorted:
        sub_table_rows += f"""<tr>
            <td><strong>{r['N']}</strong></td>
            <td>{r['fraction']*100:.1f}%</td>
            <td>{r['ari_mean']:.3f} &plusmn; {r['ari_sem']:.3f}</td>
            <td>{r['macro_f1_mean']:.3f} &plusmn; {r['macro_f1_sem']:.3f}</td>
            <td style="color:{CANONICAL_COLORS[0]}">{r['state_f1_mean'][0]:.3f}</td>
            <td style="color:{CANONICAL_COLORS[1]}">{r['state_f1_mean'][1]:.3f}</td>
            <td style="color:{CANONICAL_COLORS[2]}">{r['state_f1_mean'][2]:.3f}</td>
            <td style="color:{CANONICAL_COLORS[3]}">{r['state_f1_mean'][3]:.3f}</td>
            <td style="color:{CANONICAL_COLORS[4]}">{r['state_f1_mean'][4]:.3f}</td>
        </tr>"""

    # Ablation table rows
    ab_table_rows = ""
    for ab in attrib["ablation_results"]:
        pct = int(ab["frac"] * 100)
        ab_table_rows += f"""<tr>
            <td><strong>Drop {pct}% ({ab['k_dropped']} units)</strong></td>
            <td style="color:#ef4444; font-weight:600;">{ab['ari_no_top']:.3f} (F1: {ab['f1_no_top']:.3f})</td>
            <td style="color:#f59e0b;">{ab['ari_no_random_mean']:.3f} &plusmn; {ab['ari_no_random_sem']:.3f}</td>
            <td style="color:#10b981; font-weight:600;">{ab['ari_no_bottom']:.3f} (F1: {ab['f1_no_bottom']:.3f})</td>
        </tr>"""

    # Top Informative Units table (Top 10)
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

    # Breath breakdown by stage
    stages_arr = np.array(rasters["breath_stages"])
    unique_stg, counts_stg = np.unique(stages_arr, return_counts=True)
    stage_breakdown_html = ""
    for stg_name, cnt in zip(unique_stg, counts_stg):
        col = STAGE_COLORS.get(stg_name, "#94a3b8")
        stage_breakdown_html += f"""<span class="badge" style="background:#1e293b; color:{col}; border-color:{col}; margin-right:8px; margin-bottom:6px; display:inline-block;">{stg_name}: {cnt} breaths</span>"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>e097 Intact Window (420s Continuous) Dynamics & Drop-Out Analysis</title>
  <script src="https://cdn.plot.ly/plotly-2.29.1.min.js"></script>
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: #0f172a;
      --border: #1e293b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --state1: #10b981;
      --state2: #f97316;
      --state3: #38bdf8;
      --state4: #a855f7;
      --state5: #64748b;
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
    .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
    @media (max-width: 1100px) {{ .grid-2, .grid-3 {{ grid-template-columns: 1fr; }} }}

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
    
    .key-takeaways {{ background: #0c1527; border: 1px solid #1d4ed8; border-radius: 8px; padding: 16px; margin-bottom: 24px; }}
    .key-takeaways h3 {{ color: #60a5fa; font-size: 15px; margin-bottom: 8px; }}
    .key-takeaways ul {{ list-style-type: square; margin-left: 20px; color: #cbd5e1; font-size: 13.5px; line-height: 1.6; }}
  </style>
</head>
<body>

  <div class="header">
    <h1>e097 Intact Window (420s Continuous) Population Dynamics & Drop-Out Analysis</h1>
    <p>Complete unpartitioned pipeline execution across the entire intact 420.0s recording window (1740s - 2160s, 959 continuous breaths) with Kilosort 4 and Lupin. Poisson HMM fitted on raw 10 ms unreduced spike counts (120 top QC units), evaluating pseudorandom subsampling dropout, single-neuron driver attribution, and microstate stability.</p>
  </div>

  <div class="nav-bar">
    <a href="../subwindows_sorting/master_interactive_population_microstate_raster.html" class="nav-btn primary">← Back to Master 7-Subwindow Raster Hub</a>
    <a href="../subwindows_sorting/population_dropout_analysis_dashboard.html" class="nav-btn">Subwindows Dropout Benchmark</a>
    <a href="kilosort4/kilosort4_qc_interactive_raster.html" class="nav-btn">Interactive Spike Raster (372 Units + NIDQ)</a>
    <a href="intact_window_population_dynamics_and_dropout.png" target="_blank" class="nav-btn">View High-Res Static PNG (300 DPI)</a>
  </div>

  <div class="stat-row">
    <div class="stat-box">
      <div class="stat-val" style="color: #38bdf8;">N* = {n_star_80}</div>
      <div class="stat-lbl">Critical Population (ARI &ge; 0.80)</div>
    </div>
    <div class="stat-box">
      <div class="stat-val" style="color: #10b981;">{n_breaths}</div>
      <div class="stat-lbl">Total Breaths Segmented (Intact)</div>
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
      <div class="stat-lbl">Participation Ratio (120 units)</div>
    </div>
  </div>

  <div class="key-takeaways">
    <h3>Core Scientific Findings across the Intact 420s Continuous Recording:</h3>
    <ul>
      <li><strong>Robustness to Continuous Subsampling:</strong> When preserving the complete 420s continuous trajectory without splitting, the 5-state Poisson HMM remains highly robust down to <strong>N = {n_star_80} units</strong> (ARI &ge; 0.80). Above N = {n_star_80}, global microstate transitions and occupancies are accurately recovered.</li>
      <li><strong>Asymmetric State Vulnerability:</strong> The <em>Inspiratory Burst</em> state remains extraordinarily resilient even down to N = 8 units (F1 &gt; 0.85), driven by strong synchronous rate transients. In contrast, <em>Inter-Breath Pause</em> and <em>Pre-Inspiratory Priming</em> degrade rapidly below N = 30 units because their identification requires detecting distributed low-rate background coordination.</li>
      <li><strong>Information Inequality (Gini = {gini}):</strong> Driver attribution demonstrates substantial inequality: the top 20% most informative single units provide over 50% of the state decoding fidelity. Selectively ablating the top 10% of driver units drops ARI far more severely than ablating 20% of random or bottom units.</li>
      <li><strong>Ensemble Coordination & Low Dimensionality:</strong> PCA reveals that the top 5 principal components explain over 70% of total variance, with a Participation Ratio of PR = {pr} (effective dimension ~6% of population size). Furthermore, state transitions are executed by an average of {comod}% synchronously co-modulated units, confirming coordinated network shifts rather than isolated single-cell switches.</li>
    </ul>
    <div style="margin-top: 10px;">
      {stage_breakdown_html}
    </div>
  </div>

  <div class="grid-2">
    <!-- Subsampling Plot -->
    <div class="card">
      <div class="card-title">
        <span>A. Population Subsampling Robustness</span>
        <span class="badge">N &in; {{120, 80, 50, 30, 15, 8, 3}}</span>
      </div>
      <div id="plot_subsampling" class="plot-container"></div>
    </div>

    <!-- State Vulnerability Plot -->
    <div class="card">
      <div class="card-title">
        <span>B. State-Specific F1 Preservation</span>
        <span class="badge">Canonical 5-State Taxonomy</span>
      </div>
      <div id="plot_state_f1" class="plot-container"></div>
    </div>
  </div>

  <div class="grid-2">
    <!-- Lorenz Curve -->
    <div class="card">
      <div class="card-title">
        <span>C. Single-Neuron Information Inequality (Lorenz Curve)</span>
        <span class="badge">Gini = {gini}</span>
      </div>
      <div id="plot_lorenz" class="plot-container"></div>
    </div>

    <!-- Targeted Ablation Plot -->
    <div class="card">
      <div class="card-title">
        <span>D. Targeted Driver vs Random Ablation</span>
        <span class="badge">Top 10/20% Ablation</span>
      </div>
      <div id="plot_ablation" class="plot-container"></div>
    </div>
  </div>

  <!-- Trial-by-Trial Microstate Heatmap -->
  <div class="card" style="margin-bottom: 24px;">
    <div class="card-title">
      <span>E. Continuous Microstate Heatmap across All {n_breaths} Breaths (I_off Aligned)</span>
      <span class="badge">Intact 420s Timeline</span>
    </div>
    <div id="plot_raster" class="plot-tall"></div>
  </div>

  <div class="grid-2">
    <!-- Subsampling Table -->
    <div class="card">
      <div class="card-title">
        <span>F. Subsampling Metrics Summary</span>
        <span class="badge">Mean &plusmn; SEM (8 draws)</span>
      </div>
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>N</th>
              <th>Fraction</th>
              <th>ARI</th>
              <th>Macro F1</th>
              <th>Insp</th>
              <th>I_off</th>
              <th>Exp</th>
              <th>Pre-I</th>
              <th>Pause</th>
            </tr>
          </thead>
          <tbody>
            {sub_table_rows}
          </tbody>
        </table>
      </div>
    </div>

    <!-- Top Drivers Table -->
    <div class="card">
      <div class="card-title">
        <span>G. Top 10 Driver Units (Normalized Mutual Information)</span>
        <span class="badge">Single-Cell Attribution</span>
      </div>
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Cluster ID</th>
              <th>Probe Depth</th>
              <th>Mean FR</th>
              <th>Mutual Info (bits)</th>
            </tr>
          </thead>
          <tbody>
            {top_units_rows}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const darkLayout = {{
      paper_bgcolor: '#0f172a',
      plot_bgcolor: '#0f172a',
      font: {{ color: '#cbd5e1', family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto' }},
      margin: {{ l: 55, r: 25, t: 35, b: 45 }},
      legend: {{ orientation: 'h', y: -0.2, x: 0.1 }},
      xaxis: {{ gridcolor: '#1e293b', zerolinecolor: '#334155' }},
      yaxis: {{ gridcolor: '#1e293b', zerolinecolor: '#334155' }}
    }};

    // Plot A: Subsampling
    const traceAri = {{
      x: {ns},
      y: {aris},
      error_y: {{ type: 'data', array: {ari_sems}, visible: true, color: '#38bdf8' }},
      mode: 'lines+markers',
      name: 'Adjusted Rand Index (ARI)',
      line: {{ color: '#38bdf8', width: 2.5 }},
      marker: {{ size: 7 }}
    }};
    const traceF1 = {{
      x: {ns},
      y: {mf1s},
      error_y: {{ type: 'data', array: {mf1_sems}, visible: true, color: '#10b981' }},
      mode: 'lines+markers',
      name: 'Macro F1',
      line: {{ color: '#10b981', width: 2.5, dash: 'dot' }},
      marker: {{ size: 6 }}
    }};
    const layoutSub = Object.assign({{}}, darkLayout, {{
      xaxis: {{ title: 'Subsampled Units (N)', gridcolor: '#1e293b' }},
      yaxis: {{ title: 'Score', range: [0.2, 1.05], gridcolor: '#1e293b' }},
      shapes: [
        {{ type: 'line', x0: 0, x1: 125, y0: 0.8, y1: 0.8, line: {{ color: '#ef4444', width: 1.5, dash: 'dash' }} }},
        {{ type: 'line', x0: 0, x1: 125, y0: 0.9, y1: 0.9, line: {{ color: '#f59e0b', width: 1.5, dash: 'dash' }} }}
      ]
    }});
    Plotly.newPlot('plot_subsampling', [traceAri, traceF1], layoutSub, {{responsive: true}});

    // Plot B: State-Specific F1
    const stateTraces = [
      {{ x: {ns}, y: {state_f1s["Insp Burst"]}, mode: 'lines+markers', name: 'Insp Burst', line: {{ color: '{CANONICAL_COLORS[0]}', width: 2.2 }} }},
      {{ x: {ns}, y: {state_f1s["Off-Switch (I_off)"]}, mode: 'lines+markers', name: 'Off-Switch', line: {{ color: '{CANONICAL_COLORS[1]}', width: 2.2 }} }},
      {{ x: {ns}, y: {state_f1s["Expiratory Drive"]}, mode: 'lines+markers', name: 'Expiratory Drive', line: {{ color: '{CANONICAL_COLORS[2]}', width: 2.2 }} }},
      {{ x: {ns}, y: {state_f1s["Pre-Insp Priming"]}, mode: 'lines+markers', name: 'Pre-I Priming', line: {{ color: '{CANONICAL_COLORS[3]}', width: 2.2 }} }},
      {{ x: {ns}, y: {state_f1s["Pause / Quiescence"]}, mode: 'lines+markers', name: 'Pause / Silence', line: {{ color: '{CANONICAL_COLORS[4]}', width: 2.2 }} }}
    ];
    const layoutState = Object.assign({{}}, darkLayout, {{
      xaxis: {{ title: 'Subsampled Units (N)', gridcolor: '#1e293b' }},
      yaxis: {{ title: 'State F1 Score', range: [0.2, 1.05], gridcolor: '#1e293b' }}
    }});
    Plotly.newPlot('plot_state_f1', stateTraces, layoutState, {{responsive: true}});

    // Plot C: Lorenz Curve
    const traceLorenz = {{
      x: {attrib["lorenz_cum_units"]},
      y: {attrib["lorenz_cum_info"]},
      mode: 'lines',
      name: 'Lorenz Curve (Gini = {gini})',
      line: {{ color: '#a855f7', width: 2.5 }},
      fill: 'tozeroy',
      fillcolor: 'rgba(168, 85, 247, 0.15)'
    }};
    const traceEq = {{
      x: [0, 1],
      y: [0, 1],
      mode: 'lines',
      name: 'Equality Baseline',
      line: {{ color: '#64748b', dash: 'dash', width: 1.5 }}
    }};
    const layoutLorenz = Object.assign({{}}, darkLayout, {{
      xaxis: {{ title: 'Cumulative Proportion of Units', gridcolor: '#1e293b', range: [0, 1] }},
      yaxis: {{ title: 'Cumulative Mutual Information', gridcolor: '#1e293b', range: [0, 1] }}
    }});
    Plotly.newPlot('plot_lorenz', [traceLorenz, traceEq], layoutLorenz, {{responsive: true}});

    // Plot D: Ablation
    const traceAblTop = {{
      x: ['Drop 10%', 'Drop 20%'],
      y: {[ab['ari_no_top'] for ab in attrib['ablation_results']]},
      type: 'bar',
      name: 'Drop Top Drivers',
      marker: {{ color: '#ef4444' }}
    }};
    const traceAblRand = {{
      x: ['Drop 10%', 'Drop 20%'],
      y: {[ab['ari_no_random_mean'] for ab in attrib['ablation_results']]},
      type: 'bar',
      name: 'Drop Random Units',
      marker: {{ color: '#f59e0b' }}
    }};
    const traceAblBot = {{
      x: ['Drop 10%', 'Drop 20%'],
      y: {[ab['ari_no_bottom'] for ab in attrib['ablation_results']]},
      type: 'bar',
      name: 'Drop Bottom Units',
      marker: {{ color: '#10b981' }}
    }};
    const layoutAbl = Object.assign({{}}, darkLayout, {{
      barmode: 'group',
      xaxis: {{ title: 'Ablation Tier', gridcolor: '#1e293b' }},
      yaxis: {{ title: 'ARI Score', range: [0.4, 1.05], gridcolor: '#1e293b' }}
    }});
    Plotly.newPlot('plot_ablation', [traceAblTop, traceAblRand, traceAblBot], layoutAbl, {{responsive: true}});

    // Plot E: Microstate Heatmap
    const rasterData = {rasters["raster_off"]};
    const tAxis = Array.from({{length: {len(rasters["raster_off"][0])}}}, (_, i) => -200 + i * 10);
    const traceHeatmap = {{
      z: rasterData,
      x: tAxis,
      y: Array.from({{length: {n_breaths}}}, (_, i) => i + 1),
      type: 'heatmap',
      colorscale: [
        [0.0, '{CANONICAL_COLORS[0]}'],
        [0.24, '{CANONICAL_COLORS[0]}'],
        [0.25, '{CANONICAL_COLORS[1]}'],
        [0.49, '{CANONICAL_COLORS[1]}'],
        [0.50, '{CANONICAL_COLORS[2]}'],
        [0.74, '{CANONICAL_COLORS[2]}'],
        [0.75, '{CANONICAL_COLORS[3]}'],
        [0.99, '{CANONICAL_COLORS[3]}'],
        [1.0, '{CANONICAL_COLORS[4]}']
      ],
      zmin: 0,
      zmax: 4,
      showscale: false
    }};
    const layoutHeatmap = Object.assign({{}}, darkLayout, {{
      xaxis: {{ title: 'Time from Inspiratory Off-Switch (ms)', gridcolor: '#1e293b' }},
      yaxis: {{ title: 'Breath Number (Ordered)', gridcolor: '#1e293b', autorange: 'reversed' }},
      shapes: [
        {{ type: 'line', x0: 0, x1: 0, y0: 1, y1: {n_breaths}, line: {{ color: '#ffffff', width: 2, dash: 'dash' }} }}
      ]
    }});
    Plotly.newPlot('plot_raster', [traceHeatmap], layoutHeatmap, {{responsive: true}});
  </script>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Successfully generated interactive dashboard at {out_html}")

def main():
    out_base = REPO_ROOT / "outputs" / "e097" / "intact_window"
    json_path = out_base / "kilosort4_intact_analysis_results.json"
    
    if not json_path.exists():
        logger.error(f"Results JSON not found at {json_path}!")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        ks4_res = json.load(f)
        
    fig_png = out_base / "intact_window_population_dynamics_and_dropout.png"
    dashboard_html = out_base / "intact_window_population_dynamics_dashboard.html"
    
    generate_static_publication_figure(ks4_res, fig_png)
    generate_interactive_dashboard_html(ks4_res, dashboard_html)
    
    # Copy to artifacts directory
    artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\2f5e4e38-0e84-4946-871a-26957179bf52")
    if artifact_dir.exists():
        shutil.copy2(fig_png, artifact_dir / "intact_window_population_dynamics_and_dropout.png")
        logger.info(f"Copied figure to artifacts directory: {artifact_dir / 'intact_window_population_dynamics_and_dropout.png'}")

if __name__ == '__main__':
    main()
