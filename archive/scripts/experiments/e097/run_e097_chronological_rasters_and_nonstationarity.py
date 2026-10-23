#!/usr/bin/env python3
"""
Comprehensive Chronological Breath-Aligned Rasters, Multi-Epoch Nonstationarity Metrics,
and Split-Cluster Discovery Pipeline for e097 (226s Recording Window).

Features:
1. Dual Dataset Evaluation:
   - Motion-Corrected Consensus Dataset
   - Uncorrected Baseline Consensus Dataset (Ground-Truth Benchmark)
2. Four Physiological Epochs:
   - Epoch 1 (Pre-Laser Baseline): 0.0s - 60.0s
   - Epoch 2 (Inter-Train 1): 62.0s - 119.7s
   - Epoch 3 (Inter-Train 2): 122.0s - 164.3s
   - Epoch 4 (Post-Laser 3 Recovery): 166.5s - 226.24s
3. Three Literature-Grounded Trial Similarity Methods:
   - Schreiber Spike Timing Reliability (sigma = 10 ms, sigma = 25 ms, circular phase)
   - van Rossum Metric Distance (tau = 10 ms, 25 ms)
   - Signal vs. Noise Correlations (PETH profile match vs trial count covariation)
4. Nonstationarity & Automated Split-Candidate Discovery Engine.
5. Publication-Quality Chronological Rasters & Interactive HTML Gallery.
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root and apply compatibility patch
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import src.pipeline.compat_patch
from src.analysis.epoch_spiking_statistics import (
    DEFAULT_EPOCHS,
    calculate_cluster_multi_epoch_profile
)
from src.analysis.nonstationarity_and_split_detector import (
    evaluate_cross_cluster_split_candidates
)
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

def compute_continuous_respiratory_phase_grid(valid_onsets, valid_offsets, time_axis_s):
    """Precomputes continuous respiratory phase phi(t) in [-pi, pi] for chronological breath trials."""
    n_breaths = len(valid_onsets)
    n_bins = len(time_axis_s)
    phase_grid = np.zeros((n_breaths, n_bins), dtype=np.float32)

    for i in range(n_breaths):
        on = valid_onsets[i]
        off = valid_offsets[i]
        next_on = valid_onsets[i + 1] if i + 1 < n_breaths else on + (off - on) * 3.0
        prev_off = valid_offsets[i - 1] if i > 0 else on - (off - on) * 2.0
        t_eval = on + time_axis_s

        phi = np.full(n_bins, np.nan, dtype=np.float32)

        insp_m = (t_eval >= on) & (t_eval < off)
        if np.any(insp_m):
            phi[insp_m] = (t_eval[insp_m] - on) / (off - on + 1e-6) * np.pi

        exp_m = (t_eval >= off) & (t_eval < next_on)
        if np.any(exp_m):
            phi[exp_m] = -np.pi + (t_eval[exp_m] - off) / (next_on - off + 1e-6) * np.pi

        pre_m = t_eval < on
        if np.any(pre_m):
            phi[pre_m] = -np.pi + (t_eval[pre_m] - prev_off) / (on - prev_off + 1e-6) * np.pi

        nan_m = np.isnan(phi)
        phi[nan_m] = -np.pi
        phase_grid[i, :] = phi

    return phase_grid

def plot_single_neuron_chronological_raster_peth(
    cid, depth, fr_hz, archetype_label,
    u_spks, valid_onsets, valid_offsets, ti_arr,
    phase_grid_chrono, emg_trials_chrono,
    laser_train_bands,
    time_axis_s, time_axis_ms, dt_s,
    kernel, out_file,
    metrics_summary=None
):
    n_breaths = len(valid_onsets)
    n_bins = len(time_axis_s)
    win_pre_s = -time_axis_s[0]
    win_post_s = time_axis_s[-1]

    spike_scatter_x = []
    spike_scatter_y = []
    rate_trials = np.zeros((n_breaths, n_bins), dtype=np.float32)

    for row_idx in range(n_breaths):
        on = valid_onsets[row_idx]
        rel_spks = u_spks - on
        in_win = rel_spks[(rel_spks >= -win_pre_s) & (rel_spks <= win_post_s)]
        for s in in_win:
            spike_scatter_x.append(s * 1000.0)
            spike_scatter_y.append(row_idx + 1)

        counts, _ = np.histogram(in_win, bins=np.append(time_axis_s, time_axis_s[-1] + dt_s))
        smooth_fr = np.convolve(counts / dt_s, kernel, mode="same")
        rate_trials[row_idx, :] = smooth_fr

    pre_indices = np.arange(0, min(198, n_breaths))
    mean_fr_pre = np.mean(rate_trials[pre_indices, :], axis=0)
    std_fr_pre = np.std(rate_trials[pre_indices, :], axis=0)

    post_start_idx = min(590, n_breaths - 1)
    post_indices = np.arange(post_start_idx, n_breaths)
    mean_fr_post = np.mean(rate_trials[post_indices, :], axis=0)
    std_fr_post = np.std(rate_trials[post_indices, :], axis=0)

    mean_emg = np.mean(emg_trials_chrono[pre_indices, :], axis=0)
    emg_min, emg_max = np.min(mean_emg), np.max(mean_emg)
    norm_emg = (mean_emg - emg_min) / (emg_max - emg_min + 1e-12) if emg_max > emg_min else np.zeros_like(mean_emg)

    mean_ti_ms = np.mean(ti_arr[:198]) * 1000.0

    fig = plt.figure(figsize=(11.0, 8.8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.4, 1.0], hspace=0.20)

    cmap_phase = LinearSegmentedColormap.from_list(
        "phase_cmap_onset",
        ["#93c5fd", "#dbeafe", "#ffffff", "#fed7aa", "#f87171"],
        N=256
    )

    # Top Panel: Chronological Raster
    ax_raster = fig.add_subplot(gs[0])
    im_phase = ax_raster.imshow(
        phase_grid_chrono,
        aspect="auto",
        cmap=cmap_phase,
        vmin=-np.pi,
        vmax=np.pi,
        extent=[time_axis_ms[0], time_axis_ms[-1], 0.5, n_breaths + 0.5],
        origin="lower",
        alpha=0.60,
        interpolation="bilinear"
    )

    for tb in laser_train_bands:
        b_start = tb["start_row"]
        b_end = tb["end_row"]
        lbl = tb["label"]
        ax_raster.fill_between(
            [time_axis_ms[0], time_axis_ms[-1]],
            [b_start - 0.5, b_start - 0.5],
            [b_end + 0.5, b_end + 0.5],
            color='#ff922b', alpha=0.35, zorder=2
        )
        ax_raster.text(
            time_axis_ms[-1] - 8, (b_start + b_end) / 2.0,
            f"⚡ {lbl}",
            color="#d9480f", fontsize=8.5, fontweight="bold",
            va="center", ha="right", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff4e6", alpha=0.90, edgecolor="#ff922b")
        )

    if len(spike_scatter_x) > 0:
        ax_raster.scatter(
            spike_scatter_x,
            spike_scatter_y,
            marker='|',
            color='black',
            s=16,
            linewidth=0.85,
            alpha=0.92,
            zorder=3,
            label="Spike Times"
        )

    ioff_curve_x = ti_arr * 1000.0
    ioff_curve_y = np.arange(1, n_breaths + 1)
    ax_raster.plot(
        ioff_curve_x,
        ioff_curve_y,
        color="#b91c1c",
        linestyle="",
        marker=".",
        markersize=2.0,
        alpha=0.75,
        zorder=4,
        label=r"Inspiration Offset ($I_{\mathrm{off}}$)"
    )

    ax_raster.axvline(0, color="#1e293b", linestyle="--", linewidth=1.6, alpha=0.85, zorder=4)
    ax_raster.set_xlim(time_axis_ms[0], time_axis_ms[-1])
    ax_raster.set_ylim(0.5, n_breaths + 0.5)
    ax_raster.set_ylabel(f"Breath Trials (Chronological Order: 1 → {n_breaths})", fontsize=10.5, fontweight="bold")

    ax_time = ax_raster.twinx()
    ax_time.set_ylim(valid_onsets[0], valid_onsets[-1])
    ax_time.set_ylabel("Continuous Recording Time (s)", fontsize=10, fontweight="bold", color="#475569")
    ax_time.tick_params(axis='y', labelcolor="#475569")

    title_str = f"Cluster #{cid} ({depth:.0f} µm | FR: {fr_hz:.1f} Hz)"
    if archetype_label:
        title_str += f" — {archetype_label}"
    title_str += " — Chronological Breath-Onset Aligned Raster\n"
    if metrics_summary:
        title_str += f"[{metrics_summary}]"
    else:
        title_str += r"[Breaths 1 → 842 (0s → 226s) | Orange Bands = Laser Trains 1, 2, 3 | Red Dots = $I_{\mathrm{off}}$]"
    ax_raster.set_title(title_str, fontsize=10.5, fontweight="bold")
    ax_raster.grid(False)

    cbar = plt.colorbar(im_phase, ax=ax_raster, fraction=0.025, pad=0.09)
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$+\frac{\pi}{2}$", r"$+\pi$"], fontsize=7.5)
    ax_raster.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=0.85, fontsize=8)

    # Bottom Panel: Dual-Condition PETH
    ax_peth = fig.add_subplot(gs[1], sharex=ax_raster)
    p1 = ax_peth.plot(time_axis_ms, mean_fr_pre, color="#0284c7", linewidth=2.2, label="Pre-Laser Eupnea (Ep 1: Breaths 1-198)")
    ax_peth.fill_between(
        time_axis_ms,
        np.maximum(0, mean_fr_pre - std_fr_pre),
        mean_fr_pre + std_fr_pre,
        color="#0284c7", alpha=0.18
    )

    p2 = ax_peth.plot(time_axis_ms, mean_fr_post, color="#e11d48", linewidth=2.2, linestyle="-", label="Post-Laser Recovery (Ep 4: Breaths 591-842)")
    ax_peth.fill_between(
        time_axis_ms,
        np.maximum(0, mean_fr_post - std_fr_post),
        mean_fr_post + std_fr_post,
        color="#e11d48", alpha=0.18
    )

    ax_emg = ax_peth.twinx()
    p3 = ax_emg.plot(time_axis_ms, norm_emg, color="#64748b", linestyle=":", linewidth=1.8, label="Diaphragm EMG Envelope (Norm)")
    ax_emg.set_ylabel("Norm EMG Envelope", fontsize=9, color="#64748b")
    ax_emg.set_ylim(-0.05, 1.25)
    ax_emg.tick_params(axis='y', labelcolor="#64748b")

    ax_peth.axvline(0, color="#1e293b", linestyle="--", linewidth=1.6, alpha=0.85)
    ax_peth.axvline(mean_ti_ms, color="#b91c1c", linestyle=":", linewidth=1.6, alpha=0.85)

    max_peth = max(float(np.max(mean_fr_pre)), float(np.max(mean_fr_post)), 5.0)
    ax_peth.set_ylim(0, max_peth * 1.28)
    ax_peth.set_ylabel("Firing Rate (Hz)", fontsize=10, fontweight="bold")
    ax_peth.set_xlabel("Time Relative to Inspiration Onset (ms)", fontsize=10, fontweight="bold")
    ax_peth.grid(True, linestyle="--", alpha=0.35)

    lines = p1 + p2 + p3
    labels = [l.get_label() for l in lines]
    ax_peth.legend(lines, labels, loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=0.85, fontsize=8)

    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

def build_split_comparison_figure(
    cid_a, cid_b,
    spks_a, spks_b,
    depth_a, depth_b,
    valid_onsets, valid_offsets, ti_arr,
    phase_grid_chrono, emg_trials_chrono,
    laser_train_bands,
    time_axis_s, time_axis_ms, dt_s,
    kernel,
    pair_row,
    out_file
):
    """
    Builds an executive side-by-side diagnostic comparison figure for a high-probability split candidate pair.
    """
    n_breaths = len(valid_onsets)
    win_pre_s = -time_axis_s[0]
    win_post_s = time_axis_s[-1]

    fig = plt.figure(figsize=(16.0, 9.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2.4, 1.0], hspace=0.22, wspace=0.18)

    cmap_phase = LinearSegmentedColormap.from_list(
        "phase_cmap_onset",
        ["#93c5fd", "#dbeafe", "#ffffff", "#fed7aa", "#f87171"],
        N=256
    )

    for col_idx, (cid, u_spks, depth, ep_active, col_color) in enumerate([
        (cid_a, spks_a, depth_a, pair_row['active_epoch_a'], '#0284c7'),
        (cid_b, spks_b, depth_b, pair_row['active_epoch_b'], '#e11d48')
    ]):
        spike_scatter_x = []
        spike_scatter_y = []
        rate_trials = np.zeros((n_breaths, len(time_axis_s)), dtype=np.float32)

        for row_idx in range(n_breaths):
            on = valid_onsets[row_idx]
            rel_spks = u_spks - on
            in_win = rel_spks[(rel_spks >= -win_pre_s) & (rel_spks <= win_post_s)]
            for s in in_win:
                spike_scatter_x.append(s * 1000.0)
                spike_scatter_y.append(row_idx + 1)
            counts, _ = np.histogram(in_win, bins=np.append(time_axis_s, time_axis_s[-1] + dt_s))
            rate_trials[row_idx, :] = np.convolve(counts / dt_s, kernel, mode="same")

        ax_raster = fig.add_subplot(gs[0, col_idx])
        ax_raster.imshow(
            phase_grid_chrono,
            aspect="auto",
            cmap=cmap_phase,
            vmin=-np.pi, vmax=np.pi,
            extent=[time_axis_ms[0], time_axis_ms[-1], 0.5, n_breaths + 0.5],
            origin="lower", alpha=0.55, interpolation="bilinear"
        )
        for tb in laser_train_bands:
            ax_raster.fill_between(
                [time_axis_ms[0], time_axis_ms[-1]],
                [tb["start_row"] - 0.5, tb["start_row"] - 0.5],
                [tb["end_row"] + 0.5, tb["end_row"] + 0.5],
                color='#ff922b', alpha=0.35, zorder=2
            )

        if len(spike_scatter_x) > 0:
            ax_raster.scatter(spike_scatter_x, spike_scatter_y, marker='|', color='black', s=14, linewidth=0.8, alpha=0.9, zorder=3)

        ax_raster.axvline(0, color="#1e293b", linestyle="--", linewidth=1.5, alpha=0.85)
        ax_raster.set_xlim(time_axis_ms[0], time_axis_ms[-1])
        ax_raster.set_ylim(0.5, n_breaths + 0.5)
        ax_raster.set_ylabel(f"Breath Trials (1 → {n_breaths})", fontsize=10, fontweight="bold")
        ax_raster.set_title(f"Cluster #{cid} ({depth:.0f} µm) — Active Dominantly in Epoch {ep_active}", fontsize=11, fontweight="bold", color=col_color)
        ax_raster.grid(False)

        # Bottom PETH
        ax_peth = fig.add_subplot(gs[1, col_idx], sharex=ax_raster)
        peth_ep1 = np.mean(rate_trials[:198, :], axis=0)
        peth_ep4 = np.mean(rate_trials[591:, :], axis=0)

        ax_peth.plot(time_axis_ms, peth_ep1, color="#0284c7", linewidth=2.0, label="Epoch 1 (Pre-Laser)")
        ax_peth.plot(time_axis_ms, peth_ep4, color="#e11d48", linewidth=2.0, label="Epoch 4 (Post-Laser 3)")
        ax_peth.axvline(0, color="#1e293b", linestyle="--", linewidth=1.5, alpha=0.85)
        ax_peth.set_xlabel("Time Relative to Inspiration (ms)", fontsize=10, fontweight="bold")
        ax_peth.set_ylabel("Firing Rate (Hz)", fontsize=10, fontweight="bold")
        ax_peth.set_ylim(0, max(np.max(peth_ep1), np.max(peth_ep4), 5.0) * 1.3)
        ax_peth.grid(True, linestyle="--", alpha=0.35)
        ax_peth.legend(loc="upper right", fontsize=8)

    diag_text = (
        f"SPLIT DIAGNOSTIC METRICS: Pair #{cid_a} <--> #{cid_b} | Depth Separation: {pair_row['delta_y_um']:.1f} µm | "
        f"Temporal Complementarity: {pair_row['temporal_complementarity']:.3f} | Cross Schreiber Similarity: {pair_row['cross_cluster_schreiber']:.3f} | "
        f"Signal Corr: {pair_row['cross_cluster_signal_corr']:.3f} | Composite Split Score: {pair_row['composite_split_score']:.4f} ({pair_row['split_confidence']})"
    )
    fig.suptitle(diag_text, fontsize=11, fontweight="bold", color="#991b1b", y=0.99)
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

def build_gallery_html(index_cards, out_html, n_breaths, dataset_title):
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dataset_title} — Chronological Rasters & Spiking Statistics</title>
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: #111827;
            --card-border: #1f2937;
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --accent-blue: #38bdf8;
            --accent-orange: #fb923c;
            --badge-bg: #0369a1;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
        }}
        .header-container {{
            text-align: center;
            max-width: 1400px;
            margin: 0 auto 24px auto;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--card-border);
        }}
        h1 {{
            color: var(--accent-blue);
            font-size: 26px;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }}
        p.subtitle {{
            color: var(--text-secondary);
            font-size: 14px;
            line-height: 1.5;
            margin: 0;
        }}
        .stats-badge-container {{
            display: flex;
            justify-content: center;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 16px;
        }}
        .stat-pill {{
            background: #1e293b;
            border: 1px solid #334155;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            color: #cbd5e1;
        }}
        .stat-pill b {{
            color: var(--accent-blue);
        }}
        .stat-pill.laser b {{
            color: var(--accent-orange);
        }}
        .filter-container {{
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
            margin: 20px auto;
            max-width: 1400px;
        }}
        .filter-btn {{
            background: #1e293b;
            color: #cbd5e1;
            border: 1px solid #334155;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .filter-btn:hover, .filter-btn.active {{
            background: #38bdf8;
            color: #0b0f19;
            border-color: #38bdf8;
        }}
        .search-box {{
            background: #1e293b;
            color: #fff;
            border: 1px solid #334155;
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 13px;
            width: 240px;
            outline: none;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(540px, 1fr));
            gap: 24px;
            max-width: 1800px;
            margin: 0 auto;
        }}
        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            border: 1px solid var(--card-border);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
            border-color: #3b82f6;
        }}
        .card img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            display: block;
            background: #000;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        .unit-title {{
            font-weight: 700;
            font-size: 16px;
            color: #ffffff;
        }}
        .badge {{
            background: var(--badge-bg);
            color: #e0f2fe;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.3px;
        }}
        .badge-recruited {{
            background: #e11d48;
            color: #fff;
        }}
        .badge-qc {{
            background: #10b981;
            color: #fff;
        }}
        .metrics-row {{
            margin-top: 10px;
            font-size: 12px;
            color: #94a3b8;
            display: flex;
            justify-content: space-between;
            padding-top: 8px;
            border-top: 1px solid #1e293b;
        }}
    </style>
</head>
<body>
    <div class="header-container">
        <h1>{dataset_title} — Chronological Breath Rasters & Spiking Statistics</h1>
        <p class="subtitle">
            Breath-Onset Aligned Spike Rasters (I-on = 0 ms), ordered <b>chronologically from Breath #1 to #{n_breaths} (0s → 226s)</b>.<br>
            Demarcating <b>Laser Trains 1 (60s), 2 (120s), and 3 (164s)</b> with 4-epoch spiking statistics and trial-to-trial Schreiber reliability.
        </p>
        <div class="stats-badge-container">
            <div class="stat-pill"><b>{len(index_cards)}</b> Analyzed Clusters</div>
            <div class="stat-pill"><b>{n_breaths}</b> Chronological Breaths</div>
            <div class="stat-pill laser">⚡ <b>3 Laser Photostimulation Trains</b></div>
        </div>
    </div>

    <div class="filter-container">
        <button class="filter-btn active" onclick="filterCards('all', this)">All Units ({len(index_cards)})</button>
        <button class="filter-btn" onclick="filterCards('qc_passed', this)">⭐ Passed Allen QC</button>
        <button class="filter-btn" onclick="filterCards('botc', this)">BötC (&lt;750 µm)</button>
        <button class="filter-btn" onclick="filterCards('prebotc', this)">preBötC Core (750-1650 µm)</button>
        <button class="filter-btn" onclick="filterCards('caudal', this)">rVRG / cVRG (&gt;1650 µm)</button>
        <input type="text" id="search-input" class="search-box" placeholder="🔍 Search cluster ID or depth..." onkeyup="searchCards(this.value)">
    </div>

    <div class="grid" id="cards-grid">
"""
    for card in index_cards:
        b_class = "badge"
        if card.get("passed_qc", False):
            b_class += " badge-qc"
        elif card.get("is_recruited", False):
            b_class += " badge-recruited"

        cat = card["category"]
        m_summary = card.get("metrics_text", "")
        html += f"""
        <div class="card" data-category="{cat}" data-cid="{card['cid']}" data-depth="{card['depth_um']}">
            <div class="card-header">
                <span class="unit-title">{card['title_label']}</span>
                <span class="{b_class}">{card['depth_um']:.0f} µm | {card['firing_rate_hz']:.1f} Hz</span>
            </div>
            <a href="{card['file_name']}" target="_blank" title="Click to view high-resolution image">
                <img src="{card['file_name']}" alt="Cluster {card['cid']}" loading="lazy">
            </a>
            <div class="metrics-row">
                <span>{m_summary}</span>
            </div>
        </div>
"""
    html += """
    </div>

    <script>
        function filterCards(cat, btn) {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const cards = document.querySelectorAll('.card');
            cards.forEach(c => {
                if (cat === 'all') {
                    c.style.display = 'block';
                } else if (cat === 'qc_passed') {
                    c.style.display = (c.dataset.category === 'qc_passed') ? 'block' : 'none';
                } else if (cat === 'botc') {
                    c.style.display = (parseFloat(c.dataset.depth) < 750) ? 'block' : 'none';
                } else if (cat === 'prebotc') {
                    const d = parseFloat(c.dataset.depth);
                    c.style.display = (d >= 750 && d < 1650) ? 'block' : 'none';
                } else if (cat === 'caudal') {
                    c.style.display = (parseFloat(c.dataset.depth) >= 1650) ? 'block' : 'none';
                }
            });
        }

        function searchCards(val) {
            const query = val.toLowerCase().trim();
            const cards = document.querySelectorAll('.card');
            cards.forEach(c => {
                const text = c.textContent.toLowerCase();
                c.style.display = (text.includes(query)) ? 'block' : 'none';
            });
        }
    </script>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[+] Saved HTML Visual Gallery to: {out_html}")

def process_dataset(
    dataset_name: str,
    cons_dir: Path,
    valid_onsets: np.ndarray,
    valid_offsets: np.ndarray,
    ti_arr: np.ndarray,
    phase_grid_chrono: np.ndarray,
    emg_trials_chrono: np.ndarray,
    laser_train_bands: list,
    time_axis_s: np.ndarray,
    time_axis_ms: np.ndarray,
    dt_s: float,
    kernel: np.ndarray,
    max_rasters: int = 150
):
    print(f"\n{'='*75}\n[*] Processing Dataset: {dataset_name}\n  Directory: {cons_dir}\n{'='*75}")
    alf_dir = cons_dir / "alf"
    out_raster_dir = cons_dir / "02_single_cell_phase_rasters" / "onset_aligned_chronological"
    out_raster_dir.mkdir(parents=True, exist_ok=True)
    out_split_dir = cons_dir / "03_split_candidate_diagnostics"
    out_split_dir.mkdir(parents=True, exist_ok=True)

    spk_times = np.load(alf_dir / "spike_times.npy")
    spk_clusters = np.load(alf_dir / "spike_clusters.npy")
    
    # Load cluster metadata
    qc_file = cons_dir / "master_consensus_qc_summary.csv"
    if qc_file.exists():
        qc_df = pd.read_csv(qc_file)
    else:
        qc_df = pd.read_csv(cons_dir / "detailed_presence_ratio_comparison.csv")

    cids_all = sorted(list(np.unique(spk_clusters)))
    print(f"[+] Loaded {len(cids_all)} total clusters and {len(spk_times):,} total spikes.")

    # 1. Compute Multi-Epoch Profiles for All Clusters
    print("[*] Computing 4-Epoch Spiking Statistics and Trial Reliability for all clusters...")
    profiles = {}
    metadata = {}
    stat_rows = []

    for idx, cid in enumerate(cids_all):
        if idx % 50 == 0:
            print(f"    Processing cluster {idx}/{len(cids_all)} (CID {cid})...")
        c_mask = (spk_clusters == cid)
        c_spks = spk_times[c_mask]
        
        row_m = qc_df[qc_df["cluster_id"] == cid]
        if len(row_m) > 0:
            depth = float(row_m["depth_um"].values[0])
            fr = float(row_m["firing_rate_hz"].values[0])
            passed_qc = bool(row_m["passed_allen_qc"].values[0]) if "passed_allen_qc" in row_m.columns else True
        else:
            depth = 0.0
            fr = float(len(c_spks) / 226.243)
            passed_qc = False

        metadata[cid] = {"depth_um": depth, "firing_rate_hz": fr, "passed_qc": passed_qc}
        prof = calculate_cluster_multi_epoch_profile(
            unit_spikes=c_spks,
            all_onsets=valid_onsets,
            all_offsets=valid_offsets,
            epochs=DEFAULT_EPOCHS,
            time_axis_s=time_axis_s,
            dt_s=dt_s
        )
        profiles[cid] = prof

        for ep_id, ep_dict in prof["epoch_stats"].items():
            stat_rows.append({
                "cluster_id": cid,
                "depth_um": depth,
                "total_fr_hz": fr,
                "passed_qc": passed_qc,
                "epoch_id": ep_id,
                "epoch_name": ep_dict["epoch_name"],
                "n_spikes": ep_dict["n_spikes"],
                "n_breaths": ep_dict["n_breaths"],
                "mean_fr_hz": ep_dict["mean_fr_hz"],
                "peak_peth_fr_hz": ep_dict["peak_peth_fr_hz"],
                "modulation_depth": ep_dict["modulation_depth"],
                "preferred_phase_rad": ep_dict["preferred_phase_rad"],
                "vector_strength": ep_dict["vector_strength"],
                "archetype": ep_dict["archetype"],
                "fano_factor": ep_dict["fano_factor"],
                "cv_isi": ep_dict["cv_isi"],
                "burst_fraction": ep_dict["burst_fraction"],
                "active_breath_fraction": ep_dict["active_breath_fraction"],
                "schreiber_reliability_10ms": ep_dict["schreiber_reliability_10ms"],
                "schreiber_reliability_25ms": ep_dict["schreiber_reliability_25ms"],
                "schreiber_reliability_phase": ep_dict["schreiber_reliability_phase"],
                "van_rossum_dist_10ms": ep_dict["van_rossum_dist_10ms"],
                "van_rossum_dist_25ms": ep_dict["van_rossum_dist_25ms"],
                "within_epoch_signal_corr": ep_dict["within_epoch_signal_corr"],
                "within_epoch_noise_corr": ep_dict["within_epoch_noise_corr"],
                "stationarity_index": ep_dict["stationarity_index"],
                "drift_rho": ep_dict["drift_rho"],
                "rate_contrast_1_4": prof["rate_contrast_1_4"],
                "signal_correlation_1_4": prof["signal_correlation_1_4"],
                "schreiber_cross_1_4": prof["schreiber_cross_1_4"],
                "trajectory_type": prof["trajectory_type"]
            })

    stats_df = pd.DataFrame(stat_rows)
    csv_stats_path = cons_dir / "multi_epoch_spiking_statistics.csv"
    stats_df.to_csv(csv_stats_path, index=False)
    print(f"[+] Exported 4-Epoch Spiking Statistics to: {csv_stats_path}")

    # 2. Evaluate Cross-Cluster Split Candidates
    print("[*] Running Cross-Cluster Split-Candidate Discovery Algorithm...")
    splits_df = evaluate_cross_cluster_split_candidates(
        cluster_profiles=profiles,
        cluster_metadata=metadata,
        max_depth_delta_um=75.0,
        dt_s=dt_s
    )
    csv_splits_path = cons_dir / "cross_cluster_split_candidates.csv"
    splits_df.to_csv(csv_splits_path, index=False)
    print(f"[+] Exported Cross-Cluster Split Candidates ({len(splits_df)} pairs) to: {csv_splits_path}")

    # Display Top Split Pairs
    if len(splits_df) > 0:
        top_splits = splits_df.head(10)
        print("\n  Top Candidate Split Pairs:")
        for _, srow in top_splits.iterrows():
            print(f"    #A={int(srow['cluster_a']):3d} ({srow['depth_a_um']:5.1f} um) <--> #B={int(srow['cluster_b']):3d} ({srow['depth_b_um']:5.1f} um) "
                  f"| dy={srow['delta_y_um']:4.1f} um | TC={srow['temporal_complementarity']:.3f} | R_cross={srow['cross_cluster_schreiber']:.3f} "
                  f"| R_signal={srow['cross_cluster_signal_corr']:.3f} | Score={srow['composite_split_score']:.4f} [{srow['split_confidence']}]")

        # Generate Split Comparison Figures for Top 5 Pairs with Score > 0.02
        for s_i, (_, srow) in enumerate(splits_df.head(5).iterrows(), 1):
            if srow["composite_split_score"] < 0.02:
                continue
            ca, cb = int(srow["cluster_a"]), int(srow["cluster_b"])
            split_fig_file = out_split_dir / f"split_pair_{s_i:02d}_cluster_{ca}_vs_{cb}.png"
            build_split_comparison_figure(
                ca, cb,
                spk_times[spk_clusters == ca],
                spk_times[spk_clusters == cb],
                srow["depth_a_um"], srow["depth_b_um"],
                valid_onsets, valid_offsets, ti_arr,
                phase_grid_chrono, emg_trials_chrono,
                laser_train_bands,
                time_axis_s, time_axis_ms, dt_s,
                kernel, srow, split_fig_file
            )
            print(f"  [+] Saved split candidate figure: {split_fig_file.name}")

    # 3. Generate Chronological Breath Rasters & HTML Gallery
    # Prioritize: 1) Passed QC units, 2) Top split candidates, 3) Representative archetypes
    target_units = []
    if "passed_allen_qc" in qc_df.columns:
        qc_cids = list(qc_df[qc_df["passed_allen_qc"] == True]["cluster_id"].values)
        target_units.extend(qc_cids)
    else:
        target_units.extend(cids_all[:100])

    # Ensure top split candidate units are included
    if len(splits_df) > 0:
        for _, srow in splits_df.head(10).iterrows():
            ca, cb = int(srow["cluster_a"]), int(srow["cluster_b"])
            if ca not in target_units: target_units.append(ca)
            if cb not in target_units: target_units.append(cb)

    target_units = list(dict.fromkeys(target_units))[:max_rasters]
    print(f"[*] Generating Chronological Rasters for {len(target_units)} selected clusters...")

    index_cards = []
    for idx, cid in enumerate(target_units, 1):
        c_mask = (spk_clusters == cid)
        c_spks = spk_times[c_mask]
        depth = metadata[cid]["depth_um"]
        fr = metadata[cid]["firing_rate_hz"]
        passed_qc = metadata[cid]["passed_qc"]

        prof = profiles[cid]
        ep1_s = prof["epoch_stats"][1]
        ep4_s = prof["epoch_stats"][4]

        metrics_text = (
            f"Ep1 FR: {ep1_s['mean_fr_hz']:.1f} Hz (R={ep1_s['schreiber_reliability_25ms']:.2f}) | "
            f"Ep4 FR: {ep4_s['mean_fr_hz']:.1f} Hz (R={ep4_s['schreiber_reliability_25ms']:.2f}) | "
            f"Cross-R: {prof['schreiber_cross_1_4']:.2f}"
        )

        archetype = ep1_s["archetype"] if ep1_s["mean_fr_hz"] > 1.0 else ep4_s["archetype"]
        out_png = out_raster_dir / f"chronological_raster_cluster_{cid:03d}.png"

        plot_single_neuron_chronological_raster_peth(
            cid=cid,
            depth=depth,
            fr_hz=fr,
            archetype_label=archetype,
            u_spks=c_spks,
            valid_onsets=valid_onsets,
            valid_offsets=valid_offsets,
            ti_arr=ti_arr,
            phase_grid_chrono=phase_grid_chrono,
            emg_trials_chrono=emg_trials_chrono,
            laser_train_bands=laser_train_bands,
            time_axis_s=time_axis_s,
            time_axis_ms=time_axis_ms,
            dt_s=dt_s,
            kernel=kernel,
            out_file=out_png,
            metrics_summary=metrics_text
        )

        category = "qc_passed" if passed_qc else "all"
        index_cards.append({
            "cid": cid,
            "depth_um": depth,
            "firing_rate_hz": fr,
            "passed_qc": passed_qc,
            "category": category,
            "title_label": f"Cluster #{cid} — {archetype}",
            "file_name": out_png.name,
            "metrics_text": metrics_text
        })

    gallery_html_file = out_raster_dir / "gallery_all_units_chronological.html"
    build_gallery_html(index_cards, gallery_html_file, len(valid_onsets), dataset_title=dataset_name)

    return stats_df, splits_df

def main():
    print("=" * 80)
    print("  CHRONOLOGICAL RASTERS, MULTI-EPOCH STATISTICS & SPLIT-CLUSTER PIPELINE")
    print("=" * 80)

    # 1. Load NIDAQ Analog Data for Breaths & Laser
    nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
    fs = 10000.0
    n_ch = 9
    start_t = 796.827
    dur = 226.243
    start_samp = int(start_t * fs)
    n_samp = int(dur * fs)

    print(f"[*] Slicing NIDAQ raw analog file: {nidq_file}")
    raw = np.memmap(nidq_file, dtype="int16", mode="r")
    slice_data = raw[start_samp * n_ch : (start_samp + n_samp) * n_ch].reshape(-1, n_ch)
    laser_v = slice_data[:, 6].astype(np.float32) * (10.0 / 32768.0)
    emg_v = slice_data[:, 0].astype(np.float32) * (10.0 / 32768.0)
    t_vec = np.arange(len(emg_v)) / fs

    emg_cfg = EMGConfig(fs=fs, highpass_cutoff=300.0, lowpass_cutoff=3000.0, smoothing_window_ms=50.0, peak_threshold_sd=2.0)
    emg_proc = process_emg(emg_v, t_vec, emg_cfg)
    emg_env = emg_proc.envelope

    # 2. Time Axis
    win_pre_s = 0.15
    win_post_s = 0.35
    dt_s = 0.002
    time_axis_s = np.arange(-win_pre_s, win_post_s + dt_s, dt_s)
    time_axis_ms = time_axis_s * 1000.0

    kernel = signal.windows.gaussian(15, std=3)
    kernel /= np.sum(kernel)

    # 3. Extract Valid Breaths
    valid_pairs = []
    for on, off in zip(emg_proc.breath_onsets, emg_proc.breath_offsets):
        if on > win_pre_s and on < t_vec[-1] - win_post_s and off > on:
            valid_pairs.append((on, off, off - on))

    valid_onsets = np.array([p[0] for p in valid_pairs])
    valid_offsets = np.array([p[1] for p in valid_pairs])
    ti_arr = np.array([p[2] for p in valid_pairs])
    n_breaths = len(valid_onsets)
    print(f"[+] Extracted {n_breaths} chronological breath trials (0s to 226s).")

    # 4. Laser Bands
    diff = np.diff((laser_v > 0.5).astype(np.int32))
    pulse_starts = t_vec[np.where(diff == 1)[0] + 1]
    train_break_indices = np.where(np.diff(pulse_starts) > 1.0)[0]
    train_start_indices = [0] + [b + 1 for b in train_break_indices]
    train_starts = [pulse_starts[idx] for idx in train_start_indices]

    laser_train_bands = []
    for tr_i, ts in enumerate(train_starts, 1):
        te = ts + 1.955
        m = (valid_onsets + win_post_s >= ts) & (valid_onsets - win_pre_s <= te)
        ov_idx = np.where(m)[0]
        start_row = ov_idx[0] + 1 if len(ov_idx) > 0 else np.argmin(np.abs(valid_onsets - ts)) + 1
        end_row = ov_idx[-1] + 1 if len(ov_idx) > 0 else start_row + 1
        laser_train_bands.append({
            "train_num": tr_i, "start_time": ts, "end_time": te,
            "start_row": start_row, "end_row": end_row,
            "label": f"Train {tr_i} ({ts:.1f}s)"
        })

    # 5. Precompute Phase Grid & EMG Trials
    print("[*] Precomputing continuous respiratory phase grid and EMG traces...")
    phase_grid_chrono = compute_continuous_respiratory_phase_grid(valid_onsets, valid_offsets, time_axis_s)
    emg_trials_chrono = np.zeros((n_breaths, len(time_axis_s)), dtype=np.float32)
    for i in range(n_breaths):
        emg_trials_chrono[i, :] = np.interp(valid_onsets[i] + time_axis_s, t_vec, emg_env, left=0.0, right=0.0)

    # -------------------------------------------------------------------------
    # Dataset A: Motion-Corrected Consensus
    # -------------------------------------------------------------------------
    mc_dir = REPO_ROOT / "output_e097_prelaser_through_laser3_226s_motion_corrected" / "consensus_master"
    stats_mc, splits_mc = process_dataset(
        dataset_name="Motion-Corrected Consensus (Kriging 20µm)",
        cons_dir=mc_dir,
        valid_onsets=valid_onsets,
        valid_offsets=valid_offsets,
        ti_arr=ti_arr,
        phase_grid_chrono=phase_grid_chrono,
        emg_trials_chrono=emg_trials_chrono,
        laser_train_bands=laser_train_bands,
        time_axis_s=time_axis_s,
        time_axis_ms=time_axis_ms,
        dt_s=dt_s,
        kernel=kernel,
        max_rasters=130
    )

    # -------------------------------------------------------------------------
    # Dataset B: Uncorrected Baseline Consensus (Ground Truth Benchmark)
    # -------------------------------------------------------------------------
    uncorr_dir = REPO_ROOT / "output_e097_prelaser_through_laser3_226s" / "lupin" / "consensus_master"
    stats_uncorr, splits_uncorr = process_dataset(
        dataset_name="Uncorrected Baseline Consensus",
        cons_dir=uncorr_dir,
        valid_onsets=valid_onsets,
        valid_offsets=valid_offsets,
        ti_arr=ti_arr,
        phase_grid_chrono=phase_grid_chrono,
        emg_trials_chrono=emg_trials_chrono,
        laser_train_bands=laser_train_bands,
        time_axis_s=time_axis_s,
        time_axis_ms=time_axis_ms,
        dt_s=dt_s,
        kernel=kernel,
        max_rasters=130
    )

    print("\n" + "=" * 80)
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
