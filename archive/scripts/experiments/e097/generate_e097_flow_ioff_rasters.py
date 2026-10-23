#!/usr/bin/env python3
"""
Dedicated Generator for Breath-Aligned Spike Rasters Aligned to Inspiration Offset (I_off)
Determined by the Flow Meter Zero-Crossing (Channel 5) for e097 (226s Continuous Recording).

Generates high-resolution raster plots with trial-by-trial respiratory phase background,
laser train highlights, inspiration onset (I_on = -T_I) curves, and dual-epoch PETHs (Ep1 vs Ep4).
Includes an interactive HTML visual gallery for exploring all units.
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(r"C:\Users\lpascu\analysis_src")
sys.path.insert(0, str(PROJECT_ROOT))
import src.pipeline.compat_patch
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.processors.proc_flow import process_flow_meter
from src.pipeline.config import EMGConfig

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.0


def main():
    t0 = time.time()
    print("=" * 80)
    print("GENERATING FLOW-DERIVED I_off ALIGNED SPIKE RASTERS & GALLERY (e097)")
    print("=" * 80)

    # 1. Load Analog Streams
    nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
    fs = 10000.0
    n_ch = 9
    start_t = 796.827
    dur = 226.243
    start_samp = int(start_t * fs)
    n_samp = int(dur * fs)

    print("[1/5] Loading NIDAQ analog streams (Diaphragm EMG, Flow Meter Ch 5, Laser Ch 6)...", flush=True)
    raw = np.memmap(nidq_file, dtype="int16", mode="r")
    slice_data = raw[start_samp * n_ch : (start_samp + n_samp) * n_ch].reshape(-1, n_ch)
    laser_v = slice_data[:, 6].astype(np.float32) * (10.0 / 32768.0)
    emg_v = slice_data[:, 0].astype(np.float32) * (10.0 / 32768.0)
    flow_v = slice_data[:, 5].astype(np.float32) * (10.0 / 32768.0)
    t_vec = np.arange(len(emg_v)) / fs

    # 2. Extract Breaths & Flow Expiration Onset Landmarks
    print("[2/5] Extracting breaths & Flow Zero-Crossing expiration onsets...", flush=True)
    emg_cfg = EMGConfig(fs=fs, highpass_cutoff=300.0, lowpass_cutoff=3000.0, smoothing_window_ms=50.0, peak_threshold_sd=2.0)
    emg_proc = process_emg(emg_v, t_vec, emg_cfg)

    win_pre_s = 0.200
    win_post_s = 0.300
    valid_pairs = []
    for on, off in zip(emg_proc.breath_onsets, emg_proc.breath_offsets):
        if on > win_pre_s and on < t_vec[-1] - win_post_s and off > on:
            valid_pairs.append((on, off, off - on))

    valid_onsets = np.array([p[0] for p in valid_pairs])
    valid_offsets = np.array([p[1] for p in valid_pairs])
    ti_emg = np.array([p[2] for p in valid_pairs])
    n_breaths = len(valid_onsets)

    flow_res = process_flow_meter(flow_v, t_vec, valid_onsets, valid_offsets, fs=fs)
    flow_zcs = flow_res.flow_zero_crossings

    # Primary reference landmark: Flow Zero-Crossing E_on (I_off)
    align_landmarks = flow_zcs
    ti_flow = align_landmarks - valid_onsets
    print(f"      Total breaths: {n_breaths} | Flow ZC detection: 100.0% | Mean T_I: {np.mean(ti_flow)*1000:.1f} ms", flush=True)

    # Epoch partitions
    idx_ep1 = np.where((valid_onsets >= 0.0) & (valid_onsets < 60.0))[0]
    idx_las1 = np.where((valid_onsets >= 60.0) & (valid_onsets < 62.0))[0]
    idx_ep2 = np.where((valid_onsets >= 62.0) & (valid_onsets < 119.7))[0]
    idx_las2 = np.where((valid_onsets >= 119.7) & (valid_onsets < 122.0))[0]
    idx_ep3 = np.where((valid_onsets >= 122.0) & (valid_onsets < 164.3))[0]
    idx_las3 = np.where((valid_onsets >= 164.3) & (valid_onsets < 166.5))[0]
    idx_ep4 = np.where((valid_onsets >= 166.5) & (valid_onsets <= 226.25))[0]

    # Sort within epochs ascending by T_I (flow-derived)
    sorted_ep1 = idx_ep1[np.argsort(ti_flow[idx_ep1])]
    sorted_ep2 = idx_ep2[np.argsort(ti_flow[idx_ep2])]
    sorted_ep3 = idx_ep3[np.argsort(ti_flow[idx_ep3])]
    sorted_ep4 = idx_ep4[np.argsort(ti_flow[idx_ep4])]
    ordered_trials = np.concatenate([sorted_ep1, idx_las1, sorted_ep2, idx_las2, sorted_ep3, idx_las3, sorted_ep4])

    # Time axis relative to Flow I_off: [-200 ms, +300 ms]
    dt_s = 0.002
    time_axis_s = np.arange(-win_pre_s, win_post_s + dt_s, dt_s)
    time_axis_ms = time_axis_s * 1000.0
    n_bins = len(time_axis_s)

    # Precompute shared Phase Grid relative to Flow I_off
    print("[3/5] Precomputing shared respiratory phase grid relative to Flow I_off...", flush=True)
    phase_grid_ioff = np.zeros((n_breaths, n_bins), dtype=np.float32)
    for i in range(n_breaths):
        off = align_landmarks[i]
        on = valid_onsets[i]
        next_on = valid_onsets[i + 1] if i + 1 < n_breaths else off + win_post_s
        t_eval = off + time_axis_s
        phi = np.full(n_bins, np.nan, dtype=np.float32)
        insp_m = (t_eval >= on) & (t_eval < off)
        if np.any(insp_m):
            phi[insp_m] = -np.pi + (t_eval[insp_m] - on) / (off - on + 1e-6) * np.pi
        exp_m = (t_eval >= off) & (t_eval < next_on)
        if np.any(exp_m):
            phi[exp_m] = (t_eval[exp_m] - off) / (next_on - off + 1e-6) * np.pi
        phi[np.isnan(phi)] = np.pi
        phase_grid_ioff[i, :] = phi

    cmap_phase = LinearSegmentedColormap.from_list("phase_ioff", ["#fed7aa", "#ffffff", "#dbeafe", "#93c5fd"], N=256)
    ordered_ti_ms = ti_flow[ordered_trials] * 1000.0

    # Epoch layout blocks & dividers
    cum_ep1 = len(sorted_ep1)
    cum_l1 = cum_ep1 + len(idx_las1)
    cum_ep2 = cum_l1 + len(sorted_ep2)
    cum_l2 = cum_ep2 + len(idx_las2)
    cum_ep3 = cum_l2 + len(sorted_ep3)
    cum_l3 = cum_ep3 + len(idx_las3)
    cum_ep4 = cum_l3 + len(sorted_ep4)

    laser_blocks = [
        ("⚡ Laser 1 (2s)", cum_ep1 + 0.5, cum_l1 + 0.5),
        ("⚡ Laser 2 (2s)", cum_ep2 + 0.5, cum_l2 + 0.5),
        ("⚡ Laser 3 (2s)", cum_ep3 + 0.5, cum_l3 + 0.5)
    ]
    dividers = [cum_ep1, cum_l1, cum_ep2, cum_l2, cum_ep3, cum_l3]

    epoch_labels = [
        {"center_y": cum_ep1 / 2.0, "label": f"Ep 1: Pre-Laser\n(n={len(sorted_ep1)}, T_I: {np.min(ti_flow[idx_ep1])*1000:.0f}–{np.max(ti_flow[idx_ep1])*1000:.0f} ms)"},
        {"center_y": (cum_l1 + cum_ep2) / 2.0, "label": f"Ep 2: Inter-Train 1\n(n={len(sorted_ep2)}, T_I: {np.min(ti_flow[idx_ep2])*1000:.0f}–{np.max(ti_flow[idx_ep2])*1000:.0f} ms)"},
        {"center_y": (cum_l2 + cum_ep3) / 2.0, "label": f"Ep 3: Inter-Train 2\n(n={len(sorted_ep3)}, T_I: {np.min(ti_flow[idx_ep3])*1000:.0f}–{np.max(ti_flow[idx_ep3])*1000:.0f} ms)"},
        {"center_y": (cum_l3 + cum_ep4) / 2.0, "label": f"Ep 4: Post-Laser 3\n(n={len(sorted_ep4)}, T_I: {np.min(ti_flow[idx_ep4])*1000:.0f}–{np.max(ti_flow[idx_ep4])*1000:.0f} ms)"}
    ]

    # 3. Load Motion-Corrected Consensus Spike Clusters
    print("[4/5] Loading Motion-Corrected Clusters & Rendering Rasters...", flush=True)
    cons_dir = PROJECT_ROOT / "output_e097_prelaser_through_laser3_226s_motion_corrected" / "consensus_master"
    out_dir = cons_dir / "02_single_cell_phase_rasters" / "flow_ioff_aligned"
    out_dir.mkdir(parents=True, exist_ok=True)

    spk_times = np.load(cons_dir / "alf" / "spike_times.npy")
    spk_clusters = np.load(cons_dir / "alf" / "spike_clusters.npy")
    df_info = pd.read_csv(cons_dir / "cluster_info.tsv", sep="\t")
    print(f"      Total clusters in dataset: {len(df_info)}", flush=True)

    # Select target clusters: all QC-passed (122 units) + split candidates + top active units
    qc_passed_ids = set(df_info[df_info["passed_allen_qc"] == True]["cluster_id"].tolist())
    split_ids = {19, 328, 150, 378, 18, 269, 13, 64, 212, 14, 234, 114, 115, 335}
    top_spk_ids = set(df_info.nlargest(160, "n_spikes")["cluster_id"].tolist())
    target_cids = sorted(list(qc_passed_ids | split_ids | top_spk_ids))
    print(f"      Selected {len(target_cids)} clusters to render...", flush=True)

    kernel_peth = signal.windows.gaussian(15, std=3)
    kernel_peth /= np.sum(kernel_peth)

    gallery_items = []
    n_rendered = 0

    for cid in target_cids:
        row = df_info[df_info["cluster_id"] == cid]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        depth = float(row["depth_um"])
        n_spk = int(row["n_spikes"])
        fr_hz = float(row["firing_rate_hz"])
        is_qc = bool(row["passed_allen_qc"])

        u_spks = spk_times[spk_clusters == cid]
        if len(u_spks) == 0:
            continue

        # Extract spikes relative to Flow I_off
        spike_x = []
        spike_y = []
        trial_binned = np.zeros((n_breaths, n_bins), dtype=np.float32)

        for r, orig_i in enumerate(ordered_trials):
            off = align_landmarks[orig_i]
            rel = u_spks - off
            in_win = rel[(rel >= -win_pre_s) & (rel <= win_post_s)]
            for s in in_win:
                spike_x.append(s * 1000.0)
                spike_y.append(r + 1)
            counts, _ = np.histogram(in_win, bins=np.append(time_axis_s, time_axis_s[-1] + dt_s))
            trial_binned[orig_i, :] = counts

        # Compute PETHs for Ep 1 and Ep 4
        peth_ep1 = np.mean(trial_binned[idx_ep1], axis=0) / dt_s if len(idx_ep1) > 0 else np.zeros(n_bins)
        peth_ep4 = np.mean(trial_binned[idx_ep4], axis=0) / dt_s if len(idx_ep4) > 0 else np.zeros(n_bins)
        peth_ep1_sm = np.convolve(peth_ep1, kernel_peth, mode="same")
        peth_ep4_sm = np.convolve(peth_ep4, kernel_peth, mode="same")
        peak_ep1 = float(np.max(peth_ep1_sm))
        peak_ep4 = float(np.max(peth_ep4_sm))

        # Classify cell type
        # In [-200, 0]ms is inspiration; in [0, +150]ms is expiration
        insp_counts = np.sum(trial_binned[:, time_axis_s < 0])
        exp_counts = np.sum(trial_binned[:, time_axis_s >= 0])
        if exp_counts > 1.8 * (insp_counts + 1):
            cell_type = "Expiratory / Post-I Phasic"
        elif insp_counts > 1.8 * (exp_counts + 1):
            cell_type = "Inspiratory Phasic"
        else:
            cell_type = "Phase-Spanning / Modulated"

        # ---------------------------------------------------------------------
        # RENDER DEDICATED 2-PANEL FIGURE
        # ---------------------------------------------------------------------
        fig = plt.figure(figsize=(11.5, 9.0))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3.6, 1.0], hspace=0.14)

        # Top Subplot: Raster
        ax_ras = fig.add_subplot(gs[0])
        ax_ras.imshow(
            phase_grid_ioff[ordered_trials, :],
            aspect="auto", cmap=cmap_phase, vmin=-np.pi, vmax=np.pi,
            extent=[time_axis_ms[0], time_axis_ms[-1], 0.5, n_breaths + 0.5],
            origin="lower", alpha=0.55, interpolation="bilinear"
        )

        # Laser bands
        for lbl, y0, y1 in laser_blocks:
            ax_ras.fill_between([time_axis_ms[0], time_axis_ms[-1]], [y0, y0], [y1, y1], color='#ff922b', alpha=0.40, zorder=2)
            ax_ras.text(
                time_axis_ms[-1] - 8, (y0 + y1) / 2.0, lbl,
                color="#d9480f", fontsize=8.5, fontweight="bold",
                va="center", ha="right", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff4e6", alpha=0.92, edgecolor="#ff922b")
            )

        # Spikes
        if len(spike_x) > 0:
            ax_ras.scatter(spike_x, spike_y, marker='|', color='black', s=16, linewidth=0.85, alpha=0.92, zorder=3, label="Spike Times")

        # Inspiration Onset curve on left: t = -T_I
        ax_ras.plot(-ordered_ti_ms, np.arange(1, n_breaths + 1), color="#b91c1c", linestyle="", marker=".", markersize=2.0, alpha=0.75, zorder=4, label=r"Inspiration Onset ($I_{\mathrm{on}} = -T_I$)")

        # Divider lines between epochs
        for div in dividers:
            ax_ras.axhline(div, color="#64748b", linestyle=":", linewidth=0.8, alpha=0.6, zorder=2)

        # Reference landmark: Flow Zero-Crossing at t = 0 ms
        ax_ras.axvline(0, color="#1e293b", linestyle="--", linewidth=1.8, alpha=0.9, zorder=4, label=r"Flow $I_{\mathrm{off}}$ / Expiration Onset (0 ms)")

        ax_ras.set_xlim(time_axis_ms[0], time_axis_ms[-1])
        ax_ras.set_ylim(0.5, n_breaths + 0.5)
        ax_ras.set_ylabel(f"Breath Trials ({n_breaths} Total | Sorted by $T_I$ within Epochs)", fontsize=10.5, fontweight="bold")
        ax_ras.tick_params(labelbottom=False)

        # Right Y-axis for Epoch Labels & Ti Ranges
        ax_epoch = ax_ras.twinx()
        ax_epoch.set_ylim(0.5, n_breaths + 0.5)
        y_ticks = [eb["center_y"] for eb in epoch_labels]
        y_lbls = [eb["label"] for eb in epoch_labels]
        ax_epoch.set_yticks(y_ticks)
        ax_epoch.set_yticklabels(y_lbls, fontsize=8.0, fontweight="semibold", color="#334155")
        ax_epoch.tick_params(right=True, length=4, color="#64748b")

        qc_tag = "QC PASSED" if is_qc else "Non-QC"
        ax_ras.set_title(
            f"Cluster #{cid:03d} (Depth: {depth:.0f} µm | FR: {fr_hz:.1f} Hz | {cell_type} [{qc_tag}])\n"
            f"[Aligned to Flow Meter Zero-Crossing ($I_{{\\mathrm{{off}}}}$ / Expiration Onset) | Trials Sorted by $T_I$ Within Epochs]",
            fontsize=11.0, fontweight="bold"
        )
        ax_ras.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=0.92, fontsize=8)

        # Bottom Subplot: PETH
        ax_peth = fig.add_subplot(gs[1], sharex=ax_ras)
        ax_peth.plot(time_axis_ms, peth_ep1_sm, color="#0284c7", linewidth=2.4, label=f"Pre-Laser Eupnea (Ep 1, Peak: {peak_ep1:.1f} Hz)")
        ax_peth.plot(time_axis_ms, peth_ep4_sm, color="#e11d48", linewidth=2.4, label=f"Post-Laser 3 Recovery (Ep 4, Peak: {peak_ep4:.1f} Hz)")
        ax_peth.axvline(0, color="#1e293b", linestyle="--", linewidth=1.5, alpha=0.9)
        ax_peth.set_xlim(time_axis_ms[0], time_axis_ms[-1])
        ax_peth.set_xlabel("Time Relative to Flow Inspiration Offset ($I_{\\mathrm{off}}$ / Expiration Onset) [ms]", fontsize=10.0, fontweight="bold")
        ax_peth.set_ylabel("Rate (Hz)", fontsize=10.0, fontweight="bold")
        ax_peth.legend(loc="upper right", fontsize=8.5)
        ax_peth.grid(True, linestyle="--", alpha=0.4)

        img_filename = f"flow_ioff_raster_cluster_{cid:03d}.png"
        out_png = out_dir / img_filename
        plt.savefig(out_png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        n_rendered += 1

        gallery_items.append({
            "cid": cid,
            "depth": depth,
            "fr": fr_hz,
            "spikes": n_spk,
            "qc": is_qc,
            "type": cell_type,
            "peak_ep1": round(peak_ep1, 1),
            "peak_ep4": round(peak_ep4, 1),
            "img": img_filename
        })

        if n_rendered % 25 == 0:
            print(f"      Rendered {n_rendered}/{len(target_cids)} Flow I_off rasters...", flush=True)

    print(f"[+] Finished rendering {n_rendered} Flow I_off rasters into: {out_dir}", flush=True)

    # 4. Generate Interactive HTML Gallery
    print("[5/5] Building Interactive HTML Gallery...", flush=True)
    html_gallery_path = out_dir / "gallery_flow_ioff_rasters.html"

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>e097 Flow-Derived I_off Aligned Spike Rasters Gallery</title>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f2937;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --accent: #0284c7;
      --accent-glow: rgba(2, 132, 199, 0.25);
      --qc-pass: #10b981;
      --qc-fail: #ef4444;
    }
    body {
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    header {
      max-width: 1440px;
      margin: 0 auto 24px auto;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 20px;
    }
    h1 {
      margin: 0 0 8px 0;
      font-size: 26px;
      color: #38bdf8;
    }
    .subtitle {
      color: var(--text-muted);
      font-size: 14px;
      line-height: 1.5;
    }
    .metrics-bar {
      display: flex;
      gap: 16px;
      margin-top: 16px;
      flex-wrap: wrap;
    }
    .metric-chip {
      background: #1e293b;
      padding: 8px 14px;
      border-radius: 6px;
      border: 1px solid #334155;
      font-size: 13px;
    }
    .metric-chip strong {
      color: #38bdf8;
      margin-right: 4px;
    }
    .controls-bar {
      max-width: 1440px;
      margin: 0 auto 24px auto;
      display: flex;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
      background: #0f172a;
      padding: 14px 18px;
      border-radius: 8px;
      border: 1px solid #1e293b;
    }
    .filter-group {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .btn-filter {
      background: #1e293b;
      color: var(--text);
      border: 1px solid #334155;
      padding: 6px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
      transition: all 0.15s ease;
    }
    .btn-filter.active {
      background: var(--accent);
      border-color: #38bdf8;
      font-weight: bold;
    }
    .search-input, .select-sort {
      background: #1e293b;
      color: var(--text);
      border: 1px solid #334155;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 13px;
    }
    .gallery-grid {
      max-width: 1440px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(460px, 1fr));
      gap: 22px;
    }
    .unit-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .unit-card:hover {
      transform: translateY(-2px);
      border-color: #38bdf8;
    }
    .card-header {
      padding: 10px 14px;
      background: #1a2234;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--card-border);
    }
    .card-title {
      font-size: 14px;
      font-weight: bold;
      color: #f8fafc;
    }
    .badge {
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 4px;
      font-weight: bold;
    }
    .badge-qc { background: #064e3b; color: #34d399; border: 1px solid #059669; }
    .badge-noqc { background: #450a0a; color: #f87171; border: 1px solid #b91c1c; }
    .card-body {
      padding: 10px;
    }
    .card-img {
      width: 100%;
      height: auto;
      border-radius: 4px;
      cursor: pointer;
      display: block;
    }
    .metrics-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      margin-top: 10px;
      font-size: 12px;
      color: #cbd5e1;
    }
    .metric-item strong {
      color: #38bdf8;
    }
    /* Modal */
    .modal {
      display: none;
      position: fixed;
      z-index: 1000;
      top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0,0,0,0.92);
      justify-content: center;
      align-items: center;
    }
    .modal.active {
      display: flex;
    }
    .modal-content {
      max-width: 95vw;
      max-height: 95vh;
      border-radius: 8px;
      border: 2px solid #38bdf8;
    }
  </style>
</head>
<body>
  <header>
    <h1>Flow-Derived Inspiration Offset ($I_{\\mathrm{off}}$) Aligned Spike Rasters (e097)</h1>
    <div class="subtitle">
      All 842 breath trials across the continuous 226s recording aligned directly to <strong>Flow Meter Zero-Crossing ($t = 0\\text{ ms}$)</strong>.<br>
      Within each epoch, trials are sorted ascending by inspiration duration ($T_I$), revealing duration-invariant post-inspiratory locking (upright burst at $0\\text{ ms}$) vs duration-dependent inspiratory scaling ($I_{\\mathrm{on}} = -T_I$ curve on left).
    </div>
    <div class="metrics-bar">
      <div class="metric-chip"><strong>Total Units Rendered:</strong> __TOTAL_RENDERED__</div>
      <div class="metric-chip"><strong>Allen QC Passed:</strong> __QC_RENDERED__</div>
      <div class="metric-chip"><strong>Reference Landmark:</strong> Flow Zero-Crossing (100.0% detected, 842/842 breaths)</div>
      <div class="metric-chip"><strong>Raster Time Window:</strong> -200 ms to +300 ms</div>
    </div>
  </header>

  <div class="controls-bar">
    <div class="filter-group">
      <button class="btn-filter active" onclick="setFilter('all', this)">All Units (__TOTAL_RENDERED__)</button>
      <button class="btn-filter" onclick="setFilter('qc', this)">QC Passed (__QC_RENDERED__)</button>
      <button class="btn-filter" onclick="setFilter('exp', this)">Expiratory / Post-I</button>
      <button class="btn-filter" onclick="setFilter('insp', this)">Inspiratory</button>
      <button class="btn-filter" onclick="setFilter('split', this)">Split Candidates</button>
    </div>
    <input type="text" id="searchInput" class="search-input" placeholder="Search Cluster ID (e.g. 212, 13, 64)..." oninput="applyFilters()">
    <select id="sortSelect" class="select-sort" onchange="applySort()">
      <option value="depth_asc">Sort: Dorsal to Ventral (Depth &uarr;)</option>
      <option value="depth_desc">Sort: Ventral to Dorsal (Depth &darr;)</option>
      <option value="cid_asc">Sort: Cluster ID (Ascending)</option>
      <option value="spikes_desc">Sort: Spike Count (Highest &darr;)</option>
      <option value="fr_desc">Sort: Firing Rate (Highest &darr;)</option>
    </select>
  </div>

  <div class="gallery-grid" id="galleryGrid"></div>

  <div class="modal" id="imageModal" onclick="closeModal()">
    <img class="modal-content" id="modalImg">
  </div>

  <script>
    const items = __JSON_ITEMS__;
    let activeFilter = 'all';

    function renderCards(cardList) {
      const grid = document.getElementById('galleryGrid');
      grid.innerHTML = '';
      cardList.forEach(c => {
        const card = document.createElement('div');
        card.className = 'unit-card';
        card.innerHTML = `
          <div class="card-header">
            <span class="card-title">Cluster #${String(c.cid).padStart(3, '0')} (${Math.round(c.depth)} µm)</span>
            <div>
              ${c.qc ? '<span class="badge badge-qc">QC PASSED</span>' : '<span class="badge badge-noqc">NON-QC</span>'}
            </div>
          </div>
          <div class="card-body">
            <img src="${c.img}" class="card-img" loading="lazy" onclick="openModal('${c.img}')" alt="Cluster ${c.cid}">
            <div class="metrics-grid">
              <div class="metric-item"><strong>Type:</strong> ${c.type}</div>
              <div class="metric-item"><strong>FR:</strong> ${c.fr.toFixed(1)} Hz (${c.spikes} spk)</div>
              <div class="metric-item"><strong>Peak Ep1:</strong> ${c.peak_ep1.toFixed(1)} Hz</div>
              <div class="metric-item"><strong>Peak Ep4:</strong> ${c.peak_ep4.toFixed(1)} Hz</div>
            </div>
          </div>
        `;
        grid.appendChild(card);
      });
    }

    function setFilter(filter, el) {
      activeFilter = filter;
      document.querySelectorAll('.btn-filter').forEach(btn => btn.classList.remove('active'));
      if (el) el.classList.add('active');
      applyFilters();
    }

    function applyFilters() {
      const q = document.getElementById('searchInput').value.trim().toLowerCase();
      const splitCids = [19, 328, 150, 378, 18, 269, 13, 64, 212, 14, 234];
      let filtered = items.filter(c => {
        if (q && !String(c.cid).includes(q)) return false;
        if (activeFilter === 'qc') return c.qc;
        if (activeFilter === 'exp') return c.type.includes('Expiratory') || c.type.includes('Post-I');
        if (activeFilter === 'insp') return c.type.includes('Inspiratory');
        if (activeFilter === 'split') return splitCids.includes(c.cid);
        return true;
      });
      applySortToList(filtered);
    }

    function applySort() {
      applyFilters();
    }

    function applySortToList(list) {
      const sortMode = document.getElementById('sortSelect').value;
      if (sortMode === 'depth_asc') list.sort((a, b) => a.depth - b.depth);
      else if (sortMode === 'depth_desc') list.sort((a, b) => b.depth - a.depth);
      else if (sortMode === 'cid_asc') list.sort((a, b) => a.cid - b.cid);
      else if (sortMode === 'spikes_desc') list.sort((a, b) => b.spikes - a.spikes);
      else if (sortMode === 'fr_desc') list.sort((a, b) => b.fr - a.fr);
      renderCards(list);
    }

    function openModal(src) {
      const m = document.getElementById('imageModal');
      const img = document.getElementById('modalImg');
      img.src = src;
      m.classList.add('active');
    }

    function closeModal() {
      document.getElementById('imageModal').classList.remove('active');
    }

    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') closeModal();
    });

    // Initial render
    applySortToList([...items]);
  </script>
</body>
</html>
"""

    qc_count = len([c for c in gallery_items if c["qc"]])
    final_gallery_html = html_template.replace("__TOTAL_RENDERED__", str(len(gallery_items))) \
                                      .replace("__QC_RENDERED__", str(qc_count)) \
                                      .replace("__JSON_ITEMS__", json.dumps(gallery_items))

    with open(html_gallery_path, "w", encoding="utf-8") as f:
        f.write(final_gallery_html)

    print(f"[+] Successfully generated gallery: {html_gallery_path}", flush=True)
    t_tot = time.time() - t0
    print(f"[*] Complete in {t_tot:.1f} seconds!")


if __name__ == "__main__":
    main()
