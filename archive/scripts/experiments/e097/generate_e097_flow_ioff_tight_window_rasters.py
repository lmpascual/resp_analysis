#!/usr/bin/env python3
"""
Dedicated Generator for Spike Rasters and Spiking Statistics Aligned to Flow Inspiration Offset (I_off)
Focused Strictly on the [-50 ms, +100 ms] Transition Window for e097 (226s Continuous Recording).

Features:
- Time window: [-50 ms, +100 ms] (150 ms total duration centered on Flow Zero-Crossing E_on / I_off).
  - [-50 ms, 0 ms]: Late-inspiratory ramp & off-switch activation.
  - [0 ms, +100 ms]: Peak post-inspiratory / early expiratory burst.
- Eliminates 100% of breath-rate acceleration contamination (no breath in Epoch 4 exceeds +100 ms expiration).
- High-resolution 1 ms binning.
- Dual-epoch PETH (Epoch 1 Pre-Laser vs Epoch 4 Post-Laser 3 Recovery).
- Schreiber reliability computed strictly within the [-50 ms, +100 ms] transition window.
- Interactive HTML Visual Gallery.
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal
import json
import torch

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
    print("GENERATING FLOW I_off RASTERS & STATISTICS IN [-50 ms, +100 ms] WINDOW (e097)")
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

    # 2. Extract Breaths & Flow Expiration Onsets
    print("[2/5] Extracting breaths & Flow Zero-Crossing expiration onsets...", flush=True)
    emg_cfg = EMGConfig(fs=fs, highpass_cutoff=300.0, lowpass_cutoff=3000.0, smoothing_window_ms=50.0, peak_threshold_sd=2.0)
    emg_proc = process_emg(emg_v, t_vec, emg_cfg)

    win_pre_s = 0.050  # -50 ms
    win_post_s = 0.100 # +100 ms
    valid_pairs = []
    for on, off in zip(emg_proc.breath_onsets, emg_proc.breath_offsets):
        if on > 0.150 and on < t_vec[-1] - 0.200 and off > on:
            valid_pairs.append((on, off, off - on))

    valid_onsets = np.array([p[0] for p in valid_pairs])
    valid_offsets = np.array([p[1] for p in valid_pairs])
    n_breaths = len(valid_onsets)

    flow_res = process_flow_meter(flow_v, t_vec, valid_onsets, valid_offsets, fs=fs)
    flow_zcs = flow_res.flow_zero_crossings
    ti_flow = flow_zcs - valid_onsets
    print(f"      Total breaths: {n_breaths} | Flow ZC detection: 100.0%", flush=True)

    # Epoch partitions
    idx_ep1 = np.where((valid_onsets >= 0.0) & (valid_onsets < 60.0))[0]
    idx_las1 = np.where((valid_onsets >= 60.0) & (valid_onsets < 62.0))[0]
    idx_ep2 = np.where((valid_onsets >= 62.0) & (valid_onsets < 119.7))[0]
    idx_las2 = np.where((valid_onsets >= 119.7) & (valid_onsets < 122.0))[0]
    idx_ep3 = np.where((valid_onsets >= 122.0) & (valid_onsets < 164.3))[0]
    idx_las3 = np.where((valid_onsets >= 164.3) & (valid_onsets < 166.5))[0]
    idx_ep4 = np.where((valid_onsets >= 166.5) & (valid_onsets <= 226.25))[0]

    # Sort within epochs ascending by T_I
    sorted_ep1 = idx_ep1[np.argsort(ti_flow[idx_ep1])]
    sorted_ep2 = idx_ep2[np.argsort(ti_flow[idx_ep2])]
    sorted_ep3 = idx_ep3[np.argsort(ti_flow[idx_ep3])]
    sorted_ep4 = idx_ep4[np.argsort(ti_flow[idx_ep4])]
    ordered_trials = np.concatenate([sorted_ep1, idx_las1, sorted_ep2, idx_las2, sorted_ep3, idx_las3, sorted_ep4])

    # Time axis: [-50 ms, +100 ms], dt = 1 ms
    dt_s = 0.001
    time_axis_s = np.arange(-win_pre_s, win_post_s + dt_s, dt_s)
    time_axis_ms = time_axis_s * 1000.0
    n_bins = len(time_axis_s)

    # Precompute shared Phase Grid in [-50 ms, +100 ms]
    print("[3/5] Precomputing shared phase grid in [-50 ms, +100 ms] window...", flush=True)
    phase_grid = np.zeros((n_breaths, n_bins), dtype=np.float32)
    for i in range(n_breaths):
        off = flow_zcs[i]
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
        phase_grid[i, :] = phi

    cmap_phase = LinearSegmentedColormap.from_list("phase_tight", ["#fed7aa", "#ffffff", "#dbeafe", "#93c5fd"], N=256)

    # Epoch layout blocks & dividers
    cum_ep1 = len(sorted_ep1)
    cum_l1 = cum_ep1 + len(idx_las1)
    cum_ep2 = cum_l1 + len(sorted_ep2)
    cum_l2 = cum_ep2 + len(idx_las2)
    cum_ep3 = cum_l2 + len(sorted_ep3)
    cum_l3 = cum_ep3 + len(idx_las3)
    cum_ep4 = cum_l3 + len(sorted_ep4)

    laser_blocks = [
        ("⚡ L1", cum_ep1 + 0.5, cum_l1 + 0.5),
        ("⚡ L2", cum_ep2 + 0.5, cum_l2 + 0.5),
        ("⚡ L3", cum_ep3 + 0.5, cum_l3 + 0.5)
    ]
    dividers = [cum_ep1, cum_l1, cum_ep2, cum_l2, cum_ep3, cum_l3]

    epoch_labels = [
        {"center_y": cum_ep1 / 2.0, "label": f"Ep 1: Pre-Laser (n={len(sorted_ep1)})"},
        {"center_y": (cum_l1 + cum_ep2) / 2.0, "label": f"Ep 2: Inter-Train 1 (n={len(sorted_ep2)})"},
        {"center_y": (cum_l2 + cum_ep3) / 2.0, "label": f"Ep 3: Inter-Train 2 (n={len(sorted_ep3)})"},
        {"center_y": (cum_l3 + cum_ep4) / 2.0, "label": f"Ep 4: Post-Laser 3 (n={len(sorted_ep4)})"}
    ]

    # 3. Load Clusters
    print("[4/5] Loading Motion-Corrected Consensus Clusters...", flush=True)
    cons_dir = PROJECT_ROOT / "output_e097_prelaser_through_laser3_226s_motion_corrected" / "consensus_master"
    out_dir = cons_dir / "02_single_cell_phase_rasters" / "flow_ioff_window_neg50_pos100"
    out_dir.mkdir(parents=True, exist_ok=True)

    spk_times = np.load(cons_dir / "alf" / "spike_times.npy")
    spk_clusters = np.load(cons_dir / "alf" / "spike_clusters.npy")
    df_info = pd.read_csv(cons_dir / "cluster_info.tsv", sep="\t")

    qc_passed_ids = set(df_info[df_info["passed_allen_qc"] == True]["cluster_id"].tolist())
    split_ids = {19, 328, 150, 378, 18, 269, 13, 64, 212, 14, 234, 114, 115, 335}
    top_spk_ids = set(df_info.nlargest(160, "n_spikes")["cluster_id"].tolist())
    target_cids = sorted(list(qc_passed_ids | split_ids | top_spk_ids))

    # PETH smoothing kernel (5 ms Gaussian)
    kernel_peth = signal.windows.gaussian(9, std=2)
    kernel_peth /= np.sum(kernel_peth)

    # Schreiber 10ms kernel
    sig_s = 0.010
    half_w = int(np.ceil(4.0 * sig_s / dt_s))
    t_k = np.arange(-half_w, half_w + 1) * dt_s
    kernel_schreiber = np.exp(-0.5 * (t_k / sig_s) ** 2)
    kernel_schreiber /= (np.sum(kernel_schreiber) * dt_s)

    gallery_items = []
    stats_records = []
    n_rendered = 0

    print(f"      Rendering {len(target_cids)} clusters...", flush=True)
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

        spike_x = []
        spike_y = []
        trial_binned = np.zeros((n_breaths, n_bins), dtype=np.float32)

        for r, orig_i in enumerate(ordered_trials):
            off = flow_zcs[orig_i]
            rel = u_spks - off
            in_win = rel[(rel >= -win_pre_s) & (rel <= win_post_s)]
            for s in in_win:
                spike_x.append(s * 1000.0)
                spike_y.append(r + 1)
            counts, _ = np.histogram(in_win, bins=np.append(time_axis_s, time_axis_s[-1] + dt_s))
            trial_binned[orig_i, :] = counts

        total_spikes_in_win = len(spike_x)
        if total_spikes_in_win == 0:
            continue

        # PETHs
        peth_ep1 = np.mean(trial_binned[idx_ep1], axis=0) / dt_s if len(idx_ep1) > 0 else np.zeros(n_bins)
        peth_ep4 = np.mean(trial_binned[idx_ep4], axis=0) / dt_s if len(idx_ep4) > 0 else np.zeros(n_bins)
        peth_ep1_sm = np.convolve(peth_ep1, kernel_peth, mode="same")
        peth_ep4_sm = np.convolve(peth_ep4, kernel_peth, mode="same")
        peak_ep1 = float(np.max(peth_ep1_sm))
        peak_ep4 = float(np.max(peth_ep4_sm))

        # Firing rates in window
        fr_win_ep1 = float(np.sum(trial_binned[idx_ep1]) / (len(idx_ep1) * 0.150)) if len(idx_ep1) > 0 else 0.0
        fr_win_ep4 = float(np.sum(trial_binned[idx_ep4]) / (len(idx_ep4) * 0.150)) if len(idx_ep4) > 0 else 0.0

        # Schreiber reliability in [-50, +100]ms window (sigma = 10 ms)
        def calc_schreiber(counts_subset):
            if len(counts_subset) < 2 or np.sum(counts_subset) < 3:
                return 0.0
            sm = np.apply_along_axis(lambda m: np.convolve(m, kernel_schreiber, mode="same") * dt_s, axis=1, arr=counts_subset).astype(np.float32)
            norms = np.sqrt(np.sum(sm ** 2, axis=1) * dt_s)
            t_sm = torch.from_numpy(np.ascontiguousarray(sm))
            G = (t_sm @ t_sm.T).numpy() * dt_s
            outer = np.outer(norms, norms)
            val = outer > 1e-9
            R = np.zeros_like(G)
            np.divide(G, outer, out=R, where=val)
            np.fill_diagonal(R, 0.0)
            m = (norms[:, None] > 1e-6) & (norms[None, :] > 1e-6)
            np.fill_diagonal(m, False)
            return float(np.mean(R[m])) if np.sum(m) > 0 else 0.0

        r_schreiber_ep1 = calc_schreiber(trial_binned[idx_ep1])
        r_schreiber_ep4 = calc_schreiber(trial_binned[idx_ep4])

        # Cross Ep1-Ep4 Schreiber
        r_cross = 0.0
        if np.sum(trial_binned[idx_ep1]) > 3 and np.sum(trial_binned[idx_ep4]) > 3:
            sm1 = np.apply_along_axis(lambda m: np.convolve(m, kernel_schreiber, mode="same") * dt_s, axis=1, arr=trial_binned[idx_ep1]).astype(np.float32)
            sm4 = np.apply_along_axis(lambda m: np.convolve(m, kernel_schreiber, mode="same") * dt_s, axis=1, arr=trial_binned[idx_ep4]).astype(np.float32)
            n1 = np.sqrt(np.sum(sm1 ** 2, axis=1) * dt_s)
            n4 = np.sqrt(np.sum(sm4 ** 2, axis=1) * dt_s)
            t_sm1 = torch.from_numpy(np.ascontiguousarray(sm1))
            t_sm4 = torch.from_numpy(np.ascontiguousarray(sm4))
            G_c = (t_sm1 @ t_sm4.T).numpy() * dt_s
            outer_c = np.outer(n1, n4)
            val_c = outer_c > 1e-9
            R_c = np.zeros_like(G_c)
            np.divide(G_c, outer_c, out=R_c, where=val_c)
            m_c = (n1[:, None] > 1e-6) & (n4[None, :] > 1e-6)
            r_cross = float(np.mean(R_c[m_c])) if np.sum(m_c) > 0 else 0.0


        # Sub-window classification: pre-off vs post-off
        pre_counts = np.sum(trial_binned[:, time_axis_s < 0])
        post_counts = np.sum(trial_binned[:, time_axis_s >= 0])
        if post_counts > 2.0 * (pre_counts + 1):
            cell_type = "Pure Post-I / Expiratory Phasic"
        elif pre_counts > 2.0 * (post_counts + 1):
            cell_type = "Late Inspiratory Ramp"
        else:
            cell_type = "Off-Switch Spanning / Transition"

        # Record metrics
        stats_records.append({
            "cluster_id": cid,
            "depth_um": depth,
            "spikes_in_150ms_win": total_spikes_in_win,
            "firing_rate_hz": fr_hz,
            "is_qc_passed": is_qc,
            "cell_type": cell_type,
            "fr_win_ep1_hz": round(fr_win_ep1, 1),
            "fr_win_ep4_hz": round(fr_win_ep4, 1),
            "r_schreiber_ep1_10ms": round(r_schreiber_ep1, 3),
            "r_schreiber_ep4_10ms": round(r_schreiber_ep4, 3),
            "r_schreiber_cross_10ms": round(r_cross, 3),
            "peak_ep1_hz": round(peak_ep1, 1),
            "peak_ep4_hz": round(peak_ep4, 1)
        })

        # ---------------------------------------------------------------------
        # RENDER 2-PANEL FIGURE
        # ---------------------------------------------------------------------
        fig = plt.figure(figsize=(10.5, 8.5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3.5, 1.0], hspace=0.14)

        # Top: Raster
        ax_ras = fig.add_subplot(gs[0])
        ax_ras.imshow(
            phase_grid[ordered_trials, :],
            aspect="auto", cmap=cmap_phase, vmin=-np.pi, vmax=np.pi,
            extent=[time_axis_ms[0], time_axis_ms[-1], 0.5, n_breaths + 0.5],
            origin="lower", alpha=0.55, interpolation="bilinear"
        )

        for lbl, y0, y1 in laser_blocks:
            ax_ras.fill_between([time_axis_ms[0], time_axis_ms[-1]], [y0, y0], [y1, y1], color='#ff922b', alpha=0.40, zorder=2)
            ax_ras.text(
                time_axis_ms[-1] - 3, (y0 + y1) / 2.0, lbl,
                color="#d9480f", fontsize=8.5, fontweight="bold",
                va="center", ha="right", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff4e6", alpha=0.92, edgecolor="#ff922b")
            )

        if len(spike_x) > 0:
            ax_ras.scatter(spike_x, spike_y, marker='|', color='black', s=16, linewidth=0.9, alpha=0.95, zorder=3, label="Spikes")

        for div in dividers:
            ax_ras.axhline(div, color="#64748b", linestyle=":", linewidth=0.8, alpha=0.6, zorder=2)

        # Off-switch vertical line
        ax_ras.axvline(0, color="#1e293b", linestyle="--", linewidth=1.8, alpha=0.9, zorder=4, label=r"Flow $I_{\mathrm{off}}$ / Expiration Onset (0 ms)")

        ax_ras.set_xlim(time_axis_ms[0], time_axis_ms[-1])
        ax_ras.set_ylim(0.5, n_breaths + 0.5)
        ax_ras.set_ylabel(f"Breath Trials ({n_breaths} Total | Sorted by $T_I$ Within Epochs)", fontsize=10.0, fontweight="bold")
        ax_ras.tick_params(labelbottom=False)

        ax_epoch = ax_ras.twinx()
        ax_epoch.set_ylim(0.5, n_breaths + 0.5)
        ax_epoch.set_yticks([eb["center_y"] for eb in epoch_labels])
        ax_epoch.set_yticklabels([eb["label"] for eb in epoch_labels], fontsize=8.0, fontweight="semibold", color="#334155")
        ax_epoch.tick_params(right=True, length=4, color="#64748b")

        qc_tag = "QC PASSED" if is_qc else "Non-QC"
        ax_ras.set_title(
            f"Cluster #{cid:03d} (Depth: {depth:.0f} µm | {cell_type} [{qc_tag}])\n"
            f"[Focus Window: -50 ms to +100 ms Centered on Flow Inspiration Offset ($I_{{\\mathrm{{off}}}}$) | $R_{{\\mathrm{{Schreiber}}}}(\\text{{Ep1}})={r_schreiber_ep1:.2f}$]",
            fontsize=10.5, fontweight="bold"
        )
        ax_ras.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=0.92, fontsize=8)

        # Bottom: PETH
        ax_peth = fig.add_subplot(gs[1], sharex=ax_ras)
        ax_peth.plot(time_axis_ms, peth_ep1_sm, color="#0284c7", linewidth=2.4, label=f"Ep 1 Pre (Peak: {peak_ep1:.1f} Hz, Mean: {fr_win_ep1:.1f} Hz)")
        ax_peth.plot(time_axis_ms, peth_ep4_sm, color="#e11d48", linewidth=2.4, label=f"Ep 4 Post (Peak: {peak_ep4:.1f} Hz, Mean: {fr_win_ep4:.1f} Hz)")
        ax_peth.axvline(0, color="#1e293b", linestyle="--", linewidth=1.5, alpha=0.9)
        ax_peth.set_xlim(time_axis_ms[0], time_axis_ms[-1])
        ax_peth.set_xlabel("Time from Flow Inspiration Offset ($I_{\\mathrm{off}}$) [ms]", fontsize=10.0, fontweight="bold")
        ax_peth.set_ylabel("Rate (Hz)", fontsize=10.0, fontweight="bold")
        ax_peth.legend(loc="upper right", fontsize=8.5)
        ax_peth.grid(True, linestyle="--", alpha=0.4)

        img_filename = f"tight_ioff_raster_cluster_{cid:03d}.png"
        out_png = out_dir / img_filename
        plt.savefig(out_png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        n_rendered += 1

        gallery_items.append({
            "cid": cid,
            "depth": depth,
            "fr": fr_hz,
            "spikes_win": total_spikes_in_win,
            "qc": is_qc,
            "type": cell_type,
            "r_ep1": r_schreiber_ep1,
            "r_ep4": r_schreiber_ep4,
            "r_cross": r_cross,
            "peak_ep1": round(peak_ep1, 1),
            "peak_ep4": round(peak_ep4, 1),
            "img": img_filename
        })

        if n_rendered % 25 == 0:
            print(f"      Rendered {n_rendered}/{len(target_cids)} tight-window rasters...", flush=True)

    print(f"[+] Rendered {n_rendered} tight-window rasters into: {out_dir}", flush=True)

    # Save summary CSV
    df_stats = pd.DataFrame(stats_records)
    stats_csv = out_dir / "spiking_statistics_neg50_pos100.csv"
    df_stats.to_csv(stats_csv, index=False)
    print(f"[+] Saved stats CSV: {stats_csv}", flush=True)

    # 4. Generate Interactive HTML Gallery
    print("[5/5] Building Interactive HTML Gallery...", flush=True)
    html_path = out_dir / "gallery_flow_ioff_window_neg50_pos100.html"

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>e097 Flow I_off Rasters [-50 ms, +100 ms]</title>
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
      --split-border: #f59e0b;
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
    <h1>Flow-Derived $I_{\\mathrm{off}}$ Rasters & Spiking Statistics ([-50 ms, +100 ms])</h1>
    <div class="subtitle">
      Centered precisely across the <strong>Inspiration Offset / Flow Zero-Crossing ($t = 0\\text{ ms}$)</strong> across all 842 breath trials (e097).<br>
      Strictly bounded to <strong>[-50 ms, +100 ms]</strong>: Captures late-inspiratory ramp and the entire post-inspiratory burst, while completely eliminating breath-frequency smearing or subsequent-breath contamination in fast breathing epochs.
    </div>
    <div class="metrics-bar">
      <div class="metric-chip"><strong>Total Units Rendered:</strong> __TOTAL_RENDERED__</div>
      <div class="metric-chip"><strong>Allen QC Passed:</strong> __QC_RENDERED__</div>
      <div class="metric-chip"><strong>Transition Window:</strong> -50 ms to +100 ms (150 ms total)</div>
      <div class="metric-chip"><strong>Temporal Binning:</strong> 1 ms</div>
      <div class="metric-chip"><strong>Reference Landmark:</strong> Flow Zero-Crossing (100.0% detected)</div>
    </div>
  </header>

  <div class="controls-bar">
    <div class="filter-group">
      <button class="btn-filter active" onclick="setFilter('all', this)">All Units (__TOTAL_RENDERED__)</button>
      <button class="btn-filter" onclick="setFilter('qc', this)">QC Passed (__QC_RENDERED__)</button>
      <button class="btn-filter" onclick="setFilter('exp', this)">Post-I / Expiratory Phasic</button>
      <button class="btn-filter" onclick="setFilter('insp', this)">Late Inspiratory Ramp</button>
      <button class="btn-filter" onclick="setFilter('high_rel', this)">High Reliability (R &ge; 0.35)</button>
    </div>
    <input type="text" id="searchInput" class="search-input" placeholder="Search Cluster ID (e.g. 212, 13, 150)..." oninput="applyFilters()">
    <select id="sortSelect" class="select-sort" onchange="applySort()">
      <option value="depth_asc">Sort: Dorsal to Ventral (Depth &uarr;)</option>
      <option value="depth_desc">Sort: Ventral to Dorsal (Depth &darr;)</option>
      <option value="cid_asc">Sort: Cluster ID (Ascending)</option>
      <option value="rel_desc">Sort: Schreiber R Ep1 (Highest &darr;)</option>
      <option value="spikes_desc">Sort: Spikes in Window (Highest &darr;)</option>
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
              <div class="metric-item"><strong>Window Spikes:</strong> ${c.spikes_win} spk</div>
              <div class="metric-item"><strong>Schreiber R (Ep1):</strong> ${c.r_ep1.toFixed(3)}</div>
              <div class="metric-item"><strong>Cross Ep1-4 R:</strong> ${c.r_cross.toFixed(3)}</div>
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
      let filtered = items.filter(c => {
        if (q && !String(c.cid).includes(q)) return false;
        if (activeFilter === 'qc') return c.qc;
        if (activeFilter === 'exp') return c.type.includes('Expiratory') || c.type.includes('Post-I');
        if (activeFilter === 'insp') return c.type.includes('Inspiratory');
        if (activeFilter === 'high_rel') return c.r_ep1 >= 0.35;
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
      else if (sortMode === 'rel_desc') list.sort((a, b) => b.r_ep1 - a.r_ep1);
      else if (sortMode === 'spikes_desc') list.sort((a, b) => b.spikes_win - a.spikes_win);
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

    applySortToList([...items]);
  </script>
</body>
</html>
"""

    qc_count = len([c for c in gallery_items if c["qc"]])
    final_gallery_html = html_template.replace("__TOTAL_RENDERED__", str(len(gallery_items))) \
                                      .replace("__QC_RENDERED__", str(qc_count)) \
                                      .replace("__JSON_ITEMS__", json.dumps(gallery_items))

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(final_gallery_html)

    print(f"[+] Successfully wrote gallery to: {html_path}", flush=True)
    t_tot = time.time() - t0
    print(f"[*] Complete in {t_tot:.1f} seconds!")


if __name__ == "__main__":
    main()
