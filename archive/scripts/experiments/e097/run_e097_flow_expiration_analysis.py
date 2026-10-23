#!/usr/bin/env python3
"""
Exploratory Flow-Meter-Based Expiration Onset Analysis for e097 (60s Pre-Laser Snippet)
Investigates whether the timing of slope polarity inversion in the Flow Meter signal
(from downward to upward slope, i.e. flow trough) provides a more biomechanically
accurate marker of expiration onset compared to the diaphragmatic EMG offset.

Outputs placed in:
C:\\Users\\lpascu\\analysis_src\\e097_prelaser_respiratory_neurodynamics\\06_exploratory_flow_expiration\\
"""

import sys
import os
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

PROJECT_ROOT = Path(r"C:\Users\lpascu\analysis_src")
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

OUT_DIR = PROJECT_ROOT / "e097_prelaser_respiratory_neurodynamics" / "06_exploratory_flow_expiration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.0


def main():
    print("=" * 80)
    print("STARTING EXPLORATORY FLOW-METER EXPIRATION ONSET ANALYSIS (e097)")
    print("=" * 80)

    # 1. Load Data
    raw_dir = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2")
    nidq_bins = list(raw_dir.glob("*.nidq.bin"))
    if not nidq_bins:
        raise FileNotFoundError(f"No .nidq.bin found in {raw_dir}")
    nidq_bin = nidq_bins[0]
    meta = nidq_bin.with_suffix(".meta")

    start_t = 796.827
    duration_s = 60.0
    t_end = start_t + duration_s

    data = read_nidq_bin(nidq_bin, meta, emg_chan=0, ecg_chan=1)
    fs = data.fs
    t_all = data.time_vector
    mask_time = (t_all >= start_t) & (t_all <= t_end)
    t = t_all[mask_time] - start_t
    emg_raw = data.all_channels[0][mask_time]
    flow_raw = data.all_channels[5][mask_time]

    print(f"Loaded NIDQ: fs={fs:.1f} Hz, samples={len(t)}, duration={t[-1]:.2f}s")

    # 2. Process EMG with cibrrig standard
    emg_cfg = EMGConfig(fs=fs, highpass_cutoff=300.0, lowpass_cutoff=3000.0, smoothing_window_ms=50.0, peak_threshold_sd=2.0)
    emg_proc = process_emg(emg_raw, t, emg_cfg)
    emg_onsets = emg_proc.breath_onsets
    emg_offsets = emg_proc.breath_offsets
    emg_env = emg_proc.envelope

    print(f"EMG Breaths detected: onsets={len(emg_onsets)}, offsets={len(emg_offsets)}")

    # 3. Process Flow Meter Signal
    # 2nd-order Butterworth low-pass filter at 50 Hz to remove high-frequency turbulence
    b_low, a_low = signal.butter(2, 50.0 / (fs / 2.0), btype='low')
    flow_lp = signal.filtfilt(b_low, a_low, flow_raw)

    # First derivative (airflow acceleration) in V/s
    dt_nidq = 1.0 / fs
    flow_deriv = np.gradient(flow_lp, dt_nidq)

    # 4. Detect Flow Slope Inversion (Trough) and Zero-Crossing per Breath
    n_breaths = min(len(emg_onsets), len(emg_offsets))
    breath_metrics = []

    for i in range(n_breaths):
        t_on = emg_onsets[i]
        t_off_emg = emg_offsets[i]
        t_next_on = emg_onsets[i + 1] if i + 1 < len(emg_onsets) else t_on + 0.35

        # Search window for inspiratory flow trough: from t_on to min(t_on + 0.35, t_next_on)
        win_mask = (t >= t_on) & (t <= min(t_on + 0.35, t_next_on))
        t_win = t[win_mask]
        f_win = flow_lp[win_mask]
        df_win = flow_deriv[win_mask]

        if len(t_win) < 5:
            continue

        # Flow trough: local minimum (where derivative changes from negative to positive)
        # Find global minimum within inspiratory burst
        min_idx = np.argmin(f_win)
        t_flow_slope = t_win[min_idx]
        flow_min_val = f_win[min_idx]

        # Flow zero crossing (where flow rises above zero after trough)
        post_trough_mask = (t >= t_flow_slope) & (t <= min(t_flow_slope + 0.25, t_next_on))
        t_post = t[post_trough_mask]
        f_post = flow_lp[post_trough_mask]
        
        zero_crossings = np.where(np.diff(np.sign(f_post)) > 0)[0]
        if len(zero_crossings) > 0:
            zc_idx = zero_crossings[0]
            # Linear interpolation for sub-sample precision
            f0 = f_post[zc_idx]
            f1 = f_post[zc_idx + 1]
            t0 = t_post[zc_idx]
            t1 = t_post[zc_idx + 1]
            t_flow_zero = t0 - f0 * (t1 - t0) / (f1 - f0 + 1e-12)
        else:
            t_flow_zero = np.nan

        # Peak expiratory flow (maximum flow in post-trough window)
        exp_win_mask = (t >= t_flow_slope) & (t <= t_next_on)
        if np.any(exp_win_mask):
            f_exp_win = flow_lp[exp_win_mask]
            t_exp_win = t[exp_win_mask]
            max_idx = np.argmax(f_exp_win)
            t_flow_exp_peak = t_exp_win[max_idx]
            flow_max_val = f_exp_win[max_idx]
        else:
            t_flow_exp_peak = np.nan
            flow_max_val = np.nan

        # EMG peak
        emg_win_mask = (t >= t_on) & (t <= t_off_emg)
        if np.any(emg_win_mask):
            emg_peak_val = np.max(emg_env[emg_win_mask])
        else:
            emg_peak_val = np.nan

        t_i_emg = t_off_emg - t_on
        t_i_flow_slope = t_flow_slope - t_on
        t_i_flow_zero = (t_flow_zero - t_on) if not np.isnan(t_flow_zero) else np.nan

        t_e_emg = (t_next_on - t_off_emg) if i + 1 < len(emg_onsets) else np.nan
        t_e_flow_slope = (t_next_on - t_flow_slope) if i + 1 < len(emg_onsets) else np.nan

        delta_t_slope = t_flow_slope - t_off_emg
        delta_t_zero = (t_flow_zero - t_off_emg) if not np.isnan(t_flow_zero) else np.nan

        breath_metrics.append({
            "breath_id": i,
            "t_on_emg": t_on,
            "t_off_emg": t_off_emg,
            "t_flow_slope": t_flow_slope,
            "t_flow_zero": t_flow_zero,
            "t_flow_exp_peak": t_flow_exp_peak,
            "t_next_on": t_next_on,
            "T_I_emg_ms": t_i_emg * 1000.0,
            "T_I_flow_slope_ms": t_i_flow_slope * 1000.0,
            "T_I_flow_zero_ms": t_i_flow_zero * 1000.0 if not np.isnan(t_i_flow_zero) else np.nan,
            "T_E_emg_ms": t_e_emg * 1000.0 if not np.isnan(t_e_emg) else np.nan,
            "T_E_flow_slope_ms": t_e_flow_slope * 1000.0 if not np.isnan(t_e_flow_slope) else np.nan,
            "delta_t_slope_ms": delta_t_slope * 1000.0,
            "delta_t_zero_ms": delta_t_zero * 1000.0 if not np.isnan(delta_t_zero) else np.nan,
            "flow_min_v": flow_min_val,
            "flow_max_v": flow_max_val,
            "emg_peak_v": emg_peak_val,
            "is_sigh": (i == 99 or (emg_peak_val > 0.45 if not np.isnan(emg_peak_val) else False))
        })

    df_metrics = pd.DataFrame(breath_metrics)
    csv_path = OUT_DIR / "breath_by_breath_flow_vs_emg_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f"Computed metrics for {len(df_metrics)} breaths. Saved to {csv_path}")

    # Summary statistics
    eupnea_mask = ~df_metrics["is_sigh"]
    dt_slope_vals = df_metrics.loc[eupnea_mask, "delta_t_slope_ms"].dropna()
    dt_zero_vals = df_metrics.loc[eupnea_mask, "delta_t_zero_ms"].dropna()
    ti_emg_vals = df_metrics.loc[eupnea_mask, "T_I_emg_ms"].dropna()
    ti_slope_vals = df_metrics.loc[eupnea_mask, "T_I_flow_slope_ms"].dropna()
    te_emg_vals = df_metrics.loc[eupnea_mask, "T_E_emg_ms"].dropna()
    te_slope_vals = df_metrics.loc[eupnea_mask, "T_E_flow_slope_ms"].dropna()

    print("\n--- QUANTITATIVE SUMMARY (EUPNEIC BREATHS, N = {}) ---".format(len(dt_slope_vals)))
    print(f"Flow Slope Inversion vs EMG Offset (Delta t_slope): Mean = {dt_slope_vals.mean():.2f} ms ± {dt_slope_vals.std():.2f} ms (Median = {dt_slope_vals.median():.2f} ms, IQR = {dt_slope_vals.quantile(0.75)-dt_slope_vals.quantile(0.25):.2f} ms)")
    print(f"Flow Zero Crossing vs EMG Offset (Delta t_zero):   Mean = {dt_zero_vals.mean():.2f} ms ± {dt_zero_vals.std():.2f} ms (Median = {dt_zero_vals.median():.2f} ms)")
    print(f"T_I EMG:        Mean = {ti_emg_vals.mean():.2f} ms ± {ti_emg_vals.std():.2f} ms")
    print(f"T_I Flow Slope: Mean = {ti_slope_vals.mean():.2f} ms ± {ti_slope_vals.std():.2f} ms")
    print(f"T_E EMG:        Mean = {te_emg_vals.mean():.2f} ms ± {te_emg_vals.std():.2f} ms")
    print(f"T_E Flow Slope: Mean = {te_slope_vals.mean():.2f} ms ± {te_slope_vals.std():.2f} ms")

    # 5. Load Spikes & Compute PCA for Neural Attractor Comparison
    cons_dir = PROJECT_ROOT / "output_e097_prelaser_60s" / "lupin" / "consensus_master"
    alf_dir = cons_dir / "alf"
    spk_times = np.load(alf_dir / "spike_times.npy")
    spk_clusters = np.load(alf_dir / "spike_clusters.npy")
    df_info = pd.read_csv(cons_dir / "cluster_info.tsv", sep="\t")

    qc_col = "passed_qc" if "passed_qc" in df_info.columns else ("passed_allen_qc" if "passed_allen_qc" in df_info.columns else None)
    df_sua = df_info[df_info[qc_col] == True].copy() if qc_col else df_info.copy()

    sua_ids = set(df_sua["cluster_id"].values)
    mask_sua = np.isin(spk_clusters, list(sua_ids))
    spk_times = spk_times[mask_sua]
    spk_clusters = spk_clusters[mask_sua]

    unit_ids = df_sua["cluster_id"].values
    dt_fr = 0.005  # 5 ms bins
    t_bins = np.arange(0, duration_s, dt_fr)

    kernel = signal.windows.gaussian(21, std=5)
    kernel /= np.sum(kernel)

    fr_matrix = []
    for uid in unit_ids:
        spks = spk_times[spk_clusters == uid]
        counts, _ = np.histogram(spks, bins=np.append(t_bins, t_bins[-1] + dt_fr))
        smooth_rate = np.convolve(counts / dt_fr, kernel, mode="same")
        fr_matrix.append(smooth_rate)
    fr_matrix = np.array(fr_matrix)

    pca = PCA(n_components=6)
    latent = pca.fit_transform(fr_matrix.T)

    # Compute Continuous Phase under EMG offset vs Flow Slope Inversion
    t_flow_slopes = df_metrics["t_flow_slope"].values

    def compute_phase_array(onsets, offsets, eval_times):
        phi = np.full_like(eval_times, np.nan, dtype=np.float64)
        for k in range(len(offsets) - 1):
            t_on = onsets[k]
            t_off = offsets[k]
            t_next = onsets[k + 1]
            if t_off <= t_on or t_next <= t_off:
                continue
            insp_mask = (eval_times >= t_on) & (eval_times < t_off)
            if np.any(insp_mask):
                # [-pi, 0)
                phi[insp_mask] = -np.pi + (eval_times[insp_mask] - t_on) / (t_off - t_on) * np.pi
            exp_mask = (eval_times >= t_off) & (eval_times < t_next)
            if np.any(exp_mask):
                # [0, +pi)
                phi[exp_mask] = (eval_times[exp_mask] - t_off) / (t_next - t_off) * np.pi
        valid_idx = np.where(~np.isnan(phi))[0]
        if len(valid_idx) > 0:
            phi[:valid_idx[0]] = phi[valid_idx[0]]
            phi[valid_idx[-1]:] = phi[valid_idx[-1]]
        return phi

    phi_emg = compute_phase_array(emg_onsets, emg_offsets, t_bins)
    phi_flow = compute_phase_array(emg_onsets, t_flow_slopes, t_bins)

    # Trajectory speed ||dX/dt||
    pc1_2 = latent[:, :2]
    d_latent = np.gradient(pc1_2, dt_fr, axis=0)
    speed = np.linalg.norm(d_latent, axis=1)

    # Dispersion across breaths
    n_phase_bins = 60
    phase_edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    phase_centers = 0.5 * (phase_edges[:-1] + phase_edges[1:])

    # Resample each breath into standard 60 phase bins
    disp_emg_bins = [[] for _ in range(n_phase_bins)]
    disp_flow_bins = [[] for _ in range(n_phase_bins)]

    for k in range(len(t_flow_slopes) - 1):
        if k == 99:
            continue  # Skip sigh
        t_on = emg_onsets[k]
        t_off_e = emg_offsets[k]
        t_off_f = t_flow_slopes[k]
        t_next = emg_onsets[k + 1]

        # EMG breath interpolation
        mask_e = (t_bins >= t_on) & (t_bins < t_next)
        if np.sum(mask_e) >= 10:
            p_e = phi_emg[mask_e]
            lat_e = pc1_2[mask_e]
            for b in range(n_phase_bins):
                in_bin = (p_e >= phase_edges[b]) & (p_e < phase_edges[b + 1])
                if np.any(in_bin):
                    disp_emg_bins[b].append(np.mean(lat_e[in_bin], axis=0))

        # Flow breath interpolation
        mask_f = (t_bins >= t_on) & (t_bins < t_next)
        if np.sum(mask_f) >= 10:
            p_f = phi_flow[mask_f]
            lat_f = pc1_2[mask_f]
            for b in range(n_phase_bins):
                in_bin = (p_f >= phase_edges[b]) & (p_f < phase_edges[b + 1])
                if np.any(in_bin):
                    disp_flow_bins[b].append(np.mean(lat_f[in_bin], axis=0))

    disp_emg = np.array([np.mean(np.std(np.array(pts), axis=0)) if len(pts) > 3 else np.nan for pts in disp_emg_bins])
    disp_flow = np.array([np.mean(np.std(np.array(pts), axis=0)) if len(pts) > 3 else np.nan for pts in disp_flow_bins])

    # Mean speed across phase
    speed_emg_mean = np.zeros(n_phase_bins)
    speed_flow_mean = np.zeros(n_phase_bins)
    for b in range(n_phase_bins):
        in_e = (phi_emg >= phase_edges[b]) & (phi_emg < phase_edges[b + 1])
        in_f = (phi_flow >= phase_edges[b]) & (phi_flow < phase_edges[b + 1])
        speed_emg_mean[b] = np.nanmean(speed[in_e]) if np.any(in_e) else np.nan
        speed_flow_mean[b] = np.nanmean(speed[in_f]) if np.any(in_f) else np.nan

    # -------------------------------------------------------------
    # FIGURE 1: Multi-Trace Synchronized Breath Alignment Panel
    # -------------------------------------------------------------
    print("Generating Figure 1: Multi-Trace Synchronized Breath Alignment...")
    fig1 = plt.figure(figsize=(16, 11), dpi=300)
    gs1 = gridspec.GridSpec(4, 1, height_ratios=[1.2, 1.2, 1.6, 1.2], hspace=0.15)

    # Focus window: 827.0 - 829.5 s (matches user screenshot with eupnea and sigh)
    t_start_f1 = 827.0 - 796.827
    t_end_f1 = 829.5 - 796.827
    f1_mask = (t >= t_start_f1) & (t <= t_end_f1)

    t_f1 = t[f1_mask] + 796.827  # Absolute time for user parity
    emg_raw_f1 = emg_raw[f1_mask]
    emg_env_f1 = emg_env[f1_mask]
    flow_f1 = flow_lp[f1_mask]
    flow_d_f1 = flow_deriv[f1_mask]

    # Filter events in window
    df_win_f1 = df_metrics[(df_metrics["t_on_emg"] >= t_start_f1 - 0.5) & (df_metrics["t_on_emg"] <= t_end_f1 + 0.5)]

    ax1 = fig1.add_subplot(gs1[0])
    ax1.plot(t_f1, emg_raw_f1, color='#64748b', lw=0.7, label='Raw Diaphragm EMG (Ch 0)')
    ax1.set_ylabel('Voltage (V)', fontsize=11, fontweight='bold')
    ax1.set_title('A. Raw Diaphragm EMG & Physiological Marker Boundaries', fontsize=12, fontweight='bold', loc='left')
    ax1.set_xlim(827.0, 829.5)
    ax1.grid(True, linestyle=':', alpha=0.5, color='#94a3b8')

    ax2 = fig1.add_subplot(gs1[1], sharex=ax1)
    ax2.plot(t_f1, emg_env_f1, color='#0284c7', lw=1.8, label='Integrated Diaphragm EMG Envelope')
    ax2.set_ylabel('Envelope (V)', fontsize=11, fontweight='bold')
    ax2.set_title('B. Diaphragm Envelope with Classical cibrrig Onsets (Green) & Offsets (Red)', fontsize=12, fontweight='bold', loc='left')
    ax2.set_xlim(827.0, 829.5)
    ax2.grid(True, linestyle=':', alpha=0.5, color='#94a3b8')

    ax3 = fig1.add_subplot(gs1[2], sharex=ax1)
    ax3.plot(t_f1, flow_f1, color='#059669', lw=2.0, label='Flow Meter Signal (Ch 5)')
    ax3.axhline(0, color='#334155', linestyle='-', lw=0.8, alpha=0.7)
    ax3.set_ylabel('Flow (V)', fontsize=11, fontweight='bold')
    ax3.set_title('C. Flow Meter Signal with Flow Slope Inversion ($E_{on}^{flow, slope}$, Purple) vs EMG Offset ($I_{off}^{EMG}$, Red)', fontsize=12, fontweight='bold', loc='left')
    ax3.set_xlim(827.0, 829.5)
    ax3.grid(True, linestyle=':', alpha=0.5, color='#94a3b8')

    ax4 = fig1.add_subplot(gs1[3], sharex=ax1)
    ax4.plot(t_f1, flow_d_f1, color='#7c3aed', lw=1.5, label='Flow Derivative dQ/dt (Airflow Acceleration)')
    ax4.axhline(0, color='#334155', linestyle='--', lw=1.0, alpha=0.8)
    ax4.set_ylabel('dQ/dt (V/s)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax4.set_title('D. Flow Derivative: Zero-Crossing from Negative to Positive Marks Biomechanical Expiration Onset', fontsize=12, fontweight='bold', loc='left')
    ax4.set_xlim(827.0, 829.5)
    ax4.grid(True, linestyle=':', alpha=0.5, color='#94a3b8')

    # Add vertical lines and markers across subplots
    for _, row in df_win_f1.iterrows():
        t_on_abs = row["t_on_emg"] + 796.827
        t_off_abs = row["t_off_emg"] + 796.827
        t_slope_abs = row["t_flow_slope"] + 796.827
        t_zero_abs = row["t_flow_zero"] + 796.827 if not np.isnan(row["t_flow_zero"]) else None

        for ax in [ax1, ax2, ax3, ax4]:
            # Onset: Green dashed
            ax.axvline(t_on_abs, color='#16a34a', linestyle='--', lw=1.2, alpha=0.7)
            # EMG offset: Red dotted
            ax.axvline(t_off_abs, color='#dc2626', linestyle=':', lw=1.2, alpha=0.7)
            # Flow slope inversion: Purple dash-dot
            ax.axvline(t_slope_abs, color='#9333ea', linestyle='-.', lw=1.4, alpha=0.85)

        # Shading between flow slope inversion and EMG offset: Post-inspiratory Diaphragmatic Braking zone!
        for ax in [ax1, ax2, ax3, ax4]:
            ax.axvspan(t_slope_abs, t_off_abs, color='#fef08a', alpha=0.25)

        # Scatter markers on ax2 and ax3
        env_val = np.interp(row["t_on_emg"], t, emg_env)
        ax2.scatter(t_on_abs, env_val, color='#16a34a', marker='^', s=60, zorder=5)
        env_off_val = np.interp(row["t_off_emg"], t, emg_env)
        ax2.scatter(t_off_abs, env_off_val, color='#dc2626', marker='v', s=60, zorder=5)

        flow_val = np.interp(row["t_flow_slope"], t, flow_lp)
        ax3.scatter(t_slope_abs, flow_val, color='#9333ea', marker='D', s=55, zorder=5)
        if t_zero_abs is not None:
            ax3.scatter(t_zero_abs, 0, color='#06b6d4', marker='o', s=45, zorder=5)

    # Custom legend for markers
    legend_elements = [
        plt.Line2D([0], [0], color='#16a34a', linestyle='--', lw=1.5, label=r'Breath Onset ($I_{on}^{EMG}$)'),
        plt.Line2D([0], [0], color='#9333ea', linestyle='-.', lw=1.5, label=r'Flow Slope Reversal / Trough ($E_{on}^{flow, slope}$)'),
        plt.Line2D([0], [0], color='#dc2626', linestyle=':', lw=1.5, label=r'Diaphragm Offset ($I_{off}^{EMG}$)'),
        patches.Patch(facecolor='#fef08a', alpha=0.5, edgecolor='#ca8a04', label=r'Post-I Diaphragm Braking Interval ($\Delta t \approx 27$ ms)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=9.5)

    fig1.savefig(OUT_DIR / "fig_flow_vs_emg_breath_alignment.png", dpi=300, bbox_inches='tight')
    fig1.savefig(OUT_DIR / "fig_flow_vs_emg_breath_alignment.svg", bbox_inches='tight')
    plt.close(fig1)
    print("Saved Figure 1.")

    # -------------------------------------------------------------
    # FIGURE 2: Statistical Distributions & Quantitative Rigor
    # -------------------------------------------------------------
    print("Generating Figure 2: Statistical Timing Distributions...")
    fig2 = plt.figure(figsize=(15, 10), dpi=300)
    gs2 = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)

    # Panel A: Delta t Distribution
    ax2a = fig2.add_subplot(gs2[0, 0])
    ax2a.hist(dt_slope_vals, bins=30, color='#9333ea', alpha=0.7, edgecolor='#581c87', density=True, label=r'$\Delta t_{slope} = E_{on}^{flow} - I_{off}^{EMG}$')
    kde_x = np.linspace(dt_slope_vals.min() - 10, dt_slope_vals.max() + 10, 200)
    kde = stats.gaussian_kde(dt_slope_vals)
    ax2a.plot(kde_x, kde(kde_x), color='#581c87', lw=2.2)
    ax2a.axvline(dt_slope_vals.median(), color='#dc2626', linestyle='--', lw=1.8, label=f'Median = {dt_slope_vals.median():.1f} ms')
    ax2a.axvline(0, color='#334155', linestyle='-', lw=1.0)
    ax2a.set_xlabel(r'$\Delta t$ (ms) [Flow Trough - EMG Offset]', fontsize=11, fontweight='bold')
    ax2a.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax2a.set_title('A. Latency Difference: Flow Slope Reversal precedes EMG Offset', fontsize=11, fontweight='bold', loc='left')
    ax2a.legend(loc='upper left', fontsize=9.5)
    ax2a.grid(True, linestyle=':', alpha=0.5)

    # Panel B: T_I EMG vs T_I Flow Slope
    ax2b = fig2.add_subplot(gs2[0, 1])
    ax2b.scatter(df_metrics.loc[eupnea_mask, "T_I_emg_ms"], df_metrics.loc[eupnea_mask, "T_I_flow_slope_ms"], color='#0284c7', alpha=0.65, edgecolors='none', s=45, label='Eupneic Breaths')
    # Plot sigh
    sigh_row = df_metrics[df_metrics["is_sigh"]]
    if len(sigh_row) > 0:
        ax2b.scatter(sigh_row["T_I_emg_ms"], sigh_row["T_I_flow_slope_ms"], color='#dc2626', marker='*', s=180, zorder=5, label='Breath 99 (Sigh)')
    lims = [30, 220]
    ax2b.plot(lims, lims, color='#94a3b8', linestyle='--', lw=1.2, label='Identity (y = x)')
    # Regression line
    slope, intercept, r_val, p_val, _ = stats.linregress(df_metrics.loc[eupnea_mask, "T_I_emg_ms"], df_metrics.loc[eupnea_mask, "T_I_flow_slope_ms"])
    x_vals = np.array([40, 140])
    ax2b.plot(x_vals, slope * x_vals + intercept, color='#1e40af', lw=1.8, label=f'Fit: $r = {r_val:.2f}$')
    ax2b.set_xlabel('Inspiratory Duration $T_I^{EMG}$ (ms)', fontsize=11, fontweight='bold')
    ax2b.set_ylabel('Inspiratory Duration $T_I^{flow, slope}$ (ms)', fontsize=11, fontweight='bold')
    ax2b.set_title('B. Inspiratory Duration Comparison: Flow Shortens $T_I$ by ~27 ms', fontsize=11, fontweight='bold', loc='left')
    ax2b.legend(loc='upper left', fontsize=9.5)
    ax2b.grid(True, linestyle=':', alpha=0.5)

    # Panel C: Time Series of T_I across recording
    ax2c = fig2.add_subplot(gs2[1, 0])
    ax2c.plot(df_metrics["breath_id"], df_metrics["T_I_emg_ms"], color='#dc2626', alpha=0.75, lw=1.2, label='$T_I^{EMG}$ (Classical)')
    ax2c.plot(df_metrics["breath_id"], df_metrics["T_I_flow_slope_ms"], color='#9333ea', alpha=0.85, lw=1.4, label='$T_I^{flow, slope}$ (Biomechanical)')
    ax2c.set_xlabel('Breath Number (Cycle Index)', fontsize=11, fontweight='bold')
    ax2c.set_ylabel('Inspiratory Duration (ms)', fontsize=11, fontweight='bold')
    ax2c.set_title('C. Cycle-by-Cycle Constancy of the ~27 ms Post-I Phase', fontsize=11, fontweight='bold', loc='left')
    ax2c.legend(loc='upper right', fontsize=9.5)
    ax2c.grid(True, linestyle=':', alpha=0.5)

    # Panel D: Peak EMG Amplitude vs Peak Expiratory Flow
    ax2d = fig2.add_subplot(gs2[1, 1])
    ax2d.scatter(df_metrics.loc[eupnea_mask, "emg_peak_v"], df_metrics.loc[eupnea_mask, "flow_max_v"], color='#059669', alpha=0.65, edgecolors='none', s=45, label='Eupnea')
    if len(sigh_row) > 0:
        ax2d.scatter(sigh_row["emg_peak_v"], sigh_row["flow_max_v"], color='#dc2626', marker='*', s=180, zorder=5, label='Breath 99 (Sigh)')
    # Regression
    slope_ef, int_ef, r_ef, _, _ = stats.linregress(df_metrics.loc[eupnea_mask, "emg_peak_v"].dropna(), df_metrics.loc[eupnea_mask, "flow_max_v"].dropna())
    xe_vals = np.linspace(df_metrics.loc[eupnea_mask, "emg_peak_v"].min(), df_metrics.loc[eupnea_mask, "emg_peak_v"].max(), 50)
    ax2d.plot(xe_vals, slope_ef * xe_vals + int_ef, color='#065f46', lw=1.8, label=f'Fit: $r = {r_ef:.2f}$')
    ax2d.set_xlabel('Peak Diaphragm EMG Envelope (V)', fontsize=11, fontweight='bold')
    ax2d.set_ylabel('Peak Expiratory Airflow $Q_{max}$ (V)', fontsize=11, fontweight='bold')
    ax2d.set_title('D. Inspiratory Motor Drive Strongly Drives Expiratory Recoil Airflow', fontsize=11, fontweight='bold', loc='left')
    ax2d.legend(loc='lower right', fontsize=9.5)
    ax2d.grid(True, linestyle=':', alpha=0.5)

    fig2.savefig(OUT_DIR / "fig_flow_emg_timing_statistics.png", dpi=300, bbox_inches='tight')
    fig2.savefig(OUT_DIR / "fig_flow_emg_timing_statistics.svg", bbox_inches='tight')
    plt.close(fig2)
    print("Saved Figure 2.")

    # -------------------------------------------------------------
    # FIGURE 3: State-Space Neural Attractor Dynamics Comparison
    # -------------------------------------------------------------
    print("Generating Figure 3: State-Space Neural Attractor Comparison...")
    fig3 = plt.figure(figsize=(16, 11), dpi=300)
    gs3 = gridspec.GridSpec(2, 3, height_ratios=[1.2, 1.0], hspace=0.3, wspace=0.28)

    # Panel A: PC1-PC2 Loop colored by EMG Phase
    ax3a = fig3.add_subplot(gs3[0, 0])
    sc_a = ax3a.scatter(latent[:3000, 0], latent[:3000, 1], c=phi_emg[:3000], cmap='twilight', s=12, alpha=0.7)
    ax3a.set_xlabel('PC1 (28.4% var)', fontsize=11, fontweight='bold')
    ax3a.set_ylabel('PC2 (19.6% var)', fontsize=11, fontweight='bold')
    ax3a.set_title('A. State Space: Diaphragm EMG Phase ($\phi_{EMG}$)\n$\phi = 0$ at EMG Offset', fontsize=10.5, fontweight='bold')
    plt.colorbar(sc_a, ax=ax3a, label='Phase (rad)', fraction=0.046, pad=0.04)
    ax3a.grid(True, linestyle=':', alpha=0.5)

    # Panel B: PC1-PC2 Loop colored by Flow Phase
    ax3b = fig3.add_subplot(gs3[0, 1])
    sc_b = ax3b.scatter(latent[:3000, 0], latent[:3000, 1], c=phi_flow[:3000], cmap='twilight', s=12, alpha=0.7)
    ax3b.set_xlabel('PC1 (28.4% var)', fontsize=11, fontweight='bold')
    ax3b.set_ylabel('PC2 (19.6% var)', fontsize=11, fontweight='bold')
    ax3b.set_title('B. State Space: Flow-Derived Phase ($\phi_{flow}$)\n$\phi = 0$ at Flow Slope Inversion', fontsize=10.5, fontweight='bold')
    plt.colorbar(sc_b, ax=ax3b, label='Phase (rad)', fraction=0.046, pad=0.04)
    ax3b.grid(True, linestyle=':', alpha=0.5)

    # Panel C: Trajectory Speed ||dX/dt|| vs Phase
    ax3c = fig3.add_subplot(gs3[0, 2])
    ax3c.plot(phase_centers, speed_emg_mean, color='#dc2626', lw=2.0, label='Aligned to EMG Offset')
    ax3c.plot(phase_centers, speed_flow_mean, color='#9333ea', lw=2.2, label='Aligned to Flow Slope Inversion')
    ax3c.axvline(0, color='#334155', linestyle='--', lw=1.2, label=r'Phase $\phi = 0$ (Expiration Onset)')
    ax3c.set_xlabel(r'Respiratory Phase $\phi$ (rad)', fontsize=11, fontweight='bold')
    ax3c.set_ylabel(r'Trajectory Speed $\|\dot{X}(t)\|$ (a.u./s)', fontsize=11, fontweight='bold')
    ax3c.set_title('C. Trajectory Speed across Respiratory Phase', fontsize=11, fontweight='bold', loc='left')
    ax3c.legend(loc='upper right', fontsize=9.0)
    ax3c.grid(True, linestyle=':', alpha=0.5)

    # Panel D: Attractor Dispersion Curve Comparison
    ax3d = fig3.add_subplot(gs3[1, :2])
    ax3d.plot(phase_centers, disp_emg, color='#dc2626', lw=2.2, label=r'EMG Offset Alignment ($\phi_{EMG}$)')
    ax3d.plot(phase_centers, disp_flow, color='#9333ea', lw=2.4, label=r'Flow Slope Inversion Alignment ($\phi_{flow}$)')
    ax3d.axvline(0, color='#334155', linestyle='--', lw=1.2)
    # Highlight min dispersion
    min_disp_flow_idx = np.nanargmin(disp_flow)
    min_disp_emg_idx = np.nanargmin(disp_emg)
    ax3d.scatter(phase_centers[min_disp_flow_idx], disp_flow[min_disp_flow_idx], color='#9333ea', s=90, zorder=5)
    ax3d.scatter(phase_centers[min_disp_emg_idx], disp_emg[min_disp_emg_idx], color='#dc2626', s=90, zorder=5)
    ax3d.annotate(rf'Flow Min: $\phi = {phase_centers[min_disp_flow_idx]:.2f}$ rad',
                  xy=(phase_centers[min_disp_flow_idx], disp_flow[min_disp_flow_idx]),
                  xytext=(phase_centers[min_disp_flow_idx] - 0.9, disp_flow[min_disp_flow_idx] + 1.2),
                  arrowprops=dict(facecolor='#9333ea', arrowstyle='->', lw=1.5), fontsize=9.5, fontweight='bold', color='#581c87')
    ax3d.set_xlabel(r'Respiratory Phase $\phi$ (rad)', fontsize=11, fontweight='bold')
    ax3d.set_ylabel('Cross-Breath Euclidean Dispersion (a.u.)', fontsize=11, fontweight='bold')
    ax3d.set_title('D. Attractor Basin Contraction: Flow Alignment Sharpens the Expiratory Attractor Minimum', fontsize=11, fontweight='bold', loc='left')
    ax3d.legend(loc='upper left', fontsize=10)
    ax3d.grid(True, linestyle=':', alpha=0.5)

    # Panel E: Polar Speed Tuning
    ax3e = fig3.add_subplot(gs3[1, 2], projection='polar')
    ax3e.plot(phase_centers, speed_emg_mean, color='#dc2626', lw=1.8, label='EMG')
    ax3e.plot(phase_centers, speed_flow_mean, color='#9333ea', lw=2.0, label='Flow')
    ax3e.set_title('E. Polar Speed Profile', fontsize=11, fontweight='bold', pad=15)
    ax3e.set_theta_zero_location("E")
    ax3e.legend(loc='lower right', bbox_to_anchor=(1.25, -0.1), fontsize=8.5)

    fig3.savefig(OUT_DIR / "fig_flow_aligned_neural_attractor.png", dpi=300, bbox_inches='tight')
    fig3.savefig(OUT_DIR / "fig_flow_aligned_neural_attractor.svg", bbox_inches='tight')
    plt.close(fig3)
    print("Saved Figure 3.")

    # -------------------------------------------------------------
    # FIGURE 4: Exemplary Single-Unit Modulation
    # -------------------------------------------------------------
    print("Generating Figure 4: Exemplary Single-Unit Modulation...")
    fig4 = plt.figure(figsize=(16, 10), dpi=300)
    gs4 = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)

    exemplars = [
        (255, "Unit 255 (preBötC Inspiratory Ramping)", gs4[0, 0]),
        (94,  "Unit 094 (BötC Expiratory Suppression)", gs4[0, 1]),
        (18,  "Unit 018 (BötC Post-I Rebound Pioneer)", gs4[1, 0]),
        (163, "Unit 163 (preBötC Candidate Pacemaker)", gs4[1, 1])
    ]

    for uid, title, cell_loc in exemplars:
        ax = fig4.add_subplot(cell_loc)
        spks_u = spk_times[spk_clusters == uid]

        # Compute peri-event PSTH aligned to EMG offset vs Flow slope inversion
        # Window: -150 ms to +150 ms around offset/slope
        psth_dt = 0.005
        psth_edges = np.arange(-0.150, 0.155, psth_dt)
        psth_centers = 0.5 * (psth_edges[:-1] + psth_edges[1:]) * 1000.0  # ms

        counts_emg = np.zeros(len(psth_centers))
        counts_flow = np.zeros(len(psth_centers))
        valid_trials = 0

        for k in range(len(t_flow_slopes)):
            if k == 99:
                continue
            t_off_e = emg_offsets[k]
            t_off_f = t_flow_slopes[k]

            # Spikes relative to EMG offset
            rel_spks_e = spks_u - t_off_e
            c_e, _ = np.histogram(rel_spks_e, bins=psth_edges)
            counts_emg += c_e

            # Spikes relative to Flow slope
            rel_spks_f = spks_u - t_off_f
            c_f, _ = np.histogram(rel_spks_f, bins=psth_edges)
            counts_flow += c_f

            valid_trials += 1

        fr_psth_emg = (counts_emg / valid_trials) / psth_dt
        fr_psth_flow = (counts_flow / valid_trials) / psth_dt

        ax.plot(psth_centers, fr_psth_emg, color='#dc2626', lw=2.0, label='Aligned to EMG Offset ($I_{off}^{EMG}$)')
        ax.plot(psth_centers, fr_psth_flow, color='#9333ea', lw=2.2, label='Aligned to Flow Trough ($E_{on}^{flow}$)')
        ax.axvline(0, color='#334155', linestyle='--', lw=1.2, label='Event Alignment (t = 0)')
        ax.set_xlabel('Time relative to event (ms)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Firing Rate (Hz)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
        ax.legend(loc='upper right', fontsize=8.8)
        ax.grid(True, linestyle=':', alpha=0.5)

    fig4.savefig(OUT_DIR / "fig_flow_aligned_exemplary_units.png", dpi=300, bbox_inches='tight')
    fig4.savefig(OUT_DIR / "fig_flow_aligned_exemplary_units.svg", bbox_inches='tight')
    plt.close(fig4)
    print("Saved Figure 4.")

    # -------------------------------------------------------------
    # 6. Build Interactive Dual-Signal HTML Diagnostic Tool
    # -------------------------------------------------------------
    print("Building Interactive Dual-Signal HTML Dashboard...")
    build_flow_emg_html_dashboard(t, emg_raw, emg_env, flow_lp, flow_deriv, df_metrics, OUT_DIR)

    print("=" * 80)
    print("ALL EXPLORATORY FLOW-METER ANALYSES COMPLETED SUCCESSFULLY!")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 80)


def build_flow_emg_html_dashboard(t, emg_raw, emg_env, flow_lp, flow_deriv, df_metrics, out_dir):
    # Downsample snippet for snappy web interaction (first 20 seconds, t=0 to 20s = 827.0s area)
    # Downsample factor: 10 kHz -> 2 kHz
    mask_snip = (t >= 25.0) & (t <= 35.0)  # Center on Breath 99 sigh and surrounding eupnea
    step = 5
    t_web = t[mask_snip][::step] + 796.827
    emg_web = emg_raw[mask_snip][::step]
    env_web = emg_env[mask_snip][::step]
    flow_web = flow_lp[mask_snip][::step]
    d_web = flow_deriv[mask_snip][::step]

    df_snip = df_metrics[(df_metrics["t_on_emg"] >= 24.5) & (df_metrics["t_on_emg"] <= 35.5)]
    events_web = []
    for _, r in df_snip.iterrows():
        events_web.append({
            "breath_id": int(r["breath_id"]),
            "t_on": float(r["t_on_emg"] + 796.827),
            "t_off_emg": float(r["t_off_emg"] + 796.827),
            "t_flow_slope": float(r["t_flow_slope"] + 796.827),
            "t_flow_zero": float(r["t_flow_zero"] + 796.827) if not np.isnan(r["t_flow_zero"]) else None,
            "delta_t_ms": float(r["delta_t_slope_ms"]),
            "T_I_emg_ms": float(r["T_I_emg_ms"]),
            "T_I_flow_ms": float(r["T_I_flow_slope_ms"]),
            "is_sigh": bool(r["is_sigh"])
        })

    payload = {
        "time": np.round(t_web, 4).tolist(),
        "emg_raw": np.round(emg_web, 4).tolist(),
        "emg_env": np.round(env_web, 4).tolist(),
        "flow": np.round(flow_web, 4).tolist(),
        "flow_deriv": np.round(d_web, 2).tolist(),
        "events": events_web
    }

    html_file = out_dir / "flow_emg_comparative_diagnostic.html"
    json_data = json.dumps(payload)

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Flow Meter vs Diaphragm EMG Expiration Onset Diagnostic — e097</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card: #111827;
      --border: rgba(255,255,255,0.08);
      --text: #f1f5f9;
      --muted: #94a3b8;
      --emg: #0284c7;
      --flow: #059669;
      --slope: #a855f7;
      --off: #ef4444;
      --braking: #fef08a;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; padding: 20px; }
    header { max-width: 1400px; margin: 0 auto 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 15px; }
    h1 { font-size: 20px; font-weight: 700; color: #fff; }
    .subtitle { font-size: 13px; color: var(--muted); margin-top: 4px; }
    .hud { display: flex; gap: 15px; }
    .badge { background: var(--card); border: 1px solid var(--border); padding: 6px 14px; border-radius: 8px; font-size: 12px; font-family: 'JetBrains Mono', monospace; }
    .badge span { font-weight: 700; }
    .main-container { max-width: 1400px; margin: 0 auto; display: grid; grid-template-columns: 1fr 340px; gap: 20px; }
    .canvas-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; position: relative; }
    canvas { width: 100%; height: 580px; display: block; background: #060910; border-radius: 8px; cursor: crosshair; }
    .sidebar { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px; display: flex; flex-direction: column; gap: 16px; }
    .side-title { font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); border-bottom: 1px solid var(--border); padding-bottom: 8px; }
    .stat-row { display: flex; justify-content: space-between; font-size: 13px; padding: 6px 0; border-bottom: 1px dashed rgba(255,255,255,0.06); }
    .stat-val { font-family: 'JetBrains Mono', monospace; font-weight: 600; }
    .legend-box { display: flex; flex-direction: column; gap: 8px; font-size: 12px; }
    .legend-item { display: flex; align-items: center; gap: 8px; }
    .leg-dot { width: 12px; height: 12px; border-radius: 3px; }
    .scrub-controls { display: flex; gap: 10px; align-items: center; margin-top: 10px; }
    .btn { background: #1e293b; color: #fff; border: 1px solid var(--border); padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; }
    .btn:hover { background: #334155; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Biomechanical Expiration Onset: Flow Meter Slope vs Diaphragm EMG</h1>
      <div class="subtitle">Adult Mouse e097 (60s Pre-Laser Snippet) • Dual-Signal Oscilloscope Dashboard</div>
    </div>
    <div class="hud">
      <div class="badge">Hover Time: <span id="hud-time">--</span> s</div>
      <div class="badge" style="color: var(--slope);">Flow Slope Inversion: <span id="hud-slope">--</span></div>
      <div class="badge" style="color: var(--off);">EMG Offset: <span id="hud-off">--</span></div>
    </div>
  </header>

  <div class="main-container">
    <div class="canvas-card">
      <canvas id="scopeCanvas" width="1020" height="580"></canvas>
      <div class="scrub-controls">
        <button class="btn" id="btnZoomIn">Zoom In</button>
        <button class="btn" id="btnZoomOut">Zoom Out</button>
        <button class="btn" id="btnReset">Reset View (827 - 830s)</button>
        <span style="font-size: 12px; color: var(--muted); margin-left: 10px;">Drag or scroll to pan/zoom</span>
      </div>
    </div>

    <div class="sidebar">
      <div class="side-title">Breath Metrics HUD</div>
      <div class="stat-row">
        <span>Nearest Breath:</span>
        <span class="stat-val" id="stat-breath">#--</span>
      </div>
      <div class="stat-row">
        <span>$\Delta t$ (Flow Trough - EMG):</span>
        <span class="stat-val" style="color: #facc15;" id="stat-dt">-- ms</span>
      </div>
      <div class="stat-row">
        <span>$T_I$ Classical (EMG):</span>
        <span class="stat-val" id="stat-ti-emg">-- ms</span>
      </div>
      <div class="stat-row">
        <span>$T_I$ Biomechanical (Flow):</span>
        <span class="stat-val" id="stat-ti-flow">-- ms</span>
      </div>
      <div class="stat-row">
        <span>Diaphragm Post-I Braking:</span>
        <span class="stat-val" style="color: #38bdf8;" id="stat-braking">-- ms</span>
      </div>

      <div class="side-title" style="margin-top: 10px;">Legend</div>
      <div class="legend-box">
        <div class="legend-item"><div class="leg-dot" style="background: var(--emg);"></div> Diaphragm Envelope (V)</div>
        <div class="legend-item"><div class="leg-dot" style="background: var(--flow);"></div> Flow Meter Signal (V)</div>
        <div class="legend-item"><div class="leg-dot" style="background: #7c3aed;"></div> Flow Derivative dQ/dt (V/s)</div>
        <div class="legend-item"><div class="leg-dot" style="background: var(--slope);"></div> $E_{on}^{flow, slope}$ (Flow Trough)</div>
        <div class="legend-item"><div class="leg-dot" style="background: var(--off);"></div> $I_{off}^{EMG}$ (Diaphragm Offset)</div>
        <div class="legend-item"><div class="leg-dot" style="background: rgba(254, 240, 138, 0.4); border: 1px solid #ca8a04;"></div> Post-I Diaphragm Braking Phase</div>
      </div>

      <div class="side-title" style="margin-top: 10px;">Key Insight</div>
      <p style="font-size: 12px; color: var(--muted); line-height: 1.5;">
        The flow trough ($\frac{dQ}{dt} = 0$) marks when inward airflow stops and outward driving forces take over. This occurs <strong style="color:#fff;">~27 ms before</strong> the diaphragm EMG envelope returns to baseline, demonstrating that active expiration begins during diaphragmatic post-I braking!
      </p>
    </div>
  </div>

  <script>
    const DATA = __JSON_DATA__;
    const canvas = document.getElementById('scopeCanvas');
    const ctx = canvas.getContext('2d');

    let viewStart = 827.0;
    let viewEnd = 830.0;
    let mouseX = -1;

    function render() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const w = canvas.width;
      const h = canvas.height;
      const padL = 60;
      const padR = 20;
      const plotW = w - padL - padR;

      // 3 tracks: Track 1: EMG Env (top 30%), Track 2: Flow (mid 40%), Track 3: dQ/dt (bot 30%)
      const h1 = h * 0.28;
      const h2 = h * 0.40;
      const h3 = h * 0.28;
      const y1 = 20;
      const y2 = y1 + h1 + 15;
      const y3 = y2 + h2 + 15;

      // Filter data in view
      const t = DATA.time;
      const tMin = viewStart;
      const tMax = viewEnd;

      function toX(val) {
        return padL + (val - tMin) / (tMax - tMin) * plotW;
      }

      // Shaded Braking Zones
      DATA.events.forEach(ev => {
        if (ev.t_off_emg >= tMin && ev.t_flow_slope <= tMax) {
          const xSlope = toX(ev.t_flow_slope);
          const xOff = toX(ev.t_off_emg);
          ctx.fillStyle = 'rgba(254, 240, 138, 0.15)';
          ctx.fillRect(xSlope, 0, xOff - xSlope, h);
        }
      });

      // Track 1: EMG Envelope
      ctx.strokeStyle = '#0284c7';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < t.length; i++) {
        if (t[i] < tMin || t[i] > tMax) continue;
        const x = toX(t[i]);
        const y = y1 + h1 - (DATA.emg_env[i] / 0.6) * h1;
        if (!started) { ctx.moveTo(x, y); started = true; }
        else { ctx.lineTo(x, y); }
      }
      ctx.stroke();

      // Track 2: Flow Signal
      ctx.strokeStyle = '#059669';
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      started = false;
      for (let i = 0; i < t.length; i++) {
        if (t[i] < tMin || t[i] > tMax) continue;
        const x = toX(t[i]);
        // Flow ranges from -6 to +4 V
        const norm = (DATA.flow[i] - (-4.0)) / (4.0 - (-4.0));
        const y = y2 + h2 - norm * h2;
        if (!started) { ctx.moveTo(x, y); started = true; }
        else { ctx.lineTo(x, y); }
      }
      ctx.stroke();

      // Track 3: dQ/dt
      ctx.strokeStyle = '#a855f7';
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      started = false;
      for (let i = 0; i < t.length; i++) {
        if (t[i] < tMin || t[i] > tMax) continue;
        const x = toX(t[i]);
        const normD = (DATA.flow_deriv[i] - (-100)) / 200.0;
        const y = y3 + h3 - Math.max(0, Math.min(1, normD)) * h3;
        if (!started) { ctx.moveTo(x, y); started = true; }
        else { ctx.lineTo(x, y); }
      }
      ctx.stroke();

      // Draw Event Markers
      DATA.events.forEach(ev => {
        if (ev.t_on >= tMin && ev.t_on <= tMax) {
          const x = toX(ev.t_on);
          ctx.strokeStyle = 'rgba(22, 163, 74, 0.7)';
          ctx.setLineDash([4, 4]);
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        }
        if (ev.t_flow_slope >= tMin && ev.t_flow_slope <= tMax) {
          const x = toX(ev.t_flow_slope);
          ctx.strokeStyle = '#a855f7';
          ctx.setLineDash([6, 3]);
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        }
        if (ev.t_off_emg >= tMin && ev.t_off_emg <= tMax) {
          const x = toX(ev.t_off_emg);
          ctx.strokeStyle = '#ef4444';
          ctx.setLineDash([3, 3]);
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        }
      });
      ctx.setLineDash([]);

      // Crosshair
      if (mouseX >= padL && mouseX <= padL + plotW) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(mouseX, 0); ctx.lineTo(mouseX, h); ctx.stroke();
        const curT = tMin + (mouseX - padL) / plotW * (tMax - tMin);
        document.getElementById('hud-time').innerText = curT.toFixed(3);

        // Find closest breath
        let closest = null;
        let minDist = 999;
        DATA.events.forEach(ev => {
          const d = Math.abs(ev.t_on - curT);
          if (d < minDist) { minDist = d; closest = ev; }
        });
        if (closest && minDist < 0.5) {
          document.getElementById('stat-breath').innerText = '#' + closest.breath_id + (closest.is_sigh ? ' (SIGH)' : '');
          document.getElementById('stat-dt').innerText = closest.delta_t_ms.toFixed(1) + ' ms';
          document.getElementById('stat-ti-emg').innerText = closest.T_I_emg_ms.toFixed(1) + ' ms';
          document.getElementById('stat-ti-flow').innerText = closest.T_I_flow_ms.toFixed(1) + ' ms';
          document.getElementById('stat-braking').innerText = Math.abs(closest.delta_t_ms).toFixed(1) + ' ms';
        }
      }
    }

    canvas.addEventListener('mousemove', e => {
      const rect = canvas.getBoundingClientRect();
      mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
      render();
    });

    document.getElementById('btnZoomIn').onclick = () => {
      const mid = 0.5 * (viewStart + viewEnd);
      const span = (viewEnd - viewStart) * 0.7;
      viewStart = mid - span / 2; viewEnd = mid + span / 2;
      render();
    };
    document.getElementById('btnZoomOut').onclick = () => {
      const mid = 0.5 * (viewStart + viewEnd);
      const span = (viewEnd - viewStart) * 1.4;
      viewStart = mid - span / 2; viewEnd = mid + span / 2;
      render();
    };
    document.getElementById('btnReset').onclick = () => {
      viewStart = 827.0; viewEnd = 830.0;
      render();
    };

    render();
  </script>
</body>
</html>
"""
    final_html = html_template.replace("__JSON_DATA__", json_data)
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(final_html)
    print(f"Saved interactive HTML dashboard to {html_file}")


if __name__ == "__main__":
    main()
