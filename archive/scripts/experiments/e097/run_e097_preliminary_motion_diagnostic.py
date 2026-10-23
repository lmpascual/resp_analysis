#!/usr/bin/env python3
"""
Preliminary Movement & Probe-Wide Firing Rate Diagnostic for e097 (29m - 36m):
- 1740s to 2160s (420 seconds total)
- Physiological epoch segmentation validated by user:
    1. Baseline: 29.0 - 30.0 min (1740s - 1800s)
    2. Anoxia Challenge 1: 30.0 - 30.6 min (1800s - 1836s)
    3. Anoxia Challenge 2 (Apnea/HR nadir): 30.6 - 31.0 min (1836s - 1860s)
    4. Early Gasping (Surge): 31.0 - 32.5 min (1860s - 1950s)
    5. Late Gasping: 32.5 - 34.5 min (1950s - 2070s)
    6. Active Expiration Shift: 34.5 - 35.6 min (2070s - 2136s)
    7. Recovery: 35.6 - 36.0 min (2136s - 2160s)
- Dual-layer visualization:
    * Spike-band RMS energy heatmap (300-6000 Hz) across depth vs time
    * Localized peak scatter (y vs t) + estimated probe drift displacement Δy(t)
    * Synchronized Diaphragm EMG, Flow, and ECG / Heart Rate
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Ensure robust stdout UTF-8 encoding on Windows
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from src.pipeline.io.io_spikeglx import read_spikeglx_ap_safe

def main():
    t_start_global = time.time()
    out_dir = REPO_ROOT / "outputs" / "e097"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    t0_s = 1740.0
    t1_s = 2160.0
    dur_s = t1_s - t0_s # 420s

    subwindows = [
        {"id": "win1_baseline", "name": "Baseline (Pre-Anoxia)", "t0": 1740.0, "t1": 1800.0, "hex": "#38bdf8", "rgba": "rgba(56, 189, 248, 0.2)"},
        {"id": "win2_anox1", "name": "Anoxic Challenge 1 (Gas Switch)", "t0": 1800.0, "t1": 1836.0, "hex": "#f43f5e", "rgba": "rgba(244, 63, 94, 0.2)"},
        {"id": "win3_anox2", "name": "Anoxic Challenge 2 (Apnea / HR Nadir)", "t0": 1836.0, "t1": 1860.0, "hex": "#a855f7", "rgba": "rgba(168, 85, 247, 0.2)"},
        {"id": "win4_gasp_early", "name": "Early Gasping (Autoresuscitation Surge)", "t0": 1860.0, "t1": 1950.0, "hex": "#eab308", "rgba": "rgba(234, 179, 8, 0.2)"},
        {"id": "win5_gasp_late", "name": "Late Gasping (Tachypneic Transition)", "t0": 1950.0, "t1": 2070.0, "hex": "#f97316", "rgba": "rgba(249, 115, 22, 0.2)"},
        {"id": "win6_flow_shift", "name": "Active Expiration Shift", "t0": 2070.0, "t1": 2136.0, "hex": "#10b981", "rgba": "rgba(16, 185, 129, 0.2)"},
        {"id": "win7_recovery", "name": "Post-Challenge Recovery", "t0": 2136.0, "t1": 2160.0, "hex": "#0ea5e9", "rgba": "rgba(14, 165, 233, 0.2)"},
    ]

    print("=================================================================")
    print(f"E097 PRELIMINARY MOVEMENT & FIRING RATE DIAGNOSTIC ({t0_s/60:.1f}m - {t1_s/60:.1f}m)")
    print("=================================================================")

    # -------------------------------------------------------------
    # 1. LOAD NIDQ SIGNALS (EMG, ECG, FLOW)
    # -------------------------------------------------------------
    print("\n[Stage 1/4] Loading NIDQ physiological signals (10 kHz)...")
    nidq_bin = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
    fs_nidq = 9999.955307
    n_nidq_chans = 9
    volts_per_count = 10.0 / 32768.0

    s0_nidq = int(t0_s * fs_nidq)
    s1_nidq = int(t1_s * fs_nidq)
    n_pts_nidq = s1_nidq - s0_nidq

    raw_nidq = np.memmap(
        nidq_bin,
        dtype=np.int16,
        mode='r',
        offset=s0_nidq * n_nidq_chans * 2,
        shape=(n_pts_nidq, n_nidq_chans)
    )

    t_nidq = np.linspace(t0_s, t1_s, n_pts_nidq, endpoint=False)
    emg_raw = raw_nidq[:, 0].astype(np.float32) * volts_per_count
    ecg_raw = raw_nidq[:, 2].astype(np.float32) * volts_per_count
    flow_raw = raw_nidq[:, 5].astype(np.float32) * volts_per_count

    # Filter EMG (300-3000 Hz bandpass, rectification, envelope)
    sos_emg = signal.butter(4, [300.0 / (0.5 * fs_nidq), 3000.0 / (0.5 * fs_nidq)], btype='bandpass', output='sos')
    emg_filt = signal.sosfiltfilt(sos_emg, emg_raw)
    emg_rect = np.abs(emg_filt)
    env_win = int(0.05 * fs_nidq)
    emg_env = np.convolve(emg_rect, np.ones(env_win) / env_win, mode='same')

    # -------------------------------------------------------------
    # 1B. ROBUST ADAPTIVE ROLLING-WINDOW ECG QRS DETECTOR
    # -------------------------------------------------------------
    print("  Running adaptive rolling-window ECG QRS detector across full recording...")
    sos_ecg = signal.butter(3, [12.0 / (0.5 * fs_nidq), 80.0 / (0.5 * fs_nidq)], btype='bandpass', output='sos')
    ecg_filt = signal.sosfiltfilt(sos_ecg, ecg_raw)
    diff_ecg = np.diff(ecg_filt, prepend=ecg_filt[0]) * fs_nidq
    int_win = int(0.025 * fs_nidq)
    energy_ecg = np.convolve(diff_ecg**2, np.ones(int_win) / int_win, mode='same')

    win_len = int(1.5 * fs_nidq)
    step = int(0.75 * fs_nidq)
    n_steps = (len(energy_ecg) - win_len) // step + 1
    detected_indices = set()
    min_dist = int(0.065 * fs_nidq) # 65 ms minimum RR interval (~920 bpm ceiling)

    for s in range(n_steps):
        chunk = energy_ecg[s * step : s * step + win_len]
        p95 = np.percentile(chunk, 95)
        p20 = np.percentile(chunk, 20)
        thresh = p20 + 0.22 * (p95 - p20)
        pks, _ = signal.find_peaks(chunk, height=thresh, distance=min_dist)
        for p in pks:
            detected_indices.add(s * step + p)

    all_pks = np.array(sorted(list(detected_indices)))
    diffs = np.diff(all_pks)
    too_close = np.where(diffs < min_dist)[0]
    if len(too_close) > 0:
        keep = np.ones(len(all_pks), dtype=bool)
        for idx in too_close:
            if energy_ecg[all_pks[idx]] < energy_ecg[all_pks[idx + 1]]:
                keep[idx] = False
            else:
                keep[idx + 1] = False
        all_pks = all_pks[keep]

    ecg_times = t_nidq[all_pks]
    rr_intervals = np.diff(ecg_times)
    hr_bpm_inst = 60.0 / rr_intervals
    valid_hr = (hr_bpm_inst >= 80) & (hr_bpm_inst <= 850)
    hr_times = ecg_times[1:][valid_hr]
    hr_bpm = hr_bpm_inst[valid_hr]

    print(f"  NIDQ loaded: {len(t_nidq):,} samples. ECG detected {len(all_pks):,} beats (mean HR {np.mean(hr_bpm):.1f} bpm).")

    # -------------------------------------------------------------
    # 2. LOAD & PROCESS NEUROPIXELS AP STREAM (384 CHANNELS)
    # -------------------------------------------------------------
    print("\n[Stage 2/4] Processing Neuropixels AP stream across all 384 channels...")
    raw_dir = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_imec0")
    rec = read_spikeglx_ap_safe(raw_dir)
    fs_ap = rec.get_sampling_frequency()
    n_chans = rec.get_num_channels()

    s0_ap = int(t0_s * fs_ap)
    s1_ap = int(t1_s * fs_ap)
    rec_slice = rec.frame_slice(s0_ap, s1_ap)

    # Probe geometry: extract y coordinates for all channels
    geom = rec.get_channel_locations()
    y_coords = geom[:, 1]
    sorted_ch_idx = np.argsort(y_coords) # from bottom tip (ventral) to top (dorsal)
    sorted_y = y_coords[sorted_ch_idx]
    depth_min, depth_max = sorted_y.min(), sorted_y.max()
    print(f"  Probe channels: {n_chans}, Depth span: {depth_min:.1f} um to {depth_max:.1f} um (span {depth_max - depth_min:.1f} um)")

    # Compute Spike-Band RMS & MUA Threshold-Crossing Rate in 0.5s chunks
    dt_bin = 0.5 # 0.5s temporal resolution
    n_bins = int(dur_s / dt_bin)
    bin_edges = np.linspace(t0_s, t1_s, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    rms_matrix = np.zeros((n_chans, n_bins), dtype=np.float32)
    mua_rate_matrix = np.zeros((n_chans, n_bins), dtype=np.float32)

    cache_file = out_dir / "cache_e097_preliminary_diagnostics.npz"
    if cache_file.exists():
        print(f"  [Cache] Loading precomputed AP metrics from {cache_file}...")
        npz = np.load(cache_file)
        rms_matrix = npz["rms_matrix"]
        mua_rate_matrix = npz["mua_rate_matrix"]
        all_peak_times = npz["all_peak_times"]
        all_peak_depths = npz["all_peak_depths"]
        all_peak_amps = npz["all_peak_amps"]
        y_coords = npz["y_coords"]
        sorted_ch_idx = npz["sorted_ch_idx"]
        sorted_y = npz["sorted_y"]
        print(f"  [Cache] Loaded {len(all_peak_times):,} peaks and RMS matrix {rms_matrix.shape}!")
    else:
        # Process in 10-second processing blocks to optimize memory and speed
        block_dur = 10.0
        n_blocks = int(np.ceil(dur_s / block_dur))
        print(f"  Computing RMS & MUA matrices across {n_blocks} streaming blocks ({block_dur}s each)...")

        # Design 300-6000 Hz Butterworth bandpass filter
        sos_ap = signal.butter(3, [300.0 / (0.5 * fs_ap), 6000.0 / (0.5 * fs_ap)], btype='bandpass', output='sos')

        # Detect peaks for scatter
        all_peak_times = []
        all_peak_depths = []
        all_peak_amps = []

        pts_per_bin = int(dt_bin * fs_ap)

        for b in range(n_blocks):
            tb0 = b * block_dur
            tb1 = min(dur_s, tb0 + block_dur)
            if tb1 <= tb0:
                continue
            sb0 = int(tb0 * fs_ap)
            sb1 = int(tb1 * fs_ap)

            # Get raw traces for block (samples x channels)
            traces = rec_slice.get_traces(start_frame=sb0, end_frame=sb1, return_scaled=True)
            # Median reference per timepoint across channels (rapid CAR)
            med = np.median(traces, axis=1, keepdims=True)
            traces_car = traces - med
            # Bandpass filter
            filt = signal.sosfiltfilt(sos_ap, traces_car, axis=0)

            # Estimate channel noise (MAD)
            mad = np.median(np.abs(filt), axis=0) / 0.6745
            mad[mad == 0] = 1.0

            # Sub-bin loop for RMS and threshold crossings
            b_bins = int(np.ceil((tb1 - tb0) / dt_bin))
            for sub_b in range(b_bins):
                global_bin_idx = int(tb0 / dt_bin) + sub_b
                if global_bin_idx >= n_bins:
                    break
                i0 = sub_b * pts_per_bin
                i1 = min(filt.shape[0], i0 + pts_per_bin)
                if i1 <= i0:
                    continue
                chunk = filt[i0:i1, :]
                # RMS
                rms = np.sqrt(np.mean(chunk**2, axis=0))
                rms_matrix[:, global_bin_idx] = rms

                # MUA threshold crossings (< -4.5 MAD)
                crossings = np.sum(chunk < -4.5 * mad, axis=0)
                dur_sub = (i1 - i0) / fs_ap
                mua_rate_matrix[:, global_bin_idx] = crossings / dur_sub

            # Sparse peak detection for scatter
            for ch in range(0, n_chans, 2): # sample alternate channels
                ch_y = y_coords[ch]
                sig = filt[:, ch]
                ch_thresh = -5.5 * mad[ch]
                pks, props = signal.find_peaks(-sig, height=-ch_thresh, distance=int(0.0015 * fs_ap))
                if len(pks) > 0:
                    if len(pks) > 150:
                        sel = np.random.choice(len(pks), 150, replace=False)
                        pks = pks[sel]
                    pk_times = t0_s + tb0 + pks / fs_ap
                    pk_amps = np.abs(sig[pks])
                    all_peak_times.extend(pk_times)
                    all_peak_depths.extend([ch_y] * len(pks))
                    all_peak_amps.extend(pk_amps)

            if (b + 1) % 10 == 0 or b == n_blocks - 1:
                print(f"    Processed block {b+1}/{n_blocks} ({tb1:.1f}s / {dur_s:.1f}s)")

        all_peak_times = np.array(all_peak_times)
        all_peak_depths = np.array(all_peak_depths)
        all_peak_amps = np.array(all_peak_amps)
        print(f"  Total detected diagnostic peaks: {len(all_peak_times):,}")

        # Save cache
        print(f"  Saving cached AP metrics to {cache_file}...")
        np.savez_compressed(
            cache_file,
            rms_matrix=rms_matrix,
            mua_rate_matrix=mua_rate_matrix,
            all_peak_times=all_peak_times,
            all_peak_depths=all_peak_depths,
            all_peak_amps=all_peak_amps,
            y_coords=y_coords,
            sorted_ch_idx=sorted_ch_idx,
            sorted_y=sorted_y
        )

    # -------------------------------------------------------------
    # 3. ESTIMATE PROBE DRIFT / AXIAL DISPLACEMENT Delta-y(t)
    # -------------------------------------------------------------
    print("\n[Stage 3/4] Estimating axial motion displacement Delta-y(t)...")
    # Cross-correlation of spatial RMS activity profiles over time against baseline reference profile
    # Sort RMS matrix by depth
    sorted_rms = rms_matrix[sorted_ch_idx, :] # shape: (384, n_bins)
    baseline_bins = int(60.0 / dt_bin) # first 60s is baseline
    ref_profile = np.mean(sorted_rms[:, :baseline_bins], axis=1) # baseline spatial footprint
    ref_norm = (ref_profile - np.mean(ref_profile)) / (np.std(ref_profile) + 1e-6)

    # Estimate shift in channels via cross-correlation with parabolic interpolation
    ch_shifts = np.zeros(n_bins)
    ch_spacing_um = 15.0 # Neuropixels 2.0 row vertical spacing is 15 um

    max_lag = 15 # +/- 15 channels (~225 um search window)
    prev_shift = 0.0
    for t_idx in range(n_bins):
        prof = sorted_rms[:, t_idx]
        prof_std = np.std(prof)
        # If network activity is suppressed/silent (very low variance), hold prior shift
        if prof_std < 0.20 * np.std(ref_profile):
            ch_shifts[t_idx] = prev_shift
            continue
            
        prof_norm = (prof - np.mean(prof)) / (prof_std + 1e-6)
        corr = signal.correlate(prof_norm, ref_norm, mode='full') / len(ref_norm)
        zero_lag = len(ref_norm) - 1
        lags = np.arange(-max_lag, max_lag + 1)
        sub_corr = corr[zero_lag - max_lag : zero_lag + max_lag + 1]
        best_lag_idx = np.argmax(sub_corr)
        max_corr = sub_corr[best_lag_idx]

        # If peak correlation is too low, retain previous shift
        if max_corr < 0.30:
            ch_shifts[t_idx] = prev_shift
            continue

        best_lag = float(lags[best_lag_idx])
        # Parabolic interpolation for sub-channel shift accuracy
        if 0 < best_lag_idx < len(sub_corr) - 1:
            y_m1, y_0, y_p1 = sub_corr[best_lag_idx - 1], sub_corr[best_lag_idx], sub_corr[best_lag_idx + 1]
            denom = 2 * (2 * y_0 - y_m1 - y_p1)
            sub_shift = (y_m1 - y_p1) / denom if denom != 0 else 0
            best_lag = best_lag + float(np.clip(sub_shift, -0.5, 0.5))

        # Clamp sudden discontinuous jumps between adjacent 0.5s bins during anoxic silencing
        if abs(best_lag - prev_shift) > 2.5 and max_corr < 0.65:
            best_lag = prev_shift

        ch_shifts[t_idx] = best_lag
        prev_shift = best_lag

    displacement_um = ch_shifts * ch_spacing_um
    # Smooth displacement curve (3s rolling median to isolate mechanical drift from transient burst modulation)
    win_smooth = int(3.0 / dt_bin)
    displacement_um_smooth = signal.medfilt(displacement_um, kernel_size=win_smooth if win_smooth % 2 == 1 else win_smooth + 1)

    # Compute subwindow motion metrics
    metrics = []
    print("\n--- Subwindow Quantitative Movement & Firing Metrics ---")
    for sw in subwindows:
        mask_bins = (bin_centers >= sw["t0"]) & (bin_centers <= sw["t1"])
        mask_nidq = (t_nidq >= sw["t0"]) & (t_nidq <= sw["t1"])
        mask_hr = (hr_times >= sw["t0"]) & (hr_times <= sw["t1"])

        disp_win = displacement_um_smooth[mask_bins]
        rms_win = sorted_rms[:, mask_bins]
        mua_win = mua_rate_matrix[:, mask_bins]

        max_disp = float(np.max(np.abs(disp_win))) if len(disp_win) > 0 else 0.0
        p2p_disp = float(np.ptp(disp_win)) if len(disp_win) > 0 else 0.0
        mean_rms = float(np.mean(rms_win)) if rms_win.size > 0 else 0.0
        mean_mua = float(np.mean(mua_win)) if mua_win.size > 0 else 0.0
        mean_emg = float(np.mean(emg_env[mask_nidq])) if np.any(mask_nidq) else 0.0
        mean_hr_val = float(np.mean(hr_bpm[mask_hr])) if np.any(mask_hr) else 0.0
        min_hr_val = float(np.min(hr_bpm[mask_hr])) if np.any(mask_hr) else 0.0

        m_dict = {
            "subwindow_id": sw["id"],
            "name": sw["name"],
            "time_range_s": [sw["t0"], sw["t1"]],
            "time_range_min": [sw["t0"] / 60.0, sw["t1"] / 60.0],
            "duration_s": sw["t1"] - sw["t0"],
            "max_displacement_um": round(max_disp, 2),
            "peak_to_peak_displacement_um": round(p2p_disp, 2),
            "mean_probe_rms_uV": round(mean_rms, 2),
            "mean_mua_rate_hz": round(mean_mua, 2),
            "mean_emg_envelope": round(mean_emg, 4),
            "mean_hr_bpm": round(mean_hr_val, 1),
            "min_hr_bpm": round(min_hr_val, 1)
        }
        metrics.append(m_dict)
        print(f"  [{sw['name']}] ({(sw['t1']-sw['t0']):.0f}s): Drift P2P = {p2p_disp:.1f} um | Max = {max_disp:.1f} um | Mean MUA = {mean_mua:.1f} Hz | HR = {mean_hr_val:.0f} bpm (min {min_hr_val:.0f})")

    # Save metrics JSON
    with open(out_dir / "e097_anoxia_subwindow_motion_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved metrics to {out_dir / 'e097_anoxia_subwindow_motion_metrics.json'}")

    # -------------------------------------------------------------
    # 4. GENERATE HIGH-RESOLUTION PUBLICATION-READY FIGURE
    # -------------------------------------------------------------
    print("\n[Stage 4/4] Rendering publication-quality multi-panel diagnostics figure...")
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#07090e')

    gs = fig.add_gridspec(
        5, 2,
        width_ratios=[48, 1],
        height_ratios=[1.3, 1.2, 1.0, 3.2, 2.0],
        wspace=0.015,
        hspace=0.14,
        left=0.065, right=0.94, top=0.94, bottom=0.05
    )

    ax_emg = fig.add_subplot(gs[0, 0])
    ax_flow = fig.add_subplot(gs[1, 0], sharex=ax_emg)
    ax_hr = fig.add_subplot(gs[2, 0], sharex=ax_emg)
    ax_heat = fig.add_subplot(gs[3, 0], sharex=ax_emg)
    ax_scat = fig.add_subplot(gs[4, 0], sharex=ax_emg)
    cax = fig.add_subplot(gs[3, 1])

    axes = [ax_emg, ax_flow, ax_hr, ax_heat, ax_scat]
    for ax in axes:
        ax.set_facecolor('#0b0f19')
    cax.set_facecolor('#0b0f19')

    fig.suptitle(
        "e097 Neuropixels 2.0: Spatiotemporal Firing Rate & Mechanical Drift Diagnostic\n"
        "Anoxic Challenge (29:00 - 36:00 | 1740s - 2160s) Across Diaphragm EMG, Airflow & ECG",
        fontsize=15, fontweight='bold', y=0.98, color='#f8fafc'
    )

    # Subwindow shaded bands across all panels
    def add_epoch_shading(ax):
        for sw in subwindows:
            ax.axvspan(sw["t0"], sw["t1"], facecolor=sw["hex"], alpha=0.18, zorder=0)

    # Panel 1: Diaphragm EMG
    ax_emg = axes[0]
    add_epoch_shading(ax_emg)
    # Downsample EMG for responsive plotting
    ds_nidq = 10
    ax_emg.plot(t_nidq[::ds_nidq], emg_raw[::ds_nidq], color='#64748b', alpha=0.6, lw=0.5, label='Raw EMG')
    ax_emg.plot(t_nidq[::ds_nidq], emg_env[::ds_nidq], color='#00f5d4', lw=1.6, label='EMG Envelope (50ms)')
    ax_emg.set_ylabel("EMG (V)", fontsize=11, fontweight='bold', color='#f8fafc')
    ax_emg.set_ylim(-0.2, np.percentile(emg_env, 99.8) * 3.0)
    ax_emg.legend(loc='upper right', frameon=True, facecolor='#0f172a', edgecolor='#475569', labelcolor='white')
    ax_emg.set_title("A. Diaphragm Motor Burst Drive (EMG)", fontsize=12, fontweight='bold', loc='left', color='#f8fafc')
    ax_emg.grid(True, linestyle=':', alpha=0.3)

    # Alternate subwindow badge labels
    y_emg_max = ax_emg.get_ylim()[1]
    for idx, sw in enumerate(subwindows):
        t_mid = 0.5 * (sw["t0"] + sw["t1"])
        if idx == 6:
            y_pos = y_emg_max * 0.50 # sit neatly beneath the upper-right legend box
        else:
            y_pos = y_emg_max * (0.80 if idx % 2 == 0 else 0.64)
        ax_emg.text(
            t_mid, y_pos,
            sw["name"],
            ha='center', va='top', fontsize=8.0, fontweight='bold', color='#f8fafc',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#0f172a', alpha=0.85, edgecolor=sw["hex"], lw=1.0)
        )

    # Panel 2: Plethysmography Flow
    ax_flow = axes[1]
    add_epoch_shading(ax_flow)
    ax_flow.plot(t_nidq[::ds_nidq], flow_raw[::ds_nidq], color='#38bdf8', lw=1.0)
    ax_flow.axhline(0, color='white', linestyle='--', alpha=0.4, lw=0.8)
    ax_flow.set_ylabel("Airflow (V)", fontsize=11, fontweight='bold', color='#f8fafc')
    ax_flow.set_title("B. Respiratory Airflow (Tidal Inspiration & Expiration)", fontsize=12, fontweight='bold', loc='left', color='#f8fafc')
    ax_flow.grid(True, linestyle=':', alpha=0.3)

    # Panel 3: Heart Rate (bpm)
    ax_hr = axes[2]
    add_epoch_shading(ax_hr)
    ax_hr.scatter(hr_times, hr_bpm, color='#ff0055', s=6, alpha=0.4, label='Instantaneous HR')
    # Smooth HR line
    if len(hr_times) > 10:
        hr_smooth = signal.medfilt(hr_bpm, kernel_size=11)
        ax_hr.plot(hr_times, hr_smooth, color='#ff2a85', lw=2.0, label='Median Filtered HR')
    ax_hr.set_ylabel("Heart Rate\n(bpm)", fontsize=11, fontweight='bold', color='#f8fafc')
    ax_hr.set_ylim(150, 750)
    min_hr = float(np.min(hr_bpm)) if len(hr_bpm) > 0 else 224.0
    ax_hr.axhline(min_hr, color='yellow', linestyle=':', alpha=0.8, label=f'Nadir ({min_hr:.0f} bpm)')
    ax_hr.legend(loc='lower right', frameon=True, facecolor='#0f172a', edgecolor='#475569', labelcolor='white')
    ax_hr.set_title("C. Instantaneous Cardiac Rate (Severe Anoxic Bradycardia & Autoresuscitation)", fontsize=12, fontweight='bold', loc='left', color='#f8fafc')
    ax_hr.grid(True, linestyle=':', alpha=0.3)

    # Panel 4: Probe-Wide Spike-Band RMS Energy Heatmap
    ax_heat = axes[3]
    add_epoch_shading(ax_heat)
    # 2D pcolormesh
    extent = [t0_s, t1_s, sorted_y.min(), sorted_y.max()]
    vmax_rms = np.percentile(sorted_rms, 99.0)
    im = ax_heat.imshow(
        sorted_rms,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='inferno',
        vmin=0,
        vmax=vmax_rms,
        interpolation='nearest'
    )
    # Overlay the estimated probe drift trajectory in cyan
    ax_heat.plot(bin_centers, sorted_y.min() + 300.0 + displacement_um_smooth * 5.0, color='#00f5d4', lw=2.0, label='Drift Offset Trajectory Δy(t) (5x visual scale)')
    ax_heat.set_ylabel("Probe Depth (um)\nVentral → Dorsal", fontsize=11, fontweight='bold', color='#f8fafc')
    ax_heat.set_title("D. Probe-Wide Spike-Band RMS Activity (300-6000 Hz) Across 384 Channels (Horizontal Bands = Spatial Stationarity)", fontsize=12, fontweight='bold', loc='left', color='#f8fafc')
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Spike RMS (uV)", fontsize=10, color='#f8fafc', labelpad=8)
    cbar.ax.yaxis.set_tick_params(color='#f8fafc', labelsize=9)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#f8fafc')

    # Panel 5: Detected Peak Scatter & Drift Displacement
    ax_scat = axes[4]
    add_epoch_shading(ax_scat)
    sc = ax_scat.scatter(
        all_peak_times,
        all_peak_depths,
        c=all_peak_amps,
        cmap='magma_r',
        s=3,
        alpha=0.5,
        rasterized=True
    )
    ax_scat.set_ylabel("Peak Depth (um)", fontsize=11, fontweight='bold', color='#f8fafc')
    ax_scat.set_xlabel("Recording Time (s)", fontsize=12, fontweight='bold', color='#f8fafc')
    ax_scat.set_title("E. Localized Peak Scatter (Depth vs Time) & Estimated Displacement Δy(t)", fontsize=12, fontweight='bold', loc='left', color='#f8fafc')
    
    # Secondary y-axis on Panel 5 for quantitative displacement Δy(t) in micrometers
    ax_disp = ax_scat.twinx()
    ax_disp.plot(bin_centers, displacement_um_smooth, color='#38bdf8', lw=2.5, label='Probe Axial Displacement Δy (um)')
    ax_disp.axhline(0, color='white', linestyle='--', alpha=0.5)
    ax_disp.set_ylabel("Estimated Axial Drift (um)", color='#38bdf8', fontsize=11, fontweight='bold')
    ax_disp.tick_params(axis='y', labelcolor='#38bdf8')
    ax_disp.set_ylim(-30, 30)

    # Set x-limits and ticks
    ax_scat.set_xlim(t0_s, t1_s)
    x_ticks = np.arange(t0_s, t1_s + 1, 60.0)
    x_labels = [f"{int(t/60)}m ({int(t)}s)" for t in x_ticks]
    ax_scat.set_xticks(x_ticks)
    ax_scat.set_xticklabels(x_labels, fontsize=10)

    # Cleanly suppress bottom tick labels on upper subplots to avoid collision with titles
    for ax in [ax_emg, ax_flow, ax_hr, ax_heat]:
        ax.tick_params(labelbottom=False)

    fig_path = out_dir / "e097_anoxia_preliminary_motion_and_firing_diagnostics.png"
    plt.savefig(fig_path, dpi=250, facecolor='#07090e')
    plt.close()
    print(f"Saved diagnostic figure to {fig_path}")

    # -------------------------------------------------------------
    # 5. GENERATE STANDALONE INTERACTIVE HTML DASHBOARD
    # -------------------------------------------------------------
    print("\nGenerating interactive synchronized Plotly HTML dashboard...")
    build_interactive_html(
        t_nidq=t_nidq[::ds_nidq],
        emg_raw=emg_raw[::ds_nidq],
        emg_env=emg_env[::ds_nidq],
        flow_raw=flow_raw[::ds_nidq],
        hr_times=hr_times,
        hr_bpm=hr_bpm,
        bin_centers=bin_centers,
        sorted_y=sorted_y,
        sorted_rms=sorted_rms,
        displacement_um=displacement_um_smooth,
        all_peak_times=all_peak_times[::4], # subsample peaks for snappy browser rendering
        all_peak_depths=all_peak_depths[::4],
        subwindows=subwindows,
        metrics=metrics,
        out_html=out_dir / "e097_anoxia_preliminary_motion_and_firing_diagnostics.html"
    )

    t_elapsed = time.time() - t_start_global
    print(f"\n=================================================================")
    print(f"DIAGNOSTIC PIPELINE COMPLETE in {t_elapsed:.1f}s ({t_elapsed/60:.2f} min)!")
    print(f"Outputs:")
    print(f"  - PNG Figure:  {fig_path}")
    print(f"  - HTML Viewer: {out_dir / 'e097_anoxia_preliminary_motion_and_firing_diagnostics.html'}")
    print(f"  - Metrics JSON:{out_dir / 'e097_anoxia_subwindow_motion_metrics.json'}")
    print("=================================================================")

def build_interactive_html(t_nidq, emg_raw, emg_env, flow_raw, hr_times, hr_bpm, bin_centers, sorted_y, sorted_rms, displacement_um, all_peak_times, all_peak_depths, subwindows, metrics, out_html):
    """Generates a dark-themed, synchronized interactive Plotly report."""
    
    # Subsample traces for fast Plotly rendering (<5 MB HTML)
    step_nidq = max(1, len(t_nidq) // 4000)
    t_nidq_sub = t_nidq[::step_nidq]
    emg_raw_sub = emg_raw[::step_nidq]
    emg_env_sub = emg_env[::step_nidq]
    flow_raw_sub = flow_raw[::step_nidq]

    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.20, 0.20, 0.18, 0.42],
        subplot_titles=[
            "Diaphragm EMG & Motor Burst Drive",
            "Plethysmography Airflow (Flow Rate)",
            "Cardiac Heart Rate (bpm)",
            "Probe Axial Displacement & Peak Scatter"
        ]
    )

    # Track 1: EMG
    fig.add_trace(go.Scatter(x=t_nidq_sub, y=emg_raw_sub, mode='lines', line=dict(color='rgba(148, 163, 184, 0.4)', width=1), name='Raw EMG'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_nidq_sub, y=emg_env_sub, mode='lines', line=dict(color='#00f5d4', width=2), name='EMG Envelope'), row=1, col=1)

    # Track 2: Flow
    fig.add_trace(go.Scatter(x=t_nidq_sub, y=flow_raw_sub, mode='lines', line=dict(color='#38bdf8', width=1.5), name='Airflow'), row=2, col=1)

    # Track 3: HR
    fig.add_trace(go.Scatter(x=hr_times, y=hr_bpm, mode='markers+lines', marker=dict(size=4, color='#ff0055'), line=dict(color='#ff0055', width=1), name='Heart Rate (bpm)'), row=3, col=1)

    # Track 4: Localized Peaks Scatter & Displacement Curve
    fig.add_trace(
        go.Scattergl(
            x=all_peak_times,
            y=all_peak_depths,
            mode='markers',
            marker=dict(size=3, color='#f59e0b', opacity=0.4),
            name='Detected Peaks'
        ),
        row=4, col=1
    )
    # Displacement curve on secondary axis or overlaid
    # Scale displacement to visualize over probe depth
    mean_depth = float(np.mean(sorted_y))
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=mean_depth + displacement_um * 5.0, # scaled for visual inspection
            mode='lines',
            line=dict(color='#00f5d4', width=2.5),
            name='Axial Drift Trajectory (5x scale)',
            hoverinfo='text',
            hovertext=[f"t={t:.1f}s | Drift: {d:+.1f} µm" for t, d in zip(bin_centers, displacement_um)]
        ),
        row=4, col=1
    )

    # Add shaded shapes for subwindows
    shapes = []
    for sw in subwindows:
        shapes.append(dict(
            type="rect",
            xref="x", yref="paper",
            x0=sw["t0"], x1=sw["t1"],
            y0=0, y1=1,
            fillcolor=sw["rgba"],
            opacity=0.35,
            layer="below",
            line_width=0
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#07090e",
        plot_bgcolor="#0b0f19",
        height=950,
        shapes=shapes,
        title=dict(
            text="e097 Anoxic Challenge Preliminary Movement & Neural Activity Diagnostic (29m - 36m)",
            font=dict(size=18, color="#f1f5f9", family="Inter, sans-serif")
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Recording Time (s)", row=4, col=1)
    fig.update_yaxes(title_text="EMG (V)", row=1, col=1)
    fig.update_yaxes(title_text="Flow (V)", row=2, col=1)
    fig.update_yaxes(title_text="HR (bpm)", row=3, col=1)
    fig.update_yaxes(title_text="Probe Depth (µm)", row=4, col=1)

    # Render metrics table rows
    table_rows = ""
    for m in metrics:
        p2p = m["peak_to_peak_displacement_um"]
        badge_class = "badge-danger" if p2p > 25.0 else ("badge-warn" if p2p > 12.0 else "badge-pass")
        table_rows += f"""
        <tr>
            <td><b>{m['name']}</b></td>
            <td>{m['time_range_min'][0]:.2f} - {m['time_range_min'][1]:.2f} m ({m['time_range_s'][0]:.0f} - {m['time_range_s'][1]:.0f}s)</td>
            <td>{m['duration_s']:.0f} s</td>
            <td><span class="{badge_class}">{p2p:.1f} µm</span></td>
            <td>{m['max_displacement_um']:.1f} µm</td>
            <td>{m['mean_probe_rms_uV']:.2f} µV</td>
            <td>{m['mean_mua_rate_hz']:.1f} Hz</td>
            <td><b>{m['mean_hr_bpm']:.0f} bpm</b> (min {m['min_hr_bpm']:.0f})</td>
        </tr>
        """

    # Interactive buttons
    jump_buttons = ""
    for sw in subwindows:
        jump_buttons += f"""
        <button class="btn-jump" onclick="zoomTo({sw['t0']}, {sw['t1']})">
            {sw['name']}
        </button>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>e097 Preliminary Movement & Firing Diagnostic (29m - 36m)</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{
            background: #07090e;
            color: #f1f5f9;
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 24px;
        }}
        .header {{
            background: #0f1523;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 20px;
        }}
        h1 {{
            margin: 0 0 8px 0;
            font-size: 22px;
            color: #38bdf8;
        }}
        p.subtitle {{
            margin: 0;
            color: #94a3b8;
            font-size: 14px;
        }}
        .button-bar {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 16px;
        }}
        .btn-jump {{
            background: #1e293b;
            color: #f1f5f9;
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 6px;
            padding: 8px 14px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .btn-jump:hover {{
            background: #38bdf8;
            color: #07090e;
        }}
        .table-container {{
            background: #0f1523;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 18px 24px;
            margin-top: 24px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}
        th {{
            color: #94a3b8;
            border-bottom: 1px solid rgba(255,255,255,0.12);
            padding: 10px 12px;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .badge-pass {{
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
        }}
        .badge-warn {{
            background: rgba(245, 158, 11, 0.2);
            color: #fbbf24;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
        }}
        .badge-danger {{
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>e097 Preliminary Movement & Firing Rate Diagnostic (29m - 36m)</h1>
        <p class="subtitle">Cardiorespiratory Anoxic Challenge Stationarity, Mechanical Drift Profile & Probe-Wide Firing Metrics</p>
        <div class="button-bar">
            <span style="font-size: 12px; color: #94a3b8; line-height: 32px; margin-right: 8px;">Zoom to Epoch:</span>
            {jump_buttons}
            <button class="btn-jump" style="background: #334155;" onclick="zoomTo(1740, 2160)">Reset View (Full 7m)</button>
        </div>
    </div>

    <div id="plotly-div"></div>

    <div class="table-container">
        <h3 style="margin: 0 0 12px 0; font-size: 16px; color: #38bdf8;">Subwindow Movement & Stationarity Summary</h3>
        <table>
            <thead>
                <tr>
                    <th>Subwindow Name</th>
                    <th>Time Range</th>
                    <th>Duration</th>
                    <th>Drift (Peak-to-Peak)</th>
                    <th>Max Displacement</th>
                    <th>Spike RMS</th>
                    <th>MUA Rate</th>
                    <th>Cardiac Rate</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>

    <script>
        var plotData = {fig.to_json()};
        Plotly.newPlot('plotly-div', plotData.data, plotData.layout, {{responsive: true}});

        function zoomTo(t0, t1) {{
            Plotly.relayout('plotly-div', {{
                'xaxis4.range': [t0, t1]
            }});
        }}
    </script>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved interactive HTML to {out_html}")

if __name__ == '__main__':
    main()
