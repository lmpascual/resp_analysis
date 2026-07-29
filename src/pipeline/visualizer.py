"""
Interactive Visual Quality & Stationarity Inspection Module.
Generates diagnostic plots for EMG SNR/stationarity, ECG R-peaks, Opto/Sync alignment,
Neuropixel drift, and multi-channel raw signal overview.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless rendering for saving figure files safely
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from .models import SessionResults

def generate_diagnostic_dashboard(results: SessionResults, output_dir: Path) -> Dict[str, Path]:
    """
    Generate diagnostic plots for visual parameter verification & data stationarity inspection.
    Saves PNG plots to output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = {}

    time = results.analog_raw.time_vector
    t_max = time[-1] if len(time) > 0 else 1.0

    # -------------------------------------------------------------
    # 0. Multi-Channel Raw Overview Plot (Inspect All Channels)
    # -------------------------------------------------------------
    if results.analog_raw.all_channels:
        num_ch = len(results.analog_raw.all_channels)
        fig, axes = plt.subplots(num_ch, 1, figsize=(14, max(6, num_ch * 1.5)), sharex=True)
        if num_ch == 1:
            axes = [axes]
        
        # Subsample for fast plotting if recording is long (>100k samples)
        step = max(1, len(time) // 50000)
        time_sub = time[::step]

        for ch in range(num_ch):
            ch_signal = results.analog_raw.all_channels[ch][::step]
            ch_std = np.std(ch_signal)
            axes[ch].plot(time_sub, ch_signal, linewidth=0.8, color='navy')
            axes[ch].set_ylabel(f"Ch {ch}\n(mV)", fontsize=8)
            axes[ch].set_title(f"Channel {ch} (std: {ch_std:.2f})", fontsize=9, loc='left')
            axes[ch].grid(True, linestyle=':', alpha=0.5)

        axes[-1].set_xlabel("Time (seconds)")
        fig.suptitle(f"Multi-Channel Raw DAQ Signal Overview (0 to {t_max:.1f}s) - {results.animal_id} {results.session_id}", fontsize=12)
        plt.tight_layout()
        ch_plot_path = output_dir / f"{results.animal_id}_{results.session_id}_raw_all_channels.png"
        fig.savefig(ch_plot_path, dpi=150)
        plt.close(fig)
        generated_files['raw_all_channels'] = ch_plot_path

    # -------------------------------------------------------------
    # 1. Diaphragm EMG Transformation & SNR Stationarity Plot
    # -------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(14, 11))
    
    # Subplot 1: Full Session Raw EMG Overview (0 to T_max)
    step = max(1, len(time) // 50000)
    axes[0].plot(time[::step], results.analog_raw.emg_raw[::step], color='gray', alpha=0.7, linewidth=0.6, label='Raw EMG (Full Session)')
    axes[0].set_title(f"Diaphragm EMG Full Recording Raw Signal (0 to {t_max:.1f}s) - {results.animal_id}")
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_xlim(0, t_max)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Subplot 2: Zoomed Snapshot (First 10s) Raw vs Filtered vs Envelope
    snap_win = 10.0
    mask_snap = time <= snap_win
    axes[1].plot(time[mask_snap], results.analog_raw.emg_raw[mask_snap], label='Raw EMG', alpha=0.4, color='gray')
    axes[1].plot(time[mask_snap], results.emg.filtered[mask_snap], label='Filtered (300-3000 Hz)', alpha=0.8, color='royalblue')
    axes[1].plot(time[mask_snap], results.emg.envelope[mask_snap], label='Smoothed Envelope', color='darkred', linewidth=1.5)
    
    peak_times = results.emg.breath_onsets
    peak_mask = peak_times <= snap_win
    if np.any(peak_mask):
        axes[1].scatter(peak_times[peak_mask], np.interp(peak_times[peak_mask], time, results.emg.envelope),
                        color='limegreen', marker='^', s=60, zorder=5, label='Breath Onset')
    axes[1].set_title("High-Resolution Waveform Snapshot (0 to 10s)")
    axes[1].set_ylabel("Voltage (mV)")
    axes[1].set_xlim(0, snap_win)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # Subplot 3: Instantaneous Respiratory Rate (Full Session)
    if len(results.emg.breath_onsets) > 1:
        axes[2].plot(results.emg.breath_onsets[1:], results.emg.instantaneous_rr, color='purple', linewidth=1.2, label='Breath Rate')
    axes[2].set_title(f"Instantaneous Respiratory Rate (Mean: {np.nanmean(results.emg.instantaneous_rr):.1f} breaths/min, Total: {len(results.emg.breath_onsets)} breaths)")
    axes[2].set_ylabel("Breaths/min")
    axes[2].set_xlim(0, t_max)
    axes[2].grid(True, linestyle=':', alpha=0.6)

    # Subplot 4: Time-Resolved EMG SNR & RMS Stationarity (Full Session)
    if len(results.emg.snr_time_vector) > 0:
        axes[3].plot(results.emg.snr_time_vector, results.emg.snr_db, color='teal', linewidth=1.5, marker='o', markersize=3, label='EMG SNR (dB)')
        axes[3].axhline(y=10.0, color='red', linestyle='--', label='Min Quality Threshold (10 dB)')
    axes[3].set_title(f"EMG Signal-to-Noise Ratio (Mean SNR: {np.nanmean(results.emg.snr_db):.1f} dB) - Stationarity Check")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].set_ylabel("SNR (dB)")
    axes[3].set_xlim(0, t_max)
    axes[3].legend(loc='upper right')
    axes[3].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    emg_plot_path = output_dir / f"{results.animal_id}_{results.session_id}_emg_stationarity.png"
    fig.savefig(emg_plot_path, dpi=200)
    plt.close(fig)
    generated_files['emg_stationarity'] = emg_plot_path

    # -------------------------------------------------------------
    # 2. ECG & R-Peak Detection Diagnostic Plot
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    
    # Subplot 1: Full Session Raw ECG Overview (0 to T_max)
    axes[0].plot(time[::step], results.analog_raw.ecg_raw[::step], color='crimson', alpha=0.7, linewidth=0.6, label='Raw ECG (Full Session)')
    axes[0].set_title(f"ECG Full Recording Raw Signal (0 to {t_max:.1f}s) - {results.animal_id}")
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_xlim(0, t_max)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Subplot 2: Zoomed Snapshot (First 10s) Raw vs Filtered ECG & R-Peaks
    axes[1].plot(time[mask_snap], results.analog_raw.ecg_raw[mask_snap], label='Raw ECG', alpha=0.4, color='gray')
    axes[1].plot(time[mask_snap], results.ecg.filtered[mask_snap], label='Filtered ECG (5-100 Hz)', color='crimson', alpha=0.9)
    r_times = results.ecg.r_peaks
    r_mask = r_times <= snap_win
    if np.any(r_mask):
        axes[1].scatter(r_times[r_mask], np.interp(r_times[r_mask], time, results.ecg.filtered),
                        color='black', marker='o', s=40, zorder=5, label='Detected R-Peaks')
    axes[1].set_title("High-Resolution Waveform Snapshot (0 to 10s)")
    axes[1].set_ylabel("Voltage (mV)")
    axes[1].set_xlim(0, snap_win)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # Subplot 3: Heart Rate Tachogram across Full Session
    if len(results.ecg.r_peaks) > 1:
        axes[2].plot(results.ecg.r_peaks[1:], results.ecg.instantaneous_hr, color='darkgreen', linewidth=1.2, label='Heart Rate')
        mean_hr = np.nanmean(results.ecg.instantaneous_hr)
        axes[2].set_title(f"Heart Rate Tachogram (Mean HR: {mean_hr:.1f} bpm, R-Peaks Detected: {len(results.ecg.r_peaks)}, HRV SDNN: {results.ecg.hrv_sdnn:.2f} ms)")
    else:
        axes[2].set_title(f"Heart Rate Tachogram (No R-Peaks Detected - Verify ECG channel assignment)")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Heart Rate (bpm)")
    axes[2].set_xlim(0, t_max)
    axes[2].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    ecg_plot_path = output_dir / f"{results.animal_id}_{results.session_id}_ecg_rpeaks.png"
    fig.savefig(ecg_plot_path, dpi=200)
    plt.close(fig)
    generated_files['ecg_rpeaks'] = ecg_plot_path

    # -------------------------------------------------------------
    # 3. Opto Laser & Time Sync Pulse Alignment Plot
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    
    axes[0].plot(time[::step], results.analog_raw.sync_raw[::step], label='Sync TTL', color='gray', linewidth=0.8)
    axes[0].set_title(f"Sync Pulse Alignment ({len(results.sync.onsets)} pulses detected) - Full Session (0 to {t_max:.1f}s)")
    axes[0].set_ylabel("Voltage / TTL (V)")
    axes[0].set_xlim(0, t_max)
    axes[0].grid(True, linestyle=':', alpha=0.6)

    axes[1].plot(time[::step], results.analog_raw.opto_raw[::step], label='Opto Laser', color='orangered', linewidth=0.8)
    axes[1].set_title(f"Optic Fiber Laser Pulses ({len(results.opto.onsets)} opto pulses detected)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Laser TTL (V)")
    axes[1].set_xlim(0, t_max)
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    pulse_plot_path = output_dir / f"{results.animal_id}_{results.session_id}_opto_sync.png"
    fig.savefig(pulse_plot_path, dpi=200)
    plt.close(fig)
    generated_files['opto_sync'] = pulse_plot_path

    # -------------------------------------------------------------
    # 4. Neuropixel Depth & Stationarity Rasters (If Ephys loaded)
    # -------------------------------------------------------------
    if results.ephys is not None and len(results.ephys.unit_ids) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        for uid in results.ephys.unit_ids:
            spikes = results.ephys.unit_spiketimes[uid]
            depth = results.ephys.unit_depths.get(uid, 0.0)
            ax.scatter(spikes, np.full_like(spikes, depth), s=2, alpha=0.6, label=f'Unit {uid}' if uid < 5 else "")
        ax.set_title(f"Neuropixel Probe Depth Spike Rasters & Stationarity (0 to {t_max:.1f}s) - {results.animal_id}")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Probe Depth (µm)")
        ax.set_xlim(0, t_max)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        npx_plot_path = output_dir / f"{results.animal_id}_{results.session_id}_npx_drift.png"
        fig.savefig(npx_plot_path, dpi=200)
        plt.close(fig)
        generated_files['npx_drift'] = npx_plot_path

    print(f" -> Generated {len(generated_files)} visual diagnostic plots in {output_dir}")
    return generated_files
