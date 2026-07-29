"""
Interactive Visual Quality & Stationarity Inspection Module.
Generates diagnostic plots for EMG SNR/stationarity, ECG R-peaks, Opto/Sync alignment, and Neuropixel drift.
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
    Saves PNG/HTML plots to output_dir and displays figures if interactive.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = {}

    # -------------------------------------------------------------
    # 1. EMG Processing, Peak Detections, & SNR Stationarity Plot
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    time = results.analog_raw.time_vector
    
    # Subplot 1: Raw vs Filtered vs Envelope (First 15 seconds snapshot)
    mask = time <= 15.0
    axes[0].plot(time[mask], results.analog_raw.emg_raw[mask], label='Raw EMG', alpha=0.5, color='gray')
    axes[0].plot(time[mask], results.emg.filtered[mask], label='Bandpass Filtered (300-3000 Hz)', alpha=0.8, color='blue')
    axes[0].plot(time[mask], results.emg.envelope[mask], label='Smoother Envelope', color='darkred', linewidth=1.5)
    
    # Overlay detected breath peaks
    peak_times = results.emg.breath_onsets
    peak_mask = peak_times <= 15.0
    if np.any(peak_mask):
        axes[0].scatter(peak_times[peak_mask], np.interp(peak_times[peak_mask], time, results.emg.envelope),
                        color='green', marker='^', s=60, zorder=5, label='Breath Onset')
    axes[0].set_title(f"Diaphragm EMG Signal Transformation (Snapshot 0-15s) - {results.animal_id} {results.session_id}")
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Subplot 2: Instantaneous Respiratory Rate over Recording Session
    if len(results.emg.breath_onsets) > 1:
        axes[1].plot(results.emg.breath_onsets[1:], results.emg.instantaneous_rr, color='purple', linewidth=1.2)
    axes[1].set_title("Instantaneous Respiratory Rate (Breaths / min)")
    axes[1].set_ylabel("Breaths/min")
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # Subplot 3: Time-Resolved EMG SNR & RMS Stationarity
    if len(results.emg.snr_time_vector) > 0:
        axes[2].plot(results.emg.snr_time_vector, results.emg.snr_db, color='teal', linewidth=1.5, marker='o', markersize=3)
        axes[2].axhline(y=10.0, color='red', linestyle='--', label='Min Quality Threshold (10 dB)')
    axes[2].set_title("Time-Resolved EMG Signal-to-Noise Ratio (SNR dB) - Stationarity Evaluation")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("SNR (dB)")
    axes[2].legend(loc='upper right')
    axes[2].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    emg_plot_path = output_dir / f"{results.animal_id}_{results.session_id}_emg_stationarity.png"
    fig.savefig(emg_plot_path, dpi=200)
    plt.close(fig)
    generated_files['emg_stationarity'] = emg_plot_path

    # -------------------------------------------------------------
    # 2. ECG & R-Peak Detection Diagnostic Plot
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    
    # Subplot 1: Filtered ECG & R-Peaks (First 15 seconds)
    axes[0].plot(time[mask], results.ecg.filtered[mask], label='Filtered ECG (5-100 Hz)', color='crimson')
    r_times = results.ecg.r_peaks
    r_mask = r_times <= 15.0
    if np.any(r_mask):
        axes[0].scatter(r_times[r_mask], np.interp(r_times[r_mask], time, results.ecg.filtered),
                        color='black', marker='o', s=40, zorder=5, label='Detected R-Peaks')
    axes[0].set_title(f"ECG & R-Peak Detection Diagnostic - {results.animal_id} {results.session_id}")
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Subplot 2: Instantaneous Heart Rate Tachogram
    if len(results.ecg.r_peaks) > 1:
        axes[1].plot(results.ecg.r_peaks[1:], results.ecg.instantaneous_hr, color='darkgreen', linewidth=1.2)
    axes[1].set_title(f"Heart Rate Tachogram (Mean HR: {np.nanmean(results.ecg.instantaneous_hr):.1f} bpm, HRV SDNN: {results.ecg.hrv_sdnn:.2f} ms)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Heart Rate (bpm)")
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    ecg_plot_path = output_dir / f"{results.animal_id}_{results.session_id}_ecg_rpeaks.png"
    fig.savefig(ecg_plot_path, dpi=200)
    plt.close(fig)
    generated_files['ecg_rpeaks'] = ecg_plot_path

    # -------------------------------------------------------------
    # 3. Opto Laser & Time Sync Pulse Alignment Plot
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    
    axes[0].plot(time[mask], results.analog_raw.sync_raw[mask], label='Time Sync TTL', color='gray')
    for onset in results.sync.onsets[results.sync.onsets <= 15.0]:
        axes[0].axvline(x=onset, color='blue', linestyle=':', alpha=0.7)
    axes[0].set_title(f"Sync Pulse Edge Alignment ({len(results.sync.onsets)} pulses detected)")
    axes[0].set_ylabel("Voltage / TTL (V)")
    axes[0].grid(True, linestyle=':', alpha=0.6)

    axes[1].plot(time[mask], results.analog_raw.opto_raw[mask], label='Opto Laser Monitor', color='orangered')
    for onset in results.opto.onsets[results.opto.onsets <= 15.0]:
        axes[1].axvline(x=onset, color='red', linestyle='--', alpha=0.8)
    axes[1].set_title(f"Optic Fiber Laser Pulse Alignment ({len(results.opto.onsets)} opto pulses detected)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Laser TTL (V)")
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
        ax.set_title(f"Neuropixel Probe Depth Spike Rasters & Stationarity - {results.animal_id}")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Probe Depth (µm)")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        npx_plot_path = output_dir / f"{results.animal_id}_{results.session_id}_npx_drift.png"
        fig.savefig(npx_plot_path, dpi=200)
        plt.close(fig)
        generated_files['npx_drift'] = npx_plot_path

    print(f" -> Generated {len(generated_files)} visual diagnostic plots in {output_dir}")
    return generated_files
