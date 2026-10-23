"""
Stage 1 Preprocessing & CatGT Diagnostic Inspector:
Generates publication-quality visual diagnostics to inspect:
1. Raw vs Filtered/Loccar Traces (stack & overlay)
2. Power Spectral Density (PSD) to detect filter ringing / high-frequency distortion
3. Spatial Channel Correlation Matrix to verify loccar reference removal
"""
from pathlib import Path
from typing import Optional, List, Tuple
import logging
import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import spikeinterface.full as si

logger = logging.getLogger("PREPROC_INSPECTOR")

def compute_psd(data_1d: np.ndarray, fs: float, nperseg: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """Computes Welch Power Spectral Density in dB/Hz."""
    freqs, psd = signal.welch(data_1d, fs=fs, nperseg=min(len(data_1d), nperseg))
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-12))
    return freqs, psd_db

def plot_preprocessing_diagnostics(
    rec_raw: si.BaseRecording,
    rec_pre: si.BaseRecording,
    out_path: Path,
    time_window_s: Tuple[float, float] = (10.0, 10.5), # 500 ms snippet
    channel_indices: Optional[List[int]] = None,
    dpi: int = 300
) -> Path:
    """
    Generates a 3-panel comprehensive preprocessing & CatGT verification figure:
    [Panel A]: Multi-channel raw vs filtered trace overlays
    [Panel B]: Power Spectral Density (PSD) spectrum
    [Panel C]: Spatial cross-channel correlation matrix (Raw vs Preprocessed)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fs = rec_raw.get_sampling_frequency()
    t0_s, t1_s = time_window_s
    f0 = int(t0_s * fs)
    f1 = int(t1_s * fs)
    n_samples = f1 - f0

    total_chans = rec_raw.get_num_channels()
    if channel_indices is None:
        # Sample 16 evenly spaced channels across the probe
        channel_indices = list(np.linspace(0, total_chans - 1, 16, dtype=int))

    # 1. Extract Snippet Traces
    traces_raw = rec_raw.get_traces(start_frame=f0, end_frame=f1, channel_ids=[rec_raw.channel_ids[i] for i in channel_indices])
    traces_pre = rec_pre.get_traces(start_frame=f0, end_frame=f1, channel_ids=[rec_pre.channel_ids[i] for i in channel_indices])
    time_axis_ms = np.linspace(0, (t1_s - t0_s) * 1000, n_samples)

    # 2. Extract Longer Segment for PSD & Correlation (e.g. 5.0 seconds)
    f_psd_len = min(int(5.0 * fs), rec_raw.get_num_samples())
    sub_raw_long = rec_raw.get_traces(start_frame=0, end_frame=f_psd_len)
    sub_pre_long = rec_pre.get_traces(start_frame=0, end_frame=f_psd_len)

    # 3. Create Multi-Panel Figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0], hspace=0.3, wspace=0.25)
    fig.suptitle("Preprocessing & CatGT Artifact Inspection Diagnostic", fontsize=16, fontweight='bold', y=0.98)

    # Panel A: Multi-channel Traces Stack
    ax_traces = fig.add_subplot(gs[0, :2])
    offset = 120.0 # uV offset per channel
    for i, ch_idx in enumerate(channel_indices):
        y_offset = i * offset
        v_raw = traces_raw[:, i] - np.median(traces_raw[:, i])
        v_pre = traces_pre[:, i] - np.median(traces_pre[:, i])
        if i == 0:
            ax_traces.plot(time_axis_ms, v_raw + y_offset, color='#7f8c8d', alpha=0.5, linewidth=0.8, label='Raw AP')
            ax_traces.plot(time_axis_ms, v_pre + y_offset, color='#2980b9', alpha=0.9, linewidth=1.1, label='CatGT / Filtered + Loccar')
        else:
            ax_traces.plot(time_axis_ms, v_raw + y_offset, color='#7f8c8d', alpha=0.5, linewidth=0.8)
            ax_traces.plot(time_axis_ms, v_pre + y_offset, color='#2980b9', alpha=0.9, linewidth=1.1)
        ax_traces.text(-5, y_offset, f"Ch {ch_idx}", va='center', ha='right', fontsize=8, color='#2c3e50')

    ax_traces.set_title("A. Raw vs Preprocessed Trace Overlays (500 ms Snippet)", fontsize=13, fontweight='bold')
    ax_traces.set_xlabel("Time within snippet (ms)", fontsize=11)
    ax_traces.set_ylabel("Electrode Channels (Stacked with 120 uV spacing)", fontsize=11)
    ax_traces.set_xlim(-15, time_axis_ms[-1] + 5)
    ax_traces.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax_traces.grid(True, linestyle=':', alpha=0.5)

    # Panel B: Power Spectral Density
    ax_psd = fig.add_subplot(gs[0, 2])
    rep_ch = channel_indices[len(channel_indices)//2]
    freqs_raw, psd_raw = compute_psd(sub_raw_long[:, rep_ch], fs=fs)
    freqs_pre, psd_pre = compute_psd(sub_pre_long[:, rep_ch], fs=fs)

    ax_psd.plot(freqs_raw, psd_raw, color='#7f8c8d', alpha=0.7, label='Raw AP Stream', linewidth=1.2)
    ax_psd.plot(freqs_pre, psd_pre, color='#e74c3c', alpha=0.9, label='CatGT Filtered (300-9000 Hz)', linewidth=1.5)
    ax_psd.axvline(300, color='blue', linestyle='--', alpha=0.6, label='Highpass (300 Hz)')
    ax_psd.axvline(9000, color='purple', linestyle='--', alpha=0.6, label='Lowpass (9000 Hz)')
    ax_psd.set_xscale('log')
    ax_psd.set_xlim(10, fs/2)
    ax_psd.set_title("B. Power Spectral Density (PSD)", fontsize=13, fontweight='bold')
    ax_psd.set_xlabel("Frequency (Hz)", fontsize=11)
    ax_psd.set_ylabel("Power / Frequency (dB/Hz)", fontsize=11)
    ax_psd.legend(loc='lower left', fontsize=9)
    ax_psd.grid(True, which='both', linestyle=':', alpha=0.5)

    # Panel C1: Raw Cross-Channel Correlation
    ax_corr_raw = fig.add_subplot(gs[1, 0])
    sub_raw_sample = sub_raw_long[:, ::8] # Downsample spatial channels for fast heatmap
    corr_raw = np.corrcoef(sub_raw_sample.T)
    im1 = ax_corr_raw.imshow(corr_raw, cmap='coolwarm', vmin=-0.2, vmax=1.0, origin='lower', aspect='auto')
    ax_corr_raw.set_title("C1. Raw Channel Correlation Matrix", fontsize=12, fontweight='bold')
    ax_corr_raw.set_xlabel("Channel Index (subsampled)", fontsize=10)
    ax_corr_raw.set_ylabel("Channel Index (subsampled)", fontsize=10)
    fig.colorbar(im1, ax=ax_corr_raw, fraction=0.046, pad=0.04)

    # Panel C2: Preprocessed Cross-Channel Correlation
    ax_corr_pre = fig.add_subplot(gs[1, 1])
    sub_pre_sample = sub_pre_long[:, ::8]
    corr_pre = np.corrcoef(sub_pre_sample.T)
    im2 = ax_corr_pre.imshow(corr_pre, cmap='coolwarm', vmin=-0.2, vmax=1.0, origin='lower', aspect='auto')
    ax_corr_pre.set_title("C2. Loccar Preprocessed Correlation Matrix", fontsize=12, fontweight='bold')
    ax_corr_pre.set_xlabel("Channel Index (subsampled)", fontsize=10)
    fig.colorbar(im2, ax=ax_corr_pre, fraction=0.046, pad=0.04)

    # Panel C3: Channel Noise Level Distribution
    ax_noise = fig.add_subplot(gs[1, 2])
    noise_raw = np.median(np.abs(sub_raw_long - np.median(sub_raw_long, axis=0)), axis=0) / 0.6745
    noise_pre = np.median(np.abs(sub_pre_long - np.median(sub_pre_long, axis=0)), axis=0) / 0.6745
    ax_noise.plot(noise_raw, label='Raw Noise (MAD uV)', color='#7f8c8d', alpha=0.7)
    ax_noise.plot(noise_pre, label='Post-Loccar Noise (MAD uV)', color='#2ecc71', alpha=0.9)
    ax_noise.set_title("C3. Spatial Noise Floor Across Probe", fontsize=12, fontweight='bold')
    ax_noise.set_xlabel("Channel Index", fontsize=10)
    ax_noise.set_ylabel("Noise Floor (uV)", fontsize=10)
    ax_noise.legend(loc='upper right', fontsize=9)
    ax_noise.grid(True, linestyle=':', alpha=0.5)

    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved preprocessing diagnostic figure to: {out_path}")
    return out_path
