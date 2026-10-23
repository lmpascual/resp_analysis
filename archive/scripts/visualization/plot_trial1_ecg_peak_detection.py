"""
Trial 1 ECG Diagnostic & Diaphragm Subtraction Explanation Script.
Channel Mapping:
 - Channel 0: Raw Diaphragm EMG
 - Channel 1: Raw ECG
 - Channel 2: Integrated Diaphragm EMG
 - Channel 4: Laser 1
 - Channel 6: Laser 2
"""
import sys
from pathlib import Path
import numpy as np
import scipy.signal as signal
import scipy.interpolate as interpolate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src directory to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.io.io_nidq import read_nidq_bin

def get_qrs_peaks(sig, time_vec, fs):
    diff_sig = np.diff(sig, prepend=sig[0]) * fs
    sq_sig = diff_sig ** 2
    win_int = max(1, int(0.02 * fs))
    kernel = np.ones(win_int) / win_int
    energy = np.convolve(sq_sig, kernel, mode='same')

    win_samples = int(2.0 * fs)
    num_wins = len(energy) // win_samples
    min_dist = int(0.08 * fs)
    p_idx = []
    for w in range(num_wins):
        idx_s = w * win_samples
        idx_e = (w + 1) * win_samples if w < num_wins - 1 else len(energy)
        seg = energy[idx_s:idx_e]
        thresh = np.mean(seg) + 1.2 * np.std(seg)
        p_seg, _ = signal.find_peaks(seg, height=thresh, distance=min_dist)
        p_idx.extend(idx_s + p_seg)
    p_idx = np.array(p_idx)
    return time_vec[p_idx], sig[p_idx], energy

def main():
    meta_path = Path(r'Z:\ramirezlab\pascual\rec_data\e086_g0\e086_g0_t0.nidq.meta')
    bin_path = Path(r'Z:\ramirezlab\pascual\rec_data\e086_g0\e086_g0_t0.nidq.bin')
    output_dir = Path(r'c:\Users\lpascu\analysis_src\output_e086')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Loading NIDQ binary recording for e086...")
    analog_data = read_nidq_bin(bin_path, meta_path)
    t_full = analog_data.time_vector
    fs = analog_data.fs

    t1_anchor = 4087.701769
    win_start = t1_anchor - 2.0
    win_end = t1_anchor + 8.0

    idx_win = np.where((t_full >= win_start) & (t_full <= win_end))[0]
    t = t_full[idx_win]
    t_rel = t - t1_anchor

    emg_raw = analog_data.all_channels[0][idx_win]   # Channel 0 Raw EMG
    ecg_raw = analog_data.all_channels[1][idx_win]   # Channel 1 Raw ECG
    emg_int = analog_data.all_channels[2][idx_win]   # Channel 2 Integrated EMG

    nyq = 0.5 * fs

    # 1. Diaphragm Envelope Transformation
    win_50ms = int(0.05 * fs)
    emg_env = np.convolve(np.abs(emg_raw), np.ones(win_50ms)/win_50ms, mode='same')

    # Fit emg_env onto raw ecg_raw
    c_env, _ = np.polyfit(emg_env, ecg_raw, 1)
    ecg_raw_sub = ecg_raw - c_env * emg_env

    # 2. Wideband Filter (0.5 - 100 Hz)
    b_wide, a_wide = signal.butter(3, [0.5/nyq, 100.0/nyq], btype='bandpass')
    ecg_wide_orig = signal.filtfilt(b_wide, a_wide, ecg_raw)
    ecg_wide_sub = signal.filtfilt(b_wide, a_wide, ecg_raw_sub)

    # 3. Standard Bandpass Filter (10 - 100 Hz)
    b_hp, a_hp = signal.butter(3, [10.0/nyq, 100.0/nyq], btype='bandpass')
    w0 = 60.0 / nyq
    b_notch, a_notch = signal.iirnotch(w0, Q=30)

    ecg_hp_orig = signal.filtfilt(b_notch, a_notch, signal.filtfilt(b_hp, a_hp, ecg_raw))
    ecg_hp_sub = signal.filtfilt(b_notch, a_notch, signal.filtfilt(b_hp, a_hp, ecg_raw_sub))

    # QRS Energy Peak Detections
    t_p_std, v_p_std, energy_std = get_qrs_peaks(ecg_hp_orig, t_rel, fs)
    t_p_sub, v_p_sub, energy_sub = get_qrs_peaks(ecg_hp_sub, t_rel, fs)

    print(f"[*] Trial 1 Channel 1 Raw ECG Peak Detection:")
    print(f" -> QRS Energy Peaks detected in 10s: {len(t_p_std)} peaks")
    print(f" -> Mean Heart Rate:                 {np.mean(60.0 / np.diff(t_p_std)):.1f} bpm")

    # 4. Generate Matplotlib Figure (PNG)
    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)

    # Panel 1: Raw Signals
    axes[0].plot(t_rel, ecg_raw, color='crimson', linewidth=0.8, label='Raw ECG (Ch 1)')
    axes[0].plot(t_rel, emg_raw, color='gray', alpha=0.4, linewidth=0.6, label='Raw Diaphragm EMG (Ch 0)')
    axes[0].plot(t_rel, emg_int, color='darkgreen', linewidth=1.0, label='Integrated EMG (Ch 2)')
    axes[0].set_title('1. Trial 1 Raw Signals (Ch 1 Raw ECG, Ch 0 Raw EMG, Ch 2 Integrated EMG)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Voltage (V)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Panel 2: Wideband ECG (0.5 - 100 Hz)
    axes[1].plot(t_rel, ecg_wide_orig, color='crimson', linewidth=0.9, label='Wideband ECG (0.5-100 Hz) - Without Subtraction')
    axes[1].plot(t_rel, ecg_wide_sub, color='teal', linewidth=0.9, label='Wideband ECG (0.5-100 Hz) - WITH Envelope Subtraction')
    axes[1].set_title('2. Wideband ECG (0.5-100 Hz): Diaphragm Envelope Subtraction Flattens Baseline Swell', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Voltage (V)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # Panel 3: Standard Bandpass ECG (10 - 100 Hz)
    axes[2].plot(t_rel, ecg_hp_orig, color='royalblue', linewidth=0.9, label='Standard Bandpass ECG (10-100 Hz) - Without Subtraction')
    axes[2].plot(t_rel, ecg_hp_sub, color='purple', linewidth=0.9, linestyle='--', label='Standard Bandpass ECG (10-100 Hz) - WITH Envelope Subtraction')
    axes[2].scatter(t_p_std, v_p_std, color='black', marker='o', s=35, zorder=5, label=f'QRS Peaks (N={len(t_p_std)})')
    axes[2].set_title('3. Standard Bandpass ECG (10-100 Hz): 10 Hz Highpass Filter ITSELF Removes Low-Freq Swell', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Voltage (V)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, linestyle=':', alpha=0.6)

    # Panel 4: Pan-Tompkins QRS Derivative Energy
    axes[3].plot(t_rel, energy_std, color='darkgreen', linewidth=1.0, label='Pan-Tompkins QRS Energy (d/dt ECG)²')
    axes[3].scatter(t_p_std, energy_std[np.searchsorted(t_rel, t_p_std)], color='red', marker='^', s=45, zorder=5, label=f'Energy Peaks (N={len(t_p_std)})')
    axes[3].set_title('4. Pan-Tompkins QRS Derivative Energy: Converts Cardiac Deflections into Positive Energy Spikes', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Energy (V²/s²)')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, linestyle=':', alpha=0.6)

    # Panel 5: Heart Rate Tachogram
    if len(t_p_std) > 1:
        axes[4].plot(t_p_std[1:], 60.0 / np.diff(t_p_std), color='crimson', marker='o', markersize=4, label='QRS Energy Heart Rate (bpm)')
    axes[4].axhline(y=540.0, color='black', linestyle=':', label='Baseline HR (~540 bpm)')
    axes[4].set_title('5. Trial 1 Instantaneous Heart Rate Tachogram (Channel 1 Raw ECG)', fontsize=12, fontweight='bold')
    axes[4].set_xlabel('Time Relative to 2nd Laser Onset (seconds)', fontsize=11)
    axes[4].set_ylabel('Heart Rate (bpm)')
    axes[4].legend(loc='upper right')
    axes[4].grid(True, linestyle=':', alpha=0.6)

    for ax in axes:
        ax.axvspan(-1.482, 3.518, color='orange', alpha=0.12)
        ax.axvspan(0.0, 5.0, color='red', alpha=0.15)
        ax.axvline(x=0.0, color='red', linestyle='--', linewidth=1.2)

    axes[4].set_xlim(-2.0, 8.0)
    plt.tight_layout()

    png_path = output_dir / 'e086_trial1_ecg_peak_detection.png'
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f" [+] Saved figure to {png_path}")

    # 5. Interactive Plotly HTML
    print(" -> Generating Interactive Plotly HTML Dashboard...")
    fig_plotly = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "1. Raw ECG Ch 1, Raw EMG Ch 0, & Integrated EMG Ch 2",
            "2. Wideband ECG (0.5-100 Hz): Envelope Subtraction Flattens Baseline Swell",
            "3. Standard Bandpass ECG (10-100 Hz): 10 Hz Highpass Filter ITSELF Removes Baseline Swell",
            "4. Pan-Tompkins QRS Derivative Energy (d/dt ECG)² Spikes",
            "5. Instantaneous Heart Rate Tachogram (Channel 1 Raw ECG)"
        )
    )

    fig_plotly.add_trace(go.Scattergl(x=t_rel, y=ecg_raw, mode='lines', name='Raw ECG (Ch 1)', line=dict(color='crimson', width=1)), row=1, col=1)
    fig_plotly.add_trace(go.Scattergl(x=t_rel, y=emg_raw, mode='lines', name='Raw EMG (Ch 0)', line=dict(color='gray', width=1)), row=1, col=1)
    fig_plotly.add_trace(go.Scattergl(x=t_rel, y=emg_int, mode='lines', name='Integrated EMG (Ch 2)', line=dict(color='darkgreen', width=1)), row=1, col=1)

    fig_plotly.add_trace(go.Scattergl(x=t_rel, y=ecg_wide_orig, mode='lines', name='Wideband (Without Sub)', line=dict(color='crimson', width=1)), row=2, col=1)
    fig_plotly.add_trace(go.Scattergl(x=t_rel, y=ecg_wide_sub, mode='lines', name='Wideband (WITH Sub)', line=dict(color='teal', width=1.5)), row=2, col=1)

    fig_plotly.add_trace(go.Scattergl(x=t_rel, y=ecg_hp_orig, mode='lines', name='Standard 10-100Hz (Without Sub)', line=dict(color='royalblue', width=1)), row=3, col=1)
    fig_plotly.add_trace(go.Scattergl(x=t_rel, y=ecg_hp_sub, mode='lines', name='Standard 10-100Hz (WITH Sub)', line=dict(color='purple', width=1, dash='dash')), row=3, col=1)

    fig_plotly.add_trace(go.Scattergl(x=t_rel, y=energy_std, mode='lines', name='QRS Energy', line=dict(color='darkgreen', width=1)), row=4, col=1)
    fig_plotly.add_trace(go.Scatter(x=t_p_std, y=energy_std[np.searchsorted(t_rel, t_p_std)], mode='markers', name='QRS Energy Peaks', marker=dict(color='red', symbol='triangle-up', size=8)), row=4, col=1)

    if len(t_p_std) > 1:
        fig_plotly.add_trace(go.Scatter(x=t_p_std[1:], y=60.0/np.diff(t_p_std), mode='lines+markers', name='Heart Rate (bpm)', line=dict(color='crimson', width=2)), row=5, col=1)

    for row in range(1, 6):
        fig_plotly.add_vrect(x0=-1.482, x1=3.518, fillcolor="rgba(255, 165, 0, 0.15)", line_width=0, row=row, col=1)
        fig_plotly.add_vrect(x0=0.0, x1=5.0, fillcolor="rgba(255, 0, 0, 0.15)", line_width=0, row=row, col=1)
        fig_plotly.add_vline(x=0.0, line_width=1.5, line_dash="dash", line_color="red", row=row, col=1)

    fig_plotly.update_xaxes(title_text="Time Relative to 2nd Laser Onset (seconds)", row=5, col=1, range=[-2.0, 8.0])
    fig_plotly.update_layout(height=1200, title_text="Trial 1 ECG Peak Detection (Channel 1 Raw ECG)", hovermode="x unified", template="plotly_white")

    html_path = output_dir / 'e086_trial1_ecg_peak_detection.html'
    fig_plotly.write_html(str(html_path), include_plotlyjs='cdn')
    print(f" [+] Saved interactive HTML to {html_path}")

if __name__ == '__main__':
    main()
