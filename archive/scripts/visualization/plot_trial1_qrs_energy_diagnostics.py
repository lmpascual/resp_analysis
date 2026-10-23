"""
Trial 1 QRS Energy Diagnostic Script for Corrected Channel 1 Raw ECG.
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

def main():
    meta_path = Path(r'Z:\ramirezlab\pascual\rec_data\e086_g0\e086_g0_t0.nidq.meta')
    bin_path = Path(r'Z:\ramirezlab\pascual\rec_data\e086_g0\e086_g0_t0.nidq.bin')
    output_dir = Path(r'c:\Users\lpascu\analysis_src\output_e086')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Loading NIDQ binary recording for e086...")
    analog_data = read_nidq_bin(bin_path, meta_path)
    t_full = analog_data.time_vector
    fs = analog_data.fs

    # Trial 1 anchor: 2nd laser (Ch4) turned on at t = 4087.701769s
    t1_anchor = 4087.701769

    # Window from t_rel = -2.0s to +8.0s (4085.702s to 4095.702s)
    win_start = t1_anchor - 2.0
    win_end = t1_anchor + 8.0

    idx_win = np.where((t_full >= win_start) & (t_full <= win_end))[0]
    t = t_full[idx_win]
    t_rel = t - t1_anchor

    emg_raw = analog_data.all_channels[0][idx_win]
    ecg_raw = analog_data.all_channels[1][idx_win] # True Channel 1 Raw ECG!
    emg_int = analog_data.all_channels[2][idx_win]

    # 1. Bandpass filter (10 - 100 Hz) + Notch (60 Hz)
    nyq = 0.5 * fs
    b_ecg, a_ecg = signal.butter(3, [10.0/nyq, 100.0/nyq], btype='bandpass')
    ecg_filt = signal.filtfilt(b_ecg, a_ecg, ecg_raw)

    w0 = 60.0 / nyq
    b_notch, a_notch = signal.iirnotch(w0, Q=30)
    ecg_filt = signal.filtfilt(b_notch, a_notch, ecg_filt)

    # 2. Pan-Tompkins QRS Derivative Energy Transform: (d/dt ECG)^2
    diff_ecg = np.diff(ecg_filt, prepend=ecg_filt[0]) * fs
    sq_ecg = diff_ecg ** 2
    win_int = int(0.02 * fs)  # 20 ms integration window
    kernel = np.ones(win_int) / win_int
    ecg_energy = np.convolve(sq_ecg, kernel, mode='same')

    # Local adaptive thresholding on derivative energy
    win_samples = int(2.0 * fs)
    num_wins = len(ecg_energy) // win_samples
    min_dist = int(0.08 * fs)

    energy_peaks_idx = []
    for w in range(num_wins):
        idx_s = w * win_samples
        idx_e = (w + 1) * win_samples if w < num_wins - 1 else len(ecg_energy)
        seg = ecg_energy[idx_s:idx_e]
        thresh = np.mean(seg) + 1.2 * np.std(seg)
        p_seg, _ = signal.find_peaks(seg, height=thresh, distance=min_dist)
        energy_peaks_idx.extend(idx_s + p_seg)

    energy_peaks_idx = np.array(energy_peaks_idx)
    t_peaks = t_rel[energy_peaks_idx]
    v_peaks_filt = ecg_filt[energy_peaks_idx]
    v_peaks_energy = ecg_energy[energy_peaks_idx]

    if len(t_peaks) > 1:
        rr_sec = np.diff(t_peaks)
        hr_bpm = 60.0 / rr_sec
        hr_times = t_peaks[1:]
    else:
        hr_bpm = np.array([])
        hr_times = np.array([])

    print(f"[*] Trial 1 QRS Energy Detection Window ({win_start:.2f}s to {win_end:.2f}s) - Channel 1 Raw ECG:")
    print(f" -> QRS Energy Peaks detected: {len(t_peaks)} beats")
    print(f" -> Mean Heart Rate in 10s:   {np.mean(hr_bpm):.1f} bpm")
    if len(hr_times) > 0:
        hr_5s = hr_bpm[(hr_times >= 0.0) & (hr_times <= 5.0)]
        print(f" -> Mean Heart Rate in 5s post-onset: {np.mean(hr_5s):.1f} bpm")

    # 3. Matplotlib Figure (PNG)
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Panel 1: Raw Signals
    axes[0].plot(t_rel, ecg_raw, color='crimson', linewidth=0.8, label='Raw ECG (Ch 1)')
    axes[0].plot(t_rel, emg_raw, color='gray', alpha=0.5, linewidth=0.7, label='Raw Diaphragm EMG (Ch 0)')
    axes[0].plot(t_rel, emg_int, color='darkgreen', linewidth=1.0, label='Integrated EMG (Ch 2)')
    axes[0].set_title('Trial 1: Raw Signals (Ch 1 Raw ECG, Ch 0 Raw EMG, Ch 2 Integrated EMG)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Voltage (V)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Panel 2: Bandpass Filtered ECG (10-100 Hz)
    axes[1].plot(t_rel, ecg_filt, color='royalblue', linewidth=1.0, label='Bandpass Filtered ECG (10-100 Hz)')
    axes[1].scatter(t_peaks, v_peaks_filt, color='black', marker='o', s=35, zorder=5, label=f'QRS Peaks (N={len(t_peaks)})')
    axes[1].set_title('Trial 1: Bandpass Filtered Channel 1 Raw ECG', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Voltage (V)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # Panel 3: Pan-Tompkins QRS Derivative Energy (d/dt ECG)^2
    axes[2].plot(t_rel, ecg_energy, color='darkgreen', linewidth=1.0, label='Pan-Tompkins QRS Energy (d/dt ECG)²')
    axes[2].scatter(t_peaks, v_peaks_energy, color='red', marker='^', s=45, zorder=5, label=f'Energy Peaks (N={len(t_peaks)})')
    axes[2].set_title('Trial 1: QRS Energy Spikes (Converts Upward/Downward Deflections to Positive Energy Spikes)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Energy (V²/s²)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, linestyle=':', alpha=0.6)

    # Panel 4: Instantaneous Heart Rate Tachogram
    if len(hr_times) > 0:
        axes[3].plot(hr_times, hr_bpm, color='crimson', marker='o', markersize=4, linewidth=1.5, label='QRS Energy Heart Rate (bpm)')
    axes[3].axhline(y=540.0, color='black', linestyle=':', label='Baseline HR (~540 bpm)')
    axes[3].set_title('Trial 1: Instantaneous Heart Rate Tachogram (Channel 1 Raw ECG)', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Time Relative to 2nd Laser Onset (seconds)', fontsize=11)
    axes[3].set_ylabel('Heart Rate (bpm)')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, linestyle=':', alpha=0.6)

    # Shade laser stimulation intervals
    for ax in axes:
        ax.axvspan(-1.482, 3.518, color='orange', alpha=0.12, label='Ch 6 Laser')
        ax.axvspan(0.0, 5.0, color='red', alpha=0.15, label='Ch 4 Laser')
        ax.axvline(x=0.0, color='red', linestyle='--', linewidth=1.2)

    axes[3].set_xlim(-2.0, 8.0)
    plt.tight_layout()

    png_path = output_dir / 'e086_trial1_qrs_energy_diagnostics.png'
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f" [+] Saved figure to {png_path}")

    # 4. Interactive Plotly HTML
    print(" -> Generating Interactive Plotly HTML Dashboard...")
    fig_plotly = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Trial 1: Raw ECG Ch 1, Raw EMG Ch 0, & Integrated EMG Ch 2",
            "Trial 1: Bandpass Filtered ECG (10-100 Hz)",
            "Trial 1: Pan-Tompkins QRS Derivative Energy (d/dt ECG)² Spikes",
            "Trial 1: Instantaneous Heart Rate Tachogram (Channel 1 Raw ECG)"
        )
    )

    fig_plotly.add_trace(go.Scattergl(x=t_rel, y=ecg_raw, mode='lines', name='Raw ECG (Ch 1)', line=dict(color='crimson', width=1)), row=1, col=1)
    fig_plotly.add_trace(go.Scattergl(x=t_rel, y=emg_raw, mode='lines', name='Raw EMG (Ch 0)', line=dict(color='gray', width=1)), row=1, col=1)
    fig_plotly.add_trace(go.Scattergl(x=t_rel, y=emg_int, mode='lines', name='Integrated EMG (Ch 2)', line=dict(color='darkgreen', width=1)), row=1, col=1)

    fig_plotly.add_trace(go.Scattergl(x=t_rel, y=ecg_filt, mode='lines', name='Filtered ECG', line=dict(color='royalblue', width=1)), row=2, col=1)
    fig_plotly.add_trace(go.Scatter(x=t_peaks, y=v_peaks_filt, mode='markers', name='QRS Peaks', marker=dict(color='black', size=8)), row=2, col=1)

    fig_plotly.add_trace(go.Scattergl(x=t_rel, y=ecg_energy, mode='lines', name='QRS Energy', line=dict(color='darkgreen', width=1)), row=3, col=1)
    fig_plotly.add_trace(go.Scatter(x=t_peaks, y=v_peaks_energy, mode='markers', name='Energy Peaks', marker=dict(color='red', symbol='triangle-up', size=8)), row=3, col=1)

    fig_plotly.add_trace(go.Scatter(x=hr_times, y=hr_bpm, mode='lines+markers', name='Heart Rate (bpm)', line=dict(color='crimson', width=2)), row=4, col=1)

    for row in range(1, 5):
        fig_plotly.add_vrect(x0=-1.482, x1=3.518, fillcolor="rgba(255, 165, 0, 0.15)", line_width=0, row=row, col=1)
        fig_plotly.add_vrect(x0=0.0, x1=5.0, fillcolor="rgba(255, 0, 0, 0.15)", line_width=0, row=row, col=1)
        fig_plotly.add_vline(x=0.0, line_width=1.5, line_dash="dash", line_color="red", row=row, col=1)

    fig_plotly.update_xaxes(title_text="Time Relative to 2nd Laser Onset (seconds)", row=4, col=1)
    fig_plotly.update_layout(height=1000, title_text="Trial 1 QRS Energy Diagnostic (Channel 1 Raw ECG)", hovermode="x unified", template="plotly_white")

    html_path = output_dir / 'e086_trial1_qrs_energy_diagnostics.html'
    fig_plotly.write_html(str(html_path), include_plotlyjs='cdn')
    print(f" [+] Saved interactive HTML to {html_path}")

if __name__ == '__main__':
    main()
