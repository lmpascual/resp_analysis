"""
Optogenetics Laser Trial Alignment & Baseline Window Analysis Script for e086.
Channel Mapping:
 - Channel 0: Raw Diaphragm EMG (with detected breath peak markers at fixed Y=+0.55V, blue triangles)
 - Channel 1: Raw ECG (with detected cardiac QRS R-peak markers at fixed Y=+0.35V, small green circles close to trace)
 - Channel 4: Laser 1
 - Channel 6: Laser 2
 - Channel 8: Sync Timer

Analyzes EMG & ECG in window 2985s to end (~4665.9s).
Uses Pan-Tompkins QRS Derivative Energy transform for ECG R-peak detection on Channel 1.
Evaluates 15-second window: 5 seconds prior to trial start (-5s) to 10 seconds after trial start (+10s).
Compares trial instantaneous breathing & heart rate against a 15-second sliding baseline window average (Mean ± 1 SEM).
Includes active representative raw traces of baseline (t=4070s to 4085s), Trial 1, and Trial 2 with close fixed Y-level scatter markers for heartbeats (small green circles at Y=+0.35V) and breath onsets (blue triangles at Y=+0.55V).
"""
import sys
from pathlib import Path
import numpy as np
import scipy.interpolate as interpolate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src directory to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.processors.proc_ecg import process_ecg
from src.pipeline.config import EMGConfig, ECGConfig

def make_rate_interpolator(times, rates, default_val=140.0):
    """Robust 1D linear interpolator with boundary fill."""
    if len(times) == 0:
        return lambda x: np.full_like(x, default_val)
    elif len(times) == 1:
        return lambda x: np.full_like(x, rates[0])
    else:
        return interpolate.interp1d(
            times, rates, kind='linear', bounds_error=False,
            fill_value=(rates[0], rates[-1])
        )

def main():
    meta_path = Path(r'Z:\ramirezlab\pascual\rec_data\e086_g0\e086_g0_t0.nidq.meta')
    bin_path = Path(r'Z:\ramirezlab\pascual\rec_data\e086_g0\e086_g0_t0.nidq.bin')
    output_dir = Path(r'c:\Users\lpascu\analysis_src\output_e086')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Loading NIDQ binary recording for e086...")
    analog_data = read_nidq_bin(bin_path, meta_path)
    t_full = analog_data.time_vector
    fs = analog_data.fs
    print(f" -> Full recording duration: {t_full[-1]:.2f}s | Fs = {fs:.1f} Hz")

    # Crop window t >= 2985.0 s (00:49:45)
    win_start = 2985.0
    idx_win = np.where(t_full >= win_start)[0]
    t = t_full[idx_win]

    emg_raw = analog_data.all_channels[0][idx_win]   # Channel 0 Raw EMG
    ecg_raw = analog_data.all_channels[1][idx_win]   # Channel 1 Raw ECG
    opto1_raw = analog_data.all_channels[4][idx_win] # Channel 4 laser
    opto2_raw = analog_data.all_channels[6][idx_win] # Channel 6 laser

    print(f" -> Window 2985s to end: {t[0]:.2f}s to {t[-1]:.2f}s (duration: {t[-1]-t[0]:.2f}s)")

    # 1. Process Diaphragm EMG
    print(" -> Processing Diaphragm EMG (Filtering, Envelope, Breath Peak Detection)...")
    emg_config = EMGConfig(fs=fs, highpass_cutoff=300.0, lowpass_cutoff=3000.0, notch_freq=60.0,
                           smoothing_window_ms=50.0, peak_threshold_sd=2.0, min_inter_breath_interval_ms=150.0)
    emg_proc = process_emg(emg_raw, t, emg_config)
    breath_onsets = emg_proc.breath_onsets
    print(f"    - Detected {len(breath_onsets)} breaths in window.")

    if len(breath_onsets) > 1:
        ibis = np.diff(breath_onsets)
        rr_inst = 60.0 / ibis
        rr_times = breath_onsets[1:]
    else:
        rr_inst = np.array([])
        rr_times = np.array([])

    f_rr = make_rate_interpolator(rr_times, rr_inst, 150.0)

    # 2. Process ECG on Channel 1 with Pan-Tompkins QRS Energy Processor & Physiological Filtering
    print(" -> Processing Channel 1 Raw ECG (Pan-Tompkins QRS Energy Transform & Physiological Filtering)...")
    ecg_config = ECGConfig(enabled=True, channel_index=1, fs=fs, highpass_cutoff=10.0, lowpass_cutoff=100.0,
                           notch_freq=60.0, r_peak_threshold_sd=1.2, min_rr_interval_ms=80.0)
    ecg_proc = process_ecg(ecg_raw, t, ecg_config)
    r_peaks = ecg_proc.r_peaks
    hr_inst = ecg_proc.instantaneous_hr
    hr_times = r_peaks[1:] if len(r_peaks) > len(hr_inst) else r_peaks

    if len(hr_inst) > 0:
        print(f"    - Retained {len(hr_inst)} clean physiological QRS HR points (250-750 bpm).")
        print(f"    - Mean Baseline HR: {np.mean(hr_inst):.1f} ± {np.std(hr_inst):.1f} bpm")
    else:
        print("    - Warning: No clean QRS HR points detected.")

    f_hr = make_rate_interpolator(hr_times, hr_inst, 520.0)

    # 3. Laser Pulse Detection & Trial Alignment
    print(" -> Detecting Laser Stimulation Trials (Channels 4 & 6)...")
    opto1_onsets = t[np.where(np.diff((opto1_raw > 1.0).astype(int)) == 1)[0]]  # Laser Ch 4
    opto2_onsets = t[np.where(np.diff((opto2_raw > 0.2).astype(int)) == 1)[0]]  # Laser Ch 6

    def get_train_starts(pulse_onsets, min_gap=5.0):
        if len(pulse_onsets) == 0:
            return []
        gaps = np.diff(pulse_onsets)
        train_starts = [pulse_onsets[0]] + list(pulse_onsets[1:][gaps > min_gap])
        return train_starts

    ch4_starts = get_train_starts(opto1_onsets)
    ch6_starts = get_train_starts(opto2_onsets)

    trial1_onset_2nd = float([s for s in ch4_starts if abs(s - 4087.7) < 5.0][0])
    trial2_onset_2nd = float([s for s in ch4_starts if abs(s - 4216.3) < 5.0][0])

    trial1_onset_1st = float([s for s in ch6_starts if abs(s - 4086.2) < 5.0][0])
    trial2_onset_1st = float([s for s in ch6_starts if abs(s - 4215.4) < 5.0][0])

    print(f"    - Trial 1: 1st laser (Ch6) at {trial1_onset_1st:.3f}s | 2nd laser (Ch4) at {trial1_onset_2nd:.3f}s (t=0 anchor)")
    print(f"    - Trial 2: 1st laser (Ch6) at {trial2_onset_1st:.3f}s | 2nd laser (Ch4) at {trial2_onset_2nd:.3f}s (t=0 anchor)")

    # 4. Window Setup: 5 seconds prior (-5.0s) to 10 seconds post (+10.0s)
    t_rel_grid = np.linspace(-5.0, 10.0, 1501)  # 100 Hz relative grid for 15s (-5s to +10s)
    win_dur = 15.0
    step_sec = 1.0

    print(" -> Computing 15-second Sliding Baseline Window Statistics (Mean ± SEM)...")
    base_start = 2985.0
    base_end = 4086.0

    # Sliding 15-second windows across baseline period
    win_starts = np.arange(base_start, base_end - win_dur, step_sec)
    rr_baseline_matrix = []
    hr_baseline_matrix = []

    for w_start in win_starts:
        abs_grid = w_start + (t_rel_grid + 5.0)  # Offset grid to start at w_start
        rr_baseline_matrix.append(f_rr(abs_grid))
        hr_baseline_matrix.append(f_hr(abs_grid))

    rr_baseline_matrix = np.array(rr_baseline_matrix)
    hr_baseline_matrix = np.array(hr_baseline_matrix)

    n_base = len(rr_baseline_matrix)
    rr_base_mean = np.mean(rr_baseline_matrix, axis=0)
    rr_base_sem = np.std(rr_baseline_matrix, axis=0) / np.sqrt(n_base)

    hr_base_mean = np.mean(hr_baseline_matrix, axis=0)
    hr_base_sem = np.std(hr_baseline_matrix, axis=0) / np.sqrt(n_base)

    print(f"    - Computed baseline across {n_base} sliding 15-second windows.")
    print(f"    - Baseline Mean RR: {np.mean(rr_base_mean):.2f} ± {np.mean(rr_base_sem):.4f} breaths/min")
    print(f"    - Baseline Mean HR: {np.mean(hr_base_mean):.2f} ± {np.mean(hr_base_sem):.4f} bpm")

    # 5. Extract Trial 1 and Trial 2 Traces (t_rel in [-5, 10]s)
    t1_abs_grid = trial1_onset_2nd + t_rel_grid
    rr_trial1 = f_rr(t1_abs_grid)
    hr_trial1 = f_hr(t1_abs_grid)

    t2_abs_grid = trial2_onset_2nd + t_rel_grid
    rr_trial2 = f_rr(t2_abs_grid)
    hr_trial2 = f_hr(t2_abs_grid)

    # 6. Extract Active Raw Traces & Close Fixed Y-Level Peak Markers (-5s to +10s rel)
    # Fixed Y levels close to signals:
    # Heartbeat markers (small green circles): Y_ecg_marker = +0.35 V
    # Breath markers (blue triangles): Y_emg_marker = +0.55 V
    Y_ECG_MARKER = 0.25
    Y_EMG_MARKER = 0.55

    def get_peak_timestamps(time_peaks, t_win_start, t_win_end):
        in_win = (time_peaks >= t_win_start) & (time_peaks <= t_win_end)
        return time_peaks[in_win]

    # Active Baseline representative window right before Trial 1: t = 4070.0s to 4085.0s (15s window, anchored at 4075.0s)
    base_anchor = 4075.0
    t_base_abs_start = base_anchor - 5.0 # 4070s
    t_base_abs_end = base_anchor + 10.0  # 4085s
    idx_base_rep = np.where((t_full >= t_base_abs_start) & (t_full <= t_base_abs_end))[0]
    t_base_rep_rel = (t_full[idx_base_rep] - base_anchor)  # -5 to +10s
    emg_base_rep = analog_data.all_channels[0][idx_base_rep]
    ecg_base_rep = analog_data.all_channels[1][idx_base_rep]

    base_b_rel = get_peak_timestamps(breath_onsets, t_base_abs_start, t_base_abs_end) - base_anchor
    base_r_rel = get_peak_timestamps(r_peaks, t_base_abs_start, t_base_abs_end) - base_anchor

    # Trial 1 raw: t_rel in [-5.0, 10.0]s
    t1_abs_start = trial1_onset_2nd - 5.0
    t1_abs_end = trial1_onset_2nd + 10.0
    idx_t1_raw = np.where((t_full >= t1_abs_start) & (t_full <= t1_abs_end))[0]
    t_t1_raw_rel = t_full[idx_t1_raw] - trial1_onset_2nd
    emg_t1_raw = analog_data.all_channels[0][idx_t1_raw]
    ecg_t1_raw = analog_data.all_channels[1][idx_t1_raw]
    opto1_t1_raw = analog_data.all_channels[4][idx_t1_raw]
    opto2_t1_raw = analog_data.all_channels[6][idx_t1_raw]

    t1_b_rel = get_peak_timestamps(breath_onsets, t1_abs_start, t1_abs_end) - trial1_onset_2nd
    t1_r_rel = get_peak_timestamps(r_peaks, t1_abs_start, t1_abs_end) - trial1_onset_2nd

    # Trial 2 raw: t_rel in [-5.0, 10.0]s
    t2_abs_start = trial2_onset_2nd - 5.0
    t2_abs_end = trial2_onset_2nd + 10.0
    idx_t2_raw = np.where((t_full >= t2_abs_start) & (t_full <= t2_abs_end))[0]
    t_t2_raw_rel = t_full[idx_t2_raw] - trial2_onset_2nd
    emg_t2_raw = analog_data.all_channels[0][idx_t2_raw]
    ecg_t2_raw = analog_data.all_channels[1][idx_t2_raw]
    opto1_t2_raw = analog_data.all_channels[4][idx_t2_raw]
    opto2_t2_raw = analog_data.all_channels[6][idx_t2_raw]

    t2_b_rel = get_peak_timestamps(breath_onsets, t2_abs_start, t2_abs_end) - trial2_onset_2nd
    t2_r_rel = get_peak_timestamps(r_peaks, t2_abs_start, t2_abs_end) - trial2_onset_2nd

    # 7. Generate Multi-Panel Matplotlib Figure (PNG)
    print(" -> Generating 5-Panel Figure (-5s to +10s) with Active Baseline Raw Trace & Close Markers (PNG)...")
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    # Panel 1: Instantaneous Breathing Rate (-5s to +10s)
    axes[0].plot(t_rel_grid, rr_base_mean, color='black', linewidth=2.5, label=f'Baseline 15s Window Average (N={n_base})')
    axes[0].fill_between(t_rel_grid, rr_base_mean - rr_base_sem, rr_base_mean + rr_base_sem,
                         color='gray', alpha=0.35, label='Baseline ± 1 SEM')
    axes[0].plot(t_rel_grid, rr_trial1, color='crimson', linewidth=2.0, linestyle='-', label=f'Trial 1 (2nd Laser: {trial1_onset_2nd:.2f}s)')
    axes[0].plot(t_rel_grid, rr_trial2, color='darkviolet', linewidth=2.0, linestyle='--', label=f'Trial 2 (2nd Laser: {trial2_onset_2nd:.2f}s)')
    axes[0].axvline(x=0.0, color='red', linestyle='--', linewidth=1.5, label='t=0 (2nd Laser Onset)')
    axes[0].set_title('Subject e086: Instantaneous Respiratory Rate (-5s Prior to +10s Post 2nd Laser Onset)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Breathing Rate\n(breaths/min)', fontsize=10)
    axes[0].set_xlim(-5.0, 10.0)
    axes[0].legend(loc='upper right', framealpha=0.9, fontsize=9)
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Panel 2: Instantaneous Heart Rate (-5s to +10s)
    axes[1].plot(t_rel_grid, hr_base_mean, color='black', linewidth=2.5, label=f'Baseline 15s Window Average (N={n_base})')
    axes[1].fill_between(t_rel_grid, hr_base_mean - hr_base_sem, hr_base_mean + hr_base_sem,
                         color='gray', alpha=0.35, label='Baseline ± 1 SEM')
    axes[1].plot(t_rel_grid, hr_trial1, color='crimson', linewidth=2.0, linestyle='-', label=f'Trial 1 HR (Ch 1 ECG)')
    axes[1].plot(t_rel_grid, hr_trial2, color='darkviolet', linewidth=2.0, linestyle='--', label=f'Trial 2 HR (Ch 1 ECG)')
    axes[1].axvline(x=0.0, color='red', linestyle='--', linewidth=1.5, label='t=0 (2nd Laser Onset)')
    axes[1].set_title('Subject e086: Instantaneous Heart Rate (Channel 1 Raw ECG - QRS Energy Detection)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Heart Rate\n(bpm)', fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # Panel 3: Representative Active Baseline Raw Traces & Close Fixed Y Peak Markers
    axes[2].plot(t_base_rep_rel, ecg_base_rep, color='crimson', linewidth=0.8, label='Raw ECG (Ch 1)')
    axes[2].scatter(base_r_rel, np.full_like(base_r_rel, Y_ECG_MARKER), color='limegreen', marker='o', s=12, zorder=5, label=f'Heartbeats (N={len(base_r_rel)})')
    axes[2].plot(t_base_rep_rel, emg_base_rep, color='gray', alpha=0.6, linewidth=0.7, label='Raw Diaphragm EMG (Ch 0)')
    axes[2].scatter(base_b_rel, np.full_like(base_b_rel, Y_EMG_MARKER), color='blue', marker='v', s=20, zorder=5, label=f'Breaths (N={len(base_b_rel)})')
    axes[2].set_title('Representative Active Baseline Raw Traces (t = 4070s to 4085s, Pre-Trial 1)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Voltage (V)', fontsize=10)
    axes[2].set_xlim(-5.0, 10.0)
    axes[2].set_ylim(-3.0, 1.2)
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].grid(True, linestyle=':', alpha=0.6)

    # Panel 4: Trial 1 Raw Traces & Close Fixed Y Peak Markers (-5s to +10s rel)
    axes[3].plot(t_t1_raw_rel, ecg_t1_raw, color='crimson', linewidth=0.8, label='Raw ECG (Ch 1)')
    axes[3].scatter(t1_r_rel, np.full_like(t1_r_rel, Y_ECG_MARKER), color='limegreen', marker='o', s=12, zorder=5, label=f'Heartbeats (N={len(t1_r_rel)})')
    axes[3].plot(t_t1_raw_rel, emg_t1_raw, color='gray', alpha=0.6, linewidth=0.7, label='Raw Diaphragm EMG (Ch 0)')
    axes[3].scatter(t1_b_rel, np.full_like(t1_b_rel, Y_EMG_MARKER), color='blue', marker='v', s=20, zorder=5, label=f'Breaths (N={len(t1_b_rel)})')
    axes[3].plot(t_t1_raw_rel, opto1_t1_raw * 0.5 - 2.5, color='red', alpha=0.8, linewidth=1.0, label='Laser 1 (Ch 4)')
    axes[3].plot(t_t1_raw_rel, opto2_t1_raw * 1.5 - 3.5, color='orange', alpha=0.8, linewidth=1.0, label='Laser 2 (Ch 6)')
    axes[3].axvline(x=0.0, color='red', linestyle='--', linewidth=1.2, label='t=0 (2nd Laser Onset)')
    axes[3].set_title(f'Trial 1 Raw Signals & Peak Markers (2nd Laser Onset: {trial1_onset_2nd:.2f}s)', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Voltage (V)', fontsize=10)
    axes[3].set_xlim(-5.0, 10.0)
    axes[3].set_ylim(-4.0, 1.2)
    axes[3].legend(loc='upper right', fontsize=9)
    axes[3].grid(True, linestyle=':', alpha=0.6)

    # Panel 5: Trial 2 Raw Traces & Close Fixed Y Peak Markers (-5s to +10s rel)
    axes[4].plot(t_t2_raw_rel, ecg_t2_raw, color='crimson', linewidth=0.8, label='Raw ECG (Ch 1)')
    axes[4].scatter(t2_r_rel, np.full_like(t2_r_rel, Y_ECG_MARKER), color='limegreen', marker='o', s=12, zorder=5, label=f'Heartbeats (N={len(t2_r_rel)})')
    axes[4].plot(t_t2_raw_rel, emg_t2_raw, color='gray', alpha=0.6, linewidth=0.7, label='Raw Diaphragm EMG (Ch 0)')
    axes[4].scatter(t2_b_rel, np.full_like(t2_b_rel, Y_EMG_MARKER), color='blue', marker='v', s=20, zorder=5, label=f'Breaths (N={len(t2_b_rel)})')
    axes[4].plot(t_t2_raw_rel, opto1_t2_raw * 0.5 - 2.5, color='red', alpha=0.8, linewidth=1.0, label='Laser 1 (Ch 4)')
    axes[4].plot(t_t2_raw_rel, opto2_t2_raw * 1.5 - 3.5, color='orange', alpha=0.8, linewidth=1.0, label='Laser 2 (Ch 6)')
    axes[4].axvline(x=0.0, color='red', linestyle='--', linewidth=1.2, label='t=0 (2nd Laser Onset)')
    axes[4].set_title(f'Trial 2 Raw Signals & Peak Markers (2nd Laser Onset: {trial2_onset_2nd:.2f}s)', fontsize=12, fontweight='bold')
    axes[4].set_xlabel('Time Relative to 2nd Laser Onset (seconds)', fontsize=10)
    axes[4].set_ylabel('Voltage (V)', fontsize=10)
    axes[4].set_xlim(-5.0, 10.0)
    axes[4].set_ylim(-4.0, 1.2)
    axes[4].legend(loc='upper right', fontsize=9)
    axes[4].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    png_path = output_dir / 'e086_opto_5s_response.png'
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f" [+] Saved 5-panel figure to {png_path}")

    # 8. Generate Interactive Plotly HTML Dashboard (-5s to +10s)
    print(" -> Generating Interactive Plotly HTML Dashboard (-5s to +10s)...")
    fig_plotly = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Instantaneous Respiratory Rate (breaths/min) [-5s Prior to +10s Post 2nd Laser Onset]",
            "Instantaneous Heart Rate (bpm) [Ch 1 Raw ECG - QRS Energy]",
            "Representative Active Baseline Raw Signals & Markers (t = 4070s to 4085s)",
            f"Trial 1 Raw Signals & Markers (2nd Laser Onset: {trial1_onset_2nd:.2f}s)",
            f"Trial 2 Raw Signals & Markers (2nd Laser Onset: {trial2_onset_2nd:.2f}s)"
        )
    )

    # Subplot 1: Breathing Rate
    fig_plotly.add_trace(
        go.Scatter(x=np.concatenate([t_rel_grid, t_rel_grid[::-1]]),
                   y=np.concatenate([rr_base_mean + rr_base_sem, (rr_base_mean - rr_base_sem)[::-1]]),
                   fill='toself', fillcolor='rgba(128, 128, 128, 0.3)',
                   line=dict(color='rgba(255,255,255,0)'),
                   showlegend=True, name='Baseline ± 1 SEM'),
        row=1, col=1
    )
    fig_plotly.add_trace(go.Scatter(x=t_rel_grid, y=rr_base_mean, mode='lines', name='Baseline Average', line=dict(color='black', width=3)), row=1, col=1)
    fig_plotly.add_trace(go.Scatter(x=t_rel_grid, y=rr_trial1, mode='lines+markers', name='Trial 1 RR', line=dict(color='crimson', width=2)), row=1, col=1)
    fig_plotly.add_trace(go.Scatter(x=t_rel_grid, y=rr_trial2, mode='lines+markers', name='Trial 2 RR', line=dict(color='purple', width=2, dash='dash')), row=1, col=1)

    # Subplot 2: Heart Rate
    fig_plotly.add_trace(
        go.Scatter(x=np.concatenate([t_rel_grid, t_rel_grid[::-1]]),
                   y=np.concatenate([hr_base_mean + hr_base_sem, (hr_base_mean - hr_base_sem)[::-1]]),
                   fill='toself', fillcolor='rgba(128, 128, 128, 0.3)',
                   line=dict(color='rgba(255,255,255,0)'),
                   showlegend=False, name='Baseline ± 1 SEM'),
        row=2, col=1
    )
    fig_plotly.add_trace(go.Scatter(x=t_rel_grid, y=hr_base_mean, mode='lines', name='Baseline Average HR', line=dict(color='black', width=3)), row=2, col=1)
    fig_plotly.add_trace(go.Scatter(x=t_rel_grid, y=hr_trial1, mode='lines+markers', name='Trial 1 HR', line=dict(color='crimson', width=2)), row=2, col=1)
    fig_plotly.add_trace(go.Scatter(x=t_rel_grid, y=hr_trial2, mode='lines+markers', name='Trial 2 HR', line=dict(color='purple', width=2, dash='dash')), row=2, col=1)

    # Subplot 3: Representative Baseline Raw Traces & Markers
    fig_plotly.add_trace(go.Scattergl(x=t_base_rep_rel, y=ecg_base_rep, mode='lines', name='Baseline Raw ECG (Ch 1)', line=dict(color='crimson', width=1)), row=3, col=1)
    fig_plotly.add_trace(go.Scatter(x=base_r_rel, y=np.full_like(base_r_rel, Y_ECG_MARKER), mode='markers', name='Baseline Heartbeats', marker=dict(color='limegreen', symbol='circle', size=4)), row=3, col=1)
    fig_plotly.add_trace(go.Scattergl(x=t_base_rep_rel, y=emg_base_rep, mode='lines', name='Baseline Raw EMG (Ch 0)', line=dict(color='gray', width=1)), row=3, col=1)
    fig_plotly.add_trace(go.Scatter(x=base_b_rel, y=np.full_like(base_b_rel, Y_EMG_MARKER), mode='markers', name='Baseline Breaths', marker=dict(color='blue', symbol='triangle-down', size=6)), row=3, col=1)

    # Subplot 4: Trial 1 Raw Traces & Markers
    fig_plotly.add_trace(go.Scattergl(x=t_t1_raw_rel, y=ecg_t1_raw, mode='lines', name='Trial 1 Raw ECG (Ch 1)', line=dict(color='crimson', width=1)), row=4, col=1)
    fig_plotly.add_trace(go.Scatter(x=t1_r_rel, y=np.full_like(t1_r_rel, Y_ECG_MARKER), mode='markers', name='Trial 1 Heartbeats', marker=dict(color='limegreen', symbol='circle', size=4)), row=4, col=1)
    fig_plotly.add_trace(go.Scattergl(x=t_t1_raw_rel, y=emg_t1_raw, mode='lines', name='Trial 1 Raw EMG (Ch 0)', line=dict(color='gray', width=1)), row=4, col=1)
    fig_plotly.add_trace(go.Scatter(x=t1_b_rel, y=np.full_like(t1_b_rel, Y_EMG_MARKER), mode='markers', name='Trial 1 Breaths', marker=dict(color='blue', symbol='triangle-down', size=6)), row=4, col=1)
    fig_plotly.add_trace(go.Scattergl(x=t_t1_raw_rel, y=opto1_t1_raw * 0.5 - 2.5, mode='lines', name='Laser 1 (Ch 4)', line=dict(color='red', width=1)), row=4, col=1)
    fig_plotly.add_trace(go.Scattergl(x=t_t1_raw_rel, y=opto2_t1_raw * 1.5 - 3.5, mode='lines', name='Laser 2 (Ch 6)', line=dict(color='orange', width=1)), row=4, col=1)

    # Subplot 5: Trial 2 Raw Traces & Markers
    fig_plotly.add_trace(go.Scattergl(x=t_t2_raw_rel, y=ecg_t2_raw, mode='lines', name='Trial 2 Raw ECG (Ch 1)', line=dict(color='crimson', width=1)), row=5, col=1)
    fig_plotly.add_trace(go.Scatter(x=t2_r_rel, y=np.full_like(t2_r_rel, Y_ECG_MARKER), mode='markers', name='Trial 2 Heartbeats', marker=dict(color='limegreen', symbol='circle', size=4)), row=5, col=1)
    fig_plotly.add_trace(go.Scattergl(x=t_t2_raw_rel, y=emg_t2_raw, mode='lines', name='Trial 2 Raw EMG (Ch 0)', line=dict(color='gray', width=1)), row=5, col=1)
    fig_plotly.add_trace(go.Scatter(x=t2_b_rel, y=np.full_like(t2_b_rel, Y_EMG_MARKER), mode='markers', name='Trial 2 Breaths', marker=dict(color='blue', symbol='triangle-down', size=6)), row=5, col=1)
    fig_plotly.add_trace(go.Scattergl(x=t_t2_raw_rel, y=opto1_t2_raw * 0.5 - 2.5, mode='lines', name='Laser 1 (Ch 4)', line=dict(color='red', width=1)), row=5, col=1)
    fig_plotly.add_trace(go.Scattergl(x=t_t2_raw_rel, y=opto2_t2_raw * 1.5 - 3.5, mode='lines', name='Laser 2 (Ch 6)', line=dict(color='orange', width=1)), row=5, col=1)

    for row in range(1, 6):
        fig_plotly.add_vline(x=0.0, line_width=1.5, line_dash="dash", line_color="red", row=row, col=1)

    fig_plotly.update_xaxes(title_text="Time Relative to 2nd Laser Onset (seconds)", row=5, col=1, range=[-5.0, 10.0])
    fig_plotly.update_layout(
        height=1300,
        title_text="Subject e086: Dual Optogenetic Laser Stimulation (-5s Prior to +10s Post 2nd Laser Onset)",
        hovermode="x unified",
        template="plotly_white"
    )

    html_path = output_dir / 'e086_opto_5s_response.html'
    fig_plotly.write_html(str(html_path), include_plotlyjs='cdn')
    print(f" [+] Saved interactive HTML to {html_path}")

    # Write summary stats to file
    summary_txt = output_dir / 'e086_opto_5s_summary.txt'
    with open(summary_txt, 'w') as f:
        f.write(f"Subject e086 Optogenetics Analysis Summary (Active Baseline Raw ECG & Close Peak Markers)\n")
        f.write(f"="*85 + "\n")
        f.write(f"Baseline window: 2985.0s to 4086.0s ({n_base} sliding 15s windows)\n")
        f.write(f"Baseline Mean Breathing Rate (-5s to +10s): {np.mean(rr_base_mean):.2f} bpm (SEM: {np.mean(rr_base_sem):.4f})\n")
        f.write(f"Baseline Mean Heart Rate (-5s to +10s): {np.mean(hr_base_mean):.2f} bpm (SEM: {np.mean(hr_base_sem):.4f})\n\n")

        f.write(f"Trial 1 (2nd Laser Onset at {trial1_onset_2nd:.3f}s):\n")
        f.write(f"  - Pre-stimulation (-5s to 0s) Mean Breathing Rate: {np.mean(rr_trial1[t_rel_grid < 0]):.2f} bpm\n")
        f.write(f"  - Pre-stimulation (-5s to 0s) Mean Heart Rate:     {np.mean(hr_trial1[t_rel_grid < 0]):.2f} bpm\n")
        f.write(f"  - Post-stimulation (0s to 10s) Mean Breathing Rate: {np.mean(rr_trial1[t_rel_grid >= 0]):.2f} bpm\n")
        f.write(f"  - Post-stimulation (0s to 10s) Mean Heart Rate:     {np.mean(hr_trial1[t_rel_grid >= 0]):.2f} bpm\n")
        f.write(f"  - Close Markers: {len(t1_r_rel)} heartbeats (small green circles at Y=0.35V), {len(t1_b_rel)} breath onsets (blue triangles at Y=0.55V)\n\n")

        f.write(f"Trial 2 (2nd Laser Onset at {trial2_onset_2nd:.3f}s):\n")
        f.write(f"  - Pre-stimulation (-5s to 0s) Mean Breathing Rate: {np.mean(rr_trial2[t_rel_grid < 0]):.2f} bpm\n")
        f.write(f"  - Pre-stimulation (-5s to 0s) Mean Heart Rate:     {np.mean(hr_trial2[t_rel_grid < 0]):.2f} bpm\n")
        f.write(f"  - Post-stimulation (0s to 10s) Mean Breathing Rate: {np.mean(rr_trial2[t_rel_grid >= 0]):.2f} bpm\n")
        f.write(f"  - Post-stimulation (0s to 10s) Mean Heart Rate:     {np.mean(hr_trial2[t_rel_grid >= 0]):.2f} bpm\n")
        f.write(f"  - Close Markers: {len(t2_r_rel)} heartbeats (small green circles at Y=0.35V), {len(t2_b_rel)} breath onsets (blue triangles at Y=0.55V)\n")
    print(f" [+] Saved summary text to {summary_txt}")

if __name__ == '__main__':
    main()
