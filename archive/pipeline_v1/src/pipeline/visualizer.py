"""
Interactive Visual Quality & Stationarity Inspection Module.
Generates diagnostic plots for EMG SNR/stationarity, ECG R-peaks, Opto/Sync alignment,
Neuropixel drift, multi-channel raw signal overview, and Optogenetics Phase Response Curves (PRC).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless rendering for saving figure files safely
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from .models import SessionResults

def generate_interactive_emg_html(results: SessionResults, output_dir: Path) -> Path:
    """
    Generate an interactive, scrollable Plotly HTML file allowing full-session
    navigation and high-resolution waveform inspection of diaphragm EMG with
    translucent shaded laser pulse overlays.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    time = results.analog_raw.time_vector
    emg_raw = results.analog_raw.emg_raw
    emg_filt = results.emg.filtered
    emg_env = results.emg.envelope
    onsets = results.emg.breath_onsets

    # Subsample for WebGL rendering performance if array is massive (>300k points)
    max_pts = 300000
    if len(time) > max_pts:
        step = len(time) // max_pts
        time_sub = time[::step]
        raw_sub = emg_raw[::step]
        filt_sub = emg_filt[::step]
        env_sub = emg_env[::step]
    else:
        time_sub = time
        raw_sub = emg_raw
        filt_sub = emg_filt
        env_sub = emg_env

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"Diaphragm EMG Raw & Bandpass Filtered Waveforms - {results.animal_id} {results.session_id}",
            "Diaphragm EMG Envelope & Breath Onset Peaks",
            "Instantaneous Respiratory Rate (Breaths / min)"
        )
    )

    # Subplot 1: Raw vs Filtered EMG
    fig.add_trace(
        go.Scattergl(x=time_sub, y=raw_sub, mode='lines', name='Raw EMG',
                     line=dict(color='rgba(128, 128, 128, 0.5)', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scattergl(x=time_sub, y=filt_sub, mode='lines', name='Filtered (300-3000 Hz)',
                     line=dict(color='rgba(30, 144, 255, 0.8)', width=1)),
        row=1, col=1
    )

    # Subplot 2: Envelope & Breath Onset Peaks
    fig.add_trace(
        go.Scattergl(x=time_sub, y=env_sub, mode='lines', name='Envelope',
                     line=dict(color='darkred', width=1.5)),
        row=2, col=1
    )
    if len(onsets) > 0:
        peak_y = np.interp(onsets, time_sub, env_sub)
        fig.add_trace(
            go.Scatter(x=onsets, y=peak_y, mode='markers', name='Breath Onset',
                       marker=dict(symbol='triangle-up', color='green', size=8)),
            row=2, col=1
        )

    # Subplot 3: Instantaneous RR
    if len(onsets) > 1:
        rr = results.emg.instantaneous_rr
        fig.add_trace(
            go.Scattergl(x=onsets[1:], y=rr, mode='lines+markers', name='Breath Rate (bpm)',
                         line=dict(color='purple', width=1.5), marker=dict(size=4)),
            row=3, col=1
        )

    # Overlay translucent shaded laser pulse spans if opto pulses exist (batched for high performance)
    if results.opto is not None and len(results.opto.onsets) > 0:
        shapes = list(fig.layout.shapes) if fig.layout.shapes else []
        for i, opto_start in enumerate(results.opto.onsets[:150]):
            opto_dur = results.opto.durations[i] if i < len(results.opto.durations) else 0.1
            opto_end = opto_start + opto_dur
            shapes.append(dict(
                type="rect",
                xref="x", yref="paper",
                x0=float(opto_start), x1=float(opto_end),
                y0=0, y1=1,
                fillcolor="rgba(255, 69, 0, 0.25)",
                layer="below", line_width=0
            ))
        fig.update_layout(shapes=shapes)

    # Add Range Slider & Quick View Buttons
    t_max = time[-1] if len(time) > 0 else 1.0
    fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
    fig.update_xaxes(
        rangeslider=dict(visible=True),
        rangeselector=dict(
            buttons=list([
                dict(count=5, label="5s", step="second", stepmode="backward"),
                dict(count=15, label="15s", step="second", stepmode="backward"),
                dict(count=30, label="30s", step="second", stepmode="backward"),
                dict(count=60, label="1m", step="second", stepmode="backward"),
                dict(count=300, label="5m", step="second", stepmode="backward"),
                dict(step="all", label="Full Session")
            ])
        ),
        row=1, col=1
    )

    # Set initial zoom window to start of first opto pulse or 0-10 seconds
    if results.opto is not None and len(results.opto.onsets) > 0:
        first_opto = results.opto.onsets[0]
        init_min = max(0, first_opto - 2.0)
        init_max = min(t_max, first_opto + 8.0)
        fig.update_xaxes(range=[init_min, init_max])
    else:
        fig.update_xaxes(range=[0, min(10.0, t_max)])

    fig.update_layout(
        height=900,
        title_text=f"Interactive Diaphragm EMG & Translucent Laser Pulse Overlay - {results.animal_id} {results.session_id}",
        hovermode="x unified",
        template="plotly_white"
    )

    html_path = output_dir / f"{results.animal_id}_{results.session_id}_diaphragm_interactive.html"
    fig.write_html(str(html_path), include_plotlyjs='cdn')
    return html_path

def plot_opto_phase_response(results: SessionResults, output_dir: Path) -> Optional[Path]:
    """
    Generate Phase Response Curve (PRC) plot showing Opto Stimulus Phase (phi)
    vs. Normalized Phase Shift (delta_phi) to evaluate inspiration advances & delays.
    """
    opto_phase = results.opto_phase
    if opto_phase is None or len(opto_phase.trials) == 0:
        return None

    trials = opto_phase.trials
    phis = np.array([tr.opto_phase_phi for tr in trials])
    delta_phis = np.array([tr.phase_shift_delta for tr in trials])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Subplot 1: Scatter plot of all trials (Phase Advance < 0 vs Phase Delay > 0)
    axes[0].scatter(phis[delta_phis >= 0], delta_phis[delta_phis >= 0], color='crimson', alpha=0.7, label='Phase Delay (Postpones Next Inspiration)')
    axes[0].scatter(phis[delta_phis < 0], delta_phis[delta_phis < 0], color='blue', alpha=0.7, label='Phase Advance (Speeds Up Next Inspiration)')
    axes[0].axhline(y=0.0, color='black', linestyle='--', linewidth=1.2)
    
    # Binned trendline
    if len(opto_phase.phase_bins) > 0:
        axes[0].plot(opto_phase.phase_bins, opto_phase.mean_phase_shift_by_bin, color='black', linewidth=2.0, marker='o', label='Binned Mean PRC')

    axes[0].set_title(f"Optogenetics Respiratory Phase Response Curve (PRC) - {results.animal_id}")
    axes[0].set_ylabel("Normalized Phase Shift Δϕ\n(>0 = Delay, <0 = Advance)")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Subplot 2: Histogram of Opto Stimulus Phasing Across Respiratory Cycle
    axes[1].hist(phis, bins=np.linspace(0.0, 1.0, 11), color='orangered', edgecolor='black', alpha=0.7)
    axes[1].set_title(f"Laser Stimulus Phase Distribution ({len(trials)} opto trials total)")
    axes[1].set_xlabel("Respiratory Cycle Phase ϕ (0 = Inspiration Onset, 1 = Next Inspiration)")
    axes[1].set_ylabel("Laser Trial Count")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    prc_path = output_dir / f"{results.animal_id}_{results.session_id}_opto_prc.png"
    fig.savefig(prc_path, dpi=200)
    plt.close(fig)
    return prc_path

def generate_diagnostic_dashboard(results: SessionResults, output_dir: Path) -> Dict[str, Path]:
    """
    Generate diagnostic plots for visual parameter verification & data stationarity inspection.
    Saves PNG and interactive HTML plots to output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = {}

    time = results.analog_raw.time_vector
    t_max = time[-1] if len(time) > 0 else 1.0

    # -------------------------------------------------------------
    # 0. Interactive Scrollable Diaphragm EMG HTML Dashboard
    # -------------------------------------------------------------
    try:
        html_emg_path = generate_interactive_emg_html(results, output_dir)
        generated_files['diaphragm_interactive_html'] = html_emg_path
    except Exception as e:
        print(f" -> Note: Interactive HTML dashboard generation warning: {e}")

    # -------------------------------------------------------------
    # 0.1 Opto Phase Response Curve (PRC) Diagnostic Plot
    # -------------------------------------------------------------
    try:
        prc_path = plot_opto_phase_response(results, output_dir)
        if prc_path:
            generated_files['opto_prc'] = prc_path
    except Exception as e:
        print(f" -> Note: PRC plot generation warning: {e}")

    # -------------------------------------------------------------
    # 1. Multi-Channel Raw Overview Plot (Inspect All Channels)
    # -------------------------------------------------------------
    if results.analog_raw.all_channels:
        num_ch = len(results.analog_raw.all_channels)
        fig, axes = plt.subplots(num_ch, 1, figsize=(14, max(6, num_ch * 1.5)), sharex=True)
        if num_ch == 1:
            axes = [axes]
        
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
    # 2. Diaphragm EMG Transformation & SNR Stationarity Plot
    # -------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(14, 11))
    
    step = max(1, len(time) // 50000)
    axes[0].plot(time[::step], results.analog_raw.emg_raw[::step], color='gray', alpha=0.7, linewidth=0.6, label='Raw EMG (Full Session)')
    axes[0].set_title(f"Diaphragm EMG Full Recording Raw Signal (0 to {t_max:.1f}s) - {results.animal_id}")
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_xlim(0, t_max)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

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

    if len(results.emg.breath_onsets) > 1:
        axes[2].plot(results.emg.breath_onsets[1:], results.emg.instantaneous_rr, color='purple', linewidth=1.2, label='Breath Rate')
    axes[2].set_title(f"Instantaneous Respiratory Rate (Mean: {np.nanmean(results.emg.instantaneous_rr):.1f} breaths/min, Total: {len(results.emg.breath_onsets)} breaths)")
    axes[2].set_ylabel("Breaths/min")
    axes[2].set_xlim(0, t_max)
    axes[2].grid(True, linestyle=':', alpha=0.6)

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
    # 3. ECG & R-Peak Detection Diagnostic Plot
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    
    axes[0].plot(time[::step], results.analog_raw.ecg_raw[::step], color='crimson', alpha=0.7, linewidth=0.6, label='Raw ECG (Full Session)')
    axes[0].set_title(f"ECG Full Recording Raw Signal (0 to {t_max:.1f}s) - {results.animal_id}")
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_xlim(0, t_max)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

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

    if len(results.ecg.r_peaks) > 1:
        axes[2].plot(results.ecg.r_peaks[1:], results.ecg.instantaneous_hr, color='darkgreen', linewidth=1.2, label='Heart Rate')
        mean_hr = np.nanmean(results.ecg.instantaneous_hr)
        axes[2].set_title(f"Heart Rate Tachogram (Mean HR: {mean_hr:.1f} bpm, R-Peaks Detected: {len(results.ecg.r_peaks)}, HRV SDNN: {results.ecg.hrv_sdnn:.2f} ms)")
    else:
        axes[2].set_title("Heart Rate Tachogram (No R-Peaks Detected / ECG Not Recorded)")
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
    # 4. Opto Laser & Time Sync Pulse Alignment Plot
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
    # 5. Neuropixel Depth & Stationarity Rasters (If Ephys loaded)
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
