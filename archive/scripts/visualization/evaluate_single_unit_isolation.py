#!/usr/bin/env python3
r"""
Single-Unit Isolation Quality & Baseline Diagnostic Audit
Units: 100, 118, 55, and 13 (Subject: TP_Tacr1_1)

Generates high-resolution multi-panel figures and an interactive HTML dashboard in
c:/Users/lpascu/analysis_src/output_TP_Tacr1_1_pacemaker_isolation
"""

import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

def compute_ccg(t1, t2, bin_size=0.001, max_lag=0.05):
    """Compute cross-correlogram (or auto-correlogram if t1 is t2)"""
    lags_bins = np.arange(-max_lag, max_lag + bin_size, bin_size)
    counts = np.zeros(len(lags_bins) - 1, dtype=int)
    
    if len(t1) == 0 or len(t2) == 0:
        return lags_bins[:-1] + bin_size / 2.0, counts

    # Search sorted cross differences
    for st in t1:
        diffs = t2 - st
        diffs = diffs[(diffs >= -max_lag) & (diffs <= max_lag)]
        if len(diffs) > 0:
            c, _ = np.histogram(diffs, bins=lags_bins)
            counts += c
            
    # Zero out zero-lag bin for auto-correlogram
    if np.array_equal(t1, t2):
        zero_bin_idx = len(counts) // 2
        counts[zero_bin_idx] = 0
        
    bin_centers = (lags_bins[:-1] + lags_bins[1:]) / 2.0 * 1000.0 # ms
    return bin_centers, counts

def main():
    out_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation")
    out_dir.mkdir(parents=True, exist_ok=True)
    session_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1\alf_session")
    alf_dir = session_dir / "alf" / "probe00"
    raw_dir = Path(r"Z:\ramirezlab\pascual\rec_data\TP_Tacr1_1_npx2_g4")
    artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\0e7be222-f160-4bc2-921c-f3fdf8b7825e")

    print("=" * 90)
    print(" Executing Single Unit Isolation Quality Audit for Units 100, 118, 55, and 13")
    print(f" ALF Directory: {alf_dir}")
    print(f" Output Directory: {out_dir}")
    print("=" * 90)

    # 1. Load Spikes, Amplitudes & Templates
    spikes_times = np.load(alf_dir / "spike_times.npy")
    spikes_clusters = np.load(alf_dir / "spike_clusters.npy")
    amplitudes = np.load(alf_dir / "amplitudes.npy")
    templates = np.load(alf_dir / "templates.npy") # (n_templates, n_samples, n_channels)
    info_df = pd.read_csv(alf_dir / "cluster_info.tsv", sep="\t")

    fs_ap = 30000.0
    if spikes_times.dtype in [np.int64, np.int32] and spikes_times.max() > 100000:
        spikes_times = spikes_times / fs_ap

    MAX_TIME_S = 550.0
    spk_mask = (spikes_times >= 0.0) & (spikes_times <= MAX_TIME_S)
    spikes_times = spikes_times[spk_mask]
    spikes_clusters = spikes_clusters[spk_mask]
    amplitudes = amplitudes[spk_mask]

    target_uids = [100, 118, 55, 13]
    unit_colors = {100: '#1f77b4', 118: '#ff7f0e', 55: '#2ca02c', 13: '#d62728'} # Blue, Orange, Green, Red

    # 2. Load EMG & Laser NIDQ signals
    nidq_meta = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.meta"
    nidq_bin = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.bin"
    nidq_data = read_nidq_bin(nidq_bin, nidq_meta)

    t_nidq = nidq_data.time_vector
    fs_nidq = nidq_data.fs

    nidq_mask = (t_nidq >= 0.0) & (t_nidq <= MAX_TIME_S)
    t_nidq = t_nidq[nidq_mask]
    emg_raw = nidq_data.all_channels[0][nidq_mask]
    ch6_laser = nidq_data.all_channels[6][nidq_mask]

    emg_cfg = EMGConfig(fs=fs_nidq, highpass_cutoff=300.0, lowpass_cutoff=3000.0, notch_freq=60.0, smoothing_window_ms=20.0)
    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    breath_onsets = emg_proc.breath_onsets
    breath_offsets = emg_proc.breath_offsets

    laser_active = (ch6_laser > 0.8).astype(int)
    laser_onsets = t_nidq[np.where(np.diff(laser_active) == 1)[0]]
    laser_offsets = t_nidq[np.where(np.diff(laser_active) == -1)[0]]
    n_pulses = min(len(laser_onsets), len(laser_offsets))
    laser_onsets = laser_onsets[:n_pulses]
    laser_offsets = laser_offsets[:n_pulses]

    # Continuous Respiratory Phase phi(t)
    dt_phase = 0.005
    t_bins = np.arange(0.0, MAX_TIME_S, dt_phase)
    n_bins = len(t_bins)
    phi_vector = np.zeros(n_bins)

    n_cycles = min(len(breath_onsets) - 1, len(breath_offsets))
    for i in range(n_cycles):
        t_on = breath_onsets[i]
        t_off = breath_offsets[i] if breath_offsets[i] > t_on else t_on + 0.15
        t_next = breath_onsets[i + 1]

        idx_insp = np.where((t_bins >= t_on) & (t_bins < t_off))[0]
        if len(idx_insp) > 0: phi_vector[idx_insp] = np.linspace(0, np.pi, len(idx_insp))

        idx_exp = np.where((t_bins >= t_off) & (t_bins < t_next))[0]
        if len(idx_exp) > 0: phi_vector[idx_exp] = np.linspace(-np.pi, 0.0, len(idx_exp))

    # 3. Process Each Target Unit
    unit_stats = {}
    spiketrains = {}
    unit_amps = {}

    for uid in target_uids:
        idx_u = np.where(spikes_clusters == uid)[0]
        u_times = spikes_times[idx_u]
        u_amps = amplitudes[idx_u]
        spiketrains[uid] = u_times
        unit_amps[uid] = u_amps

        row_info = info_df[info_df['cluster_id'] == uid].iloc[0]
        
        # Calculate ISIs & Refractory Violations (< 1.5 ms)
        isis_ms = np.diff(u_times) * 1000.0
        n_viol = np.sum(isis_ms < 1.5)
        viol_ratio = n_viol / max(1, len(isis_ms))
        
        # Mean Waveform Template (90 samples at 30 kHz)
        if uid < len(templates):
            tmpl = templates[uid] # (90, n_ch)
            # Channel with max peak-to-peak amplitude
            ptp_per_ch = np.ptp(tmpl, axis=0)
            best_ch_idx = np.argmax(ptp_per_ch)
            best_wf = tmpl[:, best_ch_idx]
        else:
            best_wf = np.zeros(90)
            best_ch_idx = row_info['ch']

        # Laser phototagging ratio
        l_counts, b_counts = [], []
        for p_i in range(n_pulses):
            ton = laser_onsets[p_i]
            toff = laser_offsets[p_i]
            dur = toff - ton
            l_counts.append(np.sum((u_times >= ton) & (u_times < toff)))
            b_counts.append(np.sum((u_times >= ton - dur) & (u_times < ton)))
            
        m_l_rate = np.mean(l_counts) / max(1e-4, np.mean(laser_offsets - laser_onsets))
        m_b_rate = np.mean(b_counts) / max(1e-4, np.mean(laser_offsets - laser_onsets))
        laser_ratio = (m_l_rate + 0.1) / (m_b_rate + 0.1)

        # Firing Phase Distribution
        spk_bin_indices = np.searchsorted(t_bins, u_times) - 1
        spk_bin_indices = spk_bin_indices[(spk_bin_indices >= 0) & (spk_bin_indices < n_bins)]
        spk_phases = phi_vector[spk_bin_indices]
        
        cos_m = np.mean(np.cos(spk_phases)) if len(spk_phases) > 0 else 0
        sin_m = np.mean(np.sin(spk_phases)) if len(spk_phases) > 0 else 0
        pref_phi = np.arctan2(sin_m, cos_m)
        v_strength = np.sqrt(cos_m**2 + sin_m**2)

        unit_stats[uid] = {
            'shank': int(row_info['shank']),
            'depth': float(row_info['depth']),
            'ch': int(row_info['ch']),
            'n_spikes': len(u_times),
            'fr': len(u_times) / MAX_TIME_S,
            'peak_amp_uv': float(row_info['peak_amp_uv']),
            'n_viol': n_viol,
            'viol_ratio': viol_ratio,
            'laser_ratio': laser_ratio,
            'pref_phi': pref_phi,
            'v_strength': v_strength,
            'waveform': best_wf,
            'isis_ms': isis_ms,
            'spk_phases': spk_phases
        }

    # Print Summary Table
    print("\n" + "=" * 90)
    print(f"{'Unit':<6} | {'Shank':<5} | {'Depth(um)':<9} | {'Spikes':<7} | {'FR(Hz)':<7} | {'Amp(uV)':<8} | {'ISI <1.5ms':<10} | {'Laser Ratio':<11} | {'Vector R':<8}")
    print("-" * 90)
    for uid in target_uids:
        s = unit_stats[uid]
        print(f"{uid:<6} | {s['shank']:<5} | {s['depth']:<9.1f} | {s['n_spikes']:<7} | {s['fr']:<7.2f} | {s['peak_amp_uv']:<8.1f} | {s['n_viol']} ({s['viol_ratio']*100:.2f}%) | {s['laser_ratio']:<11.2f} | {s['v_strength']:<8.2f}")
    print("=" * 90 + "\n")

    # 4. Create High-Resolution Static PNG Figure (Matplotlib 4x4 Grid)
    fig_mpl, axs = plt.subplots(4, 4, figsize=(20, 16))
    t_wf_ms = np.linspace(-1.5, 1.5, 90) # 90 samples at 30 kHz

    for col_idx, uid in enumerate(target_uids):
        s = unit_stats[uid]
        st = spiketrains[uid]
        u_color = unit_colors[uid]

        # Row 1: Mean Waveform Template & Isolation Summary
        ax_wf = axs[0, col_idx]
        ax_wf.plot(t_wf_ms, s['waveform'], color=u_color, linewidth=2.5)
        ax_wf.axvline(0, color='gray', linestyle=':', alpha=0.6)
        ax_wf.set_title(f"Unit {uid} (Shank {s['shank']}, Depth {s['depth']} µm)\nFR: {s['fr']:.2f} Hz | Amp: {s['peak_amp_uv']:.1f} µV", fontweight='bold', fontsize=10)
        ax_wf.set_xlabel("Time (ms)")
        ax_wf.set_ylabel("Amplitude (a.u.)")
        ax_wf.grid(True, linestyle='--', alpha=0.3)

        # Row 2: ISI Histogram (0-50 ms, 0.5 ms bins)
        ax_isi = axs[1, col_idx]
        isi_bins = np.arange(0, 50.5, 0.5)
        ax_isi.hist(s['isis_ms'], bins=isi_bins, color=u_color, alpha=0.75, edgecolor='black', linewidth=0.5)
        ax_isi.axvline(1.5, color='red', linestyle='--', linewidth=2, label=f'1.5ms Cutoff ({s["n_viol"]} viol, {s["viol_ratio"]*100:.2f}%)')
        ax_isi.set_title(f"ISI Histogram (Refractory Viol: {s['viol_ratio']*100:.2f}%)", fontsize=10, fontweight='bold')
        ax_isi.set_xlabel("Inter-Spike Interval (ms)")
        ax_isi.set_ylabel("Spike Count")
        ax_isi.set_xlim([0, 50])
        ax_isi.legend(loc='upper right', fontsize=8)

        # Row 3: Amplitude & Firing Rate Timecourse Stationarity
        ax_st = axs[2, col_idx]
        # Scatter subset of amplitudes to prevent bloat
        sc_idx = np.random.choice(len(st), size=min(1000, len(st)), replace=False)
        ax_st.scatter(st[sc_idx], unit_amps[uid][sc_idx], color=u_color, alpha=0.3, s=8, label='Spike Amps')
        
        # Binned firing rate (10s bins)
        rate_bins = np.arange(0, MAX_TIME_S, 10.0)
        r_counts, _ = np.histogram(st, bins=rate_bins)
        r_rate = r_counts / 10.0
        ax_st_r = ax_st.twinx()
        ax_st_r.plot(rate_bins[:-1] + 5.0, r_rate, color='black', linewidth=1.5, label='Firing Rate (10s bin)')
        ax_st_r.set_ylabel("Rate (Hz)", color='black')
        
        ax_st.axvspan(60.55, 284.0, color='magenta', alpha=0.1, label='Laser Epoch')
        ax_st.set_title("Amplitude & Rate Stationarity (550s)", fontsize=10, fontweight='bold')
        ax_st.set_xlabel("Time in Recording (s)")
        ax_st.set_ylabel("Amplitudes (a.u.)", color=u_color)

        # Row 4: Auto-Correlogram (ACG) [-50ms to +50ms]
        ax_acg = axs[3, col_idx]
        acg_lags, acg_counts = compute_ccg(st, st, bin_size=0.001, max_lag=0.05)
        ax_acg.bar(acg_lags, acg_counts, width=1.0, color=u_color, alpha=0.85, edgecolor='none')
        ax_acg.axvline(0, color='black', linestyle='-', linewidth=1)
        ax_acg.set_title(f"Auto-Correlogram [-50ms, +50ms]", fontsize=10, fontweight='bold')
        ax_acg.set_xlabel("Lag (ms)")
        ax_acg.set_ylabel("Coincidence Count")

    plt.suptitle("Single-Unit Isolation Audit & Stationarity Diagnostics (Units 100, 118, 55, 13)", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    png_out = out_dir / "single_unit_isolation_units_100_118_55_13.png"
    plt.savefig(png_out, dpi=300)
    plt.close()

    # Copy PNG to artifact directory
    artifact_png = artifact_dir / "single_unit_isolation_units_100_118_55_13.png"
    shutil.copy2(png_out, artifact_png)
    print(f" Saved Diagnostic Figure: {png_out}")

    # 5. Interactive Plotly Dashboard
    fig_plotly = make_subplots(
        rows=3, cols=4,
        subplot_titles=[
            f"Unit {u} Waveform Template" for u in target_uids
        ] + [
            f"Unit {u} ISI Distribution" for u in target_uids
        ] + [
            f"Unit {u} Auto-Correlogram (ACG)" for u in target_uids
        ],
        vertical_spacing=0.10, horizontal_spacing=0.05
    )

    for i, uid in enumerate(target_uids):
        col = i + 1
        s = unit_stats[uid]
        st = spiketrains[uid]
        c = unit_colors[uid]

        # Row 1: Waveforms
        fig_plotly.add_trace(go.Scatter(x=t_wf_ms, y=s['waveform'], mode='lines', line=dict(color=c, width=3), name=f"Unit {uid} Waveform"), row=1, col=col)
        
        # Row 2: ISIs
        counts, bin_edges = np.histogram(s['isis_ms'], bins=np.arange(0, 50.5, 1.0))
        fig_plotly.add_trace(go.Bar(x=(bin_edges[:-1]+bin_edges[1:])/2.0, y=counts, marker_color=c, name=f"Unit {uid} ISI"), row=2, col=col)

        # Row 3: ACG
        lags, acg_c = compute_ccg(st, st, bin_size=0.001, max_lag=0.05)
        fig_plotly.add_trace(go.Bar(x=lags, y=acg_c, marker_color=c, name=f"Unit {uid} ACG"), row=3, col=col)

    fig_plotly.update_layout(
        title="<b>Subject TP_Tacr1_1: Single Unit Isolation Quality Dashboard (Units 100, 118, 55, 13)</b>",
        template="plotly_white",
        height=900,
        showlegend=False
    )

    html_out = out_dir / "single_unit_isolation_dashboard.html"
    fig_plotly.write_html(str(html_out))
    print(f" Saved Interactive Dashboard: {html_out}")

    print("\n==========================================================================================")
    print(" [SUCCESS] Phase 1 Single-Unit Isolation Audit Complete!")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
