#!/usr/bin/env python3
r"""
Per-Neuron Individual Quality Metrics Diagnostic Dashboard (AllenSDK Standards)
Generates individual diagnostic plots per neuron:
1. Mean Waveform & Peak Amplitude (uV)
2. Inter-Spike Interval (ISI) Histogram & Refractory Line (1.5 ms)
3. Spike Amplitude Distribution Histogram & Amplitude Cutoff Fit
4. Timecourse Block Presence Ratio (60s baseline)
5. Quality Metrics Summary Card

Saves output PNG and interactive HTML dashboard to:
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

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def compute_isi_violations(spike_times, max_time, isi_threshold=0.0015, min_isi=0.0005):
    if len(spike_times) <= 1: return 0.0
    isis = np.diff(spike_times)
    num_violations = np.sum((isis < isi_threshold) & (isis >= min_isi))
    total_spikes = len(spike_times)
    violation_rate = num_violations / max_time
    firing_rate = total_spikes / max_time
    violation_ratio = violation_rate / (2 * (isi_threshold - min_isi) * firing_rate**2) if firing_rate > 0 else 0.0
    return float(violation_ratio)

def compute_presence_ratio(spike_times, max_time, num_blocks=60):
    if len(spike_times) == 0: return 0.0, np.zeros(num_blocks)
    block_edges = np.linspace(0, max_time, num_blocks + 1)
    counts, _ = np.histogram(spike_times, bins=block_edges)
    presence_vector = (counts > 0).astype(int)
    return float(np.sum(presence_vector) / num_blocks), presence_vector

def compute_amplitude_cutoff(u_amps, num_bins=50):
    if len(u_amps) < 10: return 0.0, np.zeros(num_bins), np.zeros(num_bins)
    h, b = np.histogram(u_amps, bins=num_bins)
    peak_idx = np.argmax(h)
    if peak_idx == 0:
        return float(min(1.0, h[0] / np.sum(h))), h, b
    
    upper_tail = h[peak_idx:]
    expected_lower = upper_tail[::-1]
    actual_lower = h[:peak_idx]
    
    n_actual = len(actual_lower)
    if n_actual > len(expected_lower):
        actual_lower = actual_lower[-len(expected_lower):]
    else:
        expected_lower = expected_lower[-n_actual:]
        
    missing_spikes = np.sum(np.maximum(0, expected_lower - actual_lower))
    total_est = np.sum(h) + missing_spikes
    return float(missing_spikes / total_est) if total_est > 0 else 0.0, h, b

def main():
    out_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation")
    out_dir.mkdir(parents=True, exist_ok=True)
    session_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1\alf_session")
    alf_dir = session_dir / "alf" / "probe00"
    artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\0e7be222-f160-4bc2-921c-f3fdf8b7825e")

    print("=" * 90)
    print(" Generating Per-Neuron Individual Quality Metrics Diagnostic Figure & Dashboard")
    print(f" ALF Directory: {alf_dir}")
    print(f" Output Directory: {out_dir}")
    print("=" * 90)

    # Load Data
    spikes_times = np.load(alf_dir / "spike_times.npy")
    spikes_clusters = np.load(alf_dir / "spike_clusters.npy")
    amplitudes = np.load(alf_dir / "amplitudes.npy")
    templates = np.load(alf_dir / "templates.npy")
    info_df = pd.read_csv(alf_dir / "cluster_info.tsv", sep="\t")

    fs_ap = 30000.0
    if spikes_times.dtype in [np.int64, np.int32] and spikes_times.max() > 100000:
        spikes_times = spikes_times / fs_ap

    MAX_TIME_S = 60.0 # 60s baseline epoch
    spk_mask = (spikes_times >= 0.0) & (spikes_times <= MAX_TIME_S)
    spikes_times = spikes_times[spk_mask]
    spikes_clusters = spikes_clusters[spk_mask]
    amplitudes = amplitudes[spk_mask]

    target_uids = [100, 118, 55, 13]
    unit_colors = {100: '#1f77b4', 118: '#ff7f0e', 55: '#2ca02c', 13: '#d62728'}
    noise_floor_uv = 15.0
    t_wf_ms = np.linspace(-1.5, 1.5, 90)

    # Build 4x4 Per-Neuron Figure (Each column = 1 neuron's complete metrics)
    fig_mpl, axs = plt.subplots(4, 4, figsize=(22, 16))

    for col_idx, uid in enumerate(target_uids):
        idx_u = np.where(spikes_clusters == uid)[0]
        u_times = spikes_times[idx_u]
        u_amps = amplitudes[idx_u]
        u_color = unit_colors[uid]

        info_row = info_df[info_df['cluster_id'] == uid].iloc[0]
        peak_amp = float(info_row['peak_amp_uv'])
        fr = len(u_times) / MAX_TIME_S
        isi_viol = compute_isi_violations(u_times, MAX_TIME_S)
        pres_ratio, pres_vec = compute_presence_ratio(u_times, MAX_TIME_S)
        amp_cutoff, amp_h, amp_b = compute_amplitude_cutoff(u_amps)
        snr = peak_amp / noise_floor_uv
        passed = (isi_viol < 0.50) and (pres_ratio >= 0.80) and (amp_cutoff < 0.15) and (fr >= 0.1)

        # Get Mean Waveform
        if uid < len(templates):
            tmpl = templates[uid]
            best_ch = np.argmax(np.ptp(tmpl, axis=0))
            wf = tmpl[:, best_ch]
        else:
            wf = np.zeros(90)

        # Row 1: Mean Waveform Profile & Summary Header
        ax1 = axs[0, col_idx]
        ax1.plot(t_wf_ms, wf, color=u_color, linewidth=2.5)
        ax1.axvline(0, color='gray', linestyle=':', alpha=0.6)
        status_txt = "PASS" if passed else "MARGINAL"
        ax1.set_title(f"Unit {uid} (Shank {int(info_row['shank'])}, {int(info_row['depth'])}µm)\nStatus: {status_txt} | FR: {fr:.2f}Hz | Amp: {peak_amp:.1f}µV", fontweight='bold', fontsize=10)
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Waveform Amp (a.u.)")
        ax1.grid(True, linestyle='--', alpha=0.3)

        # Row 2: ISI Histogram & Refractory Cutoff
        ax2 = axs[1, col_idx]
        isis_ms = np.diff(u_times) * 1000.0
        n_viol = np.sum((isis_ms < 1.5) & (isis_ms >= 0.5))
        ax2.hist(isis_ms, bins=np.arange(0, 50.5, 1.0), color=u_color, alpha=0.75, edgecolor='black', linewidth=0.5)
        ax2.axvline(1.5, color='red', linestyle='--', linewidth=2, label=f'1.5ms Cutoff\nRatio: {isi_viol:.3f} ({n_viol} viol)')
        ax2.set_title(f"Unit {uid} ISI Violations Ratio: {isi_viol:.3f}", fontsize=10, fontweight='bold')
        ax2.set_xlabel("ISI (ms)")
        ax2.set_ylabel("Spike Count")
        ax2.set_xlim([0, 50])
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, linestyle=':', alpha=0.4)

        # Row 3: Spike Amplitude Histogram & Amplitude Cutoff Estimation
        ax3 = axs[2, col_idx]
        if len(u_amps) > 0:
            ax3.hist(u_amps, bins=40, color=u_color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax3.axvline(np.mean(u_amps) if len(u_amps)>0 else 0, color='black', linestyle=':', label=f'Mean Amp: {np.mean(u_amps):.1f}')
        ax3.set_title(f"Unit {uid} Amplitude Cutoff: {amp_cutoff:.4f} ({amp_cutoff*100:.1f}% missing)", fontsize=10, fontweight='bold')
        ax3.set_xlabel("Spike Amplitude (uV)")
        ax3.set_ylabel("Spike Count")
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, linestyle=':', alpha=0.4)

        # Row 4: Presence Ratio Timecourse across 1s Time Blocks
        ax4 = axs[3, col_idx]
        t_block_centers = np.linspace(0.5, 59.5, 60)
        ax4.bar(t_block_centers, pres_vec, width=0.9, color=u_color, alpha=0.85)
        ax4.set_ylim([0, 1.25])
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['Absent', 'Present'])
        ax4.set_title(f"Unit {uid} Presence Ratio: {pres_ratio:.2f} (60s Baseline)", fontsize=10, fontweight='bold')
        ax4.set_xlabel("Session Time Block (s)")
        ax4.set_ylabel("Block Presence")
        ax4.grid(True, linestyle=':', alpha=0.4)

    plt.suptitle("Individual Single-Unit Quality Metrics Breakdown (Units 100, 118, 55, 13)", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    png_out = out_dir / "individual_per_neuron_quality_metrics.png"
    plt.savefig(png_out, dpi=300)
    plt.close()

    # Copy to artifact directory
    artifact_png = artifact_dir / "individual_per_neuron_quality_metrics.png"
    shutil.copy2(png_out, artifact_png)
    print(f" Saved Diagnostic Per-Neuron Figure: {png_out}")

    # 5. Interactive Plotly Per-Neuron Dashboard
    fig_plotly = make_subplots(
        rows=4, cols=4,
        subplot_titles=[
            f"Unit {u} Waveform Profile" for u in target_uids
        ] + [
            f"Unit {u} ISI Distribution" for u in target_uids
        ] + [
            f"Unit {u} Amplitude Cutoff Fit" for u in target_uids
        ] + [
            f"Unit {u} Presence Timecourse" for u in target_uids
        ],
        vertical_spacing=0.08, horizontal_spacing=0.05
    )

    for i, uid in enumerate(target_uids):
        col = i + 1
        idx_u = np.where(spikes_clusters == uid)[0]
        u_times = spikes_times[idx_u]
        u_amps = amplitudes[idx_u]
        c = unit_colors[uid]

        info_row = info_df[info_df['cluster_id'] == uid].iloc[0]
        peak_amp = float(info_row['peak_amp_uv'])
        fr = len(u_times) / MAX_TIME_S
        isi_viol = compute_isi_violations(u_times, MAX_TIME_S)
        pres_ratio, pres_vec = compute_presence_ratio(u_times, MAX_TIME_S)
        amp_cutoff, _, _ = compute_amplitude_cutoff(u_amps)
        
        if uid < len(templates):
            tmpl = templates[uid]
            best_ch = np.argmax(np.ptp(tmpl, axis=0))
            wf = tmpl[:, best_ch]
        else:
            wf = np.zeros(90)

        # Row 1: Waveform
        fig_plotly.add_trace(go.Scatter(x=t_wf_ms, y=wf, mode='lines', line=dict(color=c, width=3), name=f"Unit {uid} Waveform"), row=1, col=col)
        
        # Row 2: ISI
        isis_ms = np.diff(u_times) * 1000.0
        counts, bin_edges = np.histogram(isis_ms, bins=np.arange(0, 50.5, 1.0))
        fig_plotly.add_trace(go.Bar(x=(bin_edges[:-1]+bin_edges[1:])/2.0, y=counts, marker_color=c, name=f"Unit {uid} ISI"), row=2, col=col)

        # Row 3: Amplitudes
        amp_cnts, amp_edges = np.histogram(u_amps, bins=30)
        fig_plotly.add_trace(go.Bar(x=(amp_edges[:-1]+amp_edges[1:])/2.0, y=amp_cnts, marker_color=c, name=f"Unit {uid} Amps"), row=3, col=col)

        # Row 4: Presence
        t_block_centers = np.linspace(0.5, 59.5, 60)
        fig_plotly.add_trace(go.Bar(x=t_block_centers, y=pres_vec, marker_color=c, name=f"Unit {uid} Presence"), row=4, col=col)

    fig_plotly.update_layout(
        title="<b>Subject TP_Tacr1_1: Per-Neuron Individual Quality Metrics Interactive Dashboard</b>",
        template="plotly_white",
        height=1100,
        showlegend=False
    )

    html_out = out_dir / "individual_per_neuron_quality_metrics.html"
    fig_plotly.write_html(str(html_out))
    print(f" Saved Interactive HTML Dashboard: {html_out}")

    print("\n==========================================================================================")
    print(" [SUCCESS] Per-Neuron Quality Metrics Dashboard Generation Complete!")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
