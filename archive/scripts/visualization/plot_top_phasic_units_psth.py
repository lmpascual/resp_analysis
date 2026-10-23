#!/usr/bin/env python3
r"""
Top 24 Phasic Single Units Breath-Onset Aligned PSTHs & Rasters
(Filtered for AllenSDK Quality Criteria Passed Single Units)

Sorts Allen QC passed single units by Mazurek Vector Strength (R) and generates
breath-onset aligned PSTH firing rate curves (Hz) for the top 24 most phasic neurons.

Outputs:
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\top_phasic_units_breath_psth_rasters.png
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\top_phasic_units_breath_psth_dashboard.html
"""

import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

def main():
    out_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation")
    out_dir.mkdir(parents=True, exist_ok=True)
    session_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1\alf_session")
    alf_dir = session_dir / "alf" / "probe00"
    raw_dir = Path(r"Z:\ramirezlab\pascual\rec_data\TP_Tacr1_1_npx2_g4")
    artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\0e7be222-f160-4bc2-921c-f3fdf8b7825e")

    print("=" * 90)
    print(" Computing Breath-Onset Aligned PSTHs for Top 24 Phasic Allen QC Passed Units")
    print(f" ALF Directory: {alf_dir}")
    print(f" Output Directory: {out_dir}")
    print("=" * 90)

    # 1. Load Spikes & Allen QC CSV
    qc_csv = out_dir / "allen_neuropixels_quality_metrics_baseline_60s.csv"
    if not qc_csv.exists():
        print(f" Error: {qc_csv} not found.")
        sys.exit(1)

    metrics_df = pd.read_csv(qc_csv)
    passed_df = metrics_df[metrics_df['passed_allen_qc']].copy().reset_index(drop=True)
    passed_uids = set(passed_df['cluster_id'])

    spikes_times = np.load(alf_dir / "spike_times.npy")
    spikes_clusters = np.load(alf_dir / "spike_clusters.npy")
    info_df = pd.read_csv(alf_dir / "cluster_info.tsv", sep="\t")

    fs_ap = 30000.0
    if spikes_times.dtype in [np.int64, np.int32] and spikes_times.max() > 100000:
        spikes_times = spikes_times / fs_ap

    MAX_TIME_S = 550.0
    spk_mask = (spikes_times >= 0.0) & (spikes_times <= MAX_TIME_S) & (np.isin(spikes_clusters, list(passed_uids)))
    spikes_times = spikes_times[spk_mask]
    spikes_clusters = spikes_clusters[spk_mask]

    # 2. Load NIDQ Signals & Compute Continuous Phase
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
    emg_env = emg_proc.envelope
    breath_onsets = emg_proc.breath_onsets
    breath_offsets = emg_proc.breath_offsets

    laser_active = (ch6_laser > 0.8).astype(int)
    laser_onsets = t_nidq[np.where(np.diff(laser_active) == 1)[0]]

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

    # 3. Calculate Vector Strength (R) for All Passed Units & Rank Top 24 Phasic Units
    vector_strengths = []
    preferred_phases = []

    for idx, row in passed_df.iterrows():
        uid = int(row['cluster_id'])
        u_spks = np.sort(spikes_times[spikes_clusters == uid])

        spk_bin_indices = np.searchsorted(t_bins, u_spks) - 1
        spk_bin_indices = spk_bin_indices[(spk_bin_indices >= 0) & (spk_bin_indices < n_bins)]
        spk_phases = phi_vector[spk_bin_indices]

        if len(spk_phases) < 15:
            vector_strengths.append(0.0)
            preferred_phases.append(0.0)
        else:
            cos_m = np.mean(np.cos(spk_phases))
            sin_m = np.mean(np.sin(spk_phases))
            v_strength = np.sqrt(cos_m**2 + sin_m**2)
            pref_phi = np.arctan2(sin_m, cos_m)
            vector_strengths.append(v_strength)
            preferred_phases.append(pref_phi)

    passed_df['vector_strength'] = vector_strengths
    passed_df['pref_phase'] = preferred_phases

    top_phasic_df = passed_df.sort_values(by='vector_strength', ascending=False).head(24).reset_index(drop=True)
    top_uids = top_phasic_df['cluster_id'].values

    print(f" Selected Top {len(top_uids)} Most Phasic Single Units (Vector Strength R = {top_phasic_df['vector_strength'].min():.3f} to {top_phasic_df['vector_strength'].max():.3f}).")
    for idx, r in top_phasic_df.iterrows():
        phase_label = "Inspiratory" if r['pref_phase'] >= 0 else "Expiratory"
        print(f" {idx+1:2d}. Unit {int(r['cluster_id']):<4} (Shank {int(r['shank'])}, {r['depth']:<6.1f}µm): R = {r['vector_strength']:.3f} | {phase_label} (phi = {r['pref_phase']:.2f} rad)")

    # Save Top 24 Phasic Units Metadata CSV
    top24_csv = out_dir / "top_24_phasic_single_units.csv"
    top_phasic_df.to_csv(top24_csv, index=False)
    print(f" Saved Top 24 Phasic Units CSV: {top24_csv}")

    # Filter unperturbed breath onsets
    valid_onsets = [o for o in breath_onsets if (len(laser_onsets)==0 or np.min(np.abs(o-laser_onsets))>0.35) and (0.5 <= o <= MAX_TIME_S-0.7)]

    # Time Window for PSTH
    t_pre = 0.40  # 400 ms pre-inspiration
    t_post = 0.60 # 600 ms post-inspiration
    bin_width = 0.010 # 10 ms bins
    psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)
    psth_centers = (psth_bins[:-1] + psth_bins[1:]) / 2.0 * 1000.0 # ms

    # Extract Aligned EMG Envelope
    emg_sweeps = []
    for o in valid_onsets:
        idx_o = np.searchsorted(t_nidq, o)
        i_pre = int(idx_o - t_pre * fs_nidq)
        i_post = int(idx_o + t_post * fs_nidq)
        if 0 <= i_pre and i_post <= len(emg_env):
            emg_sweeps.append(emg_env[i_pre:i_post])

    emg_sweeps = np.array(emg_sweeps)
    mean_emg = np.mean(emg_sweeps, axis=0) if len(emg_sweeps) > 0 else np.zeros(100)
    sem_emg = stats.sem(emg_sweeps, axis=0) if len(emg_sweeps) > 0 else np.zeros(100)
    t_emg_ms = np.linspace(-t_pre, t_post, len(mean_emg)) * 1000.0

    # 4. Create 6x4 Matplotlib PSTH Figure for Top 24 Phasic Units
    fig_mpl = plt.figure(figsize=(24, 25))

    for idx, uid in enumerate(top_uids):
        u_spks = np.sort(spikes_times[spikes_clusters == uid])
        r_info = top_phasic_df[top_phasic_df['cluster_id'] == uid].iloc[0]
        v_str = r_info['vector_strength']
        pref_p = r_info['pref_phase']
        phase_label = "Insp" if pref_p >= 0 else "Exp"

        trial_counts = []
        for o in valid_onsets:
            spks_in_win = u_spks[(u_spks >= o - t_pre) & (u_spks <= o + t_post)] - o
            c, _ = np.histogram(spks_in_win, bins=psth_bins)
            trial_counts.append(c / bin_width)

        trial_counts = np.array(trial_counts)
        mean_psth = np.mean(trial_counts, axis=0)
        sem_psth = stats.sem(trial_counts, axis=0)

        ax = fig_mpl.add_subplot(6, 4, idx + 1)
        bar_color = 'red' if pref_p >= 0 else '#1f77b4'
        ax.bar(psth_centers, mean_psth, width=bin_width*1000.0, color=bar_color, alpha=0.4, edgecolor=bar_color)
        ax.plot(psth_centers, mean_psth, color=bar_color, linewidth=2.5, label='PSTH Rate')
        ax.fill_between(psth_centers, np.maximum(0, mean_psth - sem_psth), mean_psth + sem_psth, color=bar_color, alpha=0.25)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Insp Onset')
        
        # Overlay EMG curve
        emg_norm = (mean_emg - np.min(mean_emg)) / (np.max(mean_emg) - np.min(mean_emg) + 1e-6) * (np.max(mean_psth) * 0.8)
        ax.plot(t_emg_ms, emg_norm, color='black', linestyle=':', linewidth=1.2, alpha=0.7)

        ax.set_title(f"Rank {idx+1}. Unit {uid} (S{int(r_info['shank'])}, {int(r_info['depth'])}µm)\nR = {v_str:.3f} ({phase_label}) | Peak: {np.max(mean_psth):.1f} Hz", fontweight='bold', fontsize=9.5)
        ax.set_xlabel("Time (ms)", fontsize=8, fontweight='bold')
        ax.set_ylabel("Rate (Hz)", fontsize=8, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.4)

    plt.suptitle("Subject TP_Tacr1_1: Peri-Inspiratory PSTHs for Top 24 Most Phasic Single Units (Allen QC Passed)", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.01, 1, 0.985])

    png_out = out_dir / "top_phasic_units_breath_psth_rasters.png"
    plt.savefig(png_out, dpi=300)
    plt.close()

    # Copy PNG to artifact directory
    artifact_png = artifact_dir / "top_phasic_units_breath_psth_rasters.png"
    shutil.copy2(png_out, artifact_png)
    print(f" Saved Diagnostic Figure: {png_out}")

    # 5. Interactive Plotly Dashboard
    fig_plotly = make_subplots(
        rows=6, cols=4,
        subplot_titles=[
            f"Rank {i+1}. Unit {u} (R = {top_phasic_df[top_phasic_df['cluster_id']==u].iloc[0]['vector_strength']:.3f})" for i, u in enumerate(top_uids)
        ],
        vertical_spacing=0.04, horizontal_spacing=0.04
    )

    for idx, uid in enumerate(top_uids):
        r_i = idx // 4 + 1
        c_i = idx % 4 + 1
        u_spks = np.sort(spikes_times[spikes_clusters == uid])
        r_info = top_phasic_df[top_phasic_df['cluster_id'] == uid].iloc[0]

        trial_counts = []
        for o in valid_onsets:
            spks_in_win = u_spks[(u_spks >= o - t_pre) & (u_spks <= o + t_post)] - o
            c, _ = np.histogram(spks_in_win, bins=psth_bins)
            trial_counts.append(c / bin_width)

        trial_counts = np.array(trial_counts)
        mean_psth = np.mean(trial_counts, axis=0)

        bar_color = 'red' if r_info['pref_phase'] >= 0 else '#1f77b4'
        fig_plotly.add_trace(go.Bar(x=psth_centers, y=mean_psth, marker_color=bar_color, opacity=0.5, name=f"Unit {uid} PSTH"), row=r_i, col=c_i)
        fig_plotly.add_trace(go.Scatter(x=psth_centers, y=mean_psth, mode='lines', line=dict(color=bar_color, width=2.5), showlegend=False), row=r_i, col=c_i)

    fig_plotly.update_layout(
        title="<b>Subject TP_Tacr1_1: Interactive PSTH Dashboard for Top 24 Most Phasic Single Units</b>",
        template="plotly_white",
        height=1400,
        showlegend=False
    )

    html_out = out_dir / "top_phasic_units_breath_psth_dashboard.html"
    fig_plotly.write_html(str(html_out))
    print(f" Saved Interactive HTML Dashboard: {html_out}")

    print("\n==========================================================================================")
    print(" [SUCCESS] Top 24 Phasic Units PSTH Analysis Complete!")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
