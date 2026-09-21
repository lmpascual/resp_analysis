#!/usr/bin/env python3
r"""
Baseline Epoch (0 - 60s) Breath-Onset Aligned PSTHs & Rasters
Units: 100, 118, 55, 13 for subject TP_Tacr1_1

Aligns single-unit spike trains strictly to unperturbed baseline diaphragm EMG inspiratory onsets
during the initial 1-minute baseline period (0 to 60.0 s).
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

def main():
    out_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation")
    out_dir.mkdir(parents=True, exist_ok=True)
    session_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1\alf_session")
    alf_dir = session_dir / "alf" / "probe00"
    raw_dir = Path(r"Z:\ramirezlab\pascual\rec_data\TP_Tacr1_1_npx2_g4")
    artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\0e7be222-f160-4bc2-921c-f3fdf8b7825e")

    print("=" * 90)
    print(" Computing 60s Baseline Epoch Breath-Onset Aligned PSTHs & Rasters")
    print(f" ALF Directory: {alf_dir}")
    print(f" Output Directory: {out_dir}")
    print("=" * 90)

    # 1. Load Spikes & Cluster Metadata
    spikes_times = np.load(alf_dir / "spike_times.npy")
    spikes_clusters = np.load(alf_dir / "spike_clusters.npy")
    info_df = pd.read_csv(alf_dir / "cluster_info.tsv", sep="\t")

    fs_ap = 30000.0
    if spikes_times.dtype in [np.int64, np.int32] and spikes_times.max() > 100000:
        spikes_times = spikes_times / fs_ap

    MAX_TIME_S = 60.0 # First 60 seconds baseline epoch
    spk_mask = (spikes_times >= 0.0) & (spikes_times <= MAX_TIME_S)
    spikes_times = spikes_times[spk_mask]
    spikes_clusters = spikes_clusters[spk_mask]

    target_uids = [100, 118, 55, 13]
    unit_colors = {100: '#1f77b4', 118: '#ff7f0e', 55: '#2ca02c', 13: '#d62728'}

    # 2. Load NIDQ Signals for 0-60s
    nidq_meta = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.meta"
    nidq_bin = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.bin"
    nidq_data = read_nidq_bin(nidq_bin, nidq_meta)

    t_nidq = nidq_data.time_vector
    fs_nidq = nidq_data.fs

    nidq_mask = (t_nidq >= 0.0) & (t_nidq <= MAX_TIME_S)
    t_nidq = t_nidq[nidq_mask]
    emg_raw = nidq_data.all_channels[0][nidq_mask]

    emg_cfg = EMGConfig(fs=fs_nidq, highpass_cutoff=300.0, lowpass_cutoff=3000.0, notch_freq=60.0, smoothing_window_ms=20.0)
    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    emg_env = emg_proc.envelope
    breath_onsets = emg_proc.breath_onsets

    # Filter baseline breath onsets within [0.5s, 59.3s]
    baseline_onsets = [o for o in breath_onsets if 0.5 <= o <= (MAX_TIME_S - 0.7)]
    n_breaths = len(baseline_onsets)
    print(f" Found {n_breaths} baseline breath cycles in the first 60 seconds.")

    # Time Window for PSTH
    t_pre = 0.40  # 400 ms pre-inspiration
    t_post = 0.60 # 600 ms post-inspiration
    bin_width = 0.010 # 10 ms bins
    psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)
    psth_centers = (psth_bins[:-1] + psth_bins[1:]) / 2.0 * 1000.0 # ms

    # Extract Aligned Baseline EMG Envelope
    emg_sweeps = []
    for o in baseline_onsets:
        idx_o = np.searchsorted(t_nidq, o)
        i_pre = int(idx_o - t_pre * fs_nidq)
        i_post = int(idx_o + t_post * fs_nidq)
        if 0 <= i_pre and i_post <= len(emg_env):
            emg_sweeps.append(emg_env[i_pre:i_post])

    emg_sweeps = np.array(emg_sweeps)
    mean_emg = np.mean(emg_sweeps, axis=0) if len(emg_sweeps) > 0 else np.zeros(100)
    sem_emg = stats.sem(emg_sweeps, axis=0) if len(emg_sweeps) > 0 else np.zeros(100)
    t_emg_ms = np.linspace(-t_pre, t_post, len(mean_emg)) * 1000.0

    # 3. Build Figure (4 Columns for Units 100, 118, 55, 13)
    fig_mpl = plt.figure(figsize=(20, 14))

    for col_idx, uid in enumerate(target_uids):
        u_spks = np.sort(spikes_times[spikes_clusters == uid])
        u_color = unit_colors[uid]
        info_row = info_df[info_df['cluster_id'] == uid].iloc[0]

        trial_spikes = []
        trial_counts = []

        for b_i, o in enumerate(baseline_onsets):
            spks_in_win = u_spks[(u_spks >= o - t_pre) & (u_spks <= o + t_post)] - o
            trial_spikes.append(spks_in_win)
            
            c, _ = np.histogram(spks_in_win, bins=psth_bins)
            trial_counts.append(c / bin_width)

        trial_counts = np.array(trial_counts) # (n_breaths, n_bins)
        mean_psth = np.mean(trial_counts, axis=0) if len(trial_counts)>0 else np.zeros(len(psth_centers))
        sem_psth = stats.sem(trial_counts, axis=0) if len(trial_counts)>0 else np.zeros(len(psth_centers))

        # Subplot 1: Baseline Diaphragm EMG Envelope
        ax_emg = fig_mpl.add_subplot(3, 4, col_idx + 1)
        ax_emg.plot(t_emg_ms, mean_emg, color='black', linewidth=1.5)
        ax_emg.fill_between(t_emg_ms, mean_emg - sem_emg, mean_emg + sem_emg, color='gray', alpha=0.3)
        ax_emg.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Insp Onset (t=0)')
        ax_emg.set_title(f"Unit {uid} (Shank {info_row['shank']}, Depth {info_row['depth']} µm)\nBaseline Rate: {len(u_spks)/MAX_TIME_S:.2f} Hz", fontweight='bold', fontsize=11)
        ax_emg.set_ylabel("Diaphragm EMG (µV)", fontsize=9, fontweight='bold')
        ax_emg.set_xlim([-t_pre * 1000, t_post * 1000])
        ax_emg.grid(True, linestyle=':', alpha=0.4)

        # Subplot 2: Baseline Breath Raster
        ax_rast = fig_mpl.add_subplot(3, 4, col_idx + 5, sharex=ax_emg)
        for b_i, rel_spks in enumerate(trial_spikes):
            if len(rel_spks) > 0:
                ax_rast.vlines(rel_spks * 1000.0, b_i + 0.5, b_i + 1.5, color=u_color, linewidth=1.2, alpha=0.85)

        ax_rast.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax_rast.set_ylabel("Baseline Breath Cycle #", fontsize=9, fontweight='bold')
        ax_rast.set_ylim([0, n_breaths + 1])
        ax_rast.set_title(f"Baseline Raster (N = {n_breaths} Cycles)", fontsize=10, fontweight='bold')

        # Subplot 3: Baseline PSTH Firing Rate
        ax_psth = fig_mpl.add_subplot(3, 4, col_idx + 9, sharex=ax_emg)
        ax_psth.bar(psth_centers, mean_psth, width=bin_width*1000.0, color=u_color, alpha=0.4, edgecolor=u_color)
        ax_psth.plot(psth_centers, mean_psth, color=u_color, linewidth=2.5, label='Mean Rate')
        ax_psth.fill_between(psth_centers, np.maximum(0, mean_psth - sem_psth), mean_psth + sem_psth, color=u_color, alpha=0.25)
        ax_psth.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax_psth.set_xlabel("Time relative to Insp Onset (ms)", fontsize=10, fontweight='bold')
        ax_psth.set_ylabel("Firing Rate (Hz)", fontsize=9, fontweight='bold')
        ax_psth.set_title("Baseline PSTH (Mean ± SEM)", fontsize=10, fontweight='bold')
        ax_psth.grid(True, linestyle=':', alpha=0.4)

    plt.suptitle("Subject TP_Tacr1_1: Baseline Epoch (0 - 60s) Breath-Onset Aligned PSTHs & Rasters", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    png_out = out_dir / "single_unit_baseline_60s_psth_rasters.png"
    plt.savefig(png_out, dpi=300)
    plt.close()

    # Copy to artifact directory
    artifact_png = artifact_dir / "single_unit_baseline_60s_psth_rasters.png"
    shutil.copy2(png_out, artifact_png)
    print(f" Saved PNG PSTH Figure (Baseline 60s): {png_out}")

    # 4. Interactive Plotly PSTH Dashboard
    fig_plotly = make_subplots(
        rows=3, cols=4,
        subplot_titles=[
            f"Unit {u} Baseline EMG" for u in target_uids
        ] + [
            f"Unit {u} Baseline Raster" for u in target_uids
        ] + [
            f"Unit {u} Baseline PSTH (Hz)" for u in target_uids
        ],
        vertical_spacing=0.08, horizontal_spacing=0.05
    )

    for i, uid in enumerate(target_uids):
        col = i + 1
        u_spks = np.sort(spikes_times[spikes_clusters == uid])
        c = unit_colors[uid]

        # Row 1: EMG Envelope
        fig_plotly.add_trace(go.Scatter(x=t_emg_ms, y=mean_emg, mode='lines', line=dict(color='black', width=2), name="EMG Envelope"), row=1, col=col)
        
        # Row 2: Raster Points
        r_x, r_y = [], []
        trial_counts = []
        for b_i, o in enumerate(baseline_onsets):
            spks_in_win = u_spks[(u_spks >= o - t_pre) & (u_spks <= o + t_post)] - o
            r_x.extend(spks_in_win * 1000.0)
            r_y.extend([b_i + 1] * len(spks_in_win))

            cnts, _ = np.histogram(spks_in_win, bins=psth_bins)
            trial_counts.append(cnts / bin_width)

        fig_plotly.add_trace(go.Scatter(x=r_x, y=r_y, mode='markers', marker=dict(color=c, size=3, opacity=0.8), name=f"Unit {uid} Raster"), row=2, col=col)

        # Row 3: PSTH Bar + Line
        trial_counts = np.array(trial_counts)
        mean_psth = np.mean(trial_counts, axis=0) if len(trial_counts)>0 else np.zeros(len(psth_centers))
        fig_plotly.add_trace(go.Bar(x=psth_centers, y=mean_psth, marker_color=c, opacity=0.6, name=f"Unit {uid} PSTH"), row=3, col=col)
        fig_plotly.add_trace(go.Scatter(x=psth_centers, y=mean_psth, mode='lines', line=dict(color=c, width=3), showlegend=False), row=3, col=col)

    fig_plotly.update_layout(
        title="<b>Subject TP_Tacr1_1: Interactive Baseline (0 - 60s) Breath-Onset PSTH & Raster Dashboard</b>",
        template="plotly_white",
        height=950,
        showlegend=False
    )

    html_out = out_dir / "single_unit_baseline_60s_psth_dashboard.html"
    fig_plotly.write_html(str(html_out))
    print(f" Saved Interactive HTML Dashboard (Baseline 60s): {html_out}")

    print("\n==========================================================================================")
    print(" [SUCCESS] Baseline 60s Breath-Onset PSTH Analysis Complete!")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
