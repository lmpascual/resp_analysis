#!/usr/bin/env python3
r"""
Trial-by-Trial Population Network Configuration Tracking
(Bush et al. 2025 & Ramirez et al. 2004 Framework)

Tracks breath-by-breath population network state vectors v_b = [r_1, ..., r_N] across N = 672 breath cycles:
1. Dimensionality Reduction (PCA on trial-by-trial population vectors)
2. Network State Drift & Reconfiguration across Baseline, Phototagging, and Recovery Epochs
3. Pairwise Breath-by-Breath Cosine Similarity Matrix (S_bb')
4. Regression: Do trial-by-trial PC1/PC2 network state coordinates predict diaphragm EMG amplitude & period?

Outputs:
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\trial_by_trial_network_configuration_metrics.csv
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\trial_by_trial_network_configuration_summary.png
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\trial_by_trial_network_configuration_dashboard.html
"""

import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
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
    print(" Running Trial-by-Trial Population Network Configuration Tracking")
    print(f" ALF Directory: {alf_dir}")
    print(f" Output Directory: {out_dir}")
    print("=" * 90)

    # 1. Load 126 Allen QC Passed Units
    qc_csv = out_dir / "allen_neuropixels_quality_metrics_baseline_60s.csv"
    if not qc_csv.exists():
        print(f" Error: {qc_csv} not found.")
        sys.exit(1)

    metrics_df = pd.read_csv(qc_csv)
    passed_df = metrics_df[metrics_df['passed_allen_qc']].copy().reset_index(drop=True)
    passed_uids = list(passed_df['cluster_id'])
    n_units = len(passed_uids)

    spikes_times = np.load(alf_dir / "spike_times.npy")
    spikes_clusters = np.load(alf_dir / "spike_clusters.npy")
    info_df = pd.read_csv(alf_dir / "cluster_info.tsv", sep="\t")

    fs_ap = 30000.0
    if spikes_times.dtype in [np.int64, np.int32] and spikes_times.max() > 100000:
        spikes_times = spikes_times / fs_ap

    MAX_TIME_S = 550.0
    spk_mask = (spikes_times >= 0.0) & (spikes_times <= MAX_TIME_S) & (np.isin(spikes_clusters, passed_uids))
    spikes_times = spikes_times[spk_mask]
    spikes_clusters = spikes_clusters[spk_mask]

    # 2. Load NIDQ Signals
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

    laser_active = (ch6_laser > 0.8).astype(int)
    laser_onsets = t_nidq[np.where(np.diff(laser_active) == 1)[0]]

    # Filter valid breath onsets across entire recording
    valid_onsets = [o for o in breath_onsets if 0.5 <= o <= MAX_TIME_S - 0.7]
    n_breaths = len(valid_onsets)

    print(f" Extracted population vectors across {n_breaths} breath cycles.")

    # 3. Construct Trial-by-Trial Population Vectors (n_breaths x n_units)
    spiketrains = {u: np.sort(spikes_times[spikes_clusters == u]) for u in passed_uids}
    
    pop_matrix = np.zeros((n_breaths, n_units))
    breath_amplitudes = []
    breath_periods = []
    epoch_labels = []

    for b_i, o in enumerate(valid_onsets):
        # Epoch assignment: Baseline (0-60s), Phototagging (60-300s), Recovery (300-550s)
        if o < 60.0:
            epoch = "Baseline (0-60s)"
        elif o < 300.0:
            epoch = "Phototagging Window (60-300s)"
        else:
            epoch = "Recovery Baseline (300-550s)"
        epoch_labels.append(epoch)

        # Diaphragm EMG peak amplitude for breath b
        idx_o = np.searchsorted(t_nidq, o)
        i_win = int(0.35 * fs_nidq)
        amp = np.max(emg_env[idx_o : idx_o + i_win]) if idx_o + i_win <= len(emg_env) else emg_env[idx_o]
        breath_amplitudes.append(amp)

        # Breath period
        if b_i < n_breaths - 1:
            breath_periods.append(valid_onsets[b_i + 1] - o)
        else:
            breath_periods.append(breath_periods[-1] if len(breath_periods)>0 else 0.5)

        # Population vector in inspiratory burst window [-100ms, +250ms]
        for u_i, uid in enumerate(passed_uids):
            u_spks = spiketrains[uid]
            spks_in_breath = u_spks[(u_spks >= o - 0.10) & (u_spks <= o + 0.25)]
            pop_matrix[b_i, u_i] = len(spks_in_breath) / 0.35 # Firing rate (Hz)

    breath_amplitudes = np.array(breath_amplitudes)
    breath_periods = np.array(breath_periods)

    # 4. Dimensionality Reduction: PCA on Trial-by-Trial Population Vectors
    # Variance stabilizing transform (sqrt)
    pop_matrix_sqrt = np.sqrt(pop_matrix)
    pca = PCA(n_components=5)
    pop_pcs = pca.fit_transform(pop_matrix_sqrt)
    exp_var = pca.explained_variance_ratio_ * 100.0

    print(f" PCA Explained Variance: PC1 = {exp_var[0]:.1f}%, PC2 = {exp_var[1]:.1f}%, PC3 = {exp_var[2]:.1f}%")

    # 5. Pairwise Trial-by-Trial Cosine Similarity Matrix
    norm_matrix = pop_matrix_sqrt / (np.linalg.norm(pop_matrix_sqrt, axis=1, keepdims=True) + 1e-8)
    sim_matrix = np.dot(norm_matrix, norm_matrix.T) # (n_breaths x n_breaths)

    # 6. Statistical Correlations: Trial-by-Trial PC1 / PC2 vs Diaphragm EMG Amplitude
    r_pc1_amp, p_pc1_amp = stats.pearsonr(pop_pcs[:, 0], breath_amplitudes)
    r_pc2_amp, p_pc2_amp = stats.pearsonr(pop_pcs[:, 1], breath_amplitudes)
    r_pc1_per, p_pc1_per = stats.pearsonr(pop_pcs[:, 0], breath_periods)

    print(f" PC1 vs Diaphragm EMG Amplitude: Pearson r = {r_pc1_amp:.4f} (p = {p_pc1_amp:.4e})")
    print(f" PC2 vs Diaphragm EMG Amplitude: Pearson r = {r_pc2_amp:.4f} (p = {p_pc2_amp:.4e})")
    print(f" PC1 vs Respiratory Period     : Pearson r = {r_pc1_per:.4f} (p = {p_pc1_per:.4e})")

    # Save Results CSV
    trial_df = pd.DataFrame({
        'breath_index': np.arange(n_breaths) + 1,
        'onset_time_s': np.round(valid_onsets, 3),
        'epoch': epoch_labels,
        'emg_amplitude_uv': np.round(breath_amplitudes, 2),
        'breath_period_s': np.round(breath_periods, 4),
        'pc1_score': np.round(pop_pcs[:, 0], 4),
        'pc2_score': np.round(pop_pcs[:, 1], 4),
        'pc3_score': np.round(pop_pcs[:, 2], 4)
    })

    csv_out = out_dir / "trial_by_trial_network_configuration_metrics.csv"
    trial_df.to_csv(csv_out, index=False)
    print(f" Saved Trial-by-Trial Metrics CSV: {csv_out}")

    # 7. Create Diagnostic Matplotlib Figure (4 Panels)
    fig_mpl = plt.figure(figsize=(22, 12))

    # Panel A: PC1 vs PC2 Trial-by-Trial Network State Trajectory
    ax1 = fig_mpl.add_subplot(2, 2, 1)
    color_map = {"Baseline (0-60s)": "#2ca02c", "Phototagging Window (60-300s)": "#ff7f0e", "Recovery Baseline (300-550s)": "#1f77b4"}
    
    for epoch_name, color in color_map.items():
        mask_e = (trial_df['epoch'] == epoch_name)
        ax1.scatter(pop_pcs[mask_e, 0], pop_pcs[mask_e, 1], color=color, label=epoch_name, alpha=0.7, s=35, edgecolor='black', linewidth=0.3)

    ax1.set_xlabel(f"Network State PC1 ({exp_var[0]:.1f}% var)", fontweight='bold', fontsize=10)
    ax1.set_ylabel(f"Network State PC2 ({exp_var[1]:.1f}% var)", fontweight='bold', fontsize=10)
    ax1.set_title("A. Trial-by-Trial Network State Latent Space (PC1 vs PC2)", fontweight='bold', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.4)

    # Panel B: Pairwise Breath-by-Breath Cosine Similarity Matrix
    ax2 = fig_mpl.add_subplot(2, 2, 2)
    im = ax2.imshow(sim_matrix, cmap='viridis', aspect='auto', origin='lower', extent=[0, n_breaths, 0, n_breaths])
    plt.colorbar(im, ax=ax2, label='Population Vector Cosine Similarity')
    ax2.set_xlabel("Breath Index #", fontweight='bold', fontsize=10)
    ax2.set_ylabel("Breath Index #", fontweight='bold', fontsize=10)
    ax2.set_title("B. Pairwise Breath-by-Breath Network Similarity Matrix", fontweight='bold', fontsize=12)

    # Panel C: Trial-by-Trial PC1 Network Score vs Diaphragm EMG Amplitude
    ax3 = fig_mpl.add_subplot(2, 2, 3)
    ax3.scatter(pop_pcs[:, 0], breath_amplitudes, color='crimson', alpha=0.6, s=35, edgecolor='black', linewidth=0.3)
    
    sl1, ic1, r1, p1, _ = stats.linregress(pop_pcs[:, 0], breath_amplitudes)
    x_fit = np.linspace(np.min(pop_pcs[:, 0]), np.max(pop_pcs[:, 0]), 100)
    ax3.plot(x_fit, sl1 * x_fit + ic1, color='black', linewidth=2.5, label=f'Fit: r = {r_pc1_amp:.3f} (p = {p_pc1_amp:.2e})')

    ax3.set_xlabel("Network Configuration PC1 Score", fontweight='bold', fontsize=10)
    ax3.set_ylabel("Peak Diaphragm EMG Amplitude (µV)", fontweight='bold', fontsize=10)
    ax3.set_title(f"C. PC1 Network Configuration vs Breath Motor Amplitude\n(Pearson r = {r_pc1_amp:.3f}, p = {p_pc1_amp:.2e})", fontweight='bold', fontsize=12)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, linestyle=':', alpha=0.4)

    # Panel D: Trial-by-Trial Network PC1 Drift Across Recording Time
    ax4 = fig_mpl.add_subplot(2, 2, 4)
    ax4.plot(valid_onsets, pop_pcs[:, 0], color='purple', linewidth=1.2, alpha=0.85, label='PC1 Network Drift')
    ax4.axvspan(0, 60, color='yellow', alpha=0.2, label='Baseline')
    ax4.axvspan(60, 300, color='magenta', alpha=0.15, label='Phototagging')
    ax4.axvspan(300, 550, color='cyan', alpha=0.15, label='Recovery')
    ax4.set_xlabel("Session Time (s)", fontweight='bold', fontsize=10)
    ax4.set_ylabel("PC1 Network Score", fontweight='bold', fontsize=10)
    ax4.set_title("D. Population Network Configuration Drift Across Recording Epochs", fontweight='bold', fontsize=12)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, linestyle=':', alpha=0.4)

    plt.suptitle("Subject TP_Tacr1_1: Trial-by-Trial Population Network Configuration Tracking", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    png_out = out_dir / "trial_by_trial_network_configuration_summary.png"
    plt.savefig(png_out, dpi=300)
    plt.close()

    # Copy PNG to artifact directory
    artifact_png = artifact_dir / "trial_by_trial_network_configuration_summary.png"
    shutil.copy2(png_out, artifact_png)
    print(f" Saved Diagnostic Figure: {png_out}")

    # 8. Interactive Plotly Dashboard
    fig_plotly = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "A. Trial-by-Trial Network Latent Space (PC1 vs PC2)",
            f"B. Pairwise Breath Similarity Matrix ({n_breaths} Breaths)",
            f"C. PC1 Network Score vs EMG Amplitude (r = {r_pc1_amp:.3f})",
            "D. PC1 Network Configuration Drift Across Session Time"
        ]
    )

    for epoch_name, color in color_map.items():
        mask_e = (trial_df['epoch'] == epoch_name)
        fig_plotly.add_trace(go.Scatter(
            x=pop_pcs[mask_e, 0],
            y=pop_pcs[mask_e, 1],
            mode='markers',
            marker=dict(color=color, size=6, opacity=0.7),
            name=epoch_name,
            hoverinfo='text',
            text=[f"Breath #{b}<br>Time: {t}s<br>EMG: {a}µV" for b, t, a in zip(trial_df.loc[mask_e, 'breath_index'], trial_df.loc[mask_e, 'onset_time_s'], trial_df.loc[mask_e, 'emg_amplitude_uv'])]
        ), row=1, col=1)

    fig_plotly.add_trace(go.Heatmap(
        z=sim_matrix, colorscale='Viridis', name="Similarity"
    ), row=1, col=2)

    fig_plotly.add_trace(go.Scatter(
        x=pop_pcs[:, 0], y=breath_amplitudes, mode='markers',
        marker=dict(color='crimson', size=5, opacity=0.6), name="PC1 vs Amp"
    ), row=2, col=1)
    fig_plotly.add_trace(go.Scatter(
        x=x_fit, y=sl1 * x_fit + ic1, mode='lines', line=dict(color='black', width=2.5), name="Fit"
    ), row=2, col=1)

    fig_plotly.add_trace(go.Scatter(
        x=valid_onsets, y=pop_pcs[:, 0], mode='lines',
        line=dict(color='purple', width=1.5), name="PC1 Drift"
    ), row=2, col=2)

    fig_plotly.update_layout(
        title="<b>Subject TP_Tacr1_1: Trial-by-Trial Network Configuration Tracking Dashboard</b>",
        template="plotly_white",
        height=850,
        showlegend=False
    )

    html_out = out_dir / "trial_by_trial_network_configuration_dashboard.html"
    fig_plotly.write_html(str(html_out))
    print(f" Saved Interactive HTML Dashboard: {html_out}")

    print("\n==========================================================================================")
    print(" [SUCCESS] Trial-by-Trial Network Configuration Tracking Complete!")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
