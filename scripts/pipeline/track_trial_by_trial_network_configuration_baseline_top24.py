#!/usr/bin/env python3
r"""
Trial-by-Trial Population Network Configuration Tracking
(STRICTLY FOR TOP 24 PHASIC UNITS DURING THE 60-SECOND BASELINE WINDOW: 0 to 60.0 s)

Tracks breath-by-breath population network state vectors v_b = [r_1, ..., r_24] across baseline breaths:
1. Dimensionality Reduction (PCA on 24-unit baseline trial-by-trial population vectors)
2. Pairwise Breath-by-Breath Cosine Similarity Matrix (100 baseline breaths)
3. Regression: Do baseline network PC1/PC2 scores predict diaphragm EMG amplitude & period?

Outputs:
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\trial_by_trial_network_baseline_24units_metrics.csv
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\trial_by_trial_network_baseline_24units_summary.png
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\trial_by_trial_network_baseline_24units_dashboard.html
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
    print(" Running Baseline (0-60s) Network Configuration Tracking (Top 24 Phasic Units Only)")
    print(f" ALF Directory: {alf_dir}")
    print(f" Output Directory: {out_dir}")
    print("=" * 90)

    # 1. Load Top 24 Phasic Units CSV
    top24_csv = out_dir / "top_24_phasic_single_units.csv"
    if not top24_csv.exists():
        print(f" Error: {top24_csv} not found.")
        sys.exit(1)

    top24_df = pd.read_csv(top24_csv)
    top24_uids = list(top24_df['cluster_id'])
    n_units = len(top24_uids)

    spikes_times = np.load(alf_dir / "spike_times.npy")
    spikes_clusters = np.load(alf_dir / "spike_clusters.npy")
    info_df = pd.read_csv(alf_dir / "cluster_info.tsv", sep="\t")

    fs_ap = 30000.0
    if spikes_times.dtype in [np.int64, np.int32] and spikes_times.max() > 100000:
        spikes_times = spikes_times / fs_ap

    MAX_TIME_S = 60.0 # First 60 seconds baseline window
    spk_mask = (spikes_times >= 0.0) & (spikes_times <= MAX_TIME_S) & (np.isin(spikes_clusters, top24_uids))
    spikes_times = spikes_times[spk_mask]
    spikes_clusters = spikes_clusters[spk_mask]

    # 2. Load NIDQ Signals for 0-60s Baseline
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
    baseline_onsets = [o for o in breath_onsets if 0.5 <= o <= MAX_TIME_S - 0.7]
    n_breaths = len(baseline_onsets)

    print(f" Analyzed {n_breaths} baseline breath cycles in the first 60 seconds.")

    # 3. Construct Baseline Population Vectors (n_breaths x 24 units)
    spiketrains = {u: np.sort(spikes_times[spikes_clusters == u]) for u in top24_uids}
    
    pop_matrix = np.zeros((n_breaths, n_units))
    breath_amplitudes = []
    breath_periods = []

    for b_i, o in enumerate(baseline_onsets):
        idx_o = np.searchsorted(t_nidq, o)
        i_win = int(0.35 * fs_nidq)
        amp = np.max(emg_env[idx_o : idx_o + i_win]) if idx_o + i_win <= len(emg_env) else emg_env[idx_o]
        breath_amplitudes.append(amp)

        if b_i < n_breaths - 1:
            breath_periods.append(baseline_onsets[b_i + 1] - o)
        else:
            breath_periods.append(breath_periods[-1] if len(breath_periods)>0 else 0.5)

        for u_i, uid in enumerate(top24_uids):
            u_spks = spiketrains[uid]
            spks_in_breath = u_spks[(u_spks >= o - 0.10) & (u_spks <= o + 0.25)]
            pop_matrix[b_i, u_i] = len(spks_in_breath) / 0.35 # Firing rate (Hz)

    breath_amplitudes = np.array(breath_amplitudes)
    breath_periods = np.array(breath_periods)

    # 4. Dimensionality Reduction: PCA on 24-Unit Baseline Population Vectors
    pop_matrix_sqrt = np.sqrt(pop_matrix)
    pca = PCA(n_components=5)
    pop_pcs = pca.fit_transform(pop_matrix_sqrt)
    exp_var = pca.explained_variance_ratio_ * 100.0

    print(f" 60s Baseline PCA Explained Variance: PC1 = {exp_var[0]:.1f}%, PC2 = {exp_var[1]:.1f}%, PC3 = {exp_var[2]:.1f}%")

    # 5. Pairwise Baseline Cosine Similarity Matrix
    norm_matrix = pop_matrix_sqrt / (np.linalg.norm(pop_matrix_sqrt, axis=1, keepdims=True) + 1e-8)
    sim_matrix = np.dot(norm_matrix, norm_matrix.T)

    # 6. Statistical Correlations for Baseline Epoch
    r_pc1_amp, p_pc1_amp = stats.pearsonr(pop_pcs[:, 0], breath_amplitudes)
    r_pc2_amp, p_pc2_amp = stats.pearsonr(pop_pcs[:, 1], breath_amplitudes)
    r_pc1_per, p_pc1_per = stats.pearsonr(pop_pcs[:, 0], breath_periods)

    print("\n" + "=" * 90)
    print(" BASELINE (0-60s) TOP 24 UNITS NETWORK CONFIGURATION CORRELATIONS")
    print("-" * 90)
    print(f" -> PC1 vs Diaphragm EMG Amplitude: Pearson r = {r_pc1_amp:.4f} (p = {p_pc1_amp:.4e})")
    print(f" -> PC2 vs Diaphragm EMG Amplitude: Pearson r = {r_pc2_amp:.4f} (p = {p_pc2_amp:.4e})")
    print(f" -> PC1 vs Respiratory Period     : Pearson r = {r_pc1_per:.4f} (p = {p_pc1_per:.4e})")
    print("=" * 90 + "\n")

    # Save CSV
    trial_df = pd.DataFrame({
        'breath_index': np.arange(n_breaths) + 1,
        'onset_time_s': np.round(baseline_onsets, 3),
        'emg_amplitude_uv': np.round(breath_amplitudes, 2),
        'breath_period_s': np.round(breath_periods, 4),
        'pc1_score': np.round(pop_pcs[:, 0], 4),
        'pc2_score': np.round(pop_pcs[:, 1], 4),
        'pc3_score': np.round(pop_pcs[:, 2], 4)
    })

    csv_out = out_dir / "trial_by_trial_network_baseline_24units_metrics.csv"
    trial_df.to_csv(csv_out, index=False)
    print(f" Saved Baseline 24-Unit Network Metrics CSV: {csv_out}")

    # 7. Generate High-Resolution Static PNG Figure (4 Panels)
    fig_mpl = plt.figure(figsize=(20, 12))

    # Panel A: PC1 vs PC2 Latent Space Scatter
    ax1 = fig_mpl.add_subplot(2, 2, 1)
    scatter = ax1.scatter(pop_pcs[:, 0], pop_pcs[:, 1], c=baseline_onsets, cmap='viridis', s=45, edgecolor='black', linewidth=0.4)
    plt.colorbar(scatter, ax=ax1, label='Baseline Session Time (s)')
    ax1.set_xlabel(f"Baseline Network PC1 ({exp_var[0]:.1f}% var)", fontweight='bold', fontsize=10)
    ax1.set_ylabel(f"Baseline Network PC2 ({exp_var[1]:.1f}% var)", fontweight='bold', fontsize=10)
    ax1.set_title("A. Baseline 24-Unit Network State (PC1 vs PC2)", fontweight='bold', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.4)

    # Panel B: Pairwise Baseline Cosine Similarity Matrix
    ax2 = fig_mpl.add_subplot(2, 2, 2)
    im = ax2.imshow(sim_matrix, cmap='magma', aspect='auto', origin='lower', extent=[0, n_breaths, 0, n_breaths])
    plt.colorbar(im, ax=ax2, label='Baseline Cosine Similarity')
    ax2.set_xlabel("Baseline Breath Index #", fontweight='bold', fontsize=10)
    ax2.set_ylabel("Baseline Breath Index #", fontweight='bold', fontsize=10)
    ax2.set_title(f"B. Baseline Breath Cosine Similarity Matrix ({n_breaths} Breaths)", fontweight='bold', fontsize=12)

    # Panel C: Baseline PC1 vs Diaphragm EMG Amplitude
    ax3 = fig_mpl.add_subplot(2, 2, 3)
    ax3.scatter(pop_pcs[:, 0], breath_amplitudes, color='#1f77b4', alpha=0.7, s=45, edgecolor='black', linewidth=0.4)
    sl1, ic1, r1, p1, _ = stats.linregress(pop_pcs[:, 0], breath_amplitudes)
    x_fit = np.linspace(np.min(pop_pcs[:, 0]), np.max(pop_pcs[:, 0]), 100)
    ax3.plot(x_fit, sl1 * x_fit + ic1, color='black', linewidth=2.5, label=f'Fit: r = {r_pc1_amp:.3f} (p = {p_pc1_amp:.2e})')
    ax3.set_xlabel("Baseline PC1 Network Score", fontweight='bold', fontsize=10)
    ax3.set_ylabel("Baseline Diaphragm EMG Amplitude (µV)", fontweight='bold', fontsize=10)
    ax3.set_title(f"C. Baseline PC1 Score vs EMG Amplitude (r = {r_pc1_amp:.3f})", fontweight='bold', fontsize=12)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, linestyle=':', alpha=0.4)

    # Panel D: Baseline PC2 vs Diaphragm EMG Amplitude
    ax4 = fig_mpl.add_subplot(2, 2, 4)
    ax4.scatter(pop_pcs[:, 1], breath_amplitudes, color='crimson', alpha=0.7, s=45, edgecolor='black', linewidth=0.4)
    sl2, ic2, r2, p2, _ = stats.linregress(pop_pcs[:, 1], breath_amplitudes)
    x_fit2 = np.linspace(np.min(pop_pcs[:, 1]), np.max(pop_pcs[:, 1]), 100)
    ax4.plot(x_fit2, sl2 * x_fit2 + ic2, color='black', linewidth=2.5, label=f'Fit: r = {r_pc2_amp:.3f} (p = {p_pc2_amp:.2e})')
    ax4.set_xlabel("Baseline PC2 Network Score", fontweight='bold', fontsize=10)
    ax4.set_ylabel("Baseline Diaphragm EMG Amplitude (µV)", fontweight='bold', fontsize=10)
    ax4.set_title(f"D. Baseline PC2 Score vs EMG Amplitude (r = {r_pc2_amp:.3f})", fontweight='bold', fontsize=12)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, linestyle=':', alpha=0.4)

    plt.suptitle("Subject TP_Tacr1_1: Baseline (0 - 60s) Top 24 Phasic Units Network Configuration Tracking", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    png_out = out_dir / "trial_by_trial_network_baseline_24units_summary.png"
    plt.savefig(png_out, dpi=300)
    plt.close()

    # Copy PNG to artifact directory
    artifact_png = artifact_dir / "trial_by_trial_network_baseline_24units_summary.png"
    shutil.copy2(png_out, artifact_png)
    print(f" Saved Diagnostic Baseline Figure: {png_out}")

    # 8. Interactive Plotly Dashboard
    fig_plotly = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "A. Baseline Network State Latent Space (PC1 vs PC2)",
            f"B. Baseline Pairwise Similarity Matrix ({n_breaths} Breaths)",
            f"C. Baseline PC1 vs EMG Amplitude (r = {r_pc1_amp:.3f})",
            f"D. Baseline PC2 vs EMG Amplitude (r = {r_pc2_amp:.3f})"
        ]
    )

    fig_plotly.add_trace(go.Scatter(
        x=pop_pcs[:, 0], y=pop_pcs[:, 1], mode='markers',
        marker=dict(color=baseline_onsets, colorscale='Viridis', size=7, showscale=True),
        hoverinfo='text',
        text=[f"Breath #{b}<br>Time: {t}s<br>EMG: {a}µV" for b, t, a in zip(trial_df['breath_index'], trial_df['onset_time_s'], trial_df['emg_amplitude_uv'])],
        name="Baseline State"
    ), row=1, col=1)

    fig_plotly.add_trace(go.Heatmap(
        z=sim_matrix, colorscale='Magma', name="Similarity"
    ), row=1, col=2)

    fig_plotly.add_trace(go.Scatter(
        x=pop_pcs[:, 0], y=breath_amplitudes, mode='markers',
        marker=dict(color='#1f77b4', size=6, opacity=0.7), name="PC1 vs Amp"
    ), row=2, col=1)
    fig_plotly.add_trace(go.Scatter(
        x=x_fit, y=sl1 * x_fit + ic1, mode='lines', line=dict(color='black', width=2.5), name="PC1 Fit"
    ), row=2, col=1)

    fig_plotly.add_trace(go.Scatter(
        x=pop_pcs[:, 1], y=breath_amplitudes, mode='markers',
        marker=dict(color='crimson', size=6, opacity=0.7), name="PC2 vs Amp"
    ), row=2, col=2)
    fig_plotly.add_trace(go.Scatter(
        x=x_fit2, y=sl2 * x_fit2 + ic2, mode='lines', line=dict(color='black', width=2.5), name="PC2 Fit"
    ), row=2, col=2)

    fig_plotly.update_layout(
        title="<b>Subject TP_Tacr1_1: Baseline (0-60s) Top 24 Phasic Units Network Tracking Dashboard</b>",
        template="plotly_white",
        height=850,
        showlegend=False
    )

    html_out = out_dir / "trial_by_trial_network_baseline_24units_dashboard.html"
    fig_plotly.write_html(str(html_out))
    print(f" Saved Interactive HTML Dashboard: {html_out}")

    print("\n==========================================================================================")
    print(" [SUCCESS] Baseline 24-Unit Network Configuration Tracking Complete!")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
