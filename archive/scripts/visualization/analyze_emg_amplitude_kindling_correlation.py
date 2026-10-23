#!/usr/bin/env python3
r"""
Correlation Analysis: Diaphragm EMG Breath Amplitude vs Synchronous Population Kindling & Individual Spiking
(Ramirez et al. 2004 & Bush et al. 2025 Framework)

Quantifies breath-by-breath relationships across N = 672 unperturbed breath cycles:
1. Diaphragm EMG Peak Amplitude (uV) vs Synchronous Population Kindling Rate (-100ms to 0ms pre-inspiration)
2. Diaphragm EMG Peak Amplitude (uV) vs Individual Single-Unit Spike Counts per breath (Top 24 Phasic Units)

Outputs:
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\emg_amplitude_kindling_correlation_metrics.csv
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\emg_amplitude_kindling_correlation_summary.png
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\emg_amplitude_kindling_correlation_dashboard.html
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
    print(" Running Correlation Analysis: EMG Breath Amplitude vs Population Kindling & Single Units")
    print(f" ALF Directory: {alf_dir}")
    print(f" Output Directory: {out_dir}")
    print("=" * 90)

    # 1. Load Top 24 Phasic Units CSV & Spikes
    top24_csv = out_dir / "top_24_phasic_single_units.csv"
    if not top24_csv.exists():
        print(f" Error: {top24_csv} not found.")
        sys.exit(1)

    top24_df = pd.read_csv(top24_csv)
    top24_uids = list(top24_df['cluster_id'])

    spikes_times = np.load(alf_dir / "spike_times.npy")
    spikes_clusters = np.load(alf_dir / "spike_clusters.npy")
    info_df = pd.read_csv(alf_dir / "cluster_info.tsv", sep="\t")

    fs_ap = 30000.0
    if spikes_times.dtype in [np.int64, np.int32] and spikes_times.max() > 100000:
        spikes_times = spikes_times / fs_ap

    MAX_TIME_S = 550.0
    spk_mask = (spikes_times >= 0.0) & (spikes_times <= MAX_TIME_S) & (np.isin(spikes_clusters, top24_uids))
    spikes_times = spikes_times[spk_mask]
    spikes_clusters = spikes_clusters[spk_mask]

    # 2. Load NIDQ EMG & Laser TTL
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

    # Unperturbed breath onsets
    valid_onsets = [o for o in breath_onsets if (len(laser_onsets)==0 or np.min(np.abs(o-laser_onsets))>0.35) and (0.5 <= o <= MAX_TIME_S-0.7)]
    n_valid_breaths = len(valid_onsets)

    # Extract Peak Diaphragm EMG Amplitude per breath
    breath_amplitudes = []
    for o in valid_onsets:
        idx_o = np.searchsorted(t_nidq, o)
        i_win = int(0.35 * fs_nidq) # 350 ms window post onset
        if idx_o + i_win <= len(emg_env):
            amp = np.max(emg_env[idx_o : idx_o + i_win])
        else:
            amp = emg_env[idx_o]
        breath_amplitudes.append(amp)

    breath_amplitudes = np.array(breath_amplitudes)

    # 3. Synchronous Population Kindling Rate per breath (-100ms to 0ms pre-inspiration)
    spiketrains = {u: np.sort(spikes_times[spikes_clusters == u]) for u in top24_uids}

    pop_kindling_spikes_per_breath = []
    unit_spikes_per_breath = {u: [] for u in top24_uids}

    for o in valid_onsets:
        # Population pre-inspiratory kindling count (-100ms to 0ms)
        pop_spk_cnt = 0
        for uid in top24_uids:
            u_spks = spiketrains[uid]
            spks_pre = u_spks[(u_spks >= o - 0.10) & (u_spks <= o)]
            pop_spk_cnt += len(spks_pre)

            # Whole breath spike count
            spks_brth = u_spks[(u_spks >= o - 0.15) & (u_spks <= o + 0.35)]
            unit_spikes_per_breath[uid].append(len(spks_brth))

        pop_kindling_spikes_per_breath.append(pop_spk_cnt)

    pop_kindling_spikes_per_breath = np.array(pop_kindling_spikes_per_breath)
    pop_kindling_rate_hz = pop_kindling_spikes_per_breath / 0.10 / len(top24_uids) # Hz per unit

    # Compute Correlation for Population Kindling vs EMG Amplitude
    r_pop, p_pop = stats.pearsonr(pop_kindling_rate_hz, breath_amplitudes)
    rho_pop, p_rho_pop = stats.spearmanr(pop_kindling_rate_hz, breath_amplitudes)

    print("\n" + "=" * 90)
    print(" SYNCHRONOUS POPULATION KINDLING vs DIAPHRAGM EMG AMPLITUDE CORRELATION")
    print("-" * 90)
    print(f" -> Pearson r   : {r_pop:.4f} (p = {p_pop:.4e})")
    print(f" -> Spearman rho: {rho_pop:.4f} (p = {p_rho_pop:.4e})")
    print("=" * 90)

    # 4. Compute Correlations for Individual Units vs EMG Amplitude
    corr_results = []
    for uid in top24_uids:
        u_spk_cnts = np.array(unit_spikes_per_breath[uid])
        r_u, p_u = stats.pearsonr(u_spk_cnts, breath_amplitudes)
        rho_u, p_rho_u = stats.spearmanr(u_spk_cnts, breath_amplitudes)
        
        info_row = top24_df[top24_df['cluster_id'] == uid].iloc[0]
        corr_results.append({
            'cluster_id': uid,
            'shank': int(info_row['shank']),
            'depth': float(info_row['depth']),
            'vector_strength': float(info_row['vector_strength']),
            'phase': 'Inspiratory' if info_row['pref_phase'] >= 0 else 'Expiratory',
            'pearson_r': round(r_u, 4),
            'pearson_p': float(p_u),
            'spearman_rho': round(rho_u, 4),
            'spearman_p': float(p_rho_u),
            'sig_star': '***' if p_u < 0.001 else '**' if p_u < 0.01 else '*' if p_u < 0.05 else 'ns'
        })

    corr_df = pd.DataFrame(corr_results)
    csv_out = out_dir / "emg_amplitude_kindling_correlation_metrics.csv"
    corr_df.to_csv(csv_out, index=False)
    print(f"\n Saved Correlation Metrics CSV: {csv_out}")

    print("\n" + "=" * 90)
    print(f"{'Unit':<6} | {'Shank':<5} | {'Depth(um)':<9} | {'Phase':<11} | {'Pearson r':<10} | {'p-value':<10} | {'Spearman rho':<12} | {'Sig':<6}")
    print("-" * 90)
    for _, r in corr_df.iterrows():
        print(f"{int(r['cluster_id']):<6} | {int(r['shank']):<5} | {r['depth']:<9.1f} | {r['phase']:<11} | {r['pearson_r']:<10.4f} | {r['pearson_p']:<10.4e} | {r['spearman_rho']:<12.4f} | {r['sig_star']:<6}")
    print("=" * 90 + "\n")

    # 5. Generate Diagnostic Matplotlib Figure (4 Panels)
    fig_mpl = plt.figure(figsize=(20, 12))

    # Panel A: Population Kindling vs EMG Amplitude Scatter & Trendline
    ax1 = fig_mpl.add_subplot(2, 2, 1)
    ax1.scatter(pop_kindling_rate_hz, breath_amplitudes, color='crimson', alpha=0.6, s=30, edgecolor='black', linewidth=0.5)
    
    # Linear Regression Line
    slope, intercept, r_value, p_value, std_err = stats.linregress(pop_kindling_rate_hz, breath_amplitudes)
    x_fit = np.linspace(np.min(pop_kindling_rate_hz), np.max(pop_kindling_rate_hz), 100)
    y_fit = slope * x_fit + intercept
    ax1.plot(x_fit, y_fit, color='black', linewidth=2.5, label=f'Fit: r = {r_pop:.3f} (p = {p_pop:.2e})')

    ax1.set_xlabel("Synchronous Population Kindling Rate (Hz/unit, -100 to 0ms)", fontweight='bold', fontsize=10)
    ax1.set_ylabel("Peak Diaphragm EMG Amplitude (µV)", fontweight='bold', fontsize=10)
    ax1.set_title(f"A. Synchronous Population Kindling vs Diaphragm EMG Amplitude\n(Pearson r = {r_pop:.3f}, p = {p_pop:.2e})", fontweight='bold', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.4)

    # Panel B: Distribution of Pearson r Across Top 24 Units
    ax2 = fig_mpl.add_subplot(2, 2, 2)
    colors_bar = ['red' if p == 'Inspiratory' else '#1f77b4' for p in corr_df['phase']]
    ax2.bar(range(len(top24_uids)), corr_df['pearson_r'], color=colors_bar, alpha=0.85, edgecolor='black')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_xticks(range(len(top24_uids)))
    ax2.set_xticklabels([f"U{u}" for u in corr_df['cluster_id']], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel("Pearson Correlation Coefficient (r)", fontweight='bold', fontsize=10)
    ax2.set_title("B. Individual Neuron Spiking Correlation with EMG Amplitude", fontweight='bold', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.4)

    # Panel C: Highlight Unit 291 (Inspiratory) vs EMG Amplitude
    ax3 = fig_mpl.add_subplot(2, 2, 3)
    u291_cnts = np.array(unit_spikes_per_breath[291])
    r291, p291 = stats.pearsonr(u291_cnts, breath_amplitudes)
    ax3.scatter(u291_cnts, breath_amplitudes, color='red', alpha=0.7, s=40, edgecolor='black')
    sl291, ic291, _, _, _ = stats.linregress(u291_cnts, breath_amplitudes)
    x_fit291 = np.linspace(np.min(u291_cnts), np.max(u291_cnts), 100)
    ax3.plot(x_fit291, sl291 * x_fit291 + ic291, color='black', linewidth=2.5, label=f'Fit: r = {r291:.3f} (p = {p291:.2e})')
    ax3.set_xlabel("Unit 291 Spikes per Breath", fontweight='bold', fontsize=10)
    ax3.set_ylabel("Peak Diaphragm EMG Amplitude (µV)", fontweight='bold', fontsize=10)
    ax3.set_title(f"C. Inspiratory Unit 291 Spiking vs Diaphragm EMG Amplitude (r = {r291:.3f})", fontweight='bold', fontsize=12)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, linestyle=':', alpha=0.4)

    # Panel D: Highlight Unit 383 (Expiratory) vs EMG Amplitude
    ax4 = fig_mpl.add_subplot(2, 2, 4)
    u383_cnts = np.array(unit_spikes_per_breath[383])
    r383, p383 = stats.pearsonr(u383_cnts, breath_amplitudes)
    ax4.scatter(u383_cnts, breath_amplitudes, color='#1f77b4', alpha=0.7, s=40, edgecolor='black')
    sl383, ic383, _, _, _ = stats.linregress(u383_cnts, breath_amplitudes)
    x_fit383 = np.linspace(np.min(u383_cnts), np.max(u383_cnts), 100)
    ax4.plot(x_fit383, sl383 * x_fit383 + ic383, color='black', linewidth=2.5, label=f'Fit: r = {r383:.3f} (p = {p383:.2e})')
    ax4.set_xlabel("Unit 383 Spikes per Breath", fontweight='bold', fontsize=10)
    ax4.set_ylabel("Peak Diaphragm EMG Amplitude (µV)", fontweight='bold', fontsize=10)
    ax4.set_title(f"D. Expiratory Unit 383 Spiking vs Diaphragm EMG Amplitude (r = {r383:.3f})", fontweight='bold', fontsize=12)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, linestyle=':', alpha=0.4)

    plt.suptitle("Subject TP_Tacr1_1: Correlation of Population Kindling & Individual Spiking with Breath Amplitude", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    png_out = out_dir / "emg_amplitude_kindling_correlation_summary.png"
    plt.savefig(png_out, dpi=300)
    plt.close()

    # Copy PNG to artifact directory
    artifact_png = artifact_dir / "emg_amplitude_kindling_correlation_summary.png"
    shutil.copy2(png_out, artifact_png)
    print(f" Saved Diagnostic Correlation Figure: {png_out}")

    # 6. Interactive Plotly Dashboard
    fig_plotly = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"A. Synchronous Population Kindling vs EMG Amplitude (r = {r_pop:.3f})",
            "B. Individual Unit Correlation Coefficients (Pearson r)",
            f"C. Inspiratory Unit 291 Spikes vs EMG Amplitude (r = {r291:.3f})",
            f"D. Expiratory Unit 383 Spikes vs EMG Amplitude (r = {r383:.3f})"
        ]
    )

    fig_plotly.add_trace(go.Scatter(
        x=pop_kindling_rate_hz, y=breath_amplitudes, mode='markers',
        marker=dict(color='crimson', size=6, opacity=0.7), name="Population Kindling"
    ), row=1, col=1)
    fig_plotly.add_trace(go.Scatter(
        x=x_fit, y=y_fit, mode='lines', line=dict(color='black', width=2.5), name="Population Fit"
    ), row=1, col=1)

    fig_plotly.add_trace(go.Bar(
        x=[f"Unit {u}" for u in corr_df['cluster_id']],
        y=corr_df['pearson_r'],
        marker_color=['red' if p == 'Inspiratory' else '#1f77b4' for p in corr_df['phase']],
        name="Pearson r"
    ), row=1, col=2)

    fig_plotly.add_trace(go.Scatter(
        x=u291_cnts, y=breath_amplitudes, mode='markers',
        marker=dict(color='red', size=6, opacity=0.7), name="Unit 291"
    ), row=2, col=1)

    fig_plotly.add_trace(go.Scatter(
        x=u383_cnts, y=breath_amplitudes, mode='markers',
        marker=dict(color='#1f77b4', size=6, opacity=0.7), name="Unit 383"
    ), row=2, col=2)

    fig_plotly.update_layout(
        title="<b>Subject TP_Tacr1_1: Population Kindling & Spiking vs Breath Amplitude Dashboard</b>",
        template="plotly_white",
        height=850,
        showlegend=False
    )

    html_out = out_dir / "emg_amplitude_kindling_correlation_dashboard.html"
    fig_plotly.write_html(str(html_out))
    print(f" Saved Interactive HTML Dashboard: {html_out}")

    print("\n==========================================================================================")
    print(" [SUCCESS] Correlation Analysis Complete!")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
