#!/usr/bin/env python3
r"""
Comparative Raster & PSTH Analysis: Top 10 MountainSort5 Phasic Units vs Matched Kilosort Units
(Bush et al. 2025 & Mazurek Vector Strength Framework)

1. Calculates Respiratory Modulation (Mazurek Vector Strength R) for all MountainSort5 Allen QC passed units.
2. Selects Top 10 most phasic MountainSort5 single units.
3. Maps each Top 10 MountainSort5 unit to its corresponding Kilosort unit by spatial depth and spike coincidences.
4. Generates side-by-side comparative spike raster plots (0-60.0s baseline) aligned to diaphragm EMG breath onsets.

Outputs:
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_mountainsort5\top10_phasic_ms5_vs_kilosort_raster.png
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_mountainsort5\top10_phasic_ms5_vs_kilosort_metrics.csv
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_mountainsort5\top10_phasic_ms5_vs_kilosort_dashboard.html
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

def compute_mazurek_vector_strength(spikes_u, onsets):
    """Computes Mazurek Vector Strength (R) and Preferred Respiratory Phase for a spike train"""
    if len(spikes_u) < 10 or len(onsets) < 3:
        return 0.0, 0.0

    phases = []
    for i in range(len(onsets) - 1):
        t_start = onsets[i]
        t_end = onsets[i + 1]
        t_dur = t_end - t_start
        if t_dur <= 0:
            continue
        
        spks_in_cycle = spikes_u[(spikes_u >= t_start) & (spikes_u < t_end)]
        if len(spks_in_cycle) > 0:
            cycle_phases = 2.0 * np.pi * (spks_in_cycle - t_start) / t_dur - np.pi
            phases.extend(cycle_phases)

    if len(phases) < 5:
        return 0.0, 0.0

    phases = np.array(phases)
    cos_sum = np.sum(np.cos(phases))
    sin_sum = np.sum(np.sin(phases))
    
    R = float(np.sqrt(cos_sum**2 + sin_sum**2) / len(phases))
    pref_phase = float(np.arctan2(sin_sum, cos_sum))
    return round(R, 4), round(pref_phase, 4)

def main():
    ms5_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_mountainsort5")
    ks_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1\alf_session\alf\probe00")
    raw_dir = Path(r"Z:\ramirezlab\pascual\rec_data\TP_Tacr1_1_npx2_g4")
    artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\0e7be222-f160-4bc2-921c-f3fdf8b7825e")

    print("=" * 90)
    print(" Running Top 10 Phasic Raster Comparison: MountainSort5 vs Kilosort")
    print(f" MS5 Directory: {ms5_dir}")
    print(f" KS Directory : {ks_dir}")
    print("=" * 90)

    # 1. Load MS5 ALF Data & QC Metrics
    ms5_times = np.load(ms5_dir / "alf" / "spike_times.npy")
    ms5_clusters = np.load(ms5_dir / "alf" / "spike_clusters.npy")
    ms5_info = pd.read_csv(ms5_dir / "mountainsort5_allen_qc_summary.csv")

    ms5_passed = ms5_info[ms5_info['passed_allen_qc']].copy().reset_index(drop=True)
    ms5_passed_uids = list(ms5_passed['cluster_id'])

    # 2. Load KS ALF Data & QC Metrics
    ks_times = np.load(ks_dir / "spike_times.npy")
    ks_clusters = np.load(ks_dir / "spike_clusters.npy")
    ks_info = pd.read_csv(ks_dir / "cluster_info.tsv", sep="\t")

    fs_ap = 30000.0
    if ks_times.dtype in [np.int64, np.int32] and ks_times.max() > 100000:
        ks_times = ks_times / fs_ap

    # 3. Load NIDQ Diaphragm EMG Signal (0 - 60.0s Baseline)
    nidq_meta = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.meta"
    nidq_bin = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.bin"
    nidq_data = read_nidq_bin(nidq_bin, nidq_meta)

    t_nidq = nidq_data.time_vector
    fs_nidq = nidq_data.fs

    MAX_TIME_S = 60.0
    nidq_mask = (t_nidq >= 0.0) & (t_nidq <= MAX_TIME_S)
    t_nidq = t_nidq[nidq_mask]
    emg_raw = nidq_data.all_channels[0][nidq_mask]

    emg_cfg = EMGConfig(fs=fs_nidq, highpass_cutoff=300.0, lowpass_cutoff=3000.0, notch_freq=60.0, smoothing_window_ms=20.0)
    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    emg_env = emg_proc.envelope
    breath_onsets = emg_proc.breath_onsets
    baseline_onsets = [o for o in breath_onsets if 0.5 <= o <= MAX_TIME_S - 0.5]

    print(f" Analyzed {len(baseline_onsets)} baseline breath cycles in the first 60s.")

    # 4. Compute Vector Strength for MS5 Allen QC Passed Units
    ms5_phasic_records = []
    for uid in ms5_passed_uids:
        u_spks = ms5_times[ms5_clusters == uid]
        u_spks_b = u_spks[(u_spks >= 0.0) & (u_spks <= MAX_TIME_S)]
        
        R, pref_phase = compute_mazurek_vector_strength(u_spks_b, baseline_onsets)
        row = ms5_passed[ms5_passed['cluster_id'] == uid].iloc[0]
        
        ms5_phasic_records.append({
            'ms5_cluster_id': int(uid),
            'depth': float(row['depth']),
            'firing_rate_60s': float(row['firing_rate']),
            'vector_strength': R,
            'pref_phase': pref_phase,
            'snr': float(row['snr'])
        })

    ms5_phasic_df = pd.DataFrame(ms5_phasic_records)
    ms5_phasic_df = ms5_phasic_df.sort_values(by='vector_strength', ascending=False).reset_index(drop=True)

    # Select Top 10 Most Phasic MS5 Units
    top10_ms5 = ms5_phasic_df.head(10).copy()

    # 5. Map each Top 10 MS5 Unit to the best matching Kilosort Unit
    matched_records = []
    for _, ms_row in top10_ms5.iterrows():
        ms_uid = int(ms_row['ms5_cluster_id'])
        ms_spks = ms5_times[ms5_clusters == ms_uid]
        ms_spks_b = ms_spks[(ms_spks >= 0.0) & (ms_spks <= MAX_TIME_S)]

        # Find KS unit with maximum spike coincidences (within 1.5ms)
        best_ks_uid = None
        best_coincidences = -1
        best_ks_spks_count = 0

        unique_ks_uids = ks_info['cluster_id'].unique()
        for ks_uid in unique_ks_uids:
            ks_spks = ks_times[ks_clusters == ks_uid]
            ks_spks_b = ks_spks[(ks_spks >= 0.0) & (ks_spks <= MAX_TIME_S)]
            if len(ks_spks_b) == 0:
                continue

            # Compute spike coincidences within 1.5 ms window
            idx_near = np.searchsorted(ks_spks_b, ms_spks_b)
            idx_near = np.clip(idx_near, 0, len(ks_spks_b) - 1)
            diffs = np.abs(ks_spks_b[idx_near] - ms_spks_b)
            coincidences = np.sum(diffs <= 0.0015)

            if coincidences > best_coincidences:
                best_coincidences = coincidences
                best_ks_uid = int(ks_uid)
                best_ks_spks_count = len(ks_spks_b)

        matched_records.append({
            'ms5_unit_id': ms_uid,
            'ms5_depth': ms_row['depth'],
            'ms5_vector_strength': ms_row['vector_strength'],
            'ms5_spikes_60s': len(ms_spks_b),
            'matched_ks_unit_id': best_ks_uid,
            'matched_ks_spikes_60s': best_ks_spks_count,
            'spike_coincidences_1.5ms': best_coincidences
        })

    matched_df = pd.DataFrame(matched_records)
    csv_out = ms5_dir / "top10_phasic_ms5_vs_kilosort_metrics.csv"
    matched_df.to_csv(csv_out, index=False)
    print(f" Saved Matched Units Metrics CSV: {csv_out}")
    print(matched_df.to_string())

    # 6. High-Resolution Diagnostic PNG Figure (Side-by-Side Raster Comparison)
    fig_mpl, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True, gridspec_kw={'height_ratios': [1, 2.5, 2.5]})

    # Subplot 1: Diaphragm EMG Signal (0-60s)
    axes[0].plot(t_nidq, emg_env, color='#00d2ff', linewidth=1.2, label='Diaphragm EMG Envelope')
    for o in baseline_onsets:
        axes[0].axvline(o, color='red', linestyle=':', alpha=0.5)
    axes[0].set_ylabel("EMG (µV)", fontweight='bold', fontsize=10)
    axes[0].set_title("A. Diaphragm EMG Activity & Breath Onsets (0 - 60.0s Baseline)", fontweight='bold', fontsize=11)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, linestyle=':', alpha=0.4)

    # Subplot 2: Top 10 MountainSort5 Phasic Units Spike Raster
    for idx, row in matched_df.iterrows():
        ms_uid = int(row['ms5_unit_id'])
        u_spks = ms5_times[ms5_clusters == ms_uid]
        u_spks_b = u_spks[(u_spks >= 0.0) & (u_spks <= MAX_TIME_S)]
        
        y_pos = idx + 1
        axes[1].vlines(u_spks_b, y_pos - 0.35, y_pos + 0.35, color='#d62728', linewidth=1.2)

    for o in baseline_onsets:
        axes[1].axvline(o, color='red', linestyle=':', alpha=0.5)

    axes[1].set_yticks(range(1, 11))
    axes[1].set_yticklabels([f"MS5 U{r['ms5_unit_id']} (R={r['ms5_vector_strength']})" for _, r in matched_df.iterrows()], fontsize=8, fontweight='bold')
    axes[1].set_ylabel("MountainSort5 Units", fontweight='bold', fontsize=10)
    axes[1].set_title("B. Top 10 MountainSort5 Phasic Single Units Spike Raster (0 - 60.0s)", fontweight='bold', fontsize=11)
    axes[1].grid(True, linestyle=':', alpha=0.4)

    # Subplot 3: Matched Kilosort Units Spike Raster
    for idx, row in matched_df.iterrows():
        ks_uid = int(row['matched_ks_unit_id'])
        u_spks = ks_times[ks_clusters == ks_uid]
        u_spks_b = u_spks[(u_spks >= 0.0) & (u_spks <= MAX_TIME_S)]
        
        y_pos = idx + 1
        axes[2].vlines(u_spks_b, y_pos - 0.35, y_pos + 0.35, color='#1f77b4', linewidth=1.2)

    for o in baseline_onsets:
        axes[2].axvline(o, color='red', linestyle=':', alpha=0.5)

    axes[2].set_yticks(range(1, 11))
    axes[2].set_yticklabels([f"KS U{r['matched_ks_unit_id']} (Matched)" for _, r in matched_df.iterrows()], fontsize=8, fontweight='bold')
    axes[2].set_xlabel("Session Time (s)", fontweight='bold', fontsize=10)
    axes[2].set_ylabel("Kilosort Matched Units", fontweight='bold', fontsize=10)
    axes[2].set_title("C. Matched Kilosort Single Units Spike Raster (0 - 60.0s)", fontweight='bold', fontsize=11)
    axes[2].grid(True, linestyle=':', alpha=0.4)

    plt.suptitle("Subject TP_Tacr1_1: Top 10 MountainSort5 Phasic Units vs Matched Kilosort Units", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    png_out = ms5_dir / "top10_phasic_ms5_vs_kilosort_raster.png"
    plt.savefig(png_out, dpi=300)
    plt.close()

    # Copy PNG to artifact directory
    artifact_png = artifact_dir / "top10_phasic_ms5_vs_kilosort_raster.png"
    shutil.copy2(png_out, artifact_png)
    print(f" Saved Comparison Raster Figure: {png_out}")

    # 7. Interactive Plotly Dashboard
    fig_plotly = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            "A. Diaphragm EMG Envelope (0 - 60.0s Baseline)",
            "B. Top 10 MountainSort5 Phasic Single Units Spike Raster",
            "C. Matched Kilosort Single Units Spike Raster"
        ]
    )

    # Subplot 1: EMG
    fig_plotly.add_trace(go.Scatter(
        x=t_nidq[::10], y=emg_env[::10], mode='lines', line=dict(color='#00d2ff', width=1.2), name="Diaphragm EMG"
    ), row=1, col=1)

    # Subplot 2: MS5 Units
    for idx, row in matched_df.iterrows():
        ms_uid = int(row['ms5_unit_id'])
        u_spks = ms5_times[ms5_clusters == ms_uid]
        u_spks_b = u_spks[(u_spks >= 0.0) & (u_spks <= MAX_TIME_S)]
        fig_plotly.add_trace(go.Scatter(
            x=u_spks_b, y=np.full(len(u_spks_b), idx + 1), mode='markers',
            marker=dict(symbol='line-ns-open', color='#d62728', size=6, line=dict(width=1.2)),
            name=f"MS5 U{ms_uid}"
        ), row=2, col=1)

    # Subplot 3: KS Units
    for idx, row in matched_df.iterrows():
        ks_uid = int(row['matched_ks_unit_id'])
        u_spks = ks_times[ks_clusters == ks_uid]
        u_spks_b = u_spks[(u_spks >= 0.0) & (u_spks <= MAX_TIME_S)]
        fig_plotly.add_trace(go.Scatter(
            x=u_spks_b, y=np.full(len(u_spks_b), idx + 1), mode='markers',
            marker=dict(symbol='line-ns-open', color='#1f77b4', size=6, line=dict(width=1.2)),
            name=f"KS U{ks_uid}"
        ), row=3, col=1)

    fig_plotly.update_layout(
        title="<b>Subject TP_Tacr1_1: Top 10 MountainSort5 Phasic Units vs Kilosort Interactive Raster</b>",
        template="plotly_white",
        height=850,
        showlegend=False
    )

    html_out = ms5_dir / "top10_phasic_ms5_vs_kilosort_dashboard.html"
    fig_plotly.write_html(str(html_out))
    print(f" Saved Interactive HTML Dashboard: {html_out}")

    print("\n==========================================================================================")
    print(" [SUCCESS] MountainSort5 vs Kilosort Phasic Raster Comparison Complete!")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
