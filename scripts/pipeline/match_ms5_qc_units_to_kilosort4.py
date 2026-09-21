#!/usr/bin/env python3
r"""
Cross-Sorter Unit Matching & Waveform Validation Analysis
Matches MountainSort5 Allen QC-Passed Single Units to Kilosort 4 Units
using Spike Time Coincidence (Jitter: 1.5 ms) and Extracted Action Potential Waveforms.

Dataset: LP21_fent_EXPT_g1 (First 300.0s AP Recording)
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import spikeinterface.full as si
import spikeinterface.extractors as se

def compute_spike_coincidences(spikes1, spikes2, delta_s=0.0015):
    """
    Computes bipartite spike coincidences between two spike trains with tolerance delta_s.
    Returns (n_coincident, agreement_jaccard, recall1, precision2)
    """
    if len(spikes1) == 0 or len(spikes2) == 0:
        return 0, 0.0, 0.0, 0.0
    
    # Fast searchsorted matching
    idx = np.searchsorted(spikes2, spikes1)
    n_coinc = 0
    used_spikes2 = set()
    
    for i, t in enumerate(spikes1):
        cand_idx = idx[i]
        best_cand = None
        min_dt = delta_s
        
        # Check adjacent candidates
        for c in (cand_idx - 1, cand_idx, cand_idx + 1):
            if 0 <= c < len(spikes2) and c not in used_spikes2:
                dt = abs(spikes2[c] - t)
                if dt < min_dt:
                    min_dt = dt
                    best_cand = c
                    
        if best_cand is not None:
            n_coinc += 1
            used_spikes2.add(best_cand)
            
    n1 = len(spikes1)
    n2 = len(spikes2)
    jaccard = n_coinc / (n1 + n2 - n_coinc) if (n1 + n2 - n_coinc) > 0 else 0.0
    recall1 = n_coinc / n1 if n1 > 0 else 0.0
    precision2 = n_coinc / n2 if n2 > 0 else 0.0
    
    return n_coinc, jaccard, recall1, precision2

def extract_unit_mean_waveform(rec, spike_frames, max_spikes=150, ms_before=1.0, ms_after=1.5):
    """
    Extracts mean action potential waveform across channels for a given unit.
    Returns (mean_waveform_2d, peak_channel_idx, time_axis_ms)
    """
    fs = rec.get_sampling_frequency()
    n_before = int(ms_before * fs / 1000.0)
    n_after = int(ms_after * fs / 1000.0)
    time_axis_ms = np.linspace(-ms_before, ms_after, n_before + n_after)
    
    total_samples = rec.get_num_samples()
    valid_frames = spike_frames[(spike_frames >= n_before) & (spike_frames < total_samples - n_after)]
    
    if len(valid_frames) == 0:
        return None, 0, time_axis_ms
        
    if len(valid_frames) > max_spikes:
        selected_frames = np.random.choice(valid_frames, size=max_spikes, replace=False)
    else:
        selected_frames = valid_frames
        
    snippets = []
    for f in selected_frames:
        chunk = rec.get_traces(start_frame=f - n_before, end_frame=f + n_after)
        snippets.append(chunk)
        
    snippets = np.array(snippets) # (n_spikes, n_samples, n_channels)
    mean_wf = np.mean(snippets, axis=0) # (n_samples, n_channels)
    
    # Peak-to-peak amplitude per channel
    ptp_per_chan = np.ptp(mean_wf, axis=0)
    peak_chan = int(np.argmax(ptp_per_chan))
    
    return mean_wf, peak_chan, time_axis_ms

def main():
    print("=" * 90)
    print(" MOUNTAINSORT 5 vs KILOSORT 4 UNIT MATCHING & WAVEFORM VALIDATION")
    print("=" * 90)
    
    base_out_dir = Path(r"c:\Users\lpascu\analysis_src\output_LP21_fent_EXPT_g1")
    scratch_dir = Path(r"C:\SCRATCH_lp21_sorting")
    rec_dir = scratch_dir / "preprocessed_recording"
    
    ms5_dir = base_out_dir / "mountainsort5" / "alf"
    ks4_dir = base_out_dir / "kilosort4" / "alf"
    
    # 1. Load Preprocessed Recording
    print(" Loading preprocessed recording from fast NVMe scratch...")
    rec = si.load(rec_dir)
    fs = rec.get_sampling_frequency()
    duration_s = rec.get_total_duration()
    print(f" Recording: {rec.get_num_channels()} channels, {fs:.1f} Hz, {duration_s:.1f} s")
    
    # 2. Load MountainSort5 Units
    ms5_times = np.load(ms5_dir / "spike_times.npy")
    ms5_clusters = np.load(ms5_dir / "spike_clusters.npy")
    ms5_metrics = pd.read_csv(base_out_dir / "mountainsort5" / "mountainsort5_qc_summary.csv")
    
    ms5_qc_passed = ms5_metrics[ms5_metrics['passed_allen_qc']].copy().reset_index(drop=True)
    print(f" MountainSort5: Total {len(ms5_metrics)} units | Allen QC Passed: {len(ms5_qc_passed)} units")
    
    # 3. Load Kilosort 4 Units
    ks4_times = np.load(ks4_dir / "spike_times.npy")
    ks4_clusters = np.load(ks4_dir / "spike_clusters.npy")
    ks4_metrics = pd.read_csv(base_out_dir / "kilosort4" / "kilosort4_qc_summary.csv")
    print(f" Kilosort 4: Total {len(ks4_metrics)} units | Allen QC Passed: {ks4_metrics['passed_allen_qc'].sum()} units")
    
    # Organize spikes per cluster
    ms5_spikes_dict = {}
    for uid in ms5_qc_passed['cluster_id']:
        ms5_spikes_dict[uid] = ms5_times[ms5_clusters == uid]
        
    ks4_spikes_dict = {}
    for uid in np.unique(ks4_clusters):
        ks4_spikes_dict[uid] = ks4_times[ks4_clusters == uid]
        
    # 4. Extract Mean Waveforms for MS5 QC units and KS4 units
    print("\n Extracting representative action potential waveforms...")
    np.random.seed(42)
    
    ms5_waveforms = {}
    ms5_peak_channels = {}
    for i, uid in enumerate(ms5_qc_passed['cluster_id']):
        spk_times = ms5_spikes_dict[uid]
        spk_frames = (spk_times * fs).astype(np.int64)
        wf, peak_ch, time_axis = extract_unit_mean_waveform(rec, spk_frames, max_spikes=120)
        ms5_waveforms[uid] = wf
        ms5_peak_channels[uid] = peak_ch
        
    ks4_waveforms = {}
    ks4_peak_channels = {}
    for uid in ks4_spikes_dict:
        spk_times = ks4_spikes_dict[uid]
        spk_frames = (spk_times * fs).astype(np.int64)
        wf, peak_ch, time_axis = extract_unit_mean_waveform(rec, spk_frames, max_spikes=120)
        ks4_waveforms[uid] = wf
        ks4_peak_channels[uid] = peak_ch
        
    # 5. Perform Spike Time Coincidence & Waveform Matching
    print(" Performing Cross-Sorter Coincidence Matching (Jitter Tolerance = 1.5 ms)...")
    DELTA_S = 0.0015 # 1.5 ms
    
    match_records = []
    
    for _, ms_row in ms5_qc_passed.iterrows():
        ms_id = int(ms_row['cluster_id'])
        ms_spks = ms5_spikes_dict[ms_id]
        ms_wf = ms5_waveforms[ms_id]
        ms_ch = ms5_peak_channels[ms_id]
        ms_n_spks = len(ms_spks)
        ms_fr = ms_row['firing_rate_hz']
        
        best_ks_id = None
        best_jaccard = 0.0
        best_coinc = 0
        best_recall = 0.0
        best_precision = 0.0
        best_wf_corr = 0.0
        best_chan_diff = 999
        
        # Test against all KS4 units
        for ks_id, ks_spks in ks4_spikes_dict.items():
            n_coinc, jaccard, recall, precision = compute_spike_coincidences(ms_spks, ks_spks, delta_s=DELTA_S)
            
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_ks_id = ks_id
                best_coinc = n_coinc
                best_recall = recall
                best_precision = precision
                
                # Compute waveform correlation on peak channel
                ks_wf = ks4_waveforms[ks_id]
                ks_ch = ks4_peak_channels[ks_id]
                best_chan_diff = abs(ms_ch - ks_ch)
                
                if ms_wf is not None and ks_wf is not None:
                    # Waveform correlation on MS5 peak channel
                    v_ms = ms_wf[:, ms_ch]
                    v_ks = ks_wf[:, ms_ch]
                    if np.std(v_ms) > 0 and np.std(v_ks) > 0:
                        r_val, _ = stats.pearsonr(v_ms, v_ks)
                        best_wf_corr = float(r_val)
                    else:
                        best_wf_corr = 0.0
                        
        # Match Classification
        if best_jaccard >= 0.50 and best_wf_corr >= 0.85 and best_chan_diff <= 8:
            match_class = "High-Confidence Match"
        elif (best_jaccard >= 0.30 or best_recall >= 0.60) and best_wf_corr >= 0.70 and best_chan_diff <= 16:
            match_class = "Moderate Match"
        elif best_jaccard >= 0.15 or (best_recall >= 0.40 and best_wf_corr >= 0.50):
            match_class = "Partial / Split Match"
        else:
            match_class = "Unmatched / Sorter-Unique"
            
        match_records.append({
            'ms5_cluster_id': ms_id,
            'ms5_firing_rate_hz': ms_fr,
            'ms5_n_spikes': ms_n_spks,
            'ms5_peak_channel': ms_ch,
            'best_ks4_cluster_id': best_ks_id if best_jaccard > 0 else -1,
            'best_ks4_peak_channel': ks4_peak_channels.get(best_ks_id, -1) if best_ks_id is not None else -1,
            'channel_difference': best_chan_diff if best_jaccard > 0 else -1,
            'n_coincident_spikes': best_coinc,
            'jaccard_agreement': round(best_jaccard, 4),
            'ms5_spike_recall': round(best_recall, 4),
            'ks4_spike_precision': round(best_precision, 4),
            'waveform_pearson_r': round(best_wf_corr, 4),
            'match_classification': match_class
        })
        
    df_matches = pd.DataFrame(match_records)
    out_csv = base_out_dir / "ms5_qc_vs_kilosort4_matches.csv"
    df_matches.to_csv(out_csv, index=False)
    print(f" Saved unit matching table to: {out_csv}")
    
    # 6. Summary Statistics
    counts = df_matches['match_classification'].value_counts()
    n_total_qc = len(df_matches)
    n_high = counts.get("High-Confidence Match", 0)
    n_mod = counts.get("Moderate Match", 0)
    n_part = counts.get("Partial / Split Match", 0)
    n_unm = counts.get("Unmatched / Sorter-Unique", 0)
    n_matched_any = n_high + n_mod + n_part
    
    print("\n" + "=" * 80)
    print(" MATCHING RESULTS SUMMARY (MountainSort5 QC-Passed Units vs Kilosort 4)")
    print("=" * 80)
    print(f" Total MS5 QC-Passed Units       : {n_total_qc}")
    print(f" -> High-Confidence Matches      : {n_high} ({n_high/n_total_qc*100:.1f}%)")
    print(f" -> Moderate Matches             : {n_mod} ({n_mod/n_total_qc*100:.1f}%)")
    print(f" -> Partial / Split Matches      : {n_part} ({n_part/n_total_qc*100:.1f}%)")
    print(f" -> Total Matched in KS4         : {n_matched_any} ({n_matched_any/n_total_qc*100:.1f}%)")
    print(f" -> Unmatched / MS5-Unique Units : {n_unm} ({n_unm/n_total_qc*100:.1f}%)")
    print(f" Mean Jaccard Agreement (Matched): {df_matches[df_matches['jaccard_agreement']>0]['jaccard_agreement'].mean():.3f}")
    print(f" Mean Waveform Correlation (r)   : {df_matches[df_matches['waveform_pearson_r']>0]['waveform_pearson_r'].mean():.3f}")
    print("=" * 80)
    
    # 7. Generate Comprehensive Publication-Grade Figure
    print(" Generating Comprehensive Diagnostic Matching Figure...")
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    fig.suptitle(
        f"Spike Sorting Unit Correspondence: MountainSort5 (QC Passed) vs Kilosort 4\n"
        f"LP21_fent_EXPT_g1 (First 5 min) | Spike Jitter Tolerance = ±1.5 ms",
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # Panel 1: Classification Pie Chart
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ['High-Confidence', 'Moderate', 'Partial/Split', 'Unmatched']
    vals = [n_high, n_mod, n_part, n_unm]
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    non_zero = [(l, v, c) for l, v, c in zip(labels, vals, colors) if v > 0]
    ax1.pie(
        [x[1] for x in non_zero], 
        labels=[f"{x[0]}\n({x[1]}, {x[1]/n_total_qc*100:.1f}%)" for x in non_zero],
        colors=[x[2] for x in non_zero],
        autopct='%1.0f%%',
        startangle=140,
        textprops={'fontsize': 10, 'fontweight': 'bold'}
    )
    ax1.set_title("A. Match Classification", fontsize=12, fontweight='bold')
    
    # Panel 2: Distribution of Jaccard Agreement Scores
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df_matches['jaccard_agreement'], bins=25, color='#3498db', alpha=0.8, edgecolor='black')
    ax2.axvline(0.5, color='#2ecc71', linestyle='--', linewidth=2, label='High-Conf (0.5)')
    ax2.set_xlabel("Spike Agreement (Jaccard Score)", fontsize=11)
    ax2.set_ylabel("Unit Count", fontsize=11)
    ax2.set_title("B. Spike Time Agreement", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Panel 3: Distribution of Waveform Pearson Correlations
    ax3 = fig.add_subplot(gs[0, 2])
    valid_wf = df_matches[df_matches['waveform_pearson_r'] > 0]['waveform_pearson_r']
    ax3.hist(valid_wf, bins=25, color='#9b59b6', alpha=0.8, edgecolor='black')
    ax3.axvline(0.85, color='#2ecc71', linestyle='--', linewidth=2, label='High-Conf (0.85)')
    ax3.set_xlabel("Waveform Pearson Correlation (r)", fontsize=11)
    ax3.set_ylabel("Unit Count", fontsize=11)
    ax3.set_title("C. Action Potential Similarity", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    # Panel 4: Scatter Plot of Agreement vs Waveform Correlation
    ax4 = fig.add_subplot(gs[0, 3])
    scatter_colors = {
        'High-Confidence Match': '#2ecc71',
        'Moderate Match': '#3498db',
        'Partial / Split Match': '#f39c12',
        'Unmatched / Sorter-Unique': '#e74c3c'
    }
    for cat, group in df_matches.groupby('match_classification'):
        ax4.scatter(
            group['jaccard_agreement'], group['waveform_pearson_r'],
            color=scatter_colors[cat], label=cat, s=40, alpha=0.85, edgecolors='none'
        )
    ax4.set_xlabel("Spike Agreement (Jaccard)", fontsize=11)
    ax4.set_ylabel("Waveform Pearson Correlation (r)", fontsize=11)
    ax4.set_title("D. Agreement vs Waveform Fit", fontsize=12, fontweight='bold')
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.1, 1.05)
    ax4.legend(fontsize=8, loc='lower right')
    ax4.grid(True, linestyle=':', alpha=0.6)
    
    # Panels 5-8: Example Overlaid Waveforms for Top 4 High-Confidence Matched Units
    top_matches = df_matches.sort_values(by=['jaccard_agreement', 'waveform_pearson_r'], ascending=False).head(4).reset_index(drop=True)
    
    for plot_i, row in top_matches.iterrows():
        ms_id = int(row['ms5_cluster_id'])
        ks_id = int(row['best_ks4_cluster_id'])
        ch = int(row['ms5_peak_channel'])
        jacc = row['jaccard_agreement']
        r_val = row['waveform_pearson_r']
        
        ax = fig.add_subplot(gs[1, plot_i])
        
        wf_ms = ms5_waveforms[ms_id][:, ch]
        wf_ks = ks4_waveforms[ks_id][:, ch]
        
        ax.plot(time_axis, wf_ms, color='#3498db', linewidth=2.5, label=f'MS5 Unit {ms_id}')
        ax.plot(time_axis, wf_ks, color='#e74c3c', linewidth=2.0, linestyle='--', label=f'KS4 Unit {ks_id}')
        ax.set_xlabel("Time (ms)", fontsize=10)
        ax.set_ylabel("Voltage (µV)", fontsize=10)
        ax.set_title(f"E{plot_i+1}. MS5 {ms_id} ↔ KS4 {ks_id}\n(J={jacc:.2f}, r={r_val:.2f}, Ch {ch})", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)
        
    # Panels 9-12: High-Resolution Spike Raster Overlay (5-second zoomed window: 10.0s - 15.0s)
    ZOOM_T0 = 10.0
    ZOOM_T1 = 15.0
    
    for plot_i, row in top_matches.iterrows():
        ms_id = int(row['ms5_cluster_id'])
        ks_id = int(row['best_ks4_cluster_id'])
        
        ax = fig.add_subplot(gs[2, plot_i])
        
        spk_ms = ms5_spikes_dict[ms_id]
        spk_ks = ks4_spikes_dict[ks_id]
        
        zoom_ms = spk_ms[(spk_ms >= ZOOM_T0) & (spk_ms <= ZOOM_T1)]
        zoom_ks = spk_ks[(spk_ks >= ZOOM_T0) & (spk_ks <= ZOOM_T1)]
        
        # Plot spike ticks
        for t in zoom_ms:
            ax.vlines(t, 0.6, 1.4, color='#3498db', linewidth=1.8)
        for t in zoom_ks:
            ax.vlines(t, 1.6, 2.4, color='#e74c3c', linewidth=1.8)
            
        ax.set_yticks([1.0, 2.0])
        ax.set_yticklabels([f'MS5 ({len(zoom_ms)})', f'KS4 ({len(zoom_ks)})'], fontsize=10, fontweight='bold')
        ax.set_xlim(ZOOM_T0, ZOOM_T1)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_title(f"F{plot_i+1}. Spike Overlap (10-15s)", fontsize=11, fontweight='bold')
        ax.grid(axis='x', linestyle=':', alpha=0.6)
        
    out_fig = base_out_dir / "ms5_vs_ks4_unit_matching_dashboard.png"
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved matching diagnostic figure to: {out_fig}")

if __name__ == '__main__':
    main()
