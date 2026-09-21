#!/usr/bin/env python3
r"""
Multi-Epoch Allen Neuropixels Single-Unit Quality Control & Persistent Unit Tracking
(Siegle et al., 2021 Nature & Bush et al., 2025 Framework)

Evaluates Allen QC criteria for every putative Kilosort cluster across 4 temporal epochs:
1. Full Session: 0.0 to 550.0 s
2. Epoch 1 (Baseline): 0.0 to 60.0 s
3. Epoch 2 (Phototagging Window): 60.0 to 300.0 s
4. Epoch 3 (Post-Phototagging / Recovery): 300.0 to 550.0 s

Identifies PERSISTENT SINGLE UNITS:
- Units that pass ALL Allen QC criteria across Epoch 1 AND Epoch 2 AND Epoch 3.

Outputs:
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_qc_epochs\allen_qc_multi_epoch_summary.csv
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_qc_epochs\persistent_qc_units_all_epochs.csv
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_qc_epochs\allen_qc_multi_epoch_summary.png
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_qc_epochs\allen_qc_multi_epoch_dashboard.html
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

def compute_isi_violations(spike_times_s, duration_s, isi_threshold_s=0.0015, min_isi_s=0.0000):
    """Computes Allen Institute Refractory Period ISI Violation Ratio"""
    n_spikes = len(spike_times_s)
    if n_spikes < 2 or duration_s <= 0:
        return 0.0, 0
    
    isis = np.diff(spike_times_s)
    violations = np.sum((isis >= min_isi_s) & (isis < isi_threshold_s))
    
    # Formula: (violations * T) / (2 * tau_ref * N^2)
    violation_rate = (violations * duration_s) / (2.0 * isi_threshold_s * (n_spikes ** 2))
    return float(violation_rate), int(violations)

def compute_presence_ratio(spike_times_s, start_time_s, end_time_s, n_bins=100):
    """Computes Presence Ratio across n_bins equal time blocks"""
    n_spikes = len(spike_times_s)
    if n_spikes == 0 or end_time_s <= start_time_s:
        return 0.0
    
    bins = np.linspace(start_time_s, end_time_s, n_bins + 1)
    counts, _ = np.histogram(spike_times_s, bins=bins)
    present_bins = np.sum(counts > 0)
    return float(present_bins / n_bins)

def compute_amplitude_cutoff(amplitudes_uv, n_bins=100):
    """Computes Amplitude Cutoff (estimated fraction of missing low-amplitude spikes)"""
    if len(amplitudes_uv) < 20:
        return 0.0
    
    hist, bin_edges = np.histogram(amplitudes_uv, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    peak_idx = np.argmax(hist)
    
    # Fit Gaussian to right tail
    try:
        peak_val = hist[peak_idx]
        tail_mask = bin_centers >= bin_centers[peak_idx]
        if np.sum(tail_mask) < 4:
            return 0.0
        
        # Estimate std from right half-width at half-max
        half_max_idx = np.where(hist[peak_idx:] <= peak_val / 2.0)[0]
        if len(half_max_idx) > 0:
            sigma = (bin_centers[peak_idx + half_max_idx[0]] - bin_centers[peak_idx]) / 1.177
        else:
            sigma = np.std(amplitudes_uv)
        
        if sigma <= 0:
            return 0.0
        
        norm_dist = stats.norm(loc=bin_centers[peak_idx], scale=sigma)
        cutoff_val = bin_edges[0]
        missing_fraction = norm_dist.cdf(cutoff_val)
        return float(np.clip(missing_fraction, 0.0, 1.0))
    except Exception:
        return 0.0

def evaluate_epoch_qc(u_spks, u_amps, start_t, end_t):
    """Evaluates Allen QC criteria for a single temporal epoch"""
    dur_t = end_t - start_t
    spk_mask = (u_spks >= start_t) & (u_spks <= end_t)
    e_spks = u_spks[spk_mask]
    e_amps = u_amps[spk_mask] if len(u_amps) == len(u_spks) else np.array([])

    n_spks = len(e_spks)
    fr = n_spks / dur_t if dur_t > 0 else 0.0
    isi_viol, n_viol = compute_isi_violations(e_spks, dur_t)
    pres_ratio = compute_presence_ratio(e_spks, start_t, end_t)
    amp_cutoff = compute_amplitude_cutoff(e_amps) if len(e_amps) > 0 else 0.0

    # Allen QC Criteria:
    # 1. ISI Violation Ratio <= 0.50
    # 2. Presence Ratio >= 0.90
    # 3. Amplitude Cutoff <= 0.10
    # 4. Firing Rate >= 0.10 Hz
    passed = (isi_viol <= 0.50) and (pres_ratio >= 0.90) and (amp_cutoff <= 0.10) and (fr >= 0.10)

    return {
        'n_spikes': n_spks,
        'firing_rate': round(fr, 3),
        'isi_violations': round(isi_viol, 4),
        'n_isi_violations': n_viol,
        'presence_ratio': round(pres_ratio, 3),
        'amplitude_cutoff': round(amp_cutoff, 4),
        'passed_qc': bool(passed)
    }

def main():
    out_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_qc_epochs")
    out_dir.mkdir(parents=True, exist_ok=True)
    session_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1\alf_session")
    alf_dir = session_dir / "alf" / "probe00"
    artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\0e7be222-f160-4bc2-921c-f3fdf8b7825e")

    print("=" * 90)
    print(" Running Multi-Epoch Allen Neuropixels Single-Unit QC & Persistent Unit Tracking")
    print(f" ALF Directory: {alf_dir}")
    print(f" Output Directory: {out_dir}")
    print("=" * 90)

    # 1. Load ALF Spikes, Clusters, and Info
    spikes_times = np.load(alf_dir / "spike_times.npy")
    spikes_clusters = np.load(alf_dir / "spike_clusters.npy")
    info_df = pd.read_csv(alf_dir / "cluster_info.tsv", sep="\t")

    spikes_amps = np.load(alf_dir / "spike_amps.npy") if (alf_dir / "spike_amps.npy").exists() else np.array([])

    fs_ap = 30000.0
    if spikes_times.dtype in [np.int64, np.int32] and spikes_times.max() > 100000:
        spikes_times = spikes_times / fs_ap

    unique_uids = np.sort(info_df['cluster_id'].unique())
    n_total_putative = len(unique_uids)
    print(f" Found {n_total_putative} total putative clusters in Kilosort output.")

    # Epoch Windows
    EPOCHS = {
        'full_session': (0.0, 550.0),
        'baseline_60s': (0.0, 60.0),
        'phototagging_240s': (60.0, 300.0),
        'post_phototagging_250s': (300.0, 550.0)
    }

    # 2. Evaluate Allen QC across Epochs
    records = []

    for uid in unique_uids:
        u_info = info_df[info_df['cluster_id'] == uid].iloc[0]
        shank = int(u_info['shank']) if 'shank' in u_info else 0
        depth = float(u_info['depth']) if 'depth' in u_info else 0.0
        snr = float(u_info['snr']) if 'snr' in u_info else 0.0
        amp_uv = float(u_info['amp']) if 'amp' in u_info else 0.0

        u_mask = (spikes_clusters == uid)
        u_spks = spikes_times[u_mask]
        u_amps = spikes_amps[u_mask] if len(spikes_amps) == len(spikes_times) else np.array([])

        qc_full = evaluate_epoch_qc(u_spks, u_amps, 0.0, 550.0)
        qc_ep1 = evaluate_epoch_qc(u_spks, u_amps, 0.0, 60.0)
        qc_ep2 = evaluate_epoch_qc(u_spks, u_amps, 60.0, 300.0)
        qc_ep3 = evaluate_epoch_qc(u_spks, u_amps, 300.0, 550.0)

        # Persistent Unit: Must pass QC in Epoch 1 AND Epoch 2 AND Epoch 3
        is_persistent = qc_ep1['passed_qc'] and qc_ep2['passed_qc'] and qc_ep3['passed_qc']

        rec = {
            'cluster_id': int(uid),
            'shank': shank,
            'depth': depth,
            'peak_amp_uv': round(amp_uv, 1),
            'snr': round(snr, 2),
            # Full Session
            'full_fr': qc_full['firing_rate'],
            'full_isi_viol': qc_full['isi_violations'],
            'full_presence': qc_full['presence_ratio'],
            'full_amp_cutoff': qc_full['amplitude_cutoff'],
            'full_passed': qc_full['passed_qc'],
            # Epoch 1 (Baseline 0-60s)
            'ep1_fr': qc_ep1['firing_rate'],
            'ep1_isi_viol': qc_ep1['isi_violations'],
            'ep1_presence': qc_ep1['presence_ratio'],
            'ep1_amp_cutoff': qc_ep1['amplitude_cutoff'],
            'ep1_passed': qc_ep1['passed_qc'],
            # Epoch 2 (Phototagging 60-300s)
            'ep2_fr': qc_ep2['firing_rate'],
            'ep2_isi_viol': qc_ep2['isi_violations'],
            'ep2_presence': qc_ep2['presence_ratio'],
            'ep2_amp_cutoff': qc_ep2['amplitude_cutoff'],
            'ep2_passed': qc_ep2['passed_qc'],
            # Epoch 3 (Post-Phototagging 300-550s)
            'ep3_fr': qc_ep3['firing_rate'],
            'ep3_isi_viol': qc_ep3['isi_violations'],
            'ep3_presence': qc_ep3['presence_ratio'],
            'ep3_amp_cutoff': qc_ep3['amplitude_cutoff'],
            'ep3_passed': qc_ep3['passed_qc'],
            # Persistent Unit Flag
            'is_persistent_across_all_epochs': is_persistent
        }
        records.append(rec)

    qc_df = pd.DataFrame(records)

    # Save Full Multi-Epoch Summary CSV
    csv_full = out_dir / "allen_qc_multi_epoch_summary.csv"
    qc_df.to_csv(csv_full, index=False)
    print(f" Saved Multi-Epoch Allen QC CSV: {csv_full}")

    # Persistent Units (Passing ALL 3 Epochs)
    persistent_df = qc_df[qc_df['is_persistent_across_all_epochs']].copy().reset_index(drop=True)
    csv_pers = out_dir / "persistent_qc_units_all_epochs.csv"
    persistent_df.to_csv(csv_pers, index=False)
    print(f" Saved Persistent Single Units CSV: {csv_pers}")

    n_full_passed = int(qc_df['full_passed'].sum())
    n_ep1_passed = int(qc_df['ep1_passed'].sum())
    n_ep2_passed = int(qc_df['ep2_passed'].sum())
    n_ep3_passed = int(qc_df['ep3_passed'].sum())
    n_persistent = len(persistent_df)

    print("\n" + "=" * 90)
    print(" ALLEN NEUROPIXELS SINGLE-UNIT QC AUDIT SUMMARY")
    print("-" * 90)
    print(f" Total Putative Kilosort Clusters     : {n_total_putative}")
    print(f" Full Session Passed Units (0-550s)   : {n_full_passed} ({n_full_passed/n_total_putative*100:.1f}%)")
    print(f" Epoch 1 Baseline Passed Units (0-60s): {n_ep1_passed} ({n_ep1_passed/n_total_putative*100:.1f}%)")
    print(f" Epoch 2 Phototagging Passed Units    : {n_ep2_passed} ({n_ep2_passed/n_total_putative*100:.1f}%)")
    print(f" Epoch 3 Post-Phototagging Passed     : {n_ep3_passed} ({n_ep3_passed/n_total_putative*100:.1f}%)")
    print(f" PERSISTENT UNITS (Passed ALL Epochs) : {n_persistent} ({n_persistent/n_total_putative*100:.1f}%)")
    print("=" * 90 + "\n")

    # 3. Diagnostic Figure (4 Panels)
    fig_mpl = plt.figure(figsize=(20, 12))

    # Panel A: Passed Unit Counts across Epochs
    ax1 = fig_mpl.add_subplot(2, 2, 1)
    epochs_labels = ['Full Session\n(0-550s)', 'Epoch 1 Baseline\n(0-60s)', 'Epoch 2 Laser\n(60-300s)', 'Epoch 3 Post-Laser\n(300-550s)', 'PERSISTENT\n(All 3 Epochs)']
    counts_bar = [n_full_passed, n_ep1_passed, n_ep2_passed, n_ep3_passed, n_persistent]
    colors_bar = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728']

    bars = ax1.bar(epochs_labels, counts_bar, color=colors_bar, alpha=0.85, edgecolor='black')
    for bar, cnt in zip(bars, counts_bar):
        ax1.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 2, f"{cnt}", ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax1.set_ylabel("Passed Single Unit Count", fontweight='bold', fontsize=10)
    ax1.set_title("A. Allen QC Passed Units Across Recording Epochs", fontweight='bold', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.4)

    # Panel B: ISI Violation Ratio Distribution for Persistent Units
    ax2 = fig_mpl.add_subplot(2, 2, 2)
    ax2.hist(persistent_df['full_isi_viol'], bins=20, color='#2ca02c', alpha=0.75, edgecolor='black')
    ax2.axvline(0.50, color='red', linestyle='--', linewidth=2, label='Allen Threshold (≤ 0.50)')
    ax2.set_xlabel("ISI Violation Ratio (Refractory Contamination)", fontweight='bold', fontsize=10)
    ax2.set_ylabel("Persistent Unit Count", fontweight='bold', fontsize=10)
    ax2.set_title(f"B. Refractory ISI Violation Ratio Distribution (N = {n_persistent})", fontweight='bold', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.4)

    # Panel C: Depth Distribution of Persistent vs Non-Persistent Units
    ax3 = fig_mpl.add_subplot(2, 2, 3)
    ax3.scatter(qc_df['depth'], qc_df['peak_amp_uv'], c=['red' if p else 'gray' for p in qc_df['is_persistent_across_all_epochs']], alpha=0.7, s=40, edgecolor='black', linewidth=0.3)
    ax3.set_xlabel("Probe Depth (µm from tip)", fontweight='bold', fontsize=10)
    ax3.set_ylabel("Peak-to-Peak Waveform Amplitude (µV)", fontweight='bold', fontsize=10)
    ax3.set_title("C. Persistent Single Units Spatial Distribution & Waveform Amplitude", fontweight='bold', fontsize=12)
    ax3.grid(True, linestyle=':', alpha=0.4)

    # Panel D: Firing Rate Stability across Epochs for Persistent Units
    ax4 = fig_mpl.add_subplot(2, 2, 4)
    for _, r in persistent_df.iterrows():
        ax4.plot([1, 2, 3], [r['ep1_fr'], r['ep2_fr'], r['ep3_fr']], color='gray', alpha=0.4, linewidth=1)
    ax4.plot([1, 2, 3], [persistent_df['ep1_fr'].median(), persistent_df['ep2_fr'].median(), persistent_df['ep3_fr'].median()], color='crimson', linewidth=3, marker='o', label='Median Firing Rate')
    ax4.set_xticks([1, 2, 3])
    ax4.set_xticklabels(['Baseline (0-60s)', 'Laser (60-300s)', 'Post-Laser (300-550s)'], fontweight='bold', fontsize=9)
    ax4.set_ylabel("Firing Rate (Hz)", fontweight='bold', fontsize=10)
    ax4.set_title("D. Firing Rate Stability Across Epochs (Persistent Units)", fontweight='bold', fontsize=12)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, linestyle=':', alpha=0.4)

    plt.suptitle("Subject TP_Tacr1_1: Multi-Epoch Allen Neuropixels Single-Unit QC Audit", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    png_out = out_dir / "allen_qc_multi_epoch_summary.png"
    plt.savefig(png_out, dpi=300)
    plt.close()

    # Copy PNG to artifact directory
    artifact_png = artifact_dir / "allen_qc_multi_epoch_summary.png"
    shutil.copy2(png_out, artifact_png)
    print(f" Saved Diagnostic Figure: {png_out}")

    # 4. Interactive Plotly Dashboard
    fig_plotly = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "A. Allen QC Passed Units Count Across Epochs",
            "B. ISI Violation Ratio Distribution (Persistent Units)",
            "C. Spatial Probe Depth vs Waveform Amplitude",
            "D. Firing Rate Stability Across Epochs"
        ]
    )

    fig_plotly.add_trace(go.Bar(
        x=epochs_labels, y=counts_bar, marker_color=colors_bar, name="Passed Units"
    ), row=1, col=1)

    fig_plotly.add_trace(go.Histogram(
        x=persistent_df['full_isi_viol'], nbinsx=20, marker_color='#2ca02c', name="ISI Violations"
    ), row=1, col=2)

    fig_plotly.add_trace(go.Scatter(
        x=qc_df['depth'], y=qc_df['peak_amp_uv'], mode='markers',
        marker=dict(color=['red' if p else 'gray' for p in qc_df['is_persistent_across_all_epochs']], size=6, opacity=0.7),
        hoverinfo='text',
        text=[f"Unit {u}<br>Persistent: {p}<br>SNR: {s}" for u, p, s in zip(qc_df['cluster_id'], qc_df['is_persistent_across_all_epochs'], qc_df['snr'])],
        name="Units Depth"
    ), row=2, col=1)

    fig_plotly.update_layout(
        title="<b>Subject TP_Tacr1_1: Multi-Epoch Allen QC Audit Interactive Dashboard</b>",
        template="plotly_white",
        height=850,
        showlegend=False
    )

    html_out = out_dir / "allen_qc_multi_epoch_dashboard.html"
    fig_plotly.write_html(str(html_out))
    print(f" Saved Interactive HTML Dashboard: {html_out}")

    print("\n==========================================================================================")
    print(" [SUCCESS] Multi-Epoch Allen QC Audit Complete!")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
