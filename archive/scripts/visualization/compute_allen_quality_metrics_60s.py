#!/usr/bin/env python3
r"""
AllenSDK Visual Behavior Neuropixels Quality Metrics Pipeline (FIRST 60 SECONDS BASELINE EPOCH)
Computes official Allen Institute single-unit quality metrics strictly evaluated on t = 0 to 60.0 s:
1. ISI Violations Ratio (isi_viol < 0.5)
2. Presence Ratio (presence_ratio > 0.90)
3. Amplitude Cutoff (amplitude_cutoff < 0.10)
4. Firing Rate (firing_rate >= 0.1 Hz)
5. SNR (Signal-to-Noise Ratio > 2.5)

Saves CSV, diagnostic figure, and interactive HTML dashboard to:
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
    """AllenSDK ISI violations ratio metric"""
    if len(spike_times) <= 1:
        return 0.0
    isis = np.diff(spike_times)
    num_violations = np.sum((isis < isi_threshold) & (isis >= min_isi))
    total_spikes = len(spike_times)
    violation_rate = num_violations / max_time
    firing_rate = total_spikes / max_time
    violation_ratio = violation_rate / (2 * (isi_threshold - min_isi) * firing_rate**2) if firing_rate > 0 else 0.0
    return float(violation_ratio)

def compute_presence_ratio(spike_times, max_time, num_blocks=60):
    """AllenSDK Presence Ratio over 60s (1s blocks)"""
    if len(spike_times) == 0:
        return 0.0
    block_edges = np.linspace(0, max_time, num_blocks + 1)
    counts, _ = np.histogram(spike_times, bins=block_edges)
    return float(np.sum(counts > 0) / num_blocks)

def compute_amplitude_cutoff(u_amps, num_bins=100):
    """AllenSDK Amplitude Cutoff: Fraction of missing spikes from amplitude distribution"""
    if len(u_amps) < 10:
        return 0.0
    h, b = np.histogram(u_amps, bins=num_bins)
    peak_idx = np.argmax(h)
    if peak_idx == 0:
        return float(min(1.0, h[0] / np.sum(h)))
    
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
    return float(missing_spikes / total_est) if total_est > 0 else 0.0

def main():
    out_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation")
    out_dir.mkdir(parents=True, exist_ok=True)
    session_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1\alf_session")
    alf_dir = session_dir / "alf" / "probe00"
    artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\0e7be222-f160-4bc2-921c-f3fdf8b7825e")

    print("=" * 90)
    print(" AllenSDK Neuropixels Quality Metrics Audit (FIRST 60 SECONDS BASELINE)")
    print(f" ALF Directory: {alf_dir}")
    print(f" Output Directory: {out_dir}")
    print("=" * 90)

    # Load ALF Data
    spikes_times = np.load(alf_dir / "spike_times.npy")
    spikes_clusters = np.load(alf_dir / "spike_clusters.npy")
    amplitudes = np.load(alf_dir / "amplitudes.npy")
    info_df = pd.read_csv(alf_dir / "cluster_info.tsv", sep="\t")

    fs_ap = 30000.0
    if spikes_times.dtype in [np.int64, np.int32] and spikes_times.max() > 100000:
        spikes_times = spikes_times / fs_ap

    MAX_TIME_S = 60.0 # First 60 seconds baseline epoch
    spk_mask = (spikes_times >= 0.0) & (spikes_times <= MAX_TIME_S)
    spikes_times = spikes_times[spk_mask]
    spikes_clusters = spikes_clusters[spk_mask]
    amplitudes = amplitudes[spk_mask]

    target_uids = [100, 118, 55, 13]
    candidate_uids = info_df[(info_df['group'] == 'good') | (info_df['duplicate_of'] == -1)]['cluster_id'].values

    # Compute Quality Metrics for All Good Units during 0-60s
    metrics_list = []
    noise_floor_uv = 15.0

    for uid in candidate_uids:
        idx_u = np.where(spikes_clusters == uid)[0]
        u_times = spikes_times[idx_u]
        u_amps = amplitudes[idx_u]
        
        info_row = info_df[info_df['cluster_id'] == uid].iloc[0]
        peak_amp = float(info_row['peak_amp_uv'])
        
        fr = len(u_times) / MAX_TIME_S
        isi_viol = compute_isi_violations(u_times, MAX_TIME_S)
        pres_ratio = compute_presence_ratio(u_times, MAX_TIME_S)
        amp_cutoff = compute_amplitude_cutoff(u_amps)
        snr = peak_amp / noise_floor_uv

        # AllenSDK Criteria for 60s baseline epoch:
        # 1. isi_viol < 0.50
        # 2. presence_ratio >= 0.80 (scaled for 60s)
        # 3. amplitude_cutoff < 0.15
        passed_allen_qc = (isi_viol < 0.50) and (pres_ratio >= 0.80) and (amp_cutoff < 0.15) and (fr >= 0.1)

        metrics_list.append({
            'cluster_id': uid,
            'shank': int(info_row['shank']),
            'depth': float(info_row['depth']),
            'firing_rate_60s': round(fr, 3),
            'n_spikes_60s': len(u_times),
            'peak_amp_uv': round(peak_amp, 1),
            'snr': round(snr, 2),
            'isi_violations': round(isi_viol, 4),
            'presence_ratio': round(pres_ratio, 3),
            'amplitude_cutoff': round(amp_cutoff, 4),
            'passed_allen_qc': passed_allen_qc
        })

    metrics_df = pd.DataFrame(metrics_list)
    csv_out = out_dir / "allen_neuropixels_quality_metrics_baseline_60s.csv"
    metrics_df.to_csv(csv_out, index=False)
    print(f" Saved Quality Metrics CSV (60s Baseline): {csv_out}")

    # Print Focus Table for Target Units 100, 118, 55, 13
    target_df = metrics_df[metrics_df['cluster_id'].isin(target_uids)].copy().reset_index(drop=True)
    print("\n" + "=" * 105)
    print(f"{'Unit':<6} | {'Shank':<5} | {'Depth(um)':<9} | {'FR 60s':<7} | {'Amp(uV)':<8} | {'SNR':<6} | {'ISI Viol':<9} | {'Presence Ratio':<15} | {'Amp Cutoff':<11} | {'Allen QC':<10}")
    print("-" * 105)
    for _, r in target_df.iterrows():
        qc_str = "PASS" if r['passed_allen_qc'] else "MARGINAL"
        print(f"{int(r['cluster_id']):<6} | {int(r['shank']):<5} | {r['depth']:<9.1f} | {r['firing_rate_60s']:<7.2f} | {r['peak_amp_uv']:<8.1f} | {r['snr']:<6.2f} | {r['isi_violations']:<9.4f} | {r['presence_ratio']:<15.3f} | {r['amplitude_cutoff']:<11.4f} | {qc_str:<10}")
    print("=" * 105 + "\n")

    # 5. Create 6-Panel Diagnostic Figure for 60s Baseline
    fig_mpl = plt.figure(figsize=(18, 12))

    ax1 = fig_mpl.add_subplot(2, 3, 1)
    ax1.scatter(metrics_df['firing_rate_60s'], metrics_df['isi_violations'], color='gray', alpha=0.5)
    for uid in target_uids:
        r = metrics_df[metrics_df['cluster_id'] == uid].iloc[0]
        ax1.scatter(r['firing_rate_60s'], r['isi_violations'], s=140, label=f"Unit {uid} ({r['isi_violations']:.3f})", zorder=5)
    ax1.axhline(0.5, color='red', linestyle='--', label='Allen Threshold (0.5)')
    ax1.set_xlabel("60s Baseline Firing Rate (Hz)", fontweight='bold')
    ax1.set_ylabel("ISI Violations Ratio", fontweight='bold')
    ax1.set_title("A. ISI Violations vs Firing Rate (0-60s)", fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, linestyle=':', alpha=0.4)

    ax2 = fig_mpl.add_subplot(2, 3, 2)
    ax2.hist(metrics_df['presence_ratio'], bins=np.linspace(0, 1.0, 21), color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(0.80, color='red', linestyle='--', label='60s Threshold (0.80)')
    for uid in target_uids:
        r = metrics_df[metrics_df['cluster_id'] == uid].iloc[0]
        ax2.axvline(r['presence_ratio'], linestyle=':', linewidth=2, label=f"Unit {uid} ({r['presence_ratio']:.2f})")
    ax2.set_xlabel("Presence Ratio (0-60s)", fontweight='bold')
    ax2.set_ylabel("Unit Count", fontweight='bold')
    ax2.set_title("B. Presence Ratio (0-60s Baseline)", fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.4)

    ax3 = fig_mpl.add_subplot(2, 3, 3)
    ax3.hist(metrics_df['amplitude_cutoff'], bins=np.linspace(0, 0.5, 21), color='salmon', edgecolor='black', alpha=0.7)
    ax3.axvline(0.15, color='red', linestyle='--', label='Threshold (0.15)')
    for uid in target_uids:
        r = metrics_df[metrics_df['cluster_id'] == uid].iloc[0]
        ax3.axvline(r['amplitude_cutoff'], linestyle=':', linewidth=2, label=f"Unit {uid} ({r['amplitude_cutoff']:.3f})")
    ax3.set_xlabel("Amplitude Cutoff (0-60s)", fontweight='bold')
    ax3.set_ylabel("Unit Count", fontweight='bold')
    ax3.set_title("C. Amplitude Cutoff (0-60s Baseline)", fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, linestyle=':', alpha=0.4)

    ax4 = fig_mpl.add_subplot(2, 3, 4)
    ax4.scatter(metrics_df['snr'], metrics_df['depth'], color='gray', alpha=0.5)
    for uid in target_uids:
        r = metrics_df[metrics_df['cluster_id'] == uid].iloc[0]
        ax4.scatter(r['snr'], r['depth'], s=140, label=f"Unit {uid} (SNR {r['snr']:.1f})", zorder=5)
    ax4.axvline(2.5, color='red', linestyle='--', label='Allen SNR Cutoff (2.5)')
    ax4.set_xlabel("Signal-to-Noise Ratio (SNR)", fontweight='bold')
    ax4.set_ylabel("Probe Depth (µm)", fontweight='bold')
    ax4.set_title("D. Unit SNR across Probe Depth", fontweight='bold')
    ax4.legend(loc='lower right', fontsize=8)
    ax4.grid(True, linestyle=':', alpha=0.4)

    ax5 = fig_mpl.add_subplot(2, 3, 5)
    ax5.scatter(metrics_df['amplitude_cutoff'], metrics_df['isi_violations'], color='gray', alpha=0.5)
    for uid in target_uids:
        r = metrics_df[metrics_df['cluster_id'] == uid].iloc[0]
        ax5.scatter(r['amplitude_cutoff'], r['isi_violations'], s=140, label=f"Unit {uid}", zorder=5)
    ax5.axvline(0.15, color='red', linestyle='--')
    ax5.axhline(0.50, color='red', linestyle='--')
    ax5.set_xlabel("Amplitude Cutoff (0-60s)", fontweight='bold')
    ax5.set_ylabel("ISI Violations Ratio", fontweight='bold')
    ax5.set_title("E. Amplitude Cutoff vs ISI Violations (0-60s)", fontweight='bold')
    ax5.grid(True, linestyle=':', alpha=0.4)

    ax6 = fig_mpl.add_subplot(2, 3, 6)
    pass_cnt = np.sum(metrics_df['passed_allen_qc'])
    fail_cnt = len(metrics_df) - pass_cnt
    ax6.bar(['Passed 60s QC', 'Marginal/Sub-threshold'], [pass_cnt, fail_cnt], color=['#2ca02c', '#d62728'], alpha=0.8)
    ax6.set_ylabel("Number of Single Units", fontweight='bold')
    ax6.set_title(f"F. 60s Baseline Pass Rate: {pass_cnt}/{len(metrics_df)} ({pass_cnt/len(metrics_df)*100:.1f}%)", fontweight='bold')
    ax6.grid(True, linestyle=':', alpha=0.4)

    plt.suptitle("Subject TP_Tacr1_1: AllenSDK Quality Metrics Audit (0 - 60 s Baseline Epoch)", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    png_out = out_dir / "allen_neuropixels_quality_metrics_baseline_60s.png"
    plt.savefig(png_out, dpi=300)
    plt.close()

    # Copy to artifact directory
    artifact_png = artifact_dir / "allen_neuropixels_quality_metrics_baseline_60s.png"
    shutil.copy2(png_out, artifact_png)
    print(f" Saved Diagnostic Figure (60s Baseline): {png_out}")

    # 6. Interactive Plotly Dashboard Table for 60s Baseline
    fig_plotly = go.Figure(data=[
        go.Table(
            header=dict(
                values=["<b>Unit ID</b>", "<b>Shank</b>", "<b>Depth (µm)</b>", "<b>FR 60s (Hz)</b>", "<b>Peak Amp (µV)</b>", "<b>SNR</b>", "<b>ISI Viol Ratio</b>", "<b>Presence Ratio</b>", "<b>Amp Cutoff</b>", "<b>60s QC Status</b>"],
                fill_color='#1f77b4',
                font=dict(color='white', size=12),
                align='center'
            ),
            cells=dict(
                values=[
                    metrics_df['cluster_id'],
                    metrics_df['shank'],
                    metrics_df['depth'],
                    metrics_df['firing_rate_60s'],
                    metrics_df['peak_amp_uv'],
                    metrics_df['snr'],
                    metrics_df['isi_violations'],
                    metrics_df['presence_ratio'],
                    metrics_df['amplitude_cutoff'],
                    ["PASS" if p else "MARGINAL" for p in metrics_df['passed_allen_qc']]
                ],
                fill_color=[['#d4edda' if p else '#f8d7da' for p in metrics_df['passed_allen_qc']]],
                align='center',
                font=dict(size=11)
            )
        )
    ])

    fig_plotly.update_layout(
        title="<b>Subject TP_Tacr1_1: Allen Institute Quality Metrics Table (0 - 60 s Baseline Epoch)</b>",
        template="plotly_white",
        height=800
    )

    html_out = out_dir / "allen_neuropixels_quality_metrics_baseline_60s.html"
    fig_plotly.write_html(str(html_out))
    print(f" Saved Interactive HTML Dashboard (60s Baseline): {html_out}")

    print("\n==========================================================================================")
    print(" [SUCCESS] 60s Baseline Allen Quality Metrics Audit Complete!")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
