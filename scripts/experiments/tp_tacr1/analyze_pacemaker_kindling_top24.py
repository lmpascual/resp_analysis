#!/usr/bin/env python3
r"""
Single-Unit Pacemaker & Population Kindling Analysis (Ramirez et al. 2004 Framework)
Refined Ectopic Burst Definition strictly locked to Diaphragm EMG INSPIRATORY ONSET.

Ectopic Expiratory Burst Definition:
- Defined relative to Diaphragm EMG INSPIRATORY ONSET (t_onset = 0 s, precise rising edge of contraction).
- Inspiratory Burst Window: [-150 ms to +350 ms] surrounding t_onset.
- True Expiratory Quiescent Window: > +350 ms AFTER t_onset_i AND > 150 ms BEFORE t_onset_{i+1}.
- Ectopic Burst: Any burst of >= 3 spikes occurring strictly during the quiescent expiratory window.

Outputs:
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\pacemaker_kindling_top24_metrics.csv
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\pacemaker_kindling_top24_summary.png
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\pacemaker_kindling_top24_dashboard.html
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

def detect_bursts_in_window(spikes_in_win, max_isi=0.040, min_spikes=3):
    """Detects distinct spike bursts based on inter-spike interval <= max_isi"""
    if len(spikes_in_win) < min_spikes:
        return 0
    isis = np.diff(spikes_in_win)
    in_burst = False
    burst_count = 0
    curr_len = 1
    
    for isi in isis:
        if isi <= max_isi:
            curr_len += 1
            if curr_len >= min_spikes and not in_burst:
                burst_count += 1
                in_burst = True
        else:
            in_burst = False
            curr_len = 1
    return burst_count

def main():
    out_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation")
    out_dir.mkdir(parents=True, exist_ok=True)
    session_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1\alf_session")
    alf_dir = session_dir / "alf" / "probe00"
    raw_dir = Path(r"Z:\ramirezlab\pascual\rec_data\TP_Tacr1_1_npx2_g4")
    artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\0e7be222-f160-4bc2-921c-f3fdf8b7825e")

    print("=" * 90)
    print(" Running Refined Pacemaker & Kindling Analysis (Onloaded to Diaphragm EMG Inspiratory Onset)")
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

    print(f" Analyzed {n_valid_breaths} unperturbed breath cycles (Onset = rising edge of diaphragm EMG).")

    # 3. Refined Ectopic Burst & Cycle-to-Cycle Recruitment Metrics
    spiketrains = {u: np.sort(spikes_times[spikes_clusters == u]) for u in top24_uids}
    
    recruitment_probs = []
    mean_spk_counts = []
    fano_factors = []
    ectopic_burst_counts = []

    t_win_pre = 0.15  # 150 ms pre-inspiration onset
    t_win_post = 0.35 # 350 ms post-inspiration onset

    for uid in top24_uids:
        u_spks = spiketrains[uid]
        counts_per_breath = []
        recruited_flag = []

        for o in valid_onsets:
            spks_in_breath = u_spks[(u_spks >= o - t_win_pre) & (u_spks <= o + t_win_post)]
            n_spks = len(spks_in_breath)
            counts_per_breath.append(n_spks)
            recruited_flag.append(1 if n_spks >= 1 else 0)

        counts_per_breath = np.array(counts_per_breath)
        recruited_flag = np.array(recruited_flag)

        p_rec = np.mean(recruited_flag)
        m_cnt = np.mean(counts_per_breath)
        v_cnt = np.var(counts_per_breath)
        ff = v_cnt / m_cnt if m_cnt > 0 else 0.0

        # Ectopic Expiratory Bursts: > +350ms after onset_i AND > 150ms before onset_{i+1}
        ectopic_cnt = 0
        for i in range(len(valid_onsets) - 1):
            t_exp_start = valid_onsets[i] + 0.35
            t_exp_end = valid_onsets[i+1] - 0.15
            if t_exp_end > t_exp_start:
                spks_exp = u_spks[(u_spks >= t_exp_start) & (u_spks <= t_exp_end)]
                ectopic_cnt += detect_bursts_in_window(spks_exp, max_isi=0.040, min_spikes=3)

        recruitment_probs.append(round(p_rec * 100.0, 1))
        mean_spk_counts.append(round(m_cnt, 2))
        fano_factors.append(round(ff, 2))
        ectopic_burst_counts.append(ectopic_cnt)

    top24_df['recruitment_prob_pct'] = recruitment_probs
    top24_df['mean_spikes_per_breath'] = mean_spk_counts
    top24_df['fano_factor'] = fano_factors
    top24_df['ectopic_exp_bursts'] = ectopic_burst_counts

    # Save CSV
    csv_out = out_dir / "pacemaker_kindling_top24_metrics.csv"
    top24_df.to_csv(csv_out, index=False)
    print(f" Saved Refined Pacemaker Metrics CSV: {csv_out}")

    # 4. Population Kindling Build-up Alignment (5ms Binned Activity Pre-Inspiration Onset)
    dt_bin = 0.005 # 5 ms
    t_psth = np.arange(-0.30, 0.45, dt_bin)
    n_psth_bins = len(t_psth) - 1
    t_psth_centers = (t_psth[:-1] + t_psth[1:]) / 2.0 * 1000.0 # ms

    pop_activity = np.zeros((n_valid_breaths, n_psth_bins))

    for b_i, o in enumerate(valid_onsets):
        for uid in top24_uids:
            u_spks = spiketrains[uid]
            spks_in_win = u_spks[(u_spks >= o - 0.30) & (u_spks <= o + 0.45)] - o
            c, _ = np.histogram(spks_in_win, bins=t_psth)
            pop_activity[b_i] += c

    mean_pop_kindling = np.mean(pop_activity, axis=0) / dt_bin / len(top24_uids) # Mean Hz per unit
    sem_pop_kindling = stats.sem(pop_activity, axis=0) / dt_bin / len(top24_uids)

    # 5. Create High-Resolution Static PNG Figure (4 Panels)
    fig_mpl = plt.figure(figsize=(20, 12))

    ax1 = fig_mpl.add_subplot(2, 2, 1)
    ax1.bar(range(len(top24_uids)), top24_df['recruitment_prob_pct'], color='#1f77b4', alpha=0.85, edgecolor='black')
    ax1.set_xticks(range(len(top24_uids)))
    ax1.set_xticklabels([f"U{u}" for u in top24_df['cluster_id']], rotation=45, ha='right', fontsize=8)
    ax1.axhline(90.0, color='red', linestyle='--', label='90% Threshold')
    ax1.set_ylabel("Recruitment Consistency (%)", fontweight='bold', fontsize=10)
    ax1.set_title("A. Breath Cycle Recruitment Consistency (Onset-Aligned)", fontweight='bold', fontsize=12)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.4)

    ax2 = fig_mpl.add_subplot(2, 2, 2)
    ax2.bar(range(len(top24_uids)), top24_df['fano_factor'], color='#ff7f0e', alpha=0.85, edgecolor='black')
    ax2.axhline(1.0, color='black', linestyle='--', label='Poisson Limit (FF = 1.0)')
    ax2.set_xticks(range(len(top24_uids)))
    ax2.set_xticklabels([f"U{u}" for u in top24_df['cluster_id']], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel("Fano Factor (Var / Mean)", fontweight='bold', fontsize=10)
    ax2.set_title("B. Spike Count Fano Factor (Stochasticity)", fontweight='bold', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.4)

    ax3 = fig_mpl.add_subplot(2, 2, 3)
    ax3.plot(t_psth_centers, mean_pop_kindling, color='crimson', linewidth=2.5, label='Population Rate (24 Units)')
    ax3.fill_between(t_psth_centers, mean_pop_kindling - sem_pop_kindling, mean_pop_kindling + sem_pop_kindling, color='crimson', alpha=0.25)
    ax3.axvline(0, color='black', linestyle='--', linewidth=1.5, label='EMG Insp Onset (t=0)')
    ax3.axvspan(-100, 0, color='yellow', alpha=0.2, label='Pre-Inspiratory Kindling Window (-100 to 0ms)')
    ax3.set_xlabel("Time relative to EMG Insp Onset (ms)", fontweight='bold', fontsize=10)
    ax3.set_ylabel("Mean Population Rate (Hz/unit)", fontweight='bold', fontsize=10)
    ax3.set_title("C. Synchronous Population Kindling Build-up at Inspiration Onset", fontweight='bold', fontsize=12)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, linestyle=':', alpha=0.4)

    ax4 = fig_mpl.add_subplot(2, 2, 4)
    ax4.scatter(top24_df['ectopic_exp_bursts'], top24_df['recruitment_prob_pct'], s=100, color='purple', alpha=0.8, edgecolor='black')
    for _, r in top24_df.iterrows():
        ax4.annotate(f"U{int(r['cluster_id'])}", (r['ectopic_exp_bursts']+0.3, r['recruitment_prob_pct']), fontsize=8)
    ax4.set_xlabel("Ectopic Expiratory Bursts (# Count in Quiescence)", fontweight='bold', fontsize=10)
    ax4.set_ylabel("Recruitment Consistency (%)", fontweight='bold', fontsize=10)
    ax4.set_title("D. Ectopic Expiratory Bursts vs Recruitment Consistency", fontweight='bold', fontsize=12)
    ax4.grid(True, linestyle=':', alpha=0.4)

    plt.suptitle("Subject TP_Tacr1_1: Pacemaker & Population Kindling (EMG Inspiration Onset Aligned)", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    png_out = out_dir / "pacemaker_kindling_top24_summary.png"
    plt.savefig(png_out, dpi=300)
    plt.close()

    # Copy PNG to artifact directory
    artifact_png = artifact_dir / "pacemaker_kindling_top24_summary.png"
    shutil.copy2(png_out, artifact_png)
    print(f" Saved Refined Diagnostic Figure: {png_out}")

    # 6. Interactive Plotly Dashboard
    fig_plotly = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "A. Cycle-to-Cycle Recruitment Consistency (%)",
            "B. Fano Factor (Var / Mean)",
            "C. Synchronous Population Kindling Build-up (Hz/unit)",
            "D. Ectopic Expiratory Bursts vs Recruitment"
        ]
    )

    fig_plotly.add_trace(go.Bar(
        x=[f"Unit {u}" for u in top24_df['cluster_id']],
        y=top24_df['recruitment_prob_pct'],
        marker_color='#1f77b4',
        name="Recruitment %"
    ), row=1, col=1)

    fig_plotly.add_trace(go.Bar(
        x=[f"Unit {u}" for u in top24_df['cluster_id']],
        y=top24_df['fano_factor'],
        marker_color='#ff7f0e',
        name="Fano Factor"
    ), row=1, col=2)

    fig_plotly.add_trace(go.Scatter(
        x=t_psth_centers,
        y=mean_pop_kindling,
        mode='lines',
        line=dict(color='crimson', width=3),
        name="Population Kindling Rate"
    ), row=2, col=1)

    fig_plotly.add_trace(go.Scatter(
        x=top24_df['ectopic_exp_bursts'],
        y=top24_df['recruitment_prob_pct'],
        mode='markers+text',
        marker=dict(color='purple', size=10),
        text=[f"Unit {u}" for u in top24_df['cluster_id']],
        textposition="top center",
        name="Ectopic Bursts vs Rec"
    ), row=2, col=2)

    fig_plotly.update_layout(
        title="<b>Subject TP_Tacr1_1: Refined Pacemaker & Population Kindling Interactive Dashboard</b>",
        template="plotly_white",
        height=850,
        showlegend=False
    )

    html_out = out_dir / "pacemaker_kindling_top24_dashboard.html"
    fig_plotly.write_html(str(html_out))
    print(f" Saved Interactive HTML Dashboard: {html_out}")

    print("\n==========================================================================================")
    print(" [SUCCESS] Refined Pacemaker & Kindling Analysis Complete!")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
