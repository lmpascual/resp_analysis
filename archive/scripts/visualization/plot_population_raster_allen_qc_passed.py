#!/usr/bin/env python3
r"""
Regenerates Population Spike Raster Plots filtered STRICTLY for single units
passing the AllenSDK Neuropixels Quality Criteria (isi_violations < 0.50, presence_ratio >= 0.90, amplitude_cutoff < 0.10).

- Standard Allen QC passed units: BLACK ticks
- Phototagged Tacr1+ units: BRIGHT RED ticks
- Divided horizontally by Shank identity (Shanks 0, 1, 2, 3)
- Translucent Magenta optogenetic laser pulse overlays

Outputs:
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\population_raster_allen_qc_passed.png
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\population_raster_allen_qc_passed.html
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
    print(" Regenerating Population Spike Raster for AllenSDK Quality Passed Single Units")
    print(f" ALF Directory: {alf_dir}")
    print(f" Output Directory: {out_dir}")
    print("=" * 90)

    # 1. Load Spikes & Allen Quality Metrics CSV
    qc_csv = out_dir / "allen_neuropixels_quality_metrics.csv"
    if not qc_csv.exists():
        print(f" Error: Quality metrics CSV {qc_csv} not found. Run compute_allen_quality_metrics.py first.")
        sys.exit(1)

    metrics_df = pd.read_csv(qc_csv)
    passed_df = metrics_df[metrics_df['passed_allen_qc']].copy().reset_index(drop=True)
    passed_uids = set(passed_df['cluster_id'])

    print(f" Found {len(passed_uids)} single units passing AllenSDK Quality Criteria.")

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

    # 2. Load NIDQ Diaphragm EMG & Laser TTL
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

    laser_active = (ch6_laser > 0.8).astype(int)
    laser_onsets = t_nidq[np.where(np.diff(laser_active) == 1)[0]]
    laser_offsets = t_nidq[np.where(np.diff(laser_active) == -1)[0]]
    n_pulses = min(len(laser_onsets), len(laser_offsets))
    laser_onsets = laser_onsets[:n_pulses]
    laser_offsets = laser_offsets[:n_pulses]
    pulse_durations = laser_offsets - laser_onsets

    # 3. Determine Tacr1+ Phototagging Status for Passed Units
    spiketrains = {u: np.sort(spikes_times[spikes_clusters == u]) for u in passed_uids}
    
    phototagged_flags = []
    laser_ratios = []

    for idx, row in passed_df.iterrows():
        uid = int(row['cluster_id'])
        u_spks = spiketrains[uid]

        l_counts, b_counts, lats = [], [], []
        for p_i in range(n_pulses):
            ton = laser_onsets[p_i]
            toff = laser_offsets[p_i]
            dur = toff - ton

            nl = np.sum((u_spks >= ton) & (u_spks < toff))
            nb = np.sum((u_spks >= ton - dur) & (u_spks < ton))
            l_counts.append(nl)
            b_counts.append(nb)

            spks_after = u_spks[u_spks >= ton]
            if len(spks_after) > 0 and spks_after[0] < toff:
                lats.append((spks_after[0] - ton) * 1000.0)

        l_counts = np.array(l_counts)
        b_counts = np.array(b_counts)
        m_l_rate = np.mean(l_counts) / max(1e-4, np.mean(pulse_durations))
        m_b_rate = np.mean(b_counts) / max(1e-4, np.mean(pulse_durations))
        ratio = (m_l_rate + 0.1) / (m_b_rate + 0.1)

        if np.sum(l_counts != b_counts) > 5:
            _, p_val = stats.wilcoxon(l_counts, b_counts)
        else:
            p_val = 1.0

        mean_lat = np.mean(lats) if len(lats) > 0 else 999.0
        is_tagged = (p_val < 0.01) and (ratio >= 2.5) and (m_l_rate > 3.0) and (mean_lat <= 11.0)
        
        phototagged_flags.append(is_tagged)
        laser_ratios.append(round(ratio, 2))

    passed_df['is_phototagged'] = phototagged_flags
    passed_df['laser_ratio'] = laser_ratios

    # Sort units by Shank and Depth
    sorted_df = passed_df.sort_values(by=['shank', 'depth'], ascending=[True, True]).reset_index(drop=True)
    sorted_uids = sorted_df['cluster_id'].values
    n_sorted = len(sorted_uids)

    n_tagged = np.sum(sorted_df['is_phototagged'])
    print(f" Sorted {n_sorted} Allen QC passed units across Shanks ({n_tagged} Phototagged Tacr1+ units).")

    # Calculate Shank Boundaries for Visual Divider Lines
    shank_boundary_indices = []
    for i in range(n_sorted - 1):
        if sorted_df.loc[i, 'shank'] != sorted_df.loc[i + 1, 'shank']:
            shank_boundary_indices.append(i + 1.5)

    # 4. Generate High-Resolution Static PNG Figure (Matplotlib)
    fig_mpl = plt.figure(figsize=(22, 12))
    gs = fig_mpl.add_gridspec(3, 1, height_ratios=[1.2, 0.4, 5.5], hspace=0.15)

    # Subplot 1: Diaphragm EMG Envelope
    ax_emg = fig_mpl.add_subplot(gs[0])
    ax_emg.plot(t_nidq, emg_env, color='black', linewidth=1.0)
    for p_i in range(n_pulses):
        ax_emg.axvspan(laser_onsets[p_i], laser_offsets[p_i], color='magenta', alpha=0.35)
    ax_emg.set_ylabel("Diaphragm EMG (µV)", fontsize=10, fontweight='bold')
    ax_emg.set_title("1. Diaphragm EMG Envelope (0 - 550 s)", fontsize=12, fontweight='bold')
    ax_emg.set_xlim([0, MAX_TIME_S])
    ax_emg.grid(True, linestyle=':', alpha=0.3)

    # Subplot 2: Compact Laser TTL Pulses
    ax_ttl = fig_mpl.add_subplot(gs[1], sharex=ax_emg)
    ax_ttl.plot(t_nidq, ch6_laser, color='magenta', linewidth=1.5)
    ax_ttl.set_ylabel("Laser TTL", fontsize=10, fontweight='bold')
    ax_ttl.set_ylim([-0.1, 1.2])
    ax_ttl.grid(True, linestyle=':', alpha=0.3)

    # Subplot 3: Population Spike Raster (Sorted by Shank & Depth)
    ax_rast = fig_mpl.add_subplot(gs[2], sharex=ax_emg)

    # Plot laser shading across raster
    for p_i in range(n_pulses):
        ax_rast.axvspan(laser_onsets[p_i], laser_offsets[p_i], color='magenta', alpha=0.35)

    # Plot Shank horizontal dividing lines
    for b_y in shank_boundary_indices:
        ax_rast.axhline(b_y, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)

    # Plot spikes per unit (Black for standard, Red for phototagged)
    y_ticks_pos = []
    y_ticks_labels = []

    for y_idx, uid in enumerate(sorted_uids):
        u_row = sorted_df[sorted_df['cluster_id'] == uid].iloc[0]
        u_spks = spiketrains[uid]
        is_tagged = u_row['is_phototagged']
        
        tick_color = 'red' if is_tagged else 'black'
        line_w = 1.4 if is_tagged else 0.9

        if len(u_spks) > 0:
            ax_rast.vlines(u_spks, y_idx + 0.6, y_idx + 1.4, color=tick_color, linewidth=line_w)

        y_ticks_pos.append(y_idx + 1)
        lbl = f"U{uid} (S{int(u_row['shank'])}, {int(u_row['depth'])}µm)"
        if is_tagged: lbl += " 🔥"
        y_ticks_labels.append(lbl)

    ax_rast.set_yticks(y_ticks_pos[::max(1, len(y_ticks_pos)//30)]) # Sample ticks for readability
    ax_rast.set_yticklabels(y_ticks_labels[::max(1, len(y_ticks_labels)//30)], fontsize=8)
    ax_rast.set_ylabel(f"Allen QC Passed Units (N = {n_sorted})", fontsize=11, fontweight='bold')
    ax_rast.set_xlabel("Session Time (s)", fontsize=11, fontweight='bold')
    ax_rast.set_title(f"3. Population Spike Raster ({n_sorted} Allen QC Passed Units | {n_tagged} Phototagged Tacr1+ in Red)", fontsize=12, fontweight='bold')
    ax_rast.set_ylim([0.2, n_sorted + 0.8])

    plt.suptitle("Subject TP_Tacr1_1: Population Spike Raster (Filtered for AllenSDK Quality Metric Passed Units)", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    png_out = out_dir / "population_raster_allen_qc_passed.png"
    plt.savefig(png_out, dpi=300)
    plt.close()

    # Copy PNG to artifact directory
    artifact_png = artifact_dir / "population_raster_allen_qc_passed.png"
    shutil.copy2(png_out, artifact_png)
    print(f" Saved Diagnostic Raster Figure: {png_out}")

    # 5. Interactive Plotly HTML Dashboard
    fig_plotly = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.18, 0.08, 0.74],
        subplot_titles=[
            "1. Diaphragm EMG Envelope",
            "2. Optogenetic Laser TTL Pulses",
            f"3. Spike Raster ({n_sorted} Allen QC Passed Single Units | Red = Tacr1+ Phototagged)"
        ],
        vertical_spacing=0.04
    )

    # Subplot 1: EMG Envelope
    fig_plotly.add_trace(go.Scatter(x=t_nidq[::10], y=emg_env[::10], mode='lines', line=dict(color='black', width=1), name="EMG Envelope"), row=1, col=1)

    # Subplot 2: Laser TTL
    fig_plotly.add_trace(go.Scatter(x=t_nidq[::10], y=ch6_laser[::10], mode='lines', line=dict(color='magenta', width=1.5), name="Laser TTL"), row=2, col=1)

    # Subplot 3: Rasters
    for y_idx, uid in enumerate(sorted_uids):
        u_row = sorted_df[sorted_df['cluster_id'] == uid].iloc[0]
        u_spks = spiketrains[uid]
        is_tagged = u_row['is_phototagged']
        c_tick = 'red' if is_tagged else 'black'
        
        # Subsample spikes for interactive rendering speed if > 2000 spikes
        if len(u_spks) > 2000:
            spk_disp = np.random.choice(u_spks, size=2000, replace=False)
        else:
            spk_disp = u_spks

        fig_plotly.add_trace(go.Scatter(
            x=spk_disp,
            y=[y_idx + 1] * len(spk_disp),
            mode='markers',
            marker=dict(symbol='line-ns-open', color=c_tick, size=5, line=dict(width=1.2)),
            hoverinfo='text',
            text=[f"Unit {uid} (Shank {int(u_row['shank'])}, {int(u_row['depth'])}µm)<br>FR: {u_row['firing_rate']}Hz | SNR: {u_row['snr']}<br>ISI Viol: {u_row['isi_violations']} | Tagged: {is_tagged}"] * len(spk_disp),
            name=f"Unit {uid}"
        ), row=3, col=1)

    # Laser Shading Shapes
    shapes = []
    for p_i in range(n_pulses):
        shapes.append(dict(
            type='rect',
            xref='x3', yref='y3',
            x0=laser_onsets[p_i], x1=laser_offsets[p_i],
            y0=0, y1=n_sorted + 1,
            fillcolor='magenta', opacity=0.30, line=dict(width=0)
        ))

    # Shank Separator Lines
    for b_y in shank_boundary_indices:
        shapes.append(dict(
            type='line',
            xref='x3', yref='y3',
            x0=0, x1=MAX_TIME_S,
            y0=b_y, y1=b_y,
            line=dict(color='blue', width=1.5, dash='dash')
        ))

    fig_plotly.update_layout(
        title="<b>Subject TP_Tacr1_1: Population Spike Raster (AllenSDK QC Passed Single Units)</b>",
        template="plotly_white",
        height=1000,
        shapes=shapes,
        showlegend=False,
        xaxis3=dict(title="Session Time (s)"),
        yaxis3=dict(title="Unit Index (Sorted by Shank & Depth)")
    )

    html_out = out_dir / "population_raster_allen_qc_passed.html"
    fig_plotly.write_html(str(html_out))
    print(f" Saved Interactive HTML Population Raster: {html_out}")

    print("\n==========================================================================================")
    print(" [SUCCESS] Population Spike Raster Regeneration Complete!")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
