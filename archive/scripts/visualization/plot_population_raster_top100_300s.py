#!/usr/bin/env python3
"""
Generate Population Spike Raster Plot & Diaphragm EMG Respiration Visualizer
for the Top 100 Kilosort4 'Good' Single Units across the First 300 Seconds.
Translucent shaded regions indicate laser pulse windows on both EMG and Raster plots.
Outputs both an interactive Plotly HTML dashboard and a high-res static PNG figure.
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

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

MANUAL_CHANNELS = [
    4, 12, 28, 32, 96, 108, 202, 210, 222, 224, 226, 290, 316,
    48, 58, 66, 82, 90, 154, 178, 186, 190, 242, 346, 364, 374,
    279, 283, 287, 343, 349, 91, 167
]

def get_shank(x):
    if x < 125: return 0
    elif x < 391: return 1
    elif x < 657: return 2
    else: return 3

def main():
    out_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1")
    session_dir = out_dir / "alf_session"
    alf_dir = session_dir / "alf" / "probe00"
    raw_dir = Path(r"Z:\ramirezlab\pascual\rec_data\TP_Tacr1_1_npx2_g4")
    artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\0e7be222-f160-4bc2-921c-f3fdf8b7825e")

    print("=" * 85)
    print(" Generating Population Spike Raster (Top 100 Units | First 300 Seconds)")
    print(f" ALF Directory: {alf_dir}")
    print(f" Output Directory: {out_dir}")
    print("=" * 85)

    # 1. Load Kilosort4 spike data & channel geometry
    spikes_times = np.load(alf_dir / "spike_times.npy")
    spikes_clusters = np.load(alf_dir / "spike_clusters.npy")
    templates = np.load(alf_dir / "templates.npy")
    channel_coords = np.load(alf_dir / "channel_positions.npy")

    # Convert spike_times to seconds if in sample counts
    fs_ap = 30000.0
    if spikes_times.dtype in [np.int64, np.int32] and spikes_times.max() > 100000:
        spikes_times = spikes_times / fs_ap

    # Filter to first 300 seconds
    MAX_TIME_S = 300.0
    time_mask_ap = (spikes_times >= 0.0) & (spikes_times <= MAX_TIME_S)
    spikes_times = spikes_times[time_mask_ap]
    spikes_clusters = spikes_clusters[time_mask_ap]

    # Load or compute cluster info dynamically for the 300s window
    info_file = alf_dir / "cluster_info.tsv"
    if info_file.exists():
        info_df = pd.read_csv(info_file, sep="\t")
    else:
        unit_ids = np.unique(spikes_clusters)
        rows = []
        for uid in unit_ids:
            u_spks = spikes_times[spikes_clusters == uid]
            n_spk = len(u_spks)
            fr = n_spk / MAX_TIME_S
            
            if uid < len(templates):
                tmpl = templates[uid]
                ptp = np.ptp(tmpl, axis=0)
                best_ch = int(np.argmax(ptp))
            else:
                best_ch = 0

            depth = float(channel_coords[best_ch, 1])
            shank = get_shank(channel_coords[best_ch, 0])

            if n_spk > 1:
                ibis = np.diff(u_spks)
                isi_v = float(np.sum(ibis < 0.0015) / max(1, len(ibis)))
            else:
                isi_v = 0.0

            group = 'good' if isi_v <= 0.02 else 'mua'
            rows.append({
                'cluster_id': uid, 'ch': best_ch, 'shank': shank, 'depth': depth,
                'n_spikes': n_spk, 'firing_rate': fr, 'isi_violations_ratio': isi_v, 'group': group
            })
        info_df = pd.DataFrame(rows)

    # Re-calculate 300s firing rate for accurate ranking in the 0-300s window
    spk_counts_300s = {}
    for uid in info_df['cluster_id']:
        spk_counts_300s[uid] = np.sum(spikes_clusters == uid)
    
    info_df['n_spikes_300s'] = info_df['cluster_id'].map(spk_counts_300s)
    info_df['firing_rate_300s'] = info_df['n_spikes_300s'] / MAX_TIME_S

    # Filter for 'good' units and take top 100 by 300s firing rate
    good_df = info_df[info_df['group'] == 'good'].copy()
    top100_df = good_df.sort_values(by='firing_rate_300s', ascending=False).head(100).reset_index(drop=True)

    # Sort units vertically by shank then depth for spatial organization
    top100_df = top100_df.sort_values(by=['shank', 'depth'], ascending=[True, True]).reset_index(drop=True)

    print(f"\nSelected Top 100 Good Units across Shanks 0-3 (First 300 Seconds):")
    print(top100_df[['cluster_id', 'ch', 'shank', 'depth', 'n_spikes_300s', 'firing_rate_300s']].head(20).to_string(index=False))

    # 2. Load NIDQ Diaphragm EMG & Laser TTL for first 300 seconds
    nidq_meta = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.meta"
    nidq_bin = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.bin"
    nidq_data = read_nidq_bin(nidq_bin, nidq_meta)

    t_nidq = nidq_data.time_vector
    fs_nidq = nidq_data.fs

    # Slice NIDQ data to first 300 seconds
    nidq_mask = (t_nidq >= 0.0) & (t_nidq <= MAX_TIME_S)
    t_nidq = t_nidq[nidq_mask]
    emg_raw = nidq_data.all_channels[0][nidq_mask]
    ch6_laser = nidq_data.all_channels[6][nidq_mask]

    # Process integrated EMG envelope
    emg_cfg = EMGConfig(fs=fs_nidq, highpass_cutoff=300.0, lowpass_cutoff=3000.0, notch_freq=60.0, smoothing_window_ms=20.0)
    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    emg_env = emg_proc.envelope
    breath_onsets = emg_proc.breath_onsets

    # Detect laser pulse onset and offset times for translucent shading
    laser_active = (ch6_laser > 0.8).astype(int)
    laser_onsets = t_nidq[np.where(np.diff(laser_active) == 1)[0]]
    laser_offsets = t_nidq[np.where(np.diff(laser_active) == -1)[0]]

    # ---------------------------------------------------------
    # 3. Create Interactive Plotly HTML Visualization (2 Panels)
    # ---------------------------------------------------------
    ds_nidq = 5
    t_nidq_ds = t_nidq[::ds_nidq]
    emg_env_ds = emg_env[::ds_nidq]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            "1. Integrated Diaphragm EMG Envelope & Breath Onset Triggers (blue triangles) [Translucent Magenta = Laser Pulses]",
            "2. Top 100 Single-Unit Spike Raster Plot (Sorted by Shank & Depth) [Translucent Magenta = Laser Pulses]"
        ),
        row_heights=[0.25, 0.75]
    )

    # Panel 1: Integrated EMG
    fig.add_trace(
        go.Scatter(
            x=t_nidq_ds,
            y=emg_env_ds,
            mode='lines',
            name='Integrated EMG (V)',
            line=dict(color='#2ca02c', width=1.5)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=breath_onsets,
            y=np.zeros(len(breath_onsets)),
            mode='markers',
            name='Inspiration Onsets',
            marker=dict(color='blue', size=6, symbol='triangle-up')
        ),
        row=1, col=1
    )

    # Add Translucent Laser Shaded Regions (vrect) to both Subplots
    for t_on, t_off in zip(laser_onsets, laser_offsets):
        for row_i in [1, 2]:
            fig.add_vrect(
                x0=t_on, x1=t_off,
                fillcolor="magenta", opacity=0.22,
                line_width=0, layer="below",
                row=row_i, col=1
            )

    # Panel 2: Population Spike Raster Plot (100 Units)
    shank_colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#9467bd', 3: '#d62728'}
    
    tick_y = []
    tick_text = []

    for y_idx, row in top100_df.iterrows():
        uid = int(row['cluster_id'])
        ch = int(row['ch'])
        sh = int(row['shank'])
        depth = float(row['depth'])
        fr = float(row['firing_rate_300s'])
        is_manual = ch in MANUAL_CHANNELS

        u_spks = spikes_times[spikes_clusters == uid]
        color = shank_colors.get(sh, '#333333')

        fig.add_trace(
            go.Scatter(
                x=u_spks,
                y=np.full(len(u_spks), y_idx + 1),
                mode='markers',
                name=f"Unit {uid} (Shank {sh})",
                marker=dict(color=color, size=3.0, symbol='line-ns-open'),
                showlegend=False
            ),
            row=2, col=1
        )

        manual_star = " ★" if is_manual else ""
        tick_y.append(y_idx + 1)
        tick_text.append(f"U{uid} (S{sh}, {depth:.0f}µm, {fr:.1f}Hz){manual_star}")

    fig.update_layout(
        title="<b>Subject TP_Tacr1_1: Top 100 Kilosort4 Good Single Units — Population Spike Raster & Integrated EMG (0-300s)</b>",
        template="plotly_white",
        height=1350,
        margin=dict(l=140, r=40, t=80, b=60),
        yaxis2=dict(
            tickmode='array',
            tickvals=tick_y,
            ticktext=tick_text,
            title="Unit ID (Shank, Depth, Firing Rate)",
            autorange="reversed",
            tickfont=dict(size=8)
        ),
        xaxis2=dict(title="Time in Recording (seconds)", type="linear")
    )
    fig.update_xaxes(range=[0.0, 300.0])

    html_out = out_dir / "population_raster_top100_300s.html"
    fig.write_html(str(html_out))
    print(f"\n -> Saved Interactive HTML Raster Plot (Top 100 | 0-300s): {html_out}")

    # ---------------------------------------------------------
    # 4. Create High-Resolution Static PNG Figure (Matplotlib)
    # ---------------------------------------------------------
    print("Generating high-resolution static PNG figure (Top 100 | 0-300s)...")
    fig_mpl, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 17), sharex=True, gridspec_kw={'height_ratios': [1.2, 5.0]})

    # Laser Shading Spans on both plots
    for t_on, t_off in zip(laser_onsets, laser_offsets):
        for ax in (ax1, ax2):
            ax.axvspan(t_on, t_off, color='magenta', alpha=0.22, zorder=0)

    # Panel 1: Integrated EMG
    ax1.plot(t_nidq_ds, emg_env_ds, color='#2ca02c', linewidth=1.2, label='Integrated EMG (V)')
    ax1.plot(breath_onsets, np.zeros(len(breath_onsets)), '^', color='blue', markersize=4, label='Breath Onsets')
    ax1.set_ylabel("Integrated EMG (V)", fontsize=10, fontweight='bold')
    ax1.set_title("Subject TP_Tacr1_1: Top 100 Kilosort4 Good Units — Population Spike Raster & Respiratory Timing (0-300s)", fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Panel 2: Raster Plot (Top 100)
    for y_idx, row in top100_df.iterrows():
        uid = int(row['cluster_id'])
        sh = int(row['shank'])
        color = shank_colors.get(sh, '#333333')
        u_spks = spikes_times[spikes_clusters == uid]

        ax2.vlines(u_spks, y_idx + 0.6, y_idx + 1.4, color=color, linewidth=0.7, alpha=0.85)

    ax2.set_yticks(tick_y)
    ax2.set_yticklabels(tick_text, fontsize=7)
    ax2.set_ylabel("Unit ID (Shank, Depth, Firing Rate)", fontsize=10, fontweight='bold')
    ax2.set_xlabel("Time in Recording (seconds)", fontsize=11, fontweight='bold')
    ax2.set_ylim(0.5, len(top100_df) + 0.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0.0, 300.0)
    ax2.grid(True, axis='x', linestyle=':', alpha=0.5)

    plt.tight_layout()
    png_out = out_dir / "population_raster_top100_300s.png"
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved High-Res PNG Raster Plot: {png_out}")

    # Copy PNG to artifact folder for markdown embedding
    artifact_png = artifact_dir / "population_raster_top100_300s.png"
    shutil.copy2(png_out, artifact_png)

    print("\n" + "=" * 85)
    print(" [SUCCESS] Top 100 Population Spike Raster & EMG Visualizer Complete!")
    print(f" HTML:     {html_out}")
    print(f" PNG:      {png_out}")
    print(f" Artifact: {artifact_png}")
    print("=" * 85)

if __name__ == "__main__":
    main()
