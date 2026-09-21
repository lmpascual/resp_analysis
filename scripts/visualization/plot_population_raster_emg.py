#!/usr/bin/env python3
"""
Generate Population Spike Raster Plot & Diaphragm EMG Respiration Visualizer.
Plots top 40 Kilosort4 'good' single units alongside integrated Diaphragm EMG
and Optogenetic Laser TTL pulses. Outputs both interactive HTML and static PNG.
"""

import sys
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

    print("=" * 85)
    print(" Generating Population Spike Raster (Top 40 Units) & Integrated EMG")
    print(f" ALF Directory: {alf_dir}")
    print(f" Output Directory: {out_dir}")
    print("=" * 85)

    # 1. Load Kilosort4 spike data & channel geometry
    spikes_times = np.load(alf_dir / "spike_times.npy")
    spikes_clusters = np.load(alf_dir / "spike_clusters.npy")
    templates = np.load(alf_dir / "templates.npy")
    channel_coords = np.load(alf_dir / "channel_positions.npy")

    # If spike_times is in samples (fs=30000), convert to seconds
    fs_ap = 30000.0
    if spikes_times.dtype in [np.int64, np.int32] and spikes_times.max() > 100000:
        spikes_times = spikes_times / fs_ap

    rec_duration = spikes_times.max()

    # Load cluster_info.tsv if exists, otherwise compute dynamically
    info_file = alf_dir / "cluster_info.tsv"
    if info_file.exists():
        info_df = pd.read_csv(info_file, sep="\t")
    else:
        logger_str = "Computing unit properties dynamically from Kilosort output..."
        print(logger_str)
        unit_ids = np.unique(spikes_clusters)
        rows = []
        for uid in unit_ids:
            u_spks = spikes_times[spikes_clusters == uid]
            n_spk = len(u_spks)
            fr = n_spk / rec_duration
            
            # Template best channel
            if uid < len(templates):
                tmpl = templates[uid]
                ptp = np.ptp(tmpl, axis=0)
                best_ch = int(np.argmax(ptp))
            else:
                best_ch = 0

            depth = float(channel_coords[best_ch, 1])
            shank = get_shank(channel_coords[best_ch, 0])

            # ISI violation ratio (< 1.5 ms)
            if n_spk > 1:
                ibis = np.diff(u_spks)
                isi_v = float(np.sum(ibis < 0.0015) / max(1, len(ibis)))
            else:
                isi_v = 0.0

            group = 'good' if isi_v <= 0.02 else 'mua'

            rows.append({
                'cluster_id': uid,
                'ch': best_ch,
                'shank': shank,
                'depth': depth,
                'n_spikes': n_spk,
                'firing_rate': fr,
                'isi_violations_ratio': isi_v,
                'group': group
            })
        info_df = pd.DataFrame(rows)
        info_df.to_csv(info_file, sep="\t", index=False)

    # Filter for 'good' units and select top 40 by firing rate
    good_df = info_df[info_df['group'] == 'good'].copy()
    top40_df = good_df.sort_values(by='firing_rate', ascending=False).head(40).reset_index(drop=True)

    # Sort units vertically by shank then depth for physical spatial organization
    top40_df = top40_df.sort_values(by=['shank', 'depth'], ascending=[True, True]).reset_index(drop=True)

    print(f"\nSelected Top 40 Good Units across Shanks 0-3:")
    print(top40_df[['cluster_id', 'ch', 'shank', 'depth', 'n_spikes', 'firing_rate']].to_string(index=False))

    # 2. Load NIDQ Diaphragm EMG & Laser TTL
    nidq_meta = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.meta"
    nidq_bin = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.bin"
    nidq_data = read_nidq_bin(nidq_bin, nidq_meta)

    t_nidq = nidq_data.time_vector
    fs_nidq = nidq_data.fs
    emg_raw = nidq_data.all_channels[0]
    ch6_laser = nidq_data.all_channels[6]

    # Process integrated EMG envelope
    emg_cfg = EMGConfig(fs=fs_nidq, highpass_cutoff=300.0, lowpass_cutoff=3000.0, notch_freq=60.0, smoothing_window_ms=20.0)
    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    emg_env = emg_proc.envelope
    breath_onsets = emg_proc.breath_onsets

    # Detect laser pulses for shading
    laser_active = (ch6_laser > 0.8).astype(int)
    laser_onsets = t_nidq[np.where(np.diff(laser_active) == 1)[0]]
    laser_offsets = t_nidq[np.where(np.diff(laser_active) == -1)[0]]

    # ---------------------------------------------------------
    # 3. Create Interactive Plotly HTML Visualization
    # ---------------------------------------------------------
    ds_nidq = 5
    t_nidq_ds = t_nidq[::ds_nidq]
    emg_env_ds = emg_env[::ds_nidq]
    laser_ds = (ch6_laser[::ds_nidq] > 0.8).astype(float)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            "1. Integrated Diaphragm EMG Envelope & Breath Onset Triggers (blue triangles)",
            "2. Optogenetic Laser Stimulation TTL (Channel 6)",
            "3. Top 40 Single-Unit Spike Raster Plot (Sorted by Shank & Depth)"
        ),
        row_heights=[0.25, 0.15, 0.60]
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

    # Panel 2: Laser TTL
    fig.add_trace(
        go.Scatter(
            x=t_nidq_ds,
            y=laser_ds,
            mode='lines',
            name='Laser TTL',
            line=dict(color='magenta', width=1.5)
        ),
        row=2, col=1
    )

    # Panel 3: Population Spike Raster Plot
    shank_colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#9467bd', 3: '#d62728'}
    
    tick_y = []
    tick_text = []

    for y_idx, row in top40_df.iterrows():
        uid = int(row['cluster_id'])
        ch = int(row['ch'])
        sh = int(row['shank'])
        depth = float(row['depth'])
        fr = float(row['firing_rate'])
        is_manual = ch in MANUAL_CHANNELS

        u_spks = spikes_times[spikes_clusters == uid]
        color = shank_colors.get(sh, '#333333')

        fig.add_trace(
            go.Scatter(
                x=u_spks,
                y=np.full(len(u_spks), y_idx + 1),
                mode='markers',
                name=f"Unit {uid} (Shank {sh})",
                marker=dict(color=color, size=3.5, symbol='line-ns-open'),
                showlegend=False
            ),
            row=3, col=1
        )

        manual_star = " ★" if is_manual else ""
        tick_y.append(y_idx + 1)
        tick_text.append(f"U{uid} (S{sh}, {depth:.0f}µm, {fr:.1f}Hz){manual_star}")

    fig.update_layout(
        title="<b>Subject TP_Tacr1_1: Top 40 Kilosort4 Good Single Units — Population Spike Raster & Integrated EMG</b>",
        template="plotly_white",
        height=950,
        margin=dict(l=140, r=40, t=80, b=60),
        yaxis3=dict(
            tickmode='array',
            tickvals=tick_y,
            ticktext=tick_text,
            title="Unit ID (Shank, Depth, Firing Rate)",
            autorange="reversed"
        ),
        xaxis3=dict(title="Time in Recording (seconds)", type="linear")
    )
    fig.update_xaxes(range=[60.0, 75.0]) # Default 15-second zoom window

    html_out = out_dir / "population_raster_emg.html"
    fig.write_html(str(html_out))
    print(f"\n -> Saved Interactive HTML Raster Plot: {html_out}")

    # ---------------------------------------------------------
    # 4. Create High-Resolution Static PNG Figure (Matplotlib)
    # ---------------------------------------------------------
    print("Generating high-resolution static PNG figure...")
    fig_mpl, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 11), sharex=True, gridspec_kw={'height_ratios': [1.2, 0.6, 3.5]})

    # Laser Shading Spans
    for t_on, t_off in zip(laser_onsets, laser_offsets):
        for ax in (ax1, ax2, ax3):
            ax.axvspan(t_on, t_off, color='magenta', alpha=0.15, zorder=0)

    # Panel 1: Integrated EMG
    ax1.plot(t_nidq_ds, emg_env_ds, color='#2ca02c', linewidth=1.2, label='Integrated EMG (V)')
    ax1.plot(breath_onsets, np.zeros(len(breath_onsets)), '^', color='blue', markersize=4, label='Breath Onsets')
    ax1.set_ylabel("Integrated EMG (V)", fontsize=10, fontweight='bold')
    ax1.set_title("Subject TP_Tacr1_1: Population Spike Raster (Top 40 Kilosort4 Good Units) & Respiratory Timing", fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Panel 2: Laser TTL
    ax2.plot(t_nidq_ds, laser_ds, color='magenta', linewidth=1.2, label='Laser TTL')
    ax2.set_ylabel("Laser TTL", fontsize=10, fontweight='bold')
    ax2.set_ylim(-0.1, 1.2)
    ax2.grid(True, linestyle=':', alpha=0.5)

    # Panel 3: Raster Plot
    for y_idx, row in top40_df.iterrows():
        uid = int(row['cluster_id'])
        sh = int(row['shank'])
        color = shank_colors.get(sh, '#333333')
        u_spks = spikes_times[spikes_clusters == uid]

        ax3.vlines(u_spks, y_idx + 0.6, y_idx + 1.4, color=color, linewidth=0.9, alpha=0.85)

    ax3.set_yticks(tick_y)
    ax3.set_yticklabels(tick_text, fontsize=8)
    ax3.set_ylabel("Unit ID (Shank, Depth, Firing Rate)", fontsize=10, fontweight='bold')
    ax3.set_xlabel("Time in Recording (seconds)", fontsize=11, fontweight='bold')
    ax3.set_ylim(0.5, len(top40_df) + 0.5)
    ax3.invert_yaxis()
    ax3.set_xlim(60.0, 75.0) # 15-second window default
    ax3.grid(True, axis='x', linestyle=':', alpha=0.5)

    plt.tight_layout()
    png_out = out_dir / "population_raster_emg.png"
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved High-Res PNG Raster Plot: {png_out}")

    print("\n" + "=" * 85)
    print(" [SUCCESS] Population Spike Raster & EMG Visualizer Complete!")
    print(f" HTML: {html_out}")
    print(f" PNG:  {png_out}")
    print("=" * 85)

if __name__ == "__main__":
    main()
