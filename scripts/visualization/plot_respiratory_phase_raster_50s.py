#!/usr/bin/env python3
"""
Generate Population Spike Raster Plot & Diaphragm EMG Respiration Visualizer
for 50 Non-Redundant Good Units evenly distributed across all 4 Shanks (Shanks 0, 1, 2, 3):
- Exactly 10 Tonic Units + 40 Phasic Units (Breath Cycle Preferences)
- Optogenetic laser pulses are shaded in vibrant translucent magenta (opacity 0.45) on both plots.
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
    print(" Generating 4-Shank Population Spike Raster (40 Phasic Units + 10 Tonic Units | 0-300s)")
    print(f" ALF Directory: {alf_dir}")
    print(f" Output Directory: {out_dir}")
    print("=" * 85)

    # 1. Load Kilosort4 spike data, cluster info, templates, & channel geometry
    spikes_times = np.load(alf_dir / "spike_times.npy")
    spikes_clusters = np.load(alf_dir / "spike_clusters.npy")
    templates = np.load(alf_dir / "templates.npy")
    template_ind = np.load(alf_dir / "template_ind.npy")
    channel_coords = np.load(alf_dir / "channel_positions.npy")
    info_df = pd.read_csv(alf_dir / "cluster_info.tsv", sep="\t")

    fs_ap = 30000.0
    if spikes_times.dtype in [np.int64, np.int32] and spikes_times.max() > 100000:
        spikes_times = spikes_times / fs_ap

    MAX_TIME_S = 300.0
    spk_mask = (spikes_times >= 0.0) & (spikes_times <= MAX_TIME_S)
    spikes_times = spikes_times[spk_mask]
    spikes_clusters = spikes_clusters[spk_mask]

    # Re-calculate 300s firing rate
    spk_counts_300s = {}
    for uid in info_df['cluster_id']:
        spk_counts_300s[uid] = np.sum(spikes_clusters == uid)
    
    info_df['n_spikes_300s'] = info_df['cluster_id'].map(spk_counts_300s)
    info_df['firing_rate_300s'] = info_df['n_spikes_300s'] / MAX_TIME_S

    # 2. Load NIDQ Diaphragm EMG & Laser TTL for 0-300s
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
    laser_offsets = t_nidq[np.where(np.diff(laser_active) == -1)[0]]

    # 3. Filter for Clean Good Units (group == 'good' and firing_rate_300s >= 1.0)
    good_df = info_df[(info_df['group'] == 'good') & (info_df['firing_rate_300s'] >= 1.0)].copy().reset_index(drop=True)
    spiketrains = {u: np.sort(spikes_times[spikes_clusters == u]) for u in good_df['cluster_id']}

    # 4. Quantify Respiratory Phase Vector Strength (R) for each good unit
    cycle_starts = breath_onsets[:-1]
    cycle_ends = breath_onsets[1:]
    cycle_durations = cycle_ends - cycle_starts

    vector_strengths = []
    phase_centroids = []
    phase_types = []

    for idx, row in good_df.iterrows():
        uid = int(row['cluster_id'])
        u_spks = spiketrains[uid]
        
        phases = []
        for spk in u_spks:
            i = np.searchsorted(cycle_starts, spk) - 1
            if 0 <= i < len(cycle_starts):
                if spk < cycle_ends[i]:
                    phases.append(2.0 * np.pi * (spk - cycle_starts[i]) / cycle_durations[i])
        
        if len(phases) < 15:
            vector_strengths.append(0.0)
            phase_centroids.append(0.0)
            phase_types.append('Tonic')
            continue

        phases = np.array(phases)
        cos_m = np.mean(np.cos(phases))
        sin_m = np.mean(np.sin(phases))
        R = np.sqrt(cos_m**2 + sin_m**2)
        vector_strengths.append(R)

        centroid = np.arctan2(sin_m, cos_m) % (2.0 * np.pi)
        phase_centroids.append(centroid)

        if R < 0.15:
            phase_types.append('Tonic')
        else:
            if centroid >= 1.70 * np.pi or centroid < 0.20 * np.pi:
                phase_types.append('Pre-Inspiration (Pre-I)')
            elif centroid < 0.70 * np.pi:
                phase_types.append('Inspiration (I)')
            elif centroid < 1.20 * np.pi:
                phase_types.append('Post-Inspiration (Post-I)')
            else:
                phase_types.append('Expiration (E2)')

    good_df['vector_strength'] = vector_strengths
    good_df['phase_centroid'] = phase_centroids
    good_df['phase_type'] = phase_types

    # 5. Selection Strategy:
    # - Select 10 Phasic Units per Shank across all 4 Shanks (Shanks 0, 1, 2, 3) = 40 Phasic Units
    # - Select 10 Tonic Units distributed across Shanks = 10 Tonic Units
    phasic_parts = []
    for sh in [0, 1, 2, 3]:
        sub_sh = good_df[(good_df['shank'] == sh) & (good_df['vector_strength'] >= 0.15)].sort_values(by='vector_strength', ascending=False).head(10)
        phasic_parts.append(sub_sh)

    top40_phasic = pd.concat(phasic_parts).reset_index(drop=True)

    # If any shank had fewer than 10 phasic units, backfill with next highest phasic units
    if len(top40_phasic) < 40:
        needed = 40 - len(top40_phasic)
        extra_phasic = good_df[(good_df['vector_strength'] >= 0.15) & (~good_df['cluster_id'].isin(top40_phasic['cluster_id']))].sort_values(by='vector_strength', ascending=False).head(needed)
        top40_phasic = pd.concat([top40_phasic, extra_phasic]).reset_index(drop=True)

    # Select 10 Tonic Units spread across shanks
    tonic_parts = []
    for sh in [0, 1, 2, 3]:
        sub_t = good_df[(good_df['shank'] == sh) & (good_df['vector_strength'] < 0.15)].sort_values(by='firing_rate_300s', ascending=False).head(3)
        tonic_parts.append(sub_t)

    top10_tonic = pd.concat(tonic_parts).sort_values(by='firing_rate_300s', ascending=False).head(10).reset_index(drop=True)

    selected_50 = pd.concat([top40_phasic, top10_tonic]).drop_duplicates(subset=['cluster_id']).reset_index(drop=True)

    # Sort units vertically by Shank (0, 1, 2, 3) then Depth for spatial organization
    selected_50 = selected_50.sort_values(by=['shank', 'depth'], ascending=[True, True]).reset_index(drop=True)

    print(f"\nFinal Selected 50 Representative Units (across all 4 Shanks 0-3):")
    print(selected_50[['cluster_id', 'ch', 'shank', 'depth', 'firing_rate_300s', 'vector_strength', 'phase_type']].to_string(index=False))

    print("\nRepresentation per Shank:")
    print(selected_50['shank'].value_counts().to_string())

    # ---------------------------------------------------------
    # 6. Create Interactive Plotly HTML Visualization (2 Panels)
    # ---------------------------------------------------------
    ds_nidq = 5
    t_nidq_ds = t_nidq[::ds_nidq]
    emg_env_ds = emg_env[::ds_nidq]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            "1. Integrated Diaphragm EMG Envelope & Breath Onset Triggers (blue triangles) [Vibrant Magenta = Laser Pulses]",
            "2. 4-Shank 50 Single-Unit Spike Raster Plot (Shanks 0, 1, 2, 3 | 40 Phasic + 10 Tonic) [Vibrant Magenta = Laser Pulses]"
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

    # Add Vibrant Magenta Shaded Laser Regions (opacity 0.45)
    for t_on, t_off in zip(laser_onsets, laser_offsets):
        for row_i in [1, 2]:
            fig.add_vrect(
                x0=t_on, x1=t_off,
                fillcolor="magenta", opacity=0.45,
                line_width=0, layer="below",
                row=row_i, col=1
            )

    # Panel 2: Population Spike Raster Plot (40 Phasic + 10 Tonic across 4 Shanks)
    shank_colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#9467bd', 3: '#d62728'}

    tick_y = []
    tick_text = []

    for y_idx, row in selected_50.iterrows():
        uid = int(row['cluster_id'])
        ch = int(row['ch'])
        sh = int(row['shank'])
        depth = float(row['depth'])
        fr = float(row['firing_rate_300s'])
        score = float(row['vector_strength'])
        ptype = str(row['phase_type'])
        is_manual = ch in MANUAL_CHANNELS

        u_spks = spiketrains[uid]
        color = shank_colors.get(sh, '#333333')

        fig.add_trace(
            go.Scatter(
                x=u_spks,
                y=np.full(len(u_spks), y_idx + 1),
                mode='markers',
                name=f"U{uid} (Shank {sh})",
                marker=dict(color=color, size=3.5, symbol='line-ns-open'),
                showlegend=False
            ),
            row=2, col=1
        )

        manual_star = " ★" if is_manual else ""
        tick_y.append(y_idx + 1)
        tick_text.append(f"U{uid} [S{sh}, {depth:.0f}µm] ({ptype}, {fr:.1f}Hz, R={score:.2f}){manual_star}")

    fig.update_layout(
        title="<b>Subject TP_Tacr1_1: 4-Shank 50 Single Units (40 Phasic + 10 Tonic) — Population Spike Raster & Integrated EMG (0-300s)</b>",
        template="plotly_white",
        height=1250,
        margin=dict(l=240, r=40, t=80, b=60),
        yaxis2=dict(
            tickmode='array',
            tickvals=tick_y,
            ticktext=tick_text,
            title="Unit ID [Shank, Depth] (Respiratory Category, Firing Rate, Vector Strength R)",
            autorange="reversed",
            tickfont=dict(size=8)
        ),
        xaxis2=dict(title="Time in Recording (seconds)", type="linear")
    )
    fig.update_xaxes(range=[0.0, 300.0])

    html_out = out_dir / "population_raster_50units_phase_diverse.html"
    fig.write_html(str(html_out))
    print(f"\n -> Saved Interactive HTML Raster Plot (4-Shank 50 Units | 0-300s): {html_out}")

    # ---------------------------------------------------------
    # 7. Create High-Resolution Static PNG Figure (Matplotlib)
    # ---------------------------------------------------------
    print("Generating high-resolution static PNG figure (4-Shank 50 Units | 0-300s)...")
    fig_mpl, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 15), sharex=True, gridspec_kw={'height_ratios': [1.2, 4.5]})

    # Vibrant laser shading (alpha=0.45)
    for t_on, t_off in zip(laser_onsets, laser_offsets):
        for ax in (ax1, ax2):
            ax.axvspan(t_on, t_off, color='magenta', alpha=0.45, zorder=0)

    # Panel 1: Integrated EMG
    ax1.plot(t_nidq_ds, emg_env_ds, color='#2ca02c', linewidth=1.2, label='Integrated EMG (V)')
    ax1.plot(breath_onsets, np.zeros(len(breath_onsets)), '^', color='blue', markersize=4, label='Breath Onsets')
    ax1.set_ylabel("Integrated EMG (V)", fontsize=10, fontweight='bold')
    ax1.set_title("Subject TP_Tacr1_1: 4-Shank 50 Single Units (40 Phasic + 10 Tonic) — Population Spike Raster & Respiratory Timing (0-300s)", fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Panel 2: Raster Plot (4 Shanks color-coded)
    for y_idx, row in selected_50.iterrows():
        uid = int(row['cluster_id'])
        sh = int(row['shank'])
        color = shank_colors.get(sh, '#333333')
        u_spks = spiketrains[uid]

        ax2.vlines(u_spks, y_idx + 0.6, y_idx + 1.4, color=color, linewidth=0.8, alpha=0.85)

    ax2.set_yticks(tick_y)
    ax2.set_yticklabels(tick_text, fontsize=7.5)
    ax2.set_ylabel("Unit ID [Shank, Depth] (Respiratory Category, Firing Rate, Vector Strength R)", fontsize=10, fontweight='bold')
    ax2.set_xlabel("Time in Recording (seconds)", fontsize=11, fontweight='bold')
    ax2.set_ylim(0.5, len(selected_50) + 0.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0.0, 300.0)
    ax2.grid(True, axis='x', linestyle=':', alpha=0.5)

    plt.tight_layout()
    png_out = out_dir / "population_raster_50units_phase_diverse.png"
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved High-Res PNG Raster Plot: {png_out}")

    # Copy PNG to artifact folder for markdown embedding
    artifact_png = artifact_dir / "population_raster_50units_phase_diverse.png"
    shutil.copy2(png_out, artifact_png)

    print("\n" + "=" * 85)
    print(" [SUCCESS] 4-Shank 50 Units Population Spike Raster & EMG Visualizer Complete!")
    print(f" HTML:     {html_out}")
    print(f" PNG:      {png_out}")
    print(f" Artifact: {artifact_png}")
    print("=" * 85)

if __name__ == "__main__":
    main()
