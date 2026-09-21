#!/usr/bin/env python3
"""
Tacr1-Ai32 Optogenetic Phototagging Analysis & Population Raster Visualizer.
- Clean color scheme: All standard units in BLACK, Phototagged Tacr1+ units in BRIGHT RED.
- Compact Panel 2 for Laser TTL.
- Horizontal dividing lines separating units by Shank identity (Shanks 0, 1, 2, 3).
- Optogenetic laser pulses shaded in vibrant translucent magenta (opacity 0.45).
Outputs both an interactive Plotly HTML dashboard and a high-res static PNG figure.
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
    print(" Running Clean Tacr1-Ai32 Phototagging Raster (Black/Red, Shank Separators, Compact TTL)")
    print(f" ALF Directory: {alf_dir}")
    print(f" Output Directory: {out_dir}")
    print("=" * 85)

    # 1. Load Kilosort4 spike data & channel geometry
    spikes_times = np.load(alf_dir / "spike_times.npy")
    spikes_clusters = np.load(alf_dir / "spike_clusters.npy")
    templates = np.load(alf_dir / "templates.npy")
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

    n_pulses = min(len(laser_onsets), len(laser_offsets))
    laser_onsets = laser_onsets[:n_pulses]
    laser_offsets = laser_offsets[:n_pulses]
    pulse_durations = laser_offsets - laser_onsets

    # 3. Filter for Clean Good Units & Deduplicate
    good_df = info_df[(info_df['group'] == 'good') & (info_df['firing_rate_300s'] >= 0.5)].copy().reset_index(drop=True)
    spiketrains = {u: np.sort(spikes_times[spikes_clusters == u]) for u in good_df['cluster_id']}

    win_s = 0.0005
    discard_uids = set()
    good_uids = list(spiketrains.keys())

    for i in range(len(good_uids)):
        u1 = good_uids[i]
        if u1 in discard_uids: continue
        st1 = spiketrains[u1]
        n1 = len(st1)

        for j in range(i + 1, len(good_uids)):
            u2 = good_uids[j]
            if u2 in discard_uids: continue
            st2 = spiketrains[u2]
            n2 = len(st2)

            left = np.searchsorted(st2, st1 - win_s)
            right = np.searchsorted(st2, st1 + win_s)
            n_coinc = np.sum(right > left)
            if n_coinc > 0:
                if n_coinc / n2 >= 0.50 and n1 > n2:
                    discard_uids.add(u2)
                elif n_coinc / n1 >= 0.50 and n2 >= n1:
                    discard_uids.add(u1)
                    break

    clean_good_df = good_df[~good_df['cluster_id'].isin(discard_uids)].copy().reset_index(drop=True)

    # 4. Phototagging & Respiratory Vector Strength
    cycle_starts = breath_onsets[:-1]
    cycle_ends = breath_onsets[1:]
    cycle_durations = cycle_ends - cycle_starts

    phototagged_flags = []
    laser_ratios = []
    latencies = []
    vector_strengths = []
    phase_types = []

    for idx, row in clean_good_df.iterrows():
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
        latencies.append(round(mean_lat, 2))

        phases = []
        for spk in u_spks:
            i = np.searchsorted(cycle_starts, spk) - 1
            if 0 <= i < len(cycle_starts):
                if spk < cycle_ends[i]:
                    phases.append(2.0 * np.pi * (spk - cycle_starts[i]) / cycle_durations[i])

        if len(phases) < 15:
            vector_strengths.append(0.0)
            phase_types.append('Tonic')
        else:
            phases = np.array(phases)
            cos_m = np.mean(np.cos(phases))
            sin_m = np.mean(np.sin(phases))
            R = np.sqrt(cos_m**2 + sin_m**2)
            vector_strengths.append(R)

            if R < 0.15:
                phase_types.append('Tonic')
            else:
                centroid = np.arctan2(sin_m, cos_m) % (2.0 * np.pi)
                if centroid >= 1.70 * np.pi or centroid < 0.20 * np.pi:
                    phase_types.append('Pre-I')
                elif centroid < 0.70 * np.pi:
                    phase_types.append('Inspiration')
                elif centroid < 1.20 * np.pi:
                    phase_types.append('Post-I')
                else:
                    phase_types.append('Expiration')

    clean_good_df['is_phototagged'] = phototagged_flags
    clean_good_df['laser_ratio'] = laser_ratios
    clean_good_df['latency_ms'] = latencies
    clean_good_df['vector_strength'] = vector_strengths
    clean_good_df['phase_type'] = phase_types

    # 5. Select 50 Units (balanced across Phototagged, Phasic, Tonic)
    tagged_sub = clean_good_df[clean_good_df['is_phototagged']].sort_values(by='laser_ratio', ascending=False).head(15)

    phasic_sub = clean_good_df[(~clean_good_df['cluster_id'].isin(tagged_sub['cluster_id'])) & (clean_good_df['vector_strength'] >= 0.15)].sort_values(by='vector_strength', ascending=False).head(25)

    tonic_sub = clean_good_df[(~clean_good_df['cluster_id'].isin(tagged_sub['cluster_id'])) & (~clean_good_df['cluster_id'].isin(phasic_sub['cluster_id'])) & (clean_good_df['vector_strength'] < 0.15)].sort_values(by='firing_rate_300s', ascending=False).head(10)

    selected_50 = pd.concat([tagged_sub, phasic_sub, tonic_sub]).drop_duplicates(subset=['cluster_id']).reset_index(drop=True)
    
    # Sort units strictly by Shank (0, 1, 2, 3) then Depth
    selected_50 = selected_50.sort_values(by=['shank', 'depth'], ascending=[True, True]).reset_index(drop=True)

    print(f"\nFinal 50 Selected Units (Sorted by Shank 0-3 then Depth):")
    print(selected_50[['cluster_id', 'ch', 'shank', 'depth', 'firing_rate_300s', 'is_phototagged', 'laser_ratio', 'latency_ms', 'phase_type']].to_string(index=False))

    # Identify Y-index boundary positions between Shanks for drawing horizontal dividing lines
    shank_boundary_y = []
    shanks_present = selected_50['shank'].values
    for idx_pos in range(len(shanks_present) - 1):
        if shanks_present[idx_pos] != shanks_present[idx_pos + 1]:
            # Midpoint between y=idx_pos+1 and y=idx_pos+2
            shank_boundary_y.append((idx_pos + 1.5, shanks_present[idx_pos], shanks_present[idx_pos + 1]))

    # ---------------------------------------------------------
    # 6. Create Interactive Plotly HTML Visualization (3 Panels)
    # ---------------------------------------------------------
    ds_nidq = 5
    t_nidq_ds = t_nidq[::ds_nidq]
    emg_env_ds = emg_env[::ds_nidq]
    laser_ds = (ch6_laser[::ds_nidq] > 0.8).astype(float)

    # Panel 2 is made compact with row_heights=[0.22, 0.06, 0.72]
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            "1. Integrated Diaphragm EMG Envelope & Breath Onsets (blue triangles) [Vibrant Magenta = Laser Pulses]",
            "2. Optogenetic Laser TTL Pulses (Channel 6)",
            "3. 50 Single-Unit Spike Raster Plot [RED = ⚡ Tacr1+ Phototagged | BLACK = Standard Units]"
        ),
        row_heights=[0.22, 0.06, 0.72]
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

    # Panel 2: Compact Laser TTL
    fig.add_trace(
        go.Scatter(
            x=t_nidq_ds,
            y=laser_ds,
            mode='lines',
            name='Laser TTL',
            line=dict(color='magenta', width=1.2)
        ),
        row=2, col=1
    )

    # Add Vibrant Magenta Shaded Laser Regions (opacity 0.45) across all subplots
    for t_on, t_off in zip(laser_onsets, laser_offsets):
        for row_i in [1, 2, 3]:
            fig.add_vrect(
                x0=t_on, x1=t_off,
                fillcolor="magenta", opacity=0.45,
                line_width=0, layer="below",
                row=row_i, col=1
            )

    # Panel 3: Population Spike Raster Plot
    # All units BLACK (#000000) except Phototagged Tacr1+ units in BRIGHT RED (#ff0000)
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
        is_tagged = bool(row['is_phototagged'])
        ratio = float(row['laser_ratio'])
        lat = float(row['latency_ms'])
        is_manual = ch in MANUAL_CHANNELS

        u_spks = spiketrains[uid]
        color = '#ff0000' if is_tagged else '#000000' # Bright RED for Phototagged, BLACK for all others

        fig.add_trace(
            go.Scatter(
                x=u_spks,
                y=np.full(len(u_spks), y_idx + 1),
                mode='markers',
                name=f"U{uid} ({'Tacr1+' if is_tagged else ptype})",
                marker=dict(color=color, size=3.8 if is_tagged else 2.8, symbol='line-ns-open'),
                showlegend=False
            ),
            row=3, col=1
        )

        manual_star = " ★" if is_manual else ""
        tag_label = f"⚡ Tacr1+ ({ratio:.0f}x, {lat:.1f}ms)" if is_tagged else f"{ptype} (R={score:.2f})"
        tick_y.append(y_idx + 1)
        tick_text.append(f"U{uid} [Shank {sh}, {depth:.0f}µm] — {tag_label}{manual_star}")

    # Add Horizontal Shank Boundary Lines in Plotly
    for b_y, sh_above, sh_below in shank_boundary_y:
        fig.add_hline(
            y=b_y,
            line=dict(color='#888888', width=1.5, dash='dash'),
            row=3, col=1
        )

    fig.update_layout(
        title="<b>Subject TP_Tacr1_1: Optogenetic Tacr1+ Phototagged & Phasic Single Units — Population Spike Raster & EMG (0-300s)</b>",
        template="plotly_white",
        height=1350,
        margin=dict(l=260, r=40, t=80, b=60),
        yaxis2=dict(title="TTL", showticklabels=False),
        yaxis3=dict(
            tickmode='array',
            tickvals=tick_y,
            ticktext=tick_text,
            title="Unit ID [Shank, Depth] — Tacr1+ Phototagged (RED) / Standard Units (BLACK)",
            autorange="reversed",
            tickfont=dict(size=8)
        ),
        xaxis3=dict(title="Time in Recording (seconds)", type="linear")
    )
    fig.update_xaxes(range=[0.0, 300.0])

    html_out = out_dir / "population_raster_tacr1_phototagged.html"
    fig.write_html(str(html_out))
    print(f"\n -> Saved Interactive HTML Raster Plot: {html_out}")

    # ---------------------------------------------------------
    # 7. Create High-Resolution Static PNG Figure (Matplotlib)
    # ---------------------------------------------------------
    print("Generating high-resolution static PNG figure...")
    fig_mpl, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 16), sharex=True, gridspec_kw={'height_ratios': [1.2, 0.3, 5.0]})

    # Vibrant laser shading (alpha=0.45)
    for t_on, t_off in zip(laser_onsets, laser_offsets):
        for ax in (ax1, ax2, ax3):
            ax.axvspan(t_on, t_off, color='magenta', alpha=0.45, zorder=0)

    # Panel 1: Integrated EMG
    ax1.plot(t_nidq_ds, emg_env_ds, color='#2ca02c', linewidth=1.2, label='Integrated EMG (V)')
    ax1.plot(breath_onsets, np.zeros(len(breath_onsets)), '^', color='blue', markersize=4, label='Breath Onsets')
    ax1.set_ylabel("Integrated EMG (V)", fontsize=10, fontweight='bold')
    ax1.set_title("Subject TP_Tacr1_1: Optogenetic Tacr1+ Phototagged & Phasic Single Units — Population Spike Raster & Respiratory Timing (0-300s)", fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Panel 2: Compact Laser TTL
    ax2.plot(t_nidq_ds, laser_ds, color='magenta', linewidth=1.2, label='Laser TTL')
    ax2.set_ylabel("Laser TTL", fontsize=9, fontweight='bold')
    ax2.set_ylim(-0.1, 1.2)
    ax2.set_yticks([0, 1])
    ax2.grid(True, linestyle=':', alpha=0.5)

    # Panel 3: Raster Plot (RED for Phototagged Tacr1+, BLACK for all others)
    for y_idx, row in selected_50.iterrows():
        uid = int(row['cluster_id'])
        is_tagged = bool(row['is_phototagged'])
        color = '#ff0000' if is_tagged else '#000000' # Bright RED for phototagged, BLACK for standard
        u_spks = spiketrains[uid]

        ax3.vlines(u_spks, y_idx + 0.6, y_idx + 1.4, color=color, linewidth=1.2 if is_tagged else 0.75, alpha=1.0 if is_tagged else 0.85)

    # Draw Horizontal Shank Boundary Lines in Matplotlib
    for b_y, sh_above, sh_below in shank_boundary_y:
        ax3.axhline(b_y, color='#444444', linestyle='--', linewidth=1.2, zorder=3)

    ax3.set_yticks(tick_y)
    ax3.set_yticklabels(tick_text, fontsize=7.5)
    ax3.set_ylabel("Unit ID [Shank, Depth] — Tacr1+ Phototagged (RED) / Standard Units (BLACK)", fontsize=10, fontweight='bold')
    ax3.set_xlabel("Time in Recording (seconds)", fontsize=11, fontweight='bold')
    ax3.set_ylim(0.5, len(selected_50) + 0.5)
    ax3.invert_yaxis()
    ax3.set_xlim(0.0, 300.0)
    ax3.grid(True, axis='x', linestyle=':', alpha=0.5)

    plt.tight_layout()
    png_out = out_dir / "population_raster_tacr1_phototagged.png"
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved High-Res PNG Raster Plot: {png_out}")

    # Copy PNG to artifact folder for markdown embedding
    artifact_png = artifact_dir / "population_raster_tacr1_phototagged.png"
    shutil.copy2(png_out, artifact_png)

    print("\n" + "=" * 85)
    print(" [SUCCESS] Clean Tacr1+ Phototagged Population Spike Raster Complete!")
    print(f" HTML:     {html_out}")
    print(f" PNG:      {png_out}")
    print(f" Artifact: {artifact_png}")
    print("=" * 85)

if __name__ == "__main__":
    main()
