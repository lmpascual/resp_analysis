#!/usr/bin/env python3
r"""
Top 24 Phasic Single Units Baseline (0 - 60s) Population Spike Raster
Plots the 60-second baseline activity of top 24 phasic single units aligned with diaphragm EMG:
- Panel 1: Diaphragm EMG Envelope (0 - 60s) with inspiration onset markers
- Panel 2: Spike Raster for Top 24 Phasic Units (0 - 60s), sorted by Shank & Depth

Outputs:
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\top_phasic_units_baseline_60s_raster.png
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\top_phasic_units_baseline_60s_raster.html
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

def main():
    out_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation")
    out_dir.mkdir(parents=True, exist_ok=True)
    session_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1\alf_session")
    alf_dir = session_dir / "alf" / "probe00"
    raw_dir = Path(r"Z:\ramirezlab\pascual\rec_data\TP_Tacr1_1_npx2_g4")
    artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\0e7be222-f160-4bc2-921c-f3fdf8b7825e")

    print("=" * 90)
    print(" Generating Baseline 60s Population Raster for Top 24 Phasic Single Units")
    print(f" ALF Directory: {alf_dir}")
    print(f" Output Directory: {out_dir}")
    print("=" * 90)

    # 1. Load Spikes & Top 24 Phasic Units CSV
    top24_csv = out_dir / "top_24_phasic_single_units.csv"
    if not top24_csv.exists():
        print(f" Warning: {top24_csv} not found, running PSTH script first.")
        from scripts.plot_top_phasic_units_psth import main as run_psth
        run_psth()

    top_df = pd.read_csv(top24_csv)
    top_phasic_uids = list(top_df['cluster_id'])

    spikes_times = np.load(alf_dir / "spike_times.npy")
    spikes_clusters = np.load(alf_dir / "spike_clusters.npy")
    info_df = pd.read_csv(alf_dir / "cluster_info.tsv", sep="\t")

    fs_ap = 30000.0
    if spikes_times.dtype in [np.int64, np.int32] and spikes_times.max() > 100000:
        spikes_times = spikes_times / fs_ap

    MAX_TIME_S = 60.0 # First 60 seconds
    spk_mask = (spikes_times >= 0.0) & (spikes_times <= MAX_TIME_S) & (np.isin(spikes_clusters, top_phasic_uids))
    spikes_times = spikes_times[spk_mask]
    spikes_clusters = spikes_clusters[spk_mask]

    # Sort units by Shank & Depth
    sorted_df = top_df.sort_values(by=['shank', 'depth'], ascending=[True, True]).reset_index(drop=True)
    sorted_uids = sorted_df['cluster_id'].values
    n_sorted = len(sorted_uids)

    # 2. Load NIDQ Diaphragm EMG for 0-60s
    nidq_meta = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.meta"
    nidq_bin = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.bin"
    nidq_data = read_nidq_bin(nidq_bin, nidq_meta)

    t_nidq = nidq_data.time_vector
    fs_nidq = nidq_data.fs

    nidq_mask = (t_nidq >= 0.0) & (t_nidq <= MAX_TIME_S)
    t_nidq = t_nidq[nidq_mask]
    emg_raw = nidq_data.all_channels[0][nidq_mask]

    emg_cfg = EMGConfig(fs=fs_nidq, highpass_cutoff=300.0, lowpass_cutoff=3000.0, notch_freq=60.0, smoothing_window_ms=20.0)
    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    emg_env = emg_proc.envelope
    breath_onsets = emg_proc.breath_onsets

    baseline_onsets = [o for o in breath_onsets if 0.0 <= o <= MAX_TIME_S]

    # Calculate Shank Boundary Lines
    shank_boundary_indices = []
    for i in range(n_sorted - 1):
        if sorted_df.loc[i, 'shank'] != sorted_df.loc[i + 1, 'shank']:
            shank_boundary_indices.append(i + 1.5)

    # 3. Generate High-Resolution Static PNG Figure
    fig_mpl = plt.figure(figsize=(22, 14))
    gs = fig_mpl.add_gridspec(2, 1, height_ratios=[1.2, 5.5], hspace=0.12)

    # Subplot 1: Diaphragm EMG Envelope (0-60s)
    ax_emg = fig_mpl.add_subplot(gs[0])
    ax_emg.plot(t_nidq, emg_env, color='black', linewidth=1.2, label='Diaphragm EMG Envelope')
    for o in baseline_onsets:
        ax_emg.axvline(o, color='red', linestyle=':', alpha=0.5)
    ax_emg.set_ylabel("Diaphragm EMG (µV)", fontsize=10, fontweight='bold')
    ax_emg.set_title("1. Diaphragm EMG Envelope (First 60 Seconds Baseline)", fontsize=12, fontweight='bold')
    ax_emg.set_xlim([0, MAX_TIME_S])
    ax_emg.legend(loc='upper right', fontsize=9)
    ax_emg.grid(True, linestyle=':', alpha=0.3)

    # Subplot 2: Population Spike Raster for Top 24 Phasic Units (0-60s)
    ax_rast = fig_mpl.add_subplot(gs[1], sharex=ax_emg)

    # Vertical breath onset lines
    for o in baseline_onsets:
        ax_rast.axvline(o, color='red', linestyle=':', alpha=0.35)

    # Horizontal shank boundary lines
    for b_y in shank_boundary_indices:
        ax_rast.axhline(b_y, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)

    y_ticks_pos, y_ticks_labels = [], []

    spiketrains = {u: np.sort(spikes_times[spikes_clusters == u]) for u in sorted_uids}

    for y_idx, uid in enumerate(sorted_uids):
        u_row = sorted_df[sorted_df['cluster_id'] == uid].iloc[0]
        u_spks = spiketrains[uid]
        is_insp = (u_row['pref_phase'] >= 0)
        
        tick_color = 'red' if is_insp else 'black'
        line_w = 1.3 if is_insp else 0.95

        if len(u_spks) > 0:
            ax_rast.vlines(u_spks, y_idx + 0.6, y_idx + 1.4, color=tick_color, linewidth=line_w)

        y_ticks_pos.append(y_idx + 1)
        phase_lbl = "Insp" if is_insp else "Exp"
        lbl = f"U{uid} (S{int(u_row['shank'])}, {int(u_row['depth'])}µm, R={u_row['vector_strength']:.2f}) - {phase_lbl}"
        y_ticks_labels.append(lbl)

    ax_rast.set_yticks(y_ticks_pos)
    ax_rast.set_yticklabels(y_ticks_labels, fontsize=8.5)
    ax_rast.set_ylabel("Top 24 Phasic Single Units (Sorted by Shank & Depth)", fontsize=11, fontweight='bold')
    ax_rast.set_xlabel("Baseline Session Time (s)", fontsize=11, fontweight='bold')
    ax_rast.set_title(f"2. Spike Raster for Top 24 Phasic Single Units (0 - 60 s Baseline | Red = Inspiratory, Black = Expiratory)", fontsize=12, fontweight='bold')
    ax_rast.set_ylim([0.2, n_sorted + 0.8])
    ax_rast.grid(True, linestyle=':', alpha=0.3)

    plt.suptitle("Subject TP_Tacr1_1: Top 24 Phasic Single Units Spike Raster (First 60 Seconds Baseline)", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    png_out = out_dir / "top_phasic_units_baseline_60s_raster.png"
    plt.savefig(png_out, dpi=300)
    plt.close()

    # Copy PNG to artifact directory
    artifact_png = artifact_dir / "top_phasic_units_baseline_60s_raster.png"
    shutil.copy2(png_out, artifact_png)
    print(f" Saved Baseline 60s Diagnostic Raster Figure: {png_out}")

    # 4. Interactive Plotly Dashboard
    fig_plotly = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.22, 0.78],
        subplot_titles=[
            "1. Diaphragm EMG Envelope (0 - 60s)",
            f"2. Spike Raster (Top {n_sorted} Phasic Single Units)"
        ],
        vertical_spacing=0.06
    )

    fig_plotly.add_trace(go.Scatter(x=t_nidq[::5], y=emg_env[::5], mode='lines', line=dict(color='black', width=1.2), name="EMG Envelope"), row=1, col=1)

    for y_idx, uid in enumerate(sorted_uids):
        u_row = sorted_df[sorted_df['cluster_id'] == uid].iloc[0]
        u_spks = spiketrains[uid]
        is_insp = (u_row['pref_phase'] >= 0)
        c_tick = 'red' if is_insp else 'black'

        fig_plotly.add_trace(go.Scatter(
            x=u_spks,
            y=[y_idx + 1] * len(u_spks),
            mode='markers',
            marker=dict(symbol='line-ns-open', color=c_tick, size=6, line=dict(width=1.3)),
            hoverinfo='text',
            text=[f"Unit {uid} (Shank {int(u_row['shank'])}, {int(u_row['depth'])}µm)<br>R: {u_row['vector_strength']:.3f} | FR 60s: {u_row['firing_rate_60s']}Hz"] * len(u_spks),
            name=f"Unit {uid}"
        ), row=2, col=1)

    shapes = []
    for o in baseline_onsets:
        shapes.append(dict(
            type='line', xref='x2', yref='y2',
            x0=o, x1=o, y0=0, y1=n_sorted + 1,
            line=dict(color='red', width=1, dash='dot')
        ))

    fig_plotly.update_layout(
        title="<b>Subject TP_Tacr1_1: Top 24 Phasic Single Units Spike Raster (0 - 60 s Baseline Epoch)</b>",
        template="plotly_white",
        height=950,
        shapes=shapes,
        showlegend=False,
        xaxis2=dict(title="Session Time (s)"),
        yaxis2=dict(title="Unit Index (Sorted by Shank & Depth)")
    )

    html_out = out_dir / "top_phasic_units_baseline_60s_raster.html"
    fig_plotly.write_html(str(html_out))
    print(f" Saved Interactive HTML Dashboard: {html_out}")

    print("\n==========================================================================================")
    print(" [SUCCESS] Baseline 60s Top 24 Phasic Raster Generation Complete!")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
