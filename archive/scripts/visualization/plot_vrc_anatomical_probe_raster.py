#!/usr/bin/env python3
"""
Generate Publication-Quality Depth-Ordered VRC Single-Unit Raster
Replicating Figure 2c from Bush & Ramirez (2024):
- 37 clean single units ordered by probe location (Rostral tip -> Caudal entry)
- Anatomical boundary bars on the right: VII (orange), BötC (teal), preBötC (gold), }* bounds, and LRNm (blue)
- Integrated Diaphragm EMG ("Dia.") trace below with a 500 ms scale bar
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

def main():
    raw_dir = Path(r"Z:\ramirezlab\pascual\rec_data\LP21_fent_EXPT_g1")
    cons_dir = PROJECT_ROOT / "LP21_fent_EXPT_g1_60s_120s_lupin_run" / "00_spike_sorting_data"
    out_dir = PROJECT_ROOT / "LP21_fent_EXPT_g1_60s_120s_lupin_run" / "04_publication_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load units and sort by depth (shallow/rostral to deep/caudal)
    spk_times = np.load(cons_dir / "alf" / "spike_times.npy")
    spk_clusters = np.load(cons_dir / "alf" / "spike_clusters.npy")
    df_info = pd.read_csv(cons_dir / "cluster_info.tsv", sep="\t")
    df_sua = df_info[df_info["passed_qc"] == True].sort_values("depth_um", ascending=True).reset_index(drop=True)
    n_units = len(df_sua)

    # 2. Load EMG
    nidq_bin = list(raw_dir.glob("*.nidq.bin"))[0]
    nidq_data = read_nidq_bin(nidq_bin, nidq_bin.with_suffix(".meta"), emg_chan=0, ecg_chan=1)
    t_nidq_raw = nidq_data.time_vector
    start_t = 60.0
    rec_dur = 60.0
    mask_n = (t_nidq_raw >= start_t) & (t_nidq_raw <= start_t + rec_dur)
    t_nidq = t_nidq_raw[mask_n] - start_t
    emg_raw = nidq_data.all_channels[0][mask_n]

    emg_cfg = EMGConfig(fs=nidq_data.fs, highpass_cutoff=300.0, lowpass_cutoff=5000.0, smoothing_window_ms=25.0)
    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    emg_env = emg_proc.envelope

    # 3. Select a representative 4-breath window (~1.75 seconds)
    # 4 beautiful, evenly-spaced breaths matching reference
    t_start = 10.08
    t_end = 11.83
    mask_win_emg = (t_nidq >= t_start) & (t_nidq <= t_end)
    t_win_emg = t_nidq[mask_win_emg] - t_start
    emg_win = emg_env[mask_win_emg]
    emg_win_norm = emg_win / (np.max(emg_win) + 1e-6)

    # 4. Create Figure
    fig = plt.figure(figsize=(4.6, 5.8), dpi=300)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4.2, 0.85], hspace=0.07)

    ax_raster = fig.add_subplot(gs[0])
    ax_emg = fig.add_subplot(gs[1], sharex=ax_raster)

    # Plot spikes
    for idx, row in df_sua.iterrows():
        cid = row["cluster_id"]
        u_spks = spk_times[spk_clusters == cid]
        in_win = u_spks[(u_spks >= t_start) & (u_spks <= t_end)] - t_start
        y_val = idx + 1
        ax_raster.scatter(in_win, np.full_like(in_win, y_val), s=8, color="#525252", alpha=0.90, edgecolors='none')

    # Spines and ticks styling for raster
    ax_raster.spines['top'].set_visible(False)
    ax_raster.spines['right'].set_visible(False)
    ax_raster.spines['bottom'].set_visible(False)
    ax_raster.spines['left'].set_linewidth(1.2)
    ax_raster.xaxis.set_visible(False)

    ax_raster.set_ylim(0.5, n_units + 0.8)
    ax_raster.set_yticks([1, n_units])
    ax_raster.set_yticklabels(["1", str(n_units)], fontsize=11, fontfamily='sans-serif')
    ax_raster.set_ylabel("Neuron ID", fontsize=12, labelpad=2, fontfamily='sans-serif')
    ax_raster.tick_params(direction='out', length=4.5, width=1.2, pad=3)

    # Add 'c' label at top-left
    fig.text(0.06, 0.94, "c", fontsize=15, fontweight='bold', ha='left', va='top', fontfamily='sans-serif')

    # Plot Diaphragm EMG below
    ax_emg.plot(t_win_emg, emg_win_norm, color="#9ca3af", linewidth=1.8)
    ax_emg.set_ylim(-0.08, 1.25)
    ax_emg.spines['top'].set_visible(False)
    ax_emg.spines['right'].set_visible(False)
    ax_emg.spines['left'].set_visible(False)
    ax_emg.spines['bottom'].set_visible(False)
    ax_emg.yaxis.set_visible(False)
    ax_emg.xaxis.set_visible(False)

    # Label "Dia."
    ax_emg.text(-0.03, 0.35, "Dia.", fontsize=11, color="#1f2937", ha='right', va='center', fontfamily='sans-serif')

    # Scale bar "500 ms"
    scale_bar_dur = 0.5  # 500 ms
    scale_bar_x0 = 0.02
    scale_bar_y0 = -0.38
    ax_emg.plot([scale_bar_x0, scale_bar_x0 + scale_bar_dur], [scale_bar_y0, scale_bar_y0], color="black", linewidth=3.2, clip_on=False)
    ax_emg.text(scale_bar_x0, scale_bar_y0 - 0.25, "500 ms", fontsize=10, color="black", ha='left', va='top', clip_on=False, fontfamily='sans-serif')

    # Set X limits
    x_len = t_end - t_start
    ax_raster.set_xlim(-0.04, x_len + 0.04)

    # -------------------------------------------------------------
    # Anatomical Boundary Bars on the Right Side
    # -------------------------------------------------------------
    # In df_sua (37 units):
    # - Units 1 to 5 (depth 30 to 428 um): VII / RTN
    # - Units 6 to 17 (depth 455 to 720 um): BötC
    # - Units 18 to 29 (depth 810 to 1035 um): preBötC
    # - Units 30 to 37 (depth 1110 to 1245 um): }* bounds
    # - LRNm: alongside BötC + preBötC (Units 6 to 29)

    bar_x1 = x_len + 0.06  # Column 1 (VII, BötC, preBötC)
    text_x1 = x_len + 0.08 # Text for Column 1
    bar_x2 = x_len + 0.21  # Column 2 (LRNm bar)
    text_x2 = x_len + 0.23 # Text for Column 2

    # VII bar (Orange)
    c_vii = "#d95f02"
    ax_raster.plot([bar_x1, bar_x1], [0.8, 5.2], color=c_vii, linewidth=4.5, solid_capstyle='butt', clip_on=False)
    ax_raster.text(text_x1, 3.0, "VII", rotation=90, color=c_vii, fontsize=11, fontweight='bold', ha='left', va='center', clip_on=False, fontfamily='sans-serif')

    # BötC bar (Teal)
    c_botc = "#00897b"
    ax_raster.plot([bar_x1, bar_x1], [5.8, 17.2], color=c_botc, linewidth=4.5, solid_capstyle='butt', clip_on=False)
    ax_raster.text(text_x1, 11.5, "BötC", rotation=90, color=c_botc, fontsize=11, fontweight='bold', ha='left', va='center', clip_on=False, fontfamily='sans-serif')

    # preBötC bar (Gold)
    c_prebotc = "#d97706"
    ax_raster.plot([bar_x1, bar_x1], [17.8, 29.2], color=c_prebotc, linewidth=4.5, solid_capstyle='butt', clip_on=False)
    ax_raster.text(text_x1, 23.5, "preBötC", rotation=90, color=c_prebotc, fontsize=11, fontweight='bold', ha='left', va='center', clip_on=False, fontfamily='sans-serif')

    # }* bounds (Text with bracket)
    ax_raster.text(text_x1 - 0.01, 33.5, "}* bounds", rotation=90, color="#1f2937", fontsize=10, fontweight='normal', ha='left', va='center', clip_on=False, fontfamily='sans-serif')

    # LRNm bar (Blue, spanning alongside BötC and preBötC)
    c_lrn = "#0284c7"
    ax_raster.plot([bar_x2, bar_x2], [5.8, 29.2], color=c_lrn, linewidth=4.5, solid_capstyle='butt', clip_on=False)
    ax_raster.text(text_x2, 17.5, "LRNm", rotation=90, color=c_lrn, fontsize=11, fontweight='bold', ha='left', va='center', clip_on=False, fontfamily='sans-serif')

    plt.subplots_adjust(left=0.18, right=0.72, top=0.91, bottom=0.11)

    out_file = out_dir / "figure_2c_vrc_depth_ordered_raster.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[SUCCESS] Saved reference-matched figure to: {out_file.resolve()}")

if __name__ == "__main__":
    main()
