#!/usr/bin/env python3
"""
Generate Breath-Triggered Peri-Event Time Histograms (PETH) for All 37 Clean Neurons
with Mean Firing Rate Line and Standard Deviation (SD) Shaded Shadow.
Synchronized 1:1 with electrophysiology slice and cibrrig burst detection.
"""
import sys
import os
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PLOT_37_PETH")


def main():
    parser = argparse.ArgumentParser(description="Plot PETH with Mean + SD for all 37 neurons")
    parser.add_argument("--consensus-dir", type=str, default="'output_LP21_60s_120s'/sliding_window_pipeline/lupin/consensus_master")
    parser.add_argument("--raw-dir", type=str, default=r"Z:\ramirezlab\pascual\rec_data\LP21_fent_EXPT_g1")
    parser.add_argument("--out-dir", type=str, default="LP21_fent_EXPT_g1_60s_120s_lupin_run/02_single_cell_phase_rasters")
    parser.add_argument("--start-time", type=float, default=60.0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds")
    parser.add_argument("--order-by", type=str, default="cluster_id", choices=["cluster_id", "depth_um", "peak_latency"])
    args = parser.parse_args()

    cons_dir_raw = args.consensus_dir.strip('"')
    cons_dir = Path(cons_dir_raw)
    if not cons_dir.exists():
        for cand in [
            PROJECT_ROOT / cons_dir_raw,
            PROJECT_ROOT / "'output_LP21_60s_120s'" / "sliding_window_pipeline" / "lupin" / "consensus_master",
            PROJECT_ROOT / "output_LP21_60s_120s" / "sliding_window_pipeline" / "lupin" / "consensus_master",
            PROJECT_ROOT / "LP21_fent_EXPT_g1_60s_120s_lupin_run" / "00_spike_sorting_data"
        ]:
            if cand.exists():
                cons_dir = cand
                break

    raw_dir = Path(args.raw_dir.strip('"'))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_t = args.start_time
    rec_dur = args.duration

    logger.info("=" * 85)
    logger.info(f" GENERATING SYNCHRONIZED PETH GRID WITH MEAN + SD SHADOW (ORDER: {args.order_by.upper()})")
    logger.info(f" Consensus Directory: {cons_dir.resolve()}")
    logger.info(f" Time Slice Window  : {start_t:.1f}s to {start_t + rec_dur:.1f}s")
    logger.info("=" * 85)

    # 1. Load Clean Single Units
    alf_dir = cons_dir / "alf"
    spk_times = np.load(alf_dir / "spike_times.npy")
    spk_clusters = np.load(alf_dir / "spike_clusters.npy")
    df_info = pd.read_csv(cons_dir / "cluster_info.tsv", sep="\t")

    qc_col = "passed_qc" if "passed_qc" in df_info.columns else ("passed_allen_qc" if "passed_allen_qc" in df_info.columns else None)
    df_sua = df_info[df_info[qc_col] == True].copy() if qc_col else df_info.copy()

    sua_ids = np.array(df_sua["cluster_id"].values)
    mask_sua = np.isin(spk_clusters, sua_ids)
    spk_times = spk_times[mask_sua]
    spk_clusters = spk_clusters[mask_sua]

    # 2. Extract Breaths from EMG Synchronized with 60s-120s Slice
    nidq_bins = list(raw_dir.glob("*.nidq.bin"))
    if not nidq_bins:
        raise FileNotFoundError(f"No .nidq.bin in {raw_dir}")
    nidq_bin = nidq_bins[0]
    nidq_data = read_nidq_bin(nidq_bin, nidq_bin.with_suffix(".meta"), emg_chan=0, ecg_chan=1)
    
    t_nidq_raw = nidq_data.time_vector
    t_end = start_t + rec_dur
    mask_n = (t_nidq_raw >= start_t) & (t_nidq_raw <= t_end)
    t_nidq = t_nidq_raw[mask_n] - start_t
    emg_raw = nidq_data.all_channels[0][mask_n]

    emg_cfg = EMGConfig(fs=nidq_data.fs, highpass_cutoff=300.0, lowpass_cutoff=5000.0, smoothing_window_ms=20.0)
    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    breath_onsets = emg_proc.breath_onsets[(emg_proc.breath_onsets >= 0.0) & (emg_proc.breath_onsets <= rec_dur)]

    logger.info(f"Loaded {len(sua_ids)} clean single units across {len(breath_onsets)} synchronized breaths.")

    # 3. Compute Firing Rate Per Breath
    dt = 0.005  # 5 ms bins
    win_pre = 0.15  # -150 ms
    win_post = 0.35  # +350 ms
    time_bins = np.arange(-win_pre, win_post + dt, dt)
    time_centers_ms = (time_bins[:-1] + time_bins[1:]) / 2.0 * 1000.0
    n_bins = len(time_centers_ms)

    valid_onsets = breath_onsets[(breath_onsets > win_pre) & (breath_onsets < rec_dur - win_post)]
    n_breaths = len(valid_onsets)

    kernel = signal.windows.gaussian(15, std=3)
    kernel /= np.sum(kernel)

    peth_stats = []
    for uid in sua_ids:
        spks = spk_times[spk_clusters == uid]
        trials = np.zeros((n_breaths, n_bins))
        for b_idx, on in enumerate(valid_onsets):
            rel = spks - on
            in_win = rel[(rel >= -win_pre) & (rel <= win_post)]
            counts, _ = np.histogram(in_win, bins=time_bins)
            rate_hz = np.convolve(counts / dt, kernel, mode="same")
            trials[b_idx, :] = rate_hz

        mean_fr = np.mean(trials, axis=0)
        std_fr = np.std(trials, axis=0)
        depth = df_sua[df_sua["cluster_id"] == uid]["depth_um"].values[0] if "depth_um" in df_sua.columns else 0.0
        overall_fr = len(spks) / rec_dur

        peth_stats.append({
            "cluster_id": int(uid),
            "depth_um": float(depth),
            "overall_fr": float(overall_fr),
            "mean_fr": mean_fr,
            "std_fr": std_fr,
            "peak_time_ms": float(time_centers_ms[np.argmax(mean_fr)])
        })

    # Sort neurons by requested order
    if args.order_by == "cluster_id":
        peth_stats.sort(key=lambda x: x["cluster_id"])
    elif args.order_by == "peak_latency":
        peth_stats.sort(key=lambda x: x["peak_time_ms"])
    else:
        peth_stats.sort(key=lambda x: x["depth_um"])

    # 4. Generate Static Grid Figure (6 cols x 7 rows = 42 subplots)
    n_units = len(peth_stats)
    n_cols = 6
    n_rows = int(np.ceil(n_units / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 3.4 * n_rows), sharex=True)
    axes_flat = axes.flatten()

    for i, p in enumerate(peth_stats):
        ax = axes_flat[i]
        mean_line = p["mean_fr"]
        std_band = p["std_fr"]

        # Line + SD Shadow
        ax.plot(time_centers_ms, mean_line, color="#0284c7", linewidth=2.2, label="Mean FR")
        ax.fill_between(
            time_centers_ms,
            np.maximum(0, mean_line - std_band),
            mean_line + std_band,
            color="#0284c7",
            alpha=0.25,
            label=r"$\pm 1$ SD"
        )

        ax.axvline(0, color="#ef4444", linestyle="--", linewidth=1.5, alpha=0.85)
        ax.set_title(f"Unit #{p['cluster_id']} ({p['depth_um']:.0f}µm)", fontsize=11, fontweight="bold")
        ax.set_ylabel("FR (sp/s)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Time from Breath Onset (ms)", fontsize=9, fontweight="bold")

    # Hide unused axes
    for j in range(n_units, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(f"Peri-Event Time Histograms (PETH): All {n_units} Clean Single Units with ±1 SD Shadow ({n_breaths} Breaths)\n[Red Dashed Line = Diaphragm Burst Onset (t = 0 ms)]", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_png = out_dir / "all_37_neurons_peth_mean_sd.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    logger.info(f"Saved Static Grid Figure: {out_png.resolve()}")

    # 5. Generate Interactive HTML Plotly Grid
    subplot_titles = [f"Unit #{p['cluster_id']} ({p['depth_um']:.0f}µm)" for p in peth_stats]
    fig_plotly = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        horizontal_spacing=0.03,
        vertical_spacing=0.06
    )

    for idx, p in enumerate(peth_stats):
        r = (idx // n_cols) + 1
        c = (idx % n_cols) + 1
        mean_line = p["mean_fr"]
        std_band = p["std_fr"]
        upper = mean_line + std_band
        lower = np.maximum(0, mean_line - std_band)

        # Upper bound (hidden line)
        fig_plotly.add_trace(
            go.Scatter(
                x=time_centers_ms,
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            ),
            row=r, col=c
        )
        # Lower bound with fill to upper
        fig_plotly.add_trace(
            go.Scatter(
                x=time_centers_ms,
                y=lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(2, 132, 199, 0.22)",
                name="±1 SD" if idx == 0 else "",
                showlegend=(idx == 0),
                hoverinfo="skip"
            ),
            row=r, col=c
        )
        # Mean Line
        fig_plotly.add_trace(
            go.Scatter(
                x=time_centers_ms,
                y=mean_line,
                mode="lines",
                line=dict(color="#0284c7", width=2.2),
                name="Mean FR (Hz)" if idx == 0 else "",
                showlegend=(idx == 0),
                hovertemplate="Time: %{x:.1f} ms<br>FR: %{y:.1f} sp/s<extra></extra>"
            ),
            row=r, col=c
        )

        # Breath Onset Line (t=0)
        fig_plotly.add_vline(
            x=0,
            line_width=1.2,
            line_dash="dash",
            line_color="rgba(239, 68, 68, 0.8)",
            row=r, col=c
        )

    fig_plotly.update_layout(
        title=f"<b>Interactive PETH Grid: All {n_units} Clean Single Units with ±1 SD Shadow ({n_breaths} Breaths)</b><br><sup>Red dashed line marks true breath onset (t = 0 ms)</sup>",
        template="plotly_white",
        height=n_rows * 230,
        margin=dict(l=40, r=30, b=40, t=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    out_html = out_dir / "all_37_neurons_peth_mean_sd_interactive.html"
    fig_plotly.write_html(str(out_html))
    logger.info(f"Saved Interactive HTML Grid: {out_html.resolve()}")

    # Also copy to root publication figures for convenience
    copy_dst_png = PROJECT_ROOT / "output_publication_figures_LP21_60s" / "all_37_neurons_peth_mean_sd.png"
    copy_dst_html = PROJECT_ROOT / "output_publication_figures_LP21_60s" / "all_37_neurons_peth_mean_sd_interactive.html"
    import shutil
    shutil.copy2(out_png, copy_dst_png)
    shutil.copy2(out_html, copy_dst_html)

    print("\n" + "=" * 85)
    print(" [SUCCESS] ALL 37 NEURONS PETH MEAN + SD REGENERATED (SYNCHRONIZED)")
    print("=" * 85)
    print(f" • Static PNG Grid : {out_png.resolve()}")
    print(f" • Interactive HTML: {out_html.resolve()}")
    print("=" * 85)


if __name__ == "__main__":
    main()
