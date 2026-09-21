#!/usr/bin/env python3
"""
Generate Individual Figures for Each Clean Neuron:
1. Top Panel: Breath-Onset Aligned Spike Raster (True Real Time - NO Warping)
   - Background of each breath row is colored by the exact breath phase phi(t) for that trial.
   - Inspiration (0 -> pi) in red/coral tones, Expiration (-pi -> 0) in blue/cyan tones.
   - Individual spike ticks overlaid in black.
2. Bottom Panel: Peri-Event Time Histogram (PETH)
   - Real-time Mean Firing Rate (spikes/s) solid line + (+/- 1 SD) standard deviation shadow.
   - Aligned Diaphragm EMG / Flow Rate reference.
"""
import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import scipy.signal as signal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("INDIVIDUAL_PHASE_RASTERS")


def compute_continuous_respiratory_phase(
    breath_onsets: np.ndarray,
    breath_offsets: np.ndarray,
    t_eval: np.ndarray
) -> np.ndarray:
    """
    Computes continuous respiratory phase phi(t) in [-pi, pi] for each time point in t_eval:
    - Inspiration: 0 to pi (from breath onset to breath offset)
    - Expiration: -pi to 0 (from breath offset to next breath onset)
    """
    phi = np.full_like(t_eval, np.nan)
    n_breaths = min(len(breath_onsets), len(breath_offsets))
    
    for i in range(n_breaths - 1):
        t_on = breath_onsets[i]
        t_off = breath_offsets[i]
        t_next_on = breath_onsets[i + 1]

        if t_off <= t_on or t_next_on <= t_off:
            continue

        # Inspiratory phase: [t_on, t_off] -> [0, pi]
        insp_mask = (t_eval >= t_on) & (t_eval < t_off)
        if np.any(insp_mask):
            phi[insp_mask] = (t_eval[insp_mask] - t_on) / (t_off - t_on) * np.pi

        # Expiratory phase: [t_off, t_next_on] -> [-pi, 0]
        exp_mask = (t_eval >= t_off) & (t_eval < t_next_on)
        if np.any(exp_mask):
            phi[exp_mask] = -np.pi + (t_eval[exp_mask] - t_off) / (t_next_on - t_off) * np.pi

    # Fill boundaries
    valid_idx = np.where(~np.isnan(phi))[0]
    if len(valid_idx) > 0:
        phi[:valid_idx[0]] = phi[valid_idx[0]]
        phi[valid_idx[-1]:] = phi[valid_idx[-1]]

    return phi


def plot_single_neuron_phase_raster_and_peth(
    unit_id: int,
    depth_um: float,
    spk_times: np.ndarray,
    breath_onsets: np.ndarray,
    breath_offsets: np.ndarray,
    t_nidq: np.ndarray,
    emg_env: np.ndarray,
    out_file: Path,
    win_pre: float = 0.15,
    win_post: float = 0.35,
    dt: float = 0.002
):
    """
    Generates a 2-panel figure for a single neuron:
    Top: Breath Raster with Phase-Colored Background
    Bottom: PETH with Mean + SD shadow and average EMG
    """
    time_axis_s = np.arange(-win_pre, win_post + dt, dt)
    time_axis_ms = time_axis_s * 1000.0
    n_time_bins = len(time_axis_s)

    valid_onsets = breath_onsets[(breath_onsets > win_pre) & (breath_onsets < t_nidq[-1] - win_post)]
    n_breaths = len(valid_onsets)

    # 1. Build Phase Background Grid & Trial Firing Rates
    phase_grid = np.zeros((n_breaths, n_time_bins))
    rate_trials = np.zeros((n_breaths, n_time_bins))
    spike_scatter_x = []
    spike_scatter_y = []

    kernel = signal.windows.gaussian(15, std=3)  # ~6 ms std for dt=2ms
    kernel /= np.sum(kernel)

    for b_idx, on in enumerate(valid_onsets):
        trial_t_eval = on + time_axis_s
        trial_phi = compute_continuous_respiratory_phase(breath_onsets, breath_offsets, trial_t_eval)
        phase_grid[b_idx, :] = trial_phi

        # Extract Spikes for this trial
        rel_spks = spk_times - on
        in_win = rel_spks[(rel_spks >= -win_pre) & (rel_spks <= win_post)]
        
        # Collect for scatter
        for s in in_win:
            spike_scatter_x.append(s * 1000.0)  # ms
            spike_scatter_y.append(b_idx + 1)   # 1-indexed trial

        # Instantaneous Firing Rate
        counts, _ = np.histogram(in_win, bins=np.append(time_axis_s, time_axis_s[-1] + dt))
        smooth_fr = np.convolve(counts / dt, kernel, mode="same")
        rate_trials[b_idx, :] = smooth_fr

    mean_fr = np.mean(rate_trials, axis=0)
    std_fr = np.std(rate_trials, axis=0)

    # Average EMG envelope for reference
    emg_trials = np.zeros((n_breaths, n_time_bins))
    for b_idx, on in enumerate(valid_onsets):
        trial_t_eval = on + time_axis_s
        emg_trials[b_idx, :] = np.interp(trial_t_eval, t_nidq, emg_env)
    mean_emg = np.mean(emg_trials, axis=0)
    mean_emg_norm = mean_emg / (np.max(mean_emg) + 1e-6)

    # 2. Plotting
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1.0], hspace=0.18)

    # -------------------------------------------------------------
    # Top Panel: Raster with Phase-Colored Background
    # -------------------------------------------------------------
    ax_raster = fig.add_subplot(gs[0])

    # Background Phase Image: Colormap RdBu_r (-pi to pi)
    # 0 to pi (Inspiration) is Red/Coral, -pi to 0 (Expiration) is Blue/Cyan
    im_phase = ax_raster.imshow(
        phase_grid,
        aspect="auto",
        cmap="coolwarm",
        vmin=-np.pi,
        vmax=np.pi,
        extent=[time_axis_ms[0], time_axis_ms[-1], 0.5, n_breaths + 0.5],
        origin="lower",
        alpha=0.45,
        interpolation="bilinear"
    )

    # Overlay Individual Spikes as crisp black ticks
    if len(spike_scatter_x) > 0:
        ax_raster.scatter(
            spike_scatter_x,
            spike_scatter_y,
            marker='|',
            color='black',
            s=45,
            linewidth=1.2,
            alpha=0.9,
            label="Spike Times"
        )

    ax_raster.axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
    ax_raster.set_xlim(time_axis_ms[0], time_axis_ms[-1])
    ax_raster.set_ylim(0.5, n_breaths + 0.5)
    ax_raster.set_ylabel("Breath Trial Index", fontsize=11, fontweight="bold")
    ax_raster.set_title(
        f"Unit #{unit_id} (Depth: {depth_um:.0f} µm) — True Real-Time Breath-Onset Raster\n"
        r"[Background: Red/Warm = Inspiration ($\phi \in [0, \pi]$) | Blue/Cool = Expiration ($\phi \in [-\pi, 0]$)]",
        fontsize=11,
        fontweight="bold"
    )
    ax_raster.grid(False)

    # Add phase colorbar
    cbar = plt.colorbar(im_phase, ax=ax_raster, fraction=0.03, pad=0.02)
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels([r"$-\pi$ (Exp)", r"$-\frac{\pi}{2}$", r"$0$ (I-on)", r"$+\frac{\pi}{2}$", r"$+\pi$ (I-off)"], fontsize=8)

    # -------------------------------------------------------------
    # Bottom Panel: PETH with Mean + SD Shadow & EMG Reference
    # -------------------------------------------------------------
    ax_peth = fig.add_subplot(gs[1], sharex=ax_raster)

    # Mean Firing Rate Line + SD Shadow
    ax_peth.plot(time_axis_ms, mean_fr, color="#1f77b4", linewidth=2.5, label="Mean Firing Rate")
    ax_peth.fill_between(
        time_axis_ms,
        np.maximum(0, mean_fr - std_fr),
        mean_fr + std_fr,
        color="#1f77b4",
        alpha=0.25,
        label=r"$\pm 1$ SD Shadow"
    )

    ax_peth.axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
    ax_peth.set_xlabel("Time from Breath Onset (ms)", fontsize=11, fontweight="bold")
    ax_peth.set_ylabel("Firing Rate (sp/s)", fontsize=11, fontweight="bold", color="#1f77b4")
    ax_peth.tick_params(axis='y', labelcolor="#1f77b4")
    ax_peth.grid(True, linestyle=":", alpha=0.6)

    # Secondary y-axis for Diaphragm EMG
    ax_emg = ax_peth.twinx()
    ax_emg.plot(time_axis_ms, mean_emg_norm, color="gray", linestyle=":", linewidth=1.8, label="Dia. EMG Envelope")
    ax_emg.set_ylabel("Dia. EMG (norm)", fontsize=10, color="dimgray")
    ax_emg.tick_params(axis='y', labelcolor="dimgray")
    ax_emg.set_ylim(-0.05, 1.2)

    # Unified Legend
    lines_peth, labels_peth = ax_peth.get_legend_handles_labels()
    lines_emg, labels_emg = ax_emg.get_legend_handles_labels()
    ax_peth.legend(lines_peth + lines_emg, labels_peth + labels_emg, loc="upper right", frameon=True, fontsize=8)

    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Individual Phase-Colored Rasters and PETHs for each clean neuron")
    parser.add_argument("--consensus-dir", type=str, default="'output_LP21_60s_120s'/sliding_window_pipeline/lupin/consensus_master")
    parser.add_argument("--raw-dir", type=str, default=r"Z:\ramirezlab\pascual\rec_data\LP21_fent_EXPT_g1")
    parser.add_argument("--out-dir", type=str, default="output_individual_neuron_rasters_LP21_60s")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--start-time", type=float, default=60.0, help="Start time in seconds")
    args = parser.parse_args()

    cons_dir_raw = args.consensus_dir.strip('"')
    cons_dir = Path(cons_dir_raw)
    if not cons_dir.exists():
        alt_path = Path(f"'{cons_dir_raw}'")
        if alt_path.exists():
            cons_dir = alt_path

    raw_dir = Path(args.raw_dir.strip('"'))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 85)
    logger.info(" GENERATING INDIVIDUAL PHASE-COLORED RASTERS & PETHS (NO WARPING)")
    logger.info(f" Consensus Directory: {cons_dir.resolve()}")
    logger.info(f" Raw Directory:       {raw_dir.resolve()}")
    logger.info(f" Output Directory:    {out_dir.resolve()}")
    logger.info("=" * 85)

    # 1. Load Clean Units
    alf_dir = cons_dir / "alf"
    spk_times = np.load(alf_dir / "spike_times.npy")
    spk_clusters = np.load(alf_dir / "spike_clusters.npy")
    df_info = pd.read_csv(cons_dir / "cluster_info.tsv", sep="\t")

    qc_col = "passed_qc" if "passed_qc" in df_info.columns else ("passed_allen_qc" if "passed_allen_qc" in df_info.columns else None)
    if qc_col:
        df_sua = df_info[df_info[qc_col] == True].copy()
    else:
        df_sua = df_info.copy()

    sua_ids = np.array(df_sua["cluster_id"].values)
    mask_sua = np.isin(spk_clusters, sua_ids)
    spk_times = spk_times[mask_sua]
    spk_clusters = spk_clusters[mask_sua]

    rec_dur = min(float(np.max(spk_times)), args.duration) if len(spk_times) > 0 else args.duration
    # Auto-detect start_time if consensus_dir contains 60s_120s or similar
    start_t = args.start_time
    if start_t == 0.0 and ("60s_120s" in str(cons_dir) or "60s_120s" in str(out_dir)):
        start_t = 60.0
        logger.info(f"Auto-detected sliced dataset offset: start_time = {start_t:.1f}s")

    # 2. Extract Breaths from EMG with Synchronized Time Slice
    nidq_bins = list(raw_dir.glob("*.nidq.bin"))
    if not nidq_bins:
        raise FileNotFoundError(f"No .nidq.bin found in {raw_dir}")
    nidq_bin = nidq_bins[0]
    nidq_data = read_nidq_bin(nidq_bin, nidq_bin.with_suffix(".meta"), emg_chan=0, ecg_chan=1)

    t_nidq_raw = nidq_data.time_vector
    t_end = start_t + rec_dur
    mask_n = (t_nidq_raw >= start_t) & (t_nidq_raw <= t_end)
    
    # Synchronize time vector: relative 0 matches 0 in spike_times
    t_nidq = t_nidq_raw[mask_n] - start_t
    emg_raw = nidq_data.all_channels[0][mask_n]

    emg_cfg = EMGConfig(fs=nidq_data.fs, highpass_cutoff=300.0, lowpass_cutoff=5000.0, smoothing_window_ms=20.0)
    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    breath_onsets = emg_proc.breath_onsets[(emg_proc.breath_onsets >= 0.0) & (emg_proc.breath_onsets <= rec_dur)]
    breath_offsets = emg_proc.breath_offsets[(emg_proc.breath_offsets >= 0.0) & (emg_proc.breath_offsets <= rec_dur)]

    logger.info(f"Loaded {len(sua_ids)} clean single units across {len(breath_onsets)} synchronized breaths (Slice: {start_t:.1f}s to {t_end:.1f}s).")

    # 3. Generate Individual Figures for All Neurons
    index_cards = []
    for idx, uid in enumerate(sua_ids):
        depth = df_sua[df_sua["cluster_id"] == uid]["depth_um"].values[0] if "depth_um" in df_sua.columns else 0.0
        u_spks = spk_times[spk_clusters == uid]
        out_png = out_dir / f"unit_{uid:03d}_depth_{int(depth)}um_raster_peth.png"
        
        plot_single_neuron_phase_raster_and_peth(
            unit_id=uid,
            depth_um=depth,
            spk_times=u_spks,
            breath_onsets=breath_onsets,
            breath_offsets=breath_offsets,
            t_nidq=t_nidq,
            emg_env=emg_proc.envelope,
            out_file=out_png,
            win_pre=0.15,
            win_post=0.35,
            dt=0.002
        )
        index_cards.append({
            "unit_id": int(uid),
            "depth_um": float(depth),
            "n_spikes": len(u_spks),
            "file_name": out_png.name
        })

    # 4. Generate an HTML Gallery for Easy Browsing
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Individual Neuron Phase-Colored Rasters & PETHs</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
        h1 { text-align: center; color: #38bdf8; margin-bottom: 5px; }
        p.subtitle { text-align: center; color: #94a3b8; font-size: 14px; margin-bottom: 25px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 20px; max-width: 1800px; margin: 0 auto; }
        .card { background: #1e293b; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5); border: 1px solid #334155; }
        .card img { width: 100%; height: auto; border-radius: 6px; display: block; }
        .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .badge { background: #0284c7; color: white; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Individual Neuron Phase-Colored Rasters & PETHs</h1>
    <p class="subtitle">37 Clean Single Units | 152 Breaths | True Real Time (No Phase Warping) | Background = Breath Phase &phi;(t)</p>
    <div class="grid">
"""
    for card in index_cards:
        html_content += f"""
        <div class="card">
            <div class="card-header">
                <span style="font-weight: bold; font-size: 16px;">Unit #{card['unit_id']}</span>
                <span class="badge">{card['depth_um']:.0f} µm | {card['n_spikes']} Spikes</span>
            </div>
            <a href="{card['file_name']}" target="_blank">
                <img src="{card['file_name']}" alt="Unit {card['unit_id']}">
            </a>
        </div>
"""
    html_content += """
    </div>
</body>
</html>
"""
    gallery_path = out_dir / "gallery_all_37_neurons.html"
    gallery_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Saved HTML Visual Gallery: {gallery_path.resolve()}")

    print("\n" + "=" * 85)
    print(" [SUCCESS] ALL 37 INDIVIDUAL NEURON PHASE RASTERS & PETHS GENERATED")
    print("=" * 85)
    print(f" • Output Folder       : {out_dir.resolve()}")
    print(f" • Interactive Gallery : {gallery_path.resolve()}")
    print("=" * 85)


if __name__ == "__main__":
    main()
