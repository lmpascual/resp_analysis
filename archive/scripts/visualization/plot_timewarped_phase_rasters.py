#!/usr/bin/env python3
"""
Generate Phase-Time-Warped Breath-Aligned Spike Rasters and PETHs for Each Clean Single Unit.
Applies piecewise linear time-warping mapped to the mean respiratory cycle:
- Inspiration [t_on, t_off] -> [0, mean_T_insp]
- Expiration [t_off, t_next_on] -> [mean_T_insp, mean_T_cycle]
- Pre-inspiration [-win_pre, 0] -> late expiration of prior breath
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

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TIMEWARPED_RASTERS")


def compute_timewarped_spikes_and_rates(
    spk_times: np.ndarray,
    breath_onsets: np.ndarray,
    breath_offsets: np.ndarray,
    mean_t_insp: float,
    mean_t_exp: float,
    mean_t_cycle: float,
    win_pre_s: float = 0.15,
    win_post_s: float = 0.40,
    dt_warp: float = 0.002
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies piecewise linear time-warping to spikes for every breath trial:
    Returns:
    - time_axis_ms: standardized time axis (ms) relative to breath onset (t=0)
    - spike_warp_x: list of warped spike times (ms)
    - spike_warp_y: list of breath trial indices (1-indexed)
    - mean_rate: mean firing rate across warped trials (sp/s)
    - std_rate: standard deviation of firing rate across warped trials (sp/s)
    """
    n_breaths = min(len(breath_onsets), len(breath_offsets))
    time_bins_s = np.arange(-win_pre_s, win_post_s + dt_warp, dt_warp)
    time_centers_ms = (time_bins_s[:-1] + time_bins_s[1:]) / 2.0 * 1000.0
    n_bins = len(time_centers_ms)

    spike_warp_x = []
    spike_warp_y = []
    trial_rates = np.zeros((n_breaths - 1, n_bins))

    kernel = signal.windows.gaussian(15, std=3)
    kernel /= np.sum(kernel)

    valid_trial_idx = 0
    for i in range(1, n_breaths - 1):
        t_on = breath_onsets[i]
        t_off = breath_offsets[i]
        t_next_on = breath_onsets[i + 1]
        t_prev_on = breath_onsets[i - 1]
        t_prev_off = breath_offsets[i - 1]

        t_insp_i = t_off - t_on
        t_exp_i = t_next_on - t_off
        t_prev_exp = t_on - t_prev_off

        if t_insp_i <= 0.01 or t_exp_i <= 0.01 or t_prev_exp <= 0.01:
            continue

        # Extract spikes in window [-win_pre_s, +win_post_s] relative to t_on
        t_win_start = t_on - win_pre_s
        t_win_end = t_on + win_post_s
        raw_spks = spk_times[(spk_times >= t_win_start) & (spk_times <= t_win_end)]

        warped_spks_trial = []
        for s in raw_spks:
            if s >= t_on and s < t_off:
                # Inspiration phase [0, mean_t_insp]
                u = (s - t_on) / t_insp_i
                t_w = u * mean_t_insp
            elif s >= t_off and s <= t_next_on:
                # Expiration phase [mean_t_insp, mean_t_cycle]
                v = (s - t_off) / t_exp_i
                t_w = mean_t_insp + v * mean_t_exp
            elif s < t_on and s >= t_prev_off:
                # Pre-inspiration (late expiration of previous breath)
                w = (s - t_prev_off) / t_prev_exp
                t_w = (w - 1.0) * mean_t_exp
            elif s < t_prev_off:
                # Further back
                t_w = -mean_t_exp - (t_prev_off - s)
            else:
                # Post next onset
                t_w = mean_t_cycle + (s - t_next_on)

            warped_spks_trial.append(t_w)
            spike_warp_x.append(t_w * 1000.0)  # ms
            spike_warp_y.append(i)

        # Compute histogram for this warped breath trial
        warped_arr = np.array(warped_spks_trial)
        counts, _ = np.histogram(warped_arr, bins=time_bins_s)
        smooth_fr = np.convolve(counts / dt_warp, kernel, mode="same")
        trial_rates[valid_trial_idx, :] = smooth_fr
        valid_trial_idx += 1

    trial_rates = trial_rates[:valid_trial_idx, :]
    mean_rate = np.mean(trial_rates, axis=0) if valid_trial_idx > 0 else np.zeros(n_bins)
    std_rate = np.std(trial_rates, axis=0) if valid_trial_idx > 0 else np.zeros(n_bins)

    return time_centers_ms, np.array(spike_warp_x), np.array(spike_warp_y), mean_rate, std_rate


def plot_single_neuron_timewarped_raster_and_peth(
    unit_id: int,
    depth_um: float,
    spk_times: np.ndarray,
    breath_onsets: np.ndarray,
    breath_offsets: np.ndarray,
    mean_t_insp: float,
    mean_t_exp: float,
    mean_t_cycle: float,
    out_file: Path,
    win_pre_s: float = 0.15,
    win_post_s: float = 0.40
):
    """
    Generates a 2-panel figure for a single neuron:
    Top: Time-Warped Breath-Aligned Raster (Background colored by standardized phase)
    Bottom: Time-Warped PETH with Mean + SD shadow
    """
    time_ms, spk_x, spk_y, mean_fr, std_fr = compute_timewarped_spikes_and_rates(
        spk_times=spk_times,
        breath_onsets=breath_onsets,
        breath_offsets=breath_offsets,
        mean_t_insp=mean_t_insp,
        mean_t_exp=mean_t_exp,
        mean_t_cycle=mean_t_cycle,
        win_pre_s=win_pre_s,
        win_post_s=win_post_s,
        dt_warp=0.002
    )

    n_trials = len(breath_onsets) - 2
    insp_end_ms = mean_t_insp * 1000.0
    cycle_end_ms = mean_t_cycle * 1000.0

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1.0], hspace=0.18)

    # -------------------------------------------------------------
    # Top Panel: Time-Warped Raster
    # -------------------------------------------------------------
    ax_raster = fig.add_subplot(gs[0])

    # Background standardized phase regions:
    # Pre-inspiration [-win_pre, 0]: Expiration (Light Blue)
    ax_raster.axvspan(-win_pre_s * 1000.0, 0, color="#dbeafe", alpha=0.7, label="Expiration")
    # Inspiration [0, mean_t_insp]: Inspiration (Light Coral / Red)
    ax_raster.axvspan(0, insp_end_ms, color="#fee2e2", alpha=0.85, label="Inspiration (Warped)")
    # Expiration [mean_t_insp, mean_t_cycle]: Expiration (Light Blue)
    ax_raster.axvspan(insp_end_ms, win_post_s * 1000.0, color="#dbeafe", alpha=0.7)

    # Scatter Warped Spikes
    if len(spk_x) > 0:
        ax_raster.scatter(
            spk_x,
            spk_y,
            marker='|',
            color='black',
            s=45,
            linewidth=1.2,
            alpha=0.9
        )

    # Landmark boundary lines
    ax_raster.axvline(0, color="#b91c1c", linestyle="--", linewidth=1.8, label="Onset (I-on: 0 ms)")
    ax_raster.axvline(insp_end_ms, color="#1d4ed8", linestyle="--", linewidth=1.8, label=f"Offset (I-off: {insp_end_ms:.0f} ms)")
    if cycle_end_ms <= win_post_s * 1000.0:
        ax_raster.axvline(cycle_end_ms, color="#b91c1c", linestyle=":", linewidth=1.5, label=f"Next Onset ({cycle_end_ms:.0f} ms)")

    ax_raster.set_xlim(-win_pre_s * 1000.0, win_post_s * 1000.0)
    ax_raster.set_ylim(0.5, n_trials + 1.5)
    ax_raster.set_ylabel("Breath Trial Index", fontsize=11, fontweight="bold")
    ax_raster.set_title(
        f"Unit #{unit_id} (Depth: {depth_um:.0f} µm) — Time-Warped Breath-Phase Raster\n"
        f"[Warped to Mean Cycle: Insp = {insp_end_ms:.0f} ms | Total Breath Period = {cycle_end_ms:.0f} ms]",
        fontsize=11,
        fontweight="bold"
    )
    ax_raster.legend(loc="upper right", frameon=True, fontsize=8, ncol=2)

    # -------------------------------------------------------------
    # Bottom Panel: Time-Warped PETH
    # -------------------------------------------------------------
    ax_peth = fig.add_subplot(gs[1], sharex=ax_raster)

    # Background shading in PETH
    ax_peth.axvspan(-win_pre_s * 1000.0, 0, color="#dbeafe", alpha=0.4)
    ax_peth.axvspan(0, insp_end_ms, color="#fee2e2", alpha=0.5)
    ax_peth.axvspan(insp_end_ms, win_post_s * 1000.0, color="#dbeafe", alpha=0.4)

    # Mean Firing Rate Line + SD Shadow
    ax_peth.plot(time_ms, mean_fr, color="#0f766e", linewidth=2.5, label="Mean Firing Rate (Warped)")
    ax_peth.fill_between(
        time_ms,
        np.maximum(0, mean_fr - std_fr),
        mean_fr + std_fr,
        color="#0f766e",
        alpha=0.25,
        label=r"$\pm 1$ SD Shadow"
    )

    ax_peth.axvline(0, color="#b91c1c", linestyle="--", linewidth=1.5)
    ax_peth.axvline(insp_end_ms, color="#1d4ed8", linestyle="--", linewidth=1.5)
    if cycle_end_ms <= win_post_s * 1000.0:
        ax_peth.axvline(cycle_end_ms, color="#b91c1c", linestyle=":", linewidth=1.2)

    ax_peth.set_xlabel("Warped Time from Breath Onset (ms)", fontsize=11, fontweight="bold")
    ax_peth.set_ylabel("Firing Rate (sp/s)", fontsize=11, fontweight="bold", color="#0f766e")
    ax_peth.tick_params(axis='y', labelcolor="#0f766e")
    ax_peth.grid(True, linestyle=":", alpha=0.6)
    ax_peth.legend(loc="upper right", frameon=True, fontsize=8)

    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Time-Warped Phase-Aligned Rasters & PETHs for Each Clean Single Unit")
    parser.add_argument("--consensus-dir", type=str, default="'output_LP21_60s_120s'/sliding_window_pipeline/lupin/consensus_master")
    parser.add_argument("--raw-dir", type=str, default=r"Z:\ramirezlab\pascual\rec_data\LP21_fent_EXPT_g1")
    parser.add_argument("--out-dir", type=str, default="output_timewarped_neuron_rasters_LP21_60s")
    parser.add_argument("--start-time", type=float, default=60.0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds")
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

    start_t = args.start_time
    rec_dur = args.duration

    logger.info("=" * 85)
    logger.info(" GENERATING TIME-WARPED PHASE-ALIGNED RASTERS & PETHS")
    logger.info(f" Consensus Directory : {cons_dir.resolve()}")
    logger.info(f" Output Directory    : {out_dir.resolve()}")
    logger.info(f" Time Slice Window   : {start_t:.1f}s to {start_t + rec_dur:.1f}s")
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

    # 2. Extract Breaths from EMG
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
    breath_offsets = emg_proc.breath_offsets[(emg_proc.breath_offsets >= 0.0) & (emg_proc.breath_offsets <= rec_dur)]

    n_breaths = min(len(breath_onsets), len(breath_offsets))
    t_insp_list = []
    t_exp_list = []
    for i in range(n_breaths - 1):
        ti = breath_offsets[i] - breath_onsets[i]
        te = breath_onsets[i + 1] - breath_offsets[i]
        if ti > 0.01 and te > 0.01:
            t_insp_list.append(ti)
            t_exp_list.append(te)

    mean_t_insp = float(np.mean(t_insp_list))
    mean_t_exp = float(np.mean(t_exp_list))
    mean_t_cycle = mean_t_insp + mean_t_exp

    logger.info(f"Mean Respiratory Parameters ({len(t_insp_list)} Breaths):")
    logger.info(f" • Mean Inspiration (T_I) : {mean_t_insp * 1000.0:.1f} ms")
    logger.info(f" • Mean Expiration (T_E)  : {mean_t_exp * 1000.0:.1f} ms")
    logger.info(f" • Mean Cycle Period (T)  : {mean_t_cycle * 1000.0:.1f} ms (Freq: {1.0 / mean_t_cycle:.2f} Hz)")

    # 3. Generate Individual Time-Warped Figures
    index_cards = []
    for idx, uid in enumerate(sua_ids):
        depth = df_sua[df_sua["cluster_id"] == uid]["depth_um"].values[0] if "depth_um" in df_sua.columns else 0.0
        u_spks = spk_times[spk_clusters == uid]
        out_png = out_dir / f"unit_{uid:03d}_depth_{int(depth)}um_timewarped_raster_peth.png"

        plot_single_neuron_timewarped_raster_and_peth(
            unit_id=uid,
            depth_um=depth,
            spk_times=u_spks,
            breath_onsets=breath_onsets,
            breath_offsets=breath_offsets,
            mean_t_insp=mean_t_insp,
            mean_t_exp=mean_t_exp,
            mean_t_cycle=mean_t_cycle,
            out_file=out_png,
            win_pre_s=0.15,
            win_post_s=0.40
        )
        index_cards.append({
            "unit_id": int(uid),
            "depth_um": float(depth),
            "n_spikes": len(u_spks),
            "file_name": out_png.name
        })

    # 4. Generate HTML Visual Gallery
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Time-Warped Phase-Aligned Rasters & PETHs</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }}
        h1 {{ text-align: center; color: #38bdf8; margin-bottom: 5px; }}
        p.subtitle {{ text-align: center; color: #94a3b8; font-size: 14px; margin-bottom: 25px; }}
        .metrics-banner {{ display: flex; justify-content: center; gap: 20px; margin-bottom: 25px; }}
        .metric-box {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 10px 20px; text-align: center; }}
        .metric-val {{ font-size: 20px; font-weight: bold; color: #38bdf8; }}
        .metric-lbl {{ font-size: 12px; color: #94a3b8; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 20px; max-width: 1800px; margin: 0 auto; }}
        .card {{ background: #1e293b; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5); border: 1px solid #334155; }}
        .card img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
        .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .badge {{ background: #0284c7; color: white; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Time-Warped Phase-Aligned Rasters & PETHs</h1>
    <p class="subtitle">37 Clean Single Units | Piecewise Linear Phase-Warping to Mean Breath Cycle</p>
    
    <div class="metrics-banner">
        <div class="metric-box">
            <div class="metric-val">{mean_t_insp * 1000.0:.1f} ms</div>
            <div class="metric-lbl">Mean Inspiration (T<sub>I</sub>)</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{mean_t_exp * 1000.0:.1f} ms</div>
            <div class="metric-lbl">Mean Expiration (T<sub>E</sub>)</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{mean_t_cycle * 1000.0:.1f} ms</div>
            <div class="metric-lbl">Mean Cycle Period (T)</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{1.0 / mean_t_cycle:.2f} Hz</div>
            <div class="metric-lbl">Breathing Frequency</div>
        </div>
    </div>

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
    gallery_path = out_dir / "gallery_timewarped_37_neurons.html"
    gallery_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Saved Time-Warped Visual Gallery: {gallery_path.resolve()}")

    print("\n" + "=" * 85)
    print(" [SUCCESS] ALL 37 TIME-WARPED NEURON RASTERS & PETHS GENERATED")
    print("=" * 85)
    print(f" • Output Folder       : {out_dir.resolve()}")
    print(f" • Interactive Gallery : {gallery_path.resolve()}")
    print("=" * 85)


if __name__ == "__main__":
    main()
