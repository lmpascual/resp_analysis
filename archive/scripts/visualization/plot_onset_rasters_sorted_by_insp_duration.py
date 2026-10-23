#!/usr/bin/env python3
"""
Generate Breath-Onset Aligned Spike Rasters and PETHs for Each Clean Single Unit,
with Trials Ordered by Inspiration Duration (T_I).

Features:
1. Top Panel: True Real-Time Breath-Onset Aligned Spike Raster
   - Trials sorted in ascending order by Inspiration Duration T_I (bottom = shortest, top = longest).
   - Background colored by continuous respiratory phase phi(t) for that trial:
     * Inspiration (0 -> +pi) in warm/red/coral tones
     * Expiration (-pi -> 0) in cool/blue/cyan tones
   - Overlay of inspiration offset curve (I_off) marking the transition between inspiration and expiration.
   - Black spike ticks overlaid with precision.
2. Bottom Panel: Peri-Event Time Histogram (PETH)
   - Real-time mean firing rate (spikes/s) solid curve + (+/- 1 SD) standard deviation shadow.
   - Secondary y-axis with synchronized Diaphragm EMG Envelope.
   - Vertical landmark guides at t=0 ms (I_on) and mean T_I (mean I_off).
3. Interactive HTML Visual Gallery for instant visual browsing of all 37 clean single units.
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
from matplotlib.colors import LinearSegmentedColormap

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ONSET_TI_SORTED_RASTERS")


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
    phi = np.full_like(t_eval, np.nan, dtype=np.float64)
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

    # Fill boundaries if any NaNs remain at outer edges
    valid_idx = np.where(~np.isnan(phi))[0]
    if len(valid_idx) > 0:
        phi[:valid_idx[0]] = phi[valid_idx[0]]
        phi[valid_idx[-1]:] = phi[valid_idx[-1]]

    return phi


def plot_single_neuron_onset_ti_sorted_raster_and_peth(
    unit_id: int,
    depth_um: float,
    firing_rate_hz: float,
    spk_times: np.ndarray,
    valid_onsets: np.ndarray,
    valid_offsets: np.ndarray,
    sort_order: np.ndarray,
    breath_onsets_all: np.ndarray,
    breath_offsets_all: np.ndarray,
    t_nidq: np.ndarray,
    emg_env: np.ndarray,
    out_file: Path,
    win_pre_s: float = 0.15,
    win_post_s: float = 0.35,
    dt_s: float = 0.002
):
    """
    Generates a 2-panel figure for a single neuron:
    Top: Breath-Onset Aligned Raster with trials ordered by inspiration duration (T_I).
    Bottom: PETH with Mean FR + SD shadow and synchronized Diaphragm EMG.
    """
    time_axis_s = np.arange(-win_pre_s, win_post_s + dt_s, dt_s)
    time_axis_ms = time_axis_s * 1000.0
    n_time_bins = len(time_axis_s)
    n_breaths = len(sort_order)

    # 1. Build Phase Background Grid, Sorted Spikes & Firing Rates
    phase_grid = np.zeros((n_breaths, n_time_bins))
    rate_trials = np.zeros((n_breaths, n_time_bins))
    emg_trials = np.zeros((n_breaths, n_time_bins))
    spike_scatter_x = []
    spike_scatter_y = []
    ioff_curve_x = []
    ioff_curve_y = []

    kernel = signal.windows.gaussian(15, std=3)  # ~6 ms std for dt=2ms
    kernel /= np.sum(kernel)

    # Calculate sorted T_I values
    sorted_ti_ms = []

    for row_idx, orig_idx in enumerate(sort_order):
        on = valid_onsets[orig_idx]
        off = valid_offsets[orig_idx]
        ti = off - on
        sorted_ti_ms.append(ti * 1000.0)

        # Inspiration offset point for this row
        ioff_curve_x.append(ti * 1000.0)
        ioff_curve_y.append(row_idx + 1)

        # Continuous phase background for this trial
        trial_t_eval = on + time_axis_s
        trial_phi = compute_continuous_respiratory_phase(breath_onsets_all, breath_offsets_all, trial_t_eval)
        phase_grid[row_idx, :] = trial_phi

        # Extract spikes in trial window
        rel_spks = spk_times - on
        in_win = rel_spks[(rel_spks >= -win_pre_s) & (rel_spks <= win_post_s)]
        for s in in_win:
            spike_scatter_x.append(s * 1000.0)  # ms
            spike_scatter_y.append(row_idx + 1) # 1-indexed trial row

        # Instantaneous firing rate for trial
        counts, _ = np.histogram(in_win, bins=np.append(time_axis_s, time_axis_s[-1] + dt_s))
        smooth_fr = np.convolve(counts / dt_s, kernel, mode="same")
        rate_trials[row_idx, :] = smooth_fr

        # EMG envelope for trial
        emg_trials[row_idx, :] = np.interp(trial_t_eval, t_nidq, emg_env, left=0.0, right=0.0)

    mean_fr = np.mean(rate_trials, axis=0)
    std_fr = np.std(rate_trials, axis=0)

    mean_emg = np.mean(emg_trials, axis=0)
    emg_min, emg_max = np.min(mean_emg), np.max(mean_emg)
    norm_emg = (mean_emg - emg_min) / (emg_max - emg_min + 1e-12) if emg_max > emg_min else np.zeros_like(mean_emg)

    mean_ti_val_ms = np.mean(sorted_ti_ms)
    min_ti_val_ms = np.min(sorted_ti_ms)
    max_ti_val_ms = np.max(sorted_ti_ms)

    # 2. Setup Figure Layout
    fig = plt.figure(figsize=(10.5, 8.2))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.3, 1.0], hspace=0.18)

    # -------------------------------------------------------------
    # Top Panel: Raster with Phase-Colored Background & T_I Sorted
    # -------------------------------------------------------------
    ax_raster = fig.add_subplot(gs[0])

    # Colormap: Cool Blue (-pi) -> White (0) -> Warm Red (+pi)
    cmap_phase = LinearSegmentedColormap.from_list(
        "phase_cmap_onset",
        ["#93c5fd", "#dbeafe", "#ffffff", "#fed7aa", "#f87171"],
        N=256
    )

    im_phase = ax_raster.imshow(
        phase_grid,
        aspect="auto",
        cmap=cmap_phase,
        vmin=-np.pi,
        vmax=np.pi,
        extent=[time_axis_ms[0], time_axis_ms[-1], 0.5, n_breaths + 0.5],
        origin="lower",
        alpha=0.60,
        interpolation="bilinear"
    )

    # Overlay Individual Spikes as crisp black vertical tick marks
    if len(spike_scatter_x) > 0:
        ax_raster.scatter(
            spike_scatter_x,
            spike_scatter_y,
            marker='|',
            color='black',
            s=45,
            linewidth=1.2,
            alpha=0.92,
            zorder=3,
            label="Spike Times"
        )

    # Overlay Inspiration Offset boundary line (I_off) across sorted trials
    ax_raster.plot(
        ioff_curve_x,
        ioff_curve_y,
        color="#b91c1c",
        linestyle="-",
        linewidth=2.2,
        alpha=0.90,
        zorder=4,
        label=r"Inspiration Offset ($I_{\mathrm{off}}$ / $T_I$)"
    )

    # Vertical line at Breath Onset (I_on: 0 ms)
    ax_raster.axvline(0, color="#1e293b", linestyle="--", linewidth=1.6, alpha=0.85, zorder=4)

    ax_raster.set_xlim(time_axis_ms[0], time_axis_ms[-1])
    ax_raster.set_ylim(0.5, n_breaths + 0.5)
    ax_raster.set_ylabel(f"Breath Trials (Sorted by $T_I$: {min_ti_val_ms:.0f} → {max_ti_val_ms:.0f} ms)", fontsize=11, fontweight="bold")
    ax_raster.set_title(
        f"Unit #{unit_id} (Depth: {depth_um:.0f} µm | FR: {firing_rate_hz:.1f} Hz) — Breath-Onset Aligned Spike Raster\n"
        r"[Trials Ordered by Inspiration Duration $T_I$ | Warm Red = Insp ($\phi \in [0, \pi]$) | Cool Blue = Exp ($\phi \in [-\pi, 0]$)]",
        fontsize=11,
        fontweight="bold"
    )
    ax_raster.grid(False)

    # Colorbar for phase
    cbar = plt.colorbar(im_phase, ax=ax_raster, fraction=0.03, pad=0.02)
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels([r"$-\pi$ (Exp)", r"$-\frac{\pi}{2}$", r"$0$ ($I_{\mathrm{on}}$)", r"$+\frac{\pi}{2}$", r"$+\pi$ ($I_{\mathrm{off}}$)"], fontsize=8)

    # Legend for raster
    ax_raster.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=0.85, fontsize=8)

    # -------------------------------------------------------------
    # Bottom Panel: PETH with Mean FR + SD Shadow & EMG Reference
    # -------------------------------------------------------------
    ax_peth = fig.add_subplot(gs[1], sharex=ax_raster)

    p_fr = ax_peth.plot(time_axis_ms, mean_fr, color="#0284c7", linewidth=2.5, label="Mean Firing Rate")
    ax_peth.fill_between(
        time_axis_ms,
        np.maximum(0, mean_fr - std_fr),
        mean_fr + std_fr,
        color="#0284c7",
        alpha=0.25,
        label=r"$\pm 1$ SD Shadow"
    )

    # Vertical landmarks on PETH
    ax_peth.axvline(0, color="#1e293b", linestyle="--", linewidth=1.5, alpha=0.85, label=r"Breath Onset ($I_{\mathrm{on}}$: 0 ms)")
    ax_peth.axvline(mean_ti_val_ms, color="#b91c1c", linestyle=":", linewidth=1.8, alpha=0.90, label=f"Mean $I_{{\\mathrm{{off}}}}$ ({mean_ti_val_ms:.1f} ms)")

    ax_peth.set_xlabel("Time from Breath Onset (ms)", fontsize=11, fontweight="bold")
    ax_peth.set_ylabel("Firing Rate (sp/s)", fontsize=11, fontweight="bold", color="#0284c7")
    ax_peth.tick_params(axis='y', labelcolor="#0284c7")
    ax_peth.grid(True, linestyle=":", alpha=0.6)

    # Secondary y-axis for Diaphragm EMG
    ax_emg = ax_peth.twinx()
    p_emg = ax_emg.plot(time_axis_ms, norm_emg, color="#475569", linestyle=":", linewidth=2.0, label="Dia. EMG Envelope")
    ax_emg.set_ylabel("Dia. EMG (norm)", fontsize=10, color="#475569")
    ax_emg.tick_params(axis='y', labelcolor="#475569")
    ax_emg.set_ylim(-0.05, 1.25)

    # Unified Legend
    lines_peth, labels_peth = ax_peth.get_legend_handles_labels()
    lines_emg, labels_emg = ax_emg.get_legend_handles_labels()
    ax_peth.legend(lines_peth + lines_emg, labels_peth + labels_emg, loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=0.85, fontsize=8)

    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def generate_gallery_onset_ti_sorted(
    consensus_dir: Path,
    raw_dir: Path,
    out_dir: Path,
    start_time: float = 60.0,
    duration: float = 60.0,
    win_pre_s: float = 0.15,
    win_post_s: float = 0.35
):
    """
    Generates all 37 unit figures sorted by inspiration duration and builds the visual HTML gallery.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 85)
    logger.info(" GENERATING BREATH-ONSET ALIGNED RASTERS ORDERED BY INSPIRATION DURATION (T_I)")
    logger.info(f" Consensus Directory : {consensus_dir.resolve()}")
    logger.info(f" Raw Directory       : {raw_dir.resolve()}")
    logger.info(f" Output Directory    : {out_dir.resolve()}")
    logger.info(f" Time Slice Window   : {start_time:.1f}s to {start_time + duration:.1f}s")
    logger.info("=" * 85)

    # 1. Load Clean Single Units
    alf_dir = consensus_dir / "alf"
    spk_times = np.load(alf_dir / "spike_times.npy")
    spk_clusters = np.load(alf_dir / "spike_clusters.npy")
    df_info = pd.read_csv(consensus_dir / "cluster_info.tsv", sep="\t")

    qc_col = "passed_qc" if "passed_qc" in df_info.columns else ("passed_allen_qc" if "passed_allen_qc" in df_info.columns else None)
    df_sua = df_info[df_info[qc_col] == True].copy() if qc_col else df_info.copy()

    sua_ids = np.array(df_sua["cluster_id"].values)
    mask_sua = np.isin(spk_clusters, sua_ids)
    spk_times = spk_times[mask_sua]
    spk_clusters = spk_clusters[mask_sua]

    rec_dur = min(float(np.max(spk_times)), duration) if len(spk_times) > 0 else duration

    # 2. Extract Breaths from EMG
    nidq_bins = list(raw_dir.glob("*.nidq.bin"))
    if not nidq_bins:
        raise FileNotFoundError(f"No .nidq.bin in {raw_dir}")
    nidq_bin = nidq_bins[0]
    nidq_data = read_nidq_bin(nidq_bin, nidq_bin.with_suffix(".meta"), emg_chan=0, ecg_chan=1)

    t_nidq_raw = nidq_data.time_vector
    t_end = start_time + rec_dur
    mask_n = (t_nidq_raw >= start_time) & (t_nidq_raw <= t_end)
    t_nidq = t_nidq_raw[mask_n] - start_time
    emg_raw = nidq_data.all_channels[0][mask_n]

    emg_cfg = EMGConfig(fs=nidq_data.fs, highpass_cutoff=300.0, lowpass_cutoff=5000.0, smoothing_window_ms=20.0)
    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    breath_onsets_all = emg_proc.breath_onsets[(emg_proc.breath_onsets >= 0.0) & (emg_proc.breath_onsets <= rec_dur)]
    breath_offsets_all = emg_proc.breath_offsets[(emg_proc.breath_offsets >= 0.0) & (emg_proc.breath_offsets <= rec_dur)]

    # Pair valid breaths
    n_pairs = min(len(breath_onsets_all), len(breath_offsets_all))
    valid_onsets_list = []
    valid_offsets_list = []
    ti_list = []

    for i in range(n_pairs):
        on = breath_onsets_all[i]
        off = breath_offsets_all[i]
        if on > win_pre_s and on < t_nidq[-1] - win_post_s and off > on:
            valid_onsets_list.append(on)
            valid_offsets_list.append(off)
            ti_list.append(off - on)

    valid_onsets = np.array(valid_onsets_list)
    valid_offsets = np.array(valid_offsets_list)
    ti_arr = np.array(ti_list)

    # Sort order: ascending order of inspiration duration T_I
    sort_order = np.argsort(ti_arr)

    logger.info(
        f"Prepared {len(sort_order)} valid breath cycles. "
        f"Inspiration duration range: {np.min(ti_arr)*1000.0:.1f} ms to {np.max(ti_arr)*1000.0:.1f} ms "
        f"(Mean: {np.mean(ti_arr)*1000.0:.1f} ms, Median: {np.median(ti_arr)*1000.0:.1f} ms)."
    )

    # 3. Generate Individual Figures
    index_cards = []
    for idx, uid in enumerate(sua_ids):
        row_unit = df_sua[df_sua["cluster_id"] == uid].iloc[0]
        depth = float(row_unit["depth_um"]) if "depth_um" in row_unit else 0.0
        fr_hz = float(row_unit["firing_rate_hz"]) if "firing_rate_hz" in row_unit else 0.0
        u_spks = spk_times[spk_clusters == uid]

        out_png = out_dir / f"unit_{uid:03d}_depth_{int(depth)}um_raster_peth.png"

        plot_single_neuron_onset_ti_sorted_raster_and_peth(
            unit_id=uid,
            depth_um=depth,
            firing_rate_hz=fr_hz,
            spk_times=u_spks,
            valid_onsets=valid_onsets,
            valid_offsets=valid_offsets,
            sort_order=sort_order,
            breath_onsets_all=breath_onsets_all,
            breath_offsets_all=breath_offsets_all,
            t_nidq=t_nidq,
            emg_env=emg_proc.envelope,
            out_file=out_png,
            win_pre_s=win_pre_s,
            win_post_s=win_post_s,
            dt_s=0.002
        )

        index_cards.append({
            "unit_id": int(uid),
            "depth_um": float(depth),
            "firing_rate_hz": float(fr_hz),
            "n_spikes": len(u_spks),
            "file_name": out_png.name
        })

    # 4. Generate Responsive HTML Visual Gallery
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Breath-Onset Aligned Rasters (Ordered by Inspiration Duration)</title>
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: #151d30;
            --card-border: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-coral: #f43f5e;
            --badge-bg: #0369a1;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
        }}
        .header-container {{
            text-align: center;
            max-width: 1200px;
            margin: 0 auto 30px auto;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--card-border);
        }}
        h1 {{
            color: var(--accent-blue);
            font-size: 26px;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }}
        p.subtitle {{
            color: var(--text-secondary);
            font-size: 14px;
            line-height: 1.5;
            margin: 0;
        }}
        .stats-badge-container {{
            display: flex;
            justify-content: center;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 16px;
        }}
        .stat-pill {{
            background: #1e293b;
            border: 1px solid #334155;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            color: #cbd5e1;
        }}
        .stat-pill b {{
            color: var(--accent-blue);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
            gap: 24px;
            max-width: 1800px;
            margin: 0 auto;
        }}
        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            border: 1px solid var(--card-border);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
            border-color: #3b82f6;
        }}
        .card img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            display: block;
            background: #000;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        .unit-title {{
            font-weight: 700;
            font-size: 17px;
            color: #ffffff;
        }}
        .badge {{
            background: var(--badge-bg);
            color: #e0f2fe;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.3px;
        }}
    </style>
</head>
<body>
    <div class="header-container">
        <h1>Single-Neuron Breath-Onset Rasters (Sorted by Inspiration Duration)</h1>
        <p class="subtitle">
            True Real-Time Spike Rasters (No Phase Warping) aligned to Breath Onset (I-on = 0 ms).<br>
            Trials are ordered in ascending order by Inspiration Duration (<i>T<sub>I</sub></i>) to reveal burst duration modulation & phase-locked termination.<br>
            Background: Warm Red = Inspiration (&phi; &in; [0, &pi;]) | Cool Blue = Expiration (&phi; &in; [-&pi;, 0]) | Red Curve = Inspiration Offset (<i>I</i><sub>off</sub>).
        </p>
        <div class="stats-badge-container">
            <div class="stat-pill"><b>37</b> Clean Single Units</div>
            <div class="stat-pill"><b>{len(sort_order)}</b> Synchronized Breaths</div>
            <div class="stat-pill">Inspiration Duration (<i>T<sub>I</sub></i>): <b>{np.min(ti_arr)*1000.0:.0f} ms → {np.max(ti_arr)*1000.0:.0f} ms</b> (Mean: <b>{np.mean(ti_arr)*1000.0:.1f} ms</b>)</div>
            <div class="stat-pill">Alignment: <b>Breath Onset (0 ms)</b></div>
        </div>
    </div>
    <div class="grid">
"""
    for card in index_cards:
        html_content += f"""
        <div class="card">
            <div class="card-header">
                <span class="unit-title">Unit #{card['unit_id']}</span>
                <span class="badge">{card['depth_um']:.0f} µm | {card['firing_rate_hz']:.1f} Hz | {card['n_spikes']} Spikes</span>
            </div>
            <a href="{card['file_name']}" target="_blank" title="Click to open full resolution image">
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
    print(" [SUCCESS] ALL 37 ONSET-ALIGNED & T_I-SORTED NEURON RASTERS & PETHS GENERATED")
    print("=" * 85)
    print(f" • Output Folder       : {out_dir.resolve()}")
    print(f" • Interactive Gallery : {gallery_path.resolve()}")
    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description="Generate Breath-Onset Aligned Rasters Sorted by Inspiration Duration")
    parser.add_argument("--consensus-dir", type=str, default="LP21_fent_EXPT_g1_60s_120s_lupin_run/00_spike_sorting_data")
    parser.add_argument("--raw-dir", type=str, default=r"Z:\ramirezlab\pascual\rec_data\LP21_fent_EXPT_g1")
    parser.add_argument("--out-dir", type=str, default="LP21_fent_EXPT_g1_60s_120s_lupin_run/02_single_cell_phase_rasters/onset_aligned_sorted_by_insp_duration")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--start-time", type=float, default=60.0)
    args = parser.parse_args()

    cons_dir_raw = args.consensus_dir.strip('"')
    cons_dir = Path(cons_dir_raw)
    if not cons_dir.exists():
        alt_path = PROJECT_ROOT / "output_LP21_60s_120s" / "sliding_window_pipeline" / "lupin" / "consensus_master"
        if alt_path.exists():
            cons_dir = alt_path

    raw_dir = Path(args.raw_dir.strip('"'))
    out_dir = Path(args.out_dir)

    generate_gallery_onset_ti_sorted(
        consensus_dir=cons_dir,
        raw_dir=raw_dir,
        out_dir=out_dir,
        start_time=args.start_time,
        duration=args.duration,
        win_pre_s=0.15,
        win_post_s=0.35
    )


if __name__ == "__main__":
    main()
