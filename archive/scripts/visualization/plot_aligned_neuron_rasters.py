#!/usr/bin/env python3
"""
Generate True Real-Time Breath-Aligned Spike Rasters and PETHs for Each Clean Single Unit.
Supports alignment to:
1. 'offset': Diaphragm burst offset (I-off / start of expiration at t=0 ms)
2. 'peak'  : Diaphragm burst peak (maximum burst amplitude at t=0 ms)
3. 'onset' : Diaphragm burst onset (I-on / start of inspiration at t=0 ms)

Includes:
- True real-time spikes (no time-warping)
- Background colored by continuous breath phase phi(t) for each individual breath trial
- Trial-averaged PETH with +/- 1 SD shadow
- Trial-averaged Diaphragm EMG Envelope
- Responsive HTML Visual Gallery
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
logger = logging.getLogger("ALIGNED_RASTERS")


def compute_continuous_phase(
    t_nidq: np.ndarray,
    breath_onsets: np.ndarray,
    breath_offsets: np.ndarray
) -> np.ndarray:
    """
    Computes continuous respiratory phase phi(t) in [-pi, pi]:
    Inspiration [on_i, off_i] -> [0, pi]
    Expiration [off_i, on_i+1] -> [-pi, 0]
    """
    phi = np.full(len(t_nidq), np.nan, dtype=np.float32)
    n_breaths = min(len(breath_onsets), len(breath_offsets))

    for i in range(n_breaths - 1):
        t_on = breath_onsets[i]
        t_off = breath_offsets[i]
        t_next_on = breath_onsets[i + 1]

        if t_off <= t_on or t_next_on <= t_off:
            continue

        # Inspiration: [t_on, t_off] -> [0, pi]
        mask_insp = (t_nidq >= t_on) & (t_nidq < t_off)
        if np.any(mask_insp):
            phi[mask_insp] = (t_nidq[mask_insp] - t_on) / (t_off - t_on) * np.pi

        # Expiration: [t_off, t_next_on] -> [-pi, 0]
        mask_exp = (t_nidq >= t_off) & (t_nidq < t_next_on)
        if np.any(mask_exp):
            phi[mask_exp] = -np.pi + (t_nidq[mask_exp] - t_off) / (t_next_on - t_off) * np.pi

    # Fill boundaries
    if not np.isnan(phi).all():
        valid_idx = np.where(~np.isnan(phi))[0]
        phi[:valid_idx[0]] = phi[valid_idx[0]]
        phi[valid_idx[-1]:] = phi[valid_idx[-1]]

    return phi


def plot_single_neuron_aligned_raster_and_peth(
    unit_id: int,
    depth_um: float,
    spk_times: np.ndarray,
    align_events: np.ndarray,
    t_nidq: np.ndarray,
    phi_nidq: np.ndarray,
    emg_envelope: np.ndarray,
    align_mode: str,
    out_file: Path,
    win_pre_s: float = 0.20,
    win_post_s: float = 0.35,
    dt_peth: float = 0.002
):
    """
    Generates a 2-panel figure for a single neuron aligned to specified landmark:
    Top: True real-time spike raster with trial-by-trial continuous phase phi(t) colored background
    Bottom: PETH with +/- 1 SD shadow and normalized trial-averaged Diaphragm EMG
    """
    n_trials = len(align_events)
    time_bins_s = np.arange(-win_pre_s, win_post_s + dt_peth, dt_peth)
    time_centers_ms = (time_bins_s[:-1] + time_bins_s[1:]) / 2.0 * 1000.0
    n_bins = len(time_centers_ms)

    trial_peths = np.zeros((n_trials, n_bins))
    trial_emg = np.zeros((n_trials, n_bins))

    # Dense time vector for 2D background phase interpolation
    n_bg_bins = 250
    dense_rel_t_s = np.linspace(-win_pre_s, win_post_s, n_bg_bins)
    phase_bg_matrix = np.full((n_trials, n_bg_bins), np.nan)

    spike_x = []
    spike_y = []

    kernel = signal.windows.gaussian(15, std=3)
    kernel /= np.sum(kernel)

    for i, t_ev in enumerate(align_events):
        # 1. Spikes
        t_start = t_ev - win_pre_s
        t_end = t_ev + win_post_s
        trial_spks = spk_times[(spk_times >= t_start) & (spk_times <= t_end)] - t_ev
        for s in trial_spks:
            spike_x.append(s * 1000.0)  # ms
            spike_y.append(i + 1)

        counts, _ = np.histogram(trial_spks, bins=time_bins_s)
        trial_peths[i, :] = np.convolve(counts / dt_peth, kernel, mode="same")

        # 2. Phase background
        trial_abs_t = dense_rel_t_s + t_ev
        phase_bg_matrix[i, :] = np.interp(trial_abs_t, t_nidq, phi_nidq, left=np.nan, right=np.nan)

        # 3. EMG envelope
        trial_emg[i, :] = np.interp(time_centers_ms / 1000.0 + t_ev, t_nidq, emg_envelope, left=0.0, right=0.0)

    mean_peth = np.mean(trial_peths, axis=0)
    std_peth = np.std(trial_peths, axis=0)

    mean_emg = np.mean(trial_emg, axis=0)
    emg_min, emg_max = np.min(mean_emg), np.max(mean_emg)
    norm_emg = (mean_emg - emg_min) / (emg_max - emg_min + 1e-12) if emg_max > emg_min else np.zeros_like(mean_emg)

    # Colormap: Cool Blue (-pi) -> White (0) -> Warm Red (+pi)
    cmap_phase = LinearSegmentedColormap.from_list(
        "phase_cmap",
        ["#93c5fd", "#dbeafe", "#ffffff", "#fed7aa", "#f87171"],
        N=256
    )

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1.0], hspace=0.18)

    # -------------------------------------------------------------
    # Top Panel: Raster with Phase-Colored Background
    # -------------------------------------------------------------
    ax_raster = fig.add_subplot(gs[0])

    im = ax_raster.imshow(
        phase_bg_matrix,
        extent=[-win_pre_s * 1000.0, win_post_s * 1000.0, 0.5, n_trials + 0.5],
        origin='lower',
        aspect='auto',
        cmap=cmap_phase,
        vmin=-np.pi,
        vmax=np.pi,
        alpha=0.65
    )

    if len(spike_x) > 0:
        ax_raster.scatter(
            spike_x,
            spike_y,
            marker='|',
            color='black',
            s=45,
            linewidth=1.2,
            alpha=0.9
        )

    # Vertical line at alignment landmark
    landmark_label = {
        "offset": "Breath Offset (I-off / Exp Onset: 0 ms)",
        "peak": "Diaphragm Peak Amplitude (0 ms)",
        "onset": "Breath Onset (I-on: 0 ms)"
    }.get(align_mode, "Alignment Landmark (0 ms)")

    ax_raster.axvline(0, color="#b91c1c" if align_mode != "offset" else "#1d4ed8", linestyle="--", linewidth=1.8, label=landmark_label)
    ax_raster.set_xlim(-win_pre_s * 1000.0, win_post_s * 1000.0)
    ax_raster.set_ylim(0.5, n_trials + 0.5)
    ax_raster.set_ylabel("Breath Trial Index", fontsize=11, fontweight="bold")
    
    title_mode = {
        "offset": "Aligned to Breath Offset (End of Inspiration / Start of Expiration)",
        "peak": "Aligned to Peak Diaphragm Burst Amplitude",
        "onset": "Aligned to Breath Onset (Start of Inspiration)"
    }.get(align_mode, f"Aligned to {align_mode}")

    ax_raster.set_title(
        f"Unit #{unit_id} (Depth: {depth_um:.0f} µm) — True Real-Time Spike Raster\n[{title_mode}]",
        fontsize=11,
        fontweight="bold"
    )

    # Add Colorbar for Phase
    cbar = plt.colorbar(im, ax=ax_raster, fraction=0.03, pad=0.02)
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels([r"$-\pi$ (Exp)", r"$-\frac{\pi}{2}$", r"$0$ (I-on)", r"$+\frac{\pi}{2}$", r"$+\pi$ (I-off)"])
    cbar.ax.tick_params(labelsize=8)

    # -------------------------------------------------------------
    # Bottom Panel: PETH + SD Shadow + Diaphragm EMG
    # -------------------------------------------------------------
    ax_peth = fig.add_subplot(gs[1], sharex=ax_raster)

    # Firing rate curve & SD shadow
    p_fr = ax_peth.plot(time_centers_ms, mean_peth, color="#0284c7", linewidth=2.5, label="Mean Firing Rate")
    ax_peth.fill_between(
        time_centers_ms,
        np.maximum(0, mean_peth - std_peth),
        mean_peth + std_peth,
        color="#0284c7",
        alpha=0.25,
        label=r"$\pm 1$ SD Shadow"
    )

    ax_peth.axvline(0, color="#b91c1c" if align_mode != "offset" else "#1d4ed8", linestyle="--", linewidth=1.5)
    ax_peth.set_xlabel(f"Time from {align_mode.capitalize()} (ms)", fontsize=11, fontweight="bold")
    ax_peth.set_ylabel("Firing Rate (sp/s)", fontsize=11, fontweight="bold", color="#0284c7")
    ax_peth.tick_params(axis='y', labelcolor="#0284c7")
    ax_peth.grid(True, linestyle=":", alpha=0.6)

    # Twin axis for normalized Diaphragm EMG
    ax_emg = ax_peth.twinx()
    p_emg = ax_emg.plot(time_centers_ms, norm_emg, color="#475569", linestyle=":", linewidth=2.0, label="Dia. EMG Envelope")
    ax_emg.set_ylabel("Dia. EMG (norm)", fontsize=10, color="#475569")
    ax_emg.tick_params(axis='y', labelcolor="#475569")
    ax_emg.set_ylim(-0.05, 1.25)

    # Combined Legend
    lines = p_fr + [ax_peth.collections[0]] + p_emg
    labels = ["Mean Firing Rate", r"$\pm 1$ SD Shadow", "Dia. EMG Envelope"]
    ax_peth.legend(lines, labels, loc="upper right", frameon=True, fontsize=8)

    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def generate_gallery_for_alignment(
    align_mode: str,
    consensus_dir: Path,
    raw_dir: Path,
    out_dir: Path,
    start_time: float = 60.0,
    duration: float = 60.0,
    win_pre_s: float = 0.20,
    win_post_s: float = 0.35
):
    """
    Runs full raster and gallery generation for a given alignment mode ('offset' or 'peak').
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 85)
    logger.info(f" GENERATING ALIGNED RASTERS & PETHS [ALIGNMENT: {align_mode.upper()}]")
    logger.info(f" Consensus Directory : {consensus_dir.resolve()}")
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

    # 2. Extract Breaths from EMG
    nidq_bins = list(raw_dir.glob("*.nidq.bin"))
    if not nidq_bins:
        raise FileNotFoundError(f"No .nidq.bin in {raw_dir}")
    nidq_bin = nidq_bins[0]
    nidq_data = read_nidq_bin(nidq_bin, nidq_bin.with_suffix(".meta"), emg_chan=0, ecg_chan=1)

    t_nidq_raw = nidq_data.time_vector
    t_end = start_time + duration
    mask_n = (t_nidq_raw >= start_time) & (t_nidq_raw <= t_end)
    t_nidq = t_nidq_raw[mask_n] - start_time
    emg_raw = nidq_data.all_channels[0][mask_n]

    emg_cfg = EMGConfig(fs=nidq_data.fs, highpass_cutoff=300.0, lowpass_cutoff=5000.0, smoothing_window_ms=20.0)
    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    breath_onsets = emg_proc.breath_onsets[(emg_proc.breath_onsets >= 0.0) & (emg_proc.breath_onsets <= duration)]
    breath_offsets = emg_proc.breath_offsets[(emg_proc.breath_offsets >= 0.0) & (emg_proc.breath_offsets <= duration)]

    # Compute Continuous Phase phi(t)
    phi_nidq = compute_continuous_phase(t_nidq, breath_onsets, breath_offsets)

    # Determine alignment events
    if align_mode == "offset":
        # Align to diaphragm burst offset (I-off)
        align_events = breath_offsets[(breath_offsets >= win_pre_s) & (breath_offsets <= duration - win_post_s)]
    elif align_mode == "peak":
        # Find peak between onset and offset for each breath
        peak_events = []
        n_b = min(len(breath_onsets), len(breath_offsets))
        for k in range(n_b):
            t_on = breath_onsets[k]
            t_off = breath_offsets[k]
            if t_off > t_on:
                mask_burst = (t_nidq >= t_on) & (t_nidq <= t_off)
                if np.any(mask_burst):
                    burst_t = t_nidq[mask_burst]
                    burst_env = emg_proc.envelope[mask_burst]
                    pk_t = burst_t[np.argmax(burst_env)]
                    if pk_t >= win_pre_s and pk_t <= duration - win_post_s:
                        peak_events.append(pk_t)
        align_events = np.array(peak_events)
    else:  # onset
        align_events = breath_onsets[(breath_onsets >= win_pre_s) & (breath_onsets <= duration - win_post_s)]

    logger.info(f"Extracted {len(align_events)} alignment events ({align_mode.upper()}) across {len(sua_ids)} clean single units.")

    # 3. Generate Individual Figures
    index_cards = []
    for idx, uid in enumerate(sua_ids):
        depth = df_sua[df_sua["cluster_id"] == uid]["depth_um"].values[0] if "depth_um" in df_sua.columns else 0.0
        u_spks = spk_times[spk_clusters == uid]
        out_png = out_dir / f"unit_{uid:03d}_depth_{int(depth)}um_aligned_{align_mode}_raster_peth.png"

        plot_single_neuron_aligned_raster_and_peth(
            unit_id=uid,
            depth_um=depth,
            spk_times=u_spks,
            align_events=align_events,
            t_nidq=t_nidq,
            phi_nidq=phi_nidq,
            emg_envelope=emg_proc.envelope,
            align_mode=align_mode,
            out_file=out_png,
            win_pre_s=win_pre_s,
            win_post_s=win_post_s,
            dt_peth=0.002
        )
        index_cards.append({
            "unit_id": int(uid),
            "depth_um": float(depth),
            "n_spikes": len(u_spks),
            "file_name": out_png.name
        })

    # 4. Generate HTML Visual Gallery
    title_text = {
        "offset": "Breath-Offset Aligned Spike Rasters & PETHs (Start of Expiration at t = 0 ms)",
        "peak": "Diaphragm Peak-Aligned Spike Rasters & PETHs (Burst Maximum at t = 0 ms)"
    }.get(align_mode, f"{align_mode.capitalize()}-Aligned Spike Rasters & PETHs")

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title_text}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }}
        h1 {{ text-align: center; color: #38bdf8; margin-bottom: 5px; }}
        p.subtitle {{ text-align: center; color: #94a3b8; font-size: 14px; margin-bottom: 25px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 20px; max-width: 1800px; margin: 0 auto; }}
        .card {{ background: #1e293b; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5); border: 1px solid #334155; }}
        .card img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
        .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .badge {{ background: #0284c7; color: white; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>{title_text}</h1>
    <p class="subtitle">37 Clean Single Units | {len(align_events)} Breaths | True Real Time (No Warping) | Background = Breath Phase &phi;(t)</p>
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
    gallery_path = out_dir / f"gallery_{align_mode}_aligned_37_neurons.html"
    gallery_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Saved Aligned Visual Gallery: {gallery_path.resolve()}")

    print("\n" + "=" * 85)
    print(f" [SUCCESS] ALL 37 {align_mode.upper()}-ALIGNED NEURON RASTERS & PETHS GENERATED")
    print("=" * 85)
    print(f" • Output Folder       : {out_dir.resolve()}")
    print(f" • Interactive Gallery : {gallery_path.resolve()}")
    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description="Generate Offset-Aligned and Peak-Aligned Spike Rasters and PETHs")
    parser.add_argument("--consensus-dir", type=str, default="'output_LP21_60s_120s'/sliding_window_pipeline/lupin/consensus_master")
    parser.add_argument("--raw-dir", type=str, default=r"Z:\ramirezlab\pascual\rec_data\LP21_fent_EXPT_g1")
    parser.add_argument("--start-time", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--mode", type=str, default="both", choices=["offset", "peak", "both"])
    args = parser.parse_args()

    cons_dir_raw = args.consensus_dir.strip('"')
    cons_dir = Path(cons_dir_raw)
    if not cons_dir.exists():
        alt_path = Path(f"'{cons_dir_raw}'")
        if alt_path.exists():
            cons_dir = alt_path

    raw_dir = Path(args.raw_dir.strip('"'))

    if args.mode in ["offset", "both"]:
        generate_gallery_for_alignment(
            align_mode="offset",
            consensus_dir=cons_dir,
            raw_dir=raw_dir,
            out_dir=Path("output_individual_neuron_rasters_aligned_offset_LP21_60s"),
            start_time=args.start_time,
            duration=args.duration,
            win_pre_s=0.25,
            win_post_s=0.30
        )

    if args.mode in ["peak", "both"]:
        generate_gallery_for_alignment(
            align_mode="peak",
            consensus_dir=cons_dir,
            raw_dir=raw_dir,
            out_dir=Path("output_individual_neuron_rasters_aligned_peak_LP21_60s"),
            start_time=args.start_time,
            duration=args.duration,
            win_pre_s=0.20,
            win_post_s=0.35
        )


if __name__ == "__main__":
    main()
