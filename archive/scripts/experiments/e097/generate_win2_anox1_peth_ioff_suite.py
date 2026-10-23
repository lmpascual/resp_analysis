#!/usr/bin/env python3
"""
Generate Breath-Aligned Peri-Event Time Histograms (PETH) and Rasters Aligned to Inspiration Offset (I_off)
for e097 Subwindow 2 (win2_anox1: 1800.0s - 1836.0s / 30:00 - 30:36 min).

Covers:
1. Kilosort 4 (92 QC-Passed Single Units)
2. Lupin SVD-OMP (120 QC-Passed Single Units)

Features:
- Primary Alignment Landmark: Flow Zero-Crossing (I_off, t = 0 ms).
- Trial Stratification:
    * Eupnea Trials (Breaths 1–80, 0.0s – 25.3s, baseline gas switch)
    * Prolonged Anoxic Apnea (25.3s – 27.6s, 2.3s pause)
    * Gasping Trials (Breaths 81–90, 27.6s – 35.5s, large ballistic autoresuscitative gasps)
- Dedicated 3-Panel Card per Unit:
    1. Cardiorespiratory Telemetry (Airflow & Diaphragm EMG envelope aligned to I_off)
    2. 90-Trial Spike Raster (Eupnea vs Gasping highlighted with I_on curve)
    3. Dual-Curve PETH (Eupnea in Blue vs Gasping in Crimson with ±SEM ribbons)
- Population Depth Heatmaps & Firing Rate Modulations
- Interactive Standalone HTML Explorer Galleries for both Sorters and Master Comparison Hub
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO_ROOT = Path(r"c:\Users\lpascu\resp_analysis")
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.processing.analog_nidq.proc_emg import process_emg
from src.pipeline.processing.analog_nidq.proc_flow import process_flow_meter
from src.pipeline.config import EMGConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PETH_IOFF_WIN2")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.0


def get_vrc_region(depth_um: float) -> str:
    """Maps Neuropixels probe depth along the Caudal-to-Rostral VRC insertion (Bush et al. 2024).
    Tip (0 um) is Rostral (VII / BötC border); Shank entry (~2850+ um) is Caudal (cVRG / obex).
    """
    if depth_um < 450:
        return "Rostral VRC / BötC border (0–450 µm)"
    elif depth_um < 800:
        return "BötC (450–800 µm)"
    elif depth_um < 1300:
        return "preBötC (800–1300 µm)"
    elif depth_um < 1900:
        return "rVRG (1300–1900 µm)"
    else:
        return "cVRG (1900–2850+ µm)"


def get_vrc_short(depth_um: float) -> str:
    if depth_um < 450: return "Rostral VRC"
    elif depth_um < 800: return "BötC"
    elif depth_um < 1300: return "preBötC"
    elif depth_um < 1900: return "rVRG"
    else: return "cVRG"


def extract_win2_breaths():
    """Extracts NIDQ signals, breath onsets, offsets, and Flow zero-crossings for win2_anox1."""
    nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
    fs = 10000.0
    n_ch = 9
    start_t = 1800.0
    dur = 36.0
    s0 = int(start_t * fs)
    s1 = int((start_t + dur) * fs)

    logger.info(f"Loading NIDAQ analog streams from {nidq_file} (t={start_t}s to {start_t+dur}s)...")
    raw = np.memmap(nidq_file, dtype="int16", mode="r")
    slice_data = raw[s0 * n_ch : s1 * n_ch].reshape(-1, n_ch)
    emg_v = slice_data[:, 0].astype(np.float32) * (10.0 / 32768.0)
    flow_v = slice_data[:, 5].astype(np.float32) * (10.0 / 32768.0)
    ecg_v = slice_data[:, 2].astype(np.float32) * (10.0 / 32768.0)
    t_vec = np.arange(len(emg_v)) / fs

    emg_cfg = EMGConfig(fs=fs, highpass_cutoff=300.0, lowpass_cutoff=3000.0, smoothing_window_ms=50.0, peak_threshold_sd=2.0)
    emg_proc = process_emg(emg_v, t_vec, emg_cfg)
    valid_pairs = [(on, off, off - on) for on, off in zip(emg_proc.breath_onsets, emg_proc.breath_offsets) if on > 0.05 and on < t_vec[-1] - 0.05 and off > on]
    valid_onsets = np.array([p[0] for p in valid_pairs])
    valid_offsets = np.array([p[1] for p in valid_pairs])

    flow_res = process_flow_meter(flow_v, t_vec, valid_onsets, valid_offsets, fs=fs)
    flow_zcs = flow_res.flow_zero_crossings

    n_breaths = len(valid_onsets)
    logger.info(f"Detected {n_breaths} valid breaths in win2_anox1.")

    # Classify Eupnea vs Gasping trials
    # Breaths 0..79: Eupneic challenge breaths
    # Breaths 80..89: Post-apnea gasps (>27.0s)
    gasp_mask = np.zeros(n_breaths, dtype=bool)
    for i in range(n_breaths):
        if valid_onsets[i] >= 27.0:
            gasp_mask[i] = True

    logger.info(f"Eupnea trials: {np.sum(~gasp_mask)} | Gasping trials: {np.sum(gasp_mask)}")

    return {
        "fs": fs,
        "t_vec": t_vec,
        "emg_v": emg_v,
        "emg_env": emg_proc.envelope,
        "flow_v": flow_v,
        "ecg_v": ecg_v,
        "valid_onsets": valid_onsets,
        "valid_offsets": valid_offsets,
        "flow_zcs": flow_zcs,
        "gasp_mask": gasp_mask,
        "n_breaths": n_breaths
    }


def compute_analog_averages(breath_data, win_pre_s=0.25, win_post_s=0.35, dt_s=0.001):
    """Computes mean and SEM traces for Flow and Diaphragm EMG aligned to I_off."""
    fs = breath_data["fs"]
    flow_v = breath_data["flow_v"]
    emg_env = breath_data["emg_env"]
    flow_zcs = breath_data["flow_zcs"]
    gasp_mask = breath_data["gasp_mask"]

    time_axis_ms = np.arange(-win_pre_s, win_post_s, dt_s) * 1000.0
    n_pts = len(time_axis_ms)

    flow_eup, flow_gasp = [], []
    emg_eup, emg_gasp = [], []

    step = int(fs * dt_s)

    for i, t_zc in enumerate(flow_zcs):
        i0 = int((t_zc - win_pre_s) * fs)
        i1 = int((t_zc + win_post_s) * fs)
        if i0 >= 0 and i1 <= len(flow_v):
            f_slice = flow_v[i0:i1:step][:n_pts]
            e_slice = emg_env[i0:i1:step][:n_pts]
            if len(f_slice) == n_pts:
                if gasp_mask[i]:
                    flow_gasp.append(f_slice)
                    emg_gasp.append(e_slice)
                else:
                    flow_eup.append(f_slice)
                    emg_eup.append(e_slice)

    flow_eup_arr = np.array(flow_eup)
    flow_gasp_arr = np.array(flow_gasp)
    emg_eup_arr = np.array(emg_eup)
    emg_gasp_arr = np.array(emg_gasp)

    return {
        "time_axis_ms": time_axis_ms,
        "flow_eup_mean": np.mean(flow_eup_arr, axis=0),
        "flow_eup_sem": np.std(flow_eup_arr, axis=0) / np.sqrt(len(flow_eup_arr)),
        "flow_gasp_mean": np.mean(flow_gasp_arr, axis=0),
        "flow_gasp_sem": np.std(flow_gasp_arr, axis=0) / np.sqrt(len(flow_gasp_arr)),
        "emg_eup_mean": np.mean(emg_eup_arr, axis=0),
        "emg_eup_sem": np.std(emg_eup_arr, axis=0) / np.sqrt(len(emg_eup_arr)),
        "emg_gasp_mean": np.mean(emg_gasp_arr, axis=0),
        "emg_gasp_sem": np.std(emg_gasp_arr, axis=0) / np.sqrt(len(emg_gasp_arr)),
    }


def render_peth_suite_for_sorter(sorter_key: str, sorter_label: str, breath_data: Dict[str, Any], analog_avg: Dict[str, Any]):
    """Processes all QC-passed units for a sorter, rendering raster/PETH cards, population plots, and HTML gallery."""
    base_dir = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / "win2_anox1" / sorter_key
    out_dir = base_dir / "peth_ioff_aligned"
    cards_dir = out_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Spikes & QC Table
    spk_times = np.load(base_dir / "alf" / "spike_times.npy")
    spk_clusters = np.load(base_dir / "alf" / "spike_clusters.npy")
    qc_csv = base_dir / f"{sorter_key}_qc_summary.csv"
    if not qc_csv.exists():
        qc_csv = base_dir / "cluster_info.tsv"
    sep = "\t" if qc_csv.suffix == ".tsv" else ","
    df_metrics = pd.read_csv(qc_csv, sep=sep)

    # Filter for QC passed
    if "passed_allen_qc" in df_metrics.columns:
        df_qc = df_metrics[df_metrics["passed_allen_qc"] == True].copy()
    else:
        df_qc = df_metrics.copy()

    # Sort units from ventral to dorsal (descending depth)
    df_qc = df_qc.sort_values(by="depth_um", ascending=False).reset_index(drop=True)
    n_units = len(df_qc)
    logger.info(f"Rendering PETH suite for {sorter_label}: {n_units} QC-passed single units...")

    # Parameters for PETH
    win_pre_s = 0.250   # -250 ms
    win_post_s = 0.350  # +350 ms
    dt_peth_s = 0.005   # 5 ms bins
    peth_edges_s = np.arange(-win_pre_s, win_post_s + dt_peth_s, dt_peth_s)
    peth_centers_ms = (peth_edges_s[:-1] + peth_edges_s[1:]) / 2.0 * 1000.0
    n_bins = len(peth_centers_ms)

    flow_zcs = breath_data["flow_zcs"]
    valid_onsets = breath_data["valid_onsets"]
    gasp_mask = breath_data["gasp_mask"]
    n_breaths = breath_data["n_breaths"]
    ti_ms = (valid_onsets - flow_zcs) * 1000.0  # inspiration onset relative to I_off (< 0)

    eup_indices = np.where(~gasp_mask)[0]
    gasp_indices = np.where(gasp_mask)[0]

    unit_records = []
    pop_peth_eup = np.zeros((n_units, n_bins), dtype=np.float32)
    pop_peth_gasp = np.zeros((n_units, n_bins), dtype=np.float32)

    t0_render = time.time()

    for idx, row in df_qc.iterrows():
        cid = int(row["cluster_id"])
        depth = float(row.get("depth_um", 0.0))
        fr_hz = float(row.get("firing_rate_hz", 0.0))
        n_spk = int(row.get("n_spikes", 0)) if "n_spikes" in row else int((spk_clusters == cid).sum())
        snr = float(row.get("snr", 0.0))
        peak_ch = int(row.get("peak_channel", 0)) if "peak_channel" in row else 0

        u_spks = spk_times[spk_clusters == cid]

        # Extract spikes per trial
        trial_spike_times_ms = []  # list of arrays for raster
        binned_trials = np.zeros((n_breaths, n_bins), dtype=np.float32)

        for tr in range(n_breaths):
            t_ref = flow_zcs[tr]
            rel_s = u_spks - t_ref
            in_win = rel_s[(rel_s >= -win_pre_s) & (rel_s <= win_post_s)]
            trial_spike_times_ms.append(in_win * 1000.0)
            counts, _ = np.histogram(in_win, bins=peth_edges_s)
            binned_trials[tr, :] = counts / dt_peth_s  # in Hz

        # PETHs for Eupnea and Gasping
        peth_eup_raw = np.mean(binned_trials[eup_indices], axis=0)
        peth_gasp_raw = np.mean(binned_trials[gasp_indices], axis=0)

        sem_eup_raw = np.std(binned_trials[eup_indices], axis=0) / np.sqrt(len(eup_indices))
        sem_gasp_raw = np.std(binned_trials[gasp_indices], axis=0) / np.sqrt(len(gasp_indices))

        # Smooth curves
        peth_eup_sm = gaussian_filter1d(peth_eup_raw, sigma=2.0)
        peth_gasp_sm = gaussian_filter1d(peth_gasp_raw, sigma=2.0)
        sem_eup_sm = gaussian_filter1d(sem_eup_raw, sigma=2.0)
        sem_gasp_sm = gaussian_filter1d(sem_gasp_raw, sigma=2.0)

        pop_peth_eup[idx, :] = peth_eup_sm
        pop_peth_gasp[idx, :] = peth_gasp_sm

        # Metric stats
        peak_eup = float(np.max(peth_eup_sm))
        peak_gasp = float(np.max(peth_gasp_sm))
        mean_eup_fr = float(np.mean(peth_eup_sm))
        mean_gasp_fr = float(np.mean(peth_gasp_sm))

        # Modulation during gasping
        gasp_ratio = (mean_gasp_fr + 0.1) / (mean_eup_fr + 0.1)
        gasp_mod_idx = (mean_gasp_fr - mean_eup_fr) / (mean_gasp_fr + mean_eup_fr + 1e-6)

        # Phasic preference relative to I_off
        insp_mask = peth_centers_ms < 0.0
        post_i_mask = (peth_centers_ms >= 0.0) & (peth_centers_ms <= 150.0)
        late_exp_mask = peth_centers_ms > 150.0

        fr_insp = np.mean(peth_eup_sm[insp_mask])
        fr_post_i = np.mean(peth_eup_sm[post_i_mask])
        fr_late_exp = np.mean(peth_eup_sm[late_exp_mask])

        if fr_insp > 1.4 * max(fr_post_i, fr_late_exp, 1.0):
            phase_type = "Inspiratory Phasic"
            phase_color = "#0284c7"
        elif fr_post_i > 1.4 * max(fr_insp, fr_late_exp, 1.0):
            phase_type = "Post-I / Expiratory Phasic"
            phase_color = "#059669"
        elif fr_late_exp > 1.4 * max(fr_insp, fr_post_i, 1.0):
            phase_type = "Late Expiratory (E2)"
            phase_color = "#d97706"
        else:
            phase_type = "Phase-Spanning / Modulated"
            phase_color = "#64748b"

        # Gasp category
        if gasp_ratio >= 1.5:
            gasp_cat = f"Gasp-Recruited (+{gasp_ratio:.1f}x)"
            gasp_badge_color = "#e11d48"
        elif gasp_ratio <= 0.60:
            gasp_cat = f"Gasp-Suppressed ({gasp_ratio:.2f}x)"
            gasp_badge_color = "#7c3aed"
        else:
            gasp_cat = "Gasp-Stationary"
            gasp_badge_color = "#475569"

        # -------------------------------------------------------------
        # BUILD HIGH-RESOLUTION 3-PANEL CARD
        # -------------------------------------------------------------
        fig = plt.figure(figsize=(10.5, 9.5), dpi=180)
        gs = gridspec.GridSpec(3, 1, height_ratios=[1.2, 3.2, 1.8], hspace=0.18)

        # 1. Analog Reference Panel (Flow & EMG)
        ax_analog = fig.add_subplot(gs[0])
        ax_analog.set_facecolor("#f8fafc")
        t_ax = analog_avg["time_axis_ms"]
        
        # Plot Eupnea vs Gasp Flow
        ax_analog.plot(t_ax, analog_avg["flow_eup_mean"], color="#0284c7", lw=1.6, label="Eupnea Flow (V)")
        ax_analog.fill_between(t_ax, analog_avg["flow_eup_mean"] - analog_avg["flow_eup_sem"], analog_avg["flow_eup_mean"] + analog_avg["flow_eup_sem"], color="#0284c7", alpha=0.18)

        ax_analog.plot(t_ax, analog_avg["flow_gasp_mean"], color="#e11d48", lw=1.8, label="Gasp Flow (V)")
        ax_analog.fill_between(t_ax, analog_avg["flow_gasp_mean"] - analog_avg["flow_gasp_sem"], analog_avg["flow_gasp_mean"] + analog_avg["flow_gasp_sem"], color="#e11d48", alpha=0.18)

        # Diaphragm EMG on twin axis
        ax_emg = ax_analog.twinx()
        ax_emg.plot(t_ax, analog_avg["emg_eup_mean"], color="#0284c7", lw=1.2, ls="--", alpha=0.75, label="Eupnea Diaphragm EMG")
        ax_emg.plot(t_ax, analog_avg["emg_gasp_mean"], color="#e11d48", lw=1.4, ls="--", alpha=0.75, label="Gasp Diaphragm EMG")
        ax_emg.set_ylabel("Dia. EMG (V)", fontsize=8.5, color="#475569", fontweight="bold")
        ax_emg.tick_params(axis="y", labelsize=8, labelcolor="#475569")

        ax_analog.axvline(0.0, color="#64748b", ls="--", lw=1.2, zorder=5)
        ax_analog.set_xlim(-win_pre_s * 1000.0, win_post_s * 1000.0)
        ax_analog.set_ylabel("Airflow (V)", fontsize=8.5, fontweight="bold", color="#0f172a")
        ax_analog.tick_params(axis="x", labelbottom=False)
        ax_analog.tick_params(axis="y", labelsize=8)
        ax_analog.grid(True, ls=":", alpha=0.5)
        vrc_loc = get_vrc_short(depth)
        ax_analog.set_title(
            f"e097 {sorter_label} | Unit #{cid:03d} | Depth: {depth:.1f} µm [{vrc_loc}] (Ch {peak_ch}) | FR: {fr_hz:.1f} Hz (SNR: {snr:.1f})\n"
            f"Classification: {phase_type} | {gasp_cat}",
            fontsize=11.0, fontweight="bold", pad=8, color="#0f172a"
        )
        lines1, labels1 = ax_analog.get_legend_handles_labels()
        lines2, labels2 = ax_emg.get_legend_handles_labels()
        ax_analog.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7.5, framealpha=0.85, ncol=2)

        # 2. Trial-by-Trial Spike Raster Panel
        ax_ras = fig.add_subplot(gs[1], sharex=ax_analog)
        ax_ras.set_facecolor("#ffffff")

        # Shading for Eupnea vs Gasping
        ax_ras.axhspan(0.5, 80.5, color="#f0f9ff", alpha=0.9, zorder=0)  # Eupnea light blue
        ax_ras.axhspan(80.5, 90.5, color="#ffe4e6", alpha=0.9, zorder=0)  # Gasping light rose

        # Horizontal divider and apnea marker
        ax_ras.axhline(80.5, color="#e11d48", lw=1.8, ls="-", zorder=3)
        ax_ras.text(
            -win_pre_s * 1000.0 + 8, 80.8,
            "⚡ 2.3s APNEA / HR NADIR ➔ GASPING TRIALS (81–90)",
            color="#be123c", fontsize=8.5, fontweight="bold", va="bottom", zorder=4
        )
        ax_ras.text(
            -win_pre_s * 1000.0 + 8, 1.2,
            "EUPNEIC BREATHS (1–80)",
            color="#0369a1", fontsize=8.0, fontweight="bold", va="bottom", zorder=4
        )

        # Plot inspiration onset points (I_on = -T_I)
        y_trials = np.arange(1, n_breaths + 1)
        ax_ras.plot(ti_ms, y_trials, color="#0891b2", lw=1.5, ls="-", zorder=2, label="Insp. Onset ($I_{on} = -T_I$)")

        # Plot spikes
        for tr in range(n_breaths):
            spk_tr = trial_spike_times_ms[tr]
            if len(spk_tr) > 0:
                ax_ras.scatter(spk_tr, np.full_like(spk_tr, tr + 1), s=4.5, c="#0f172a", alpha=0.85, edgecolors="none", rasterized=True, zorder=3)

        ax_ras.axvline(0.0, color="#64748b", ls="--", lw=1.4, zorder=5)
        ax_ras.set_ylabel("Breath Trial Number (1–90)", fontsize=9, fontweight="bold", color="#0f172a")
        ax_ras.set_ylim(0.5, n_breaths + 0.5)
        ax_ras.tick_params(axis="x", labelbottom=False)
        ax_ras.tick_params(axis="y", labelsize=8)
        ax_ras.grid(True, ls=":", alpha=0.4)
        ax_ras.legend(loc="upper left", fontsize=8, framealpha=0.85)

        # 3. Dual-Curve PETH Panel
        ax_peth = fig.add_subplot(gs[2], sharex=ax_analog)
        ax_peth.set_facecolor("#f8fafc")

        # Eupnea PETH
        ax_peth.plot(peth_centers_ms, peth_eup_sm, color="#0284c7", lw=2.2, label=f"Eupnea (n=80 trials, Peak={peak_eup:.1f} Hz)")
        ax_peth.fill_between(peth_centers_ms, np.maximum(0, peth_eup_sm - sem_eup_sm), peth_eup_sm + sem_eup_sm, color="#0284c7", alpha=0.22)

        # Gasping PETH
        ax_peth.plot(peth_centers_ms, peth_gasp_sm, color="#e11d48", lw=2.4, label=f"Gasping (n=10 trials, Peak={peak_gasp:.1f} Hz, {gasp_ratio:.1f}x)")
        ax_peth.fill_between(peth_centers_ms, np.maximum(0, peth_gasp_sm - sem_gasp_sm), peth_gasp_sm + sem_gasp_sm, color="#e11d48", alpha=0.22)

        ax_peth.axvline(0.0, color="#64748b", ls="--", lw=1.4, zorder=5)
        ax_peth.set_xlabel("Time Relative to Inspiration Offset ($I_{off}$) [ms]", fontsize=9.5, fontweight="bold", color="#0f172a")
        ax_peth.set_ylabel("Firing Rate (Hz)", fontsize=9, fontweight="bold", color="#0f172a")
        ax_peth.set_ylim(bottom=0.0)
        ax_peth.tick_params(axis="both", labelsize=8)
        ax_peth.grid(True, ls=":", alpha=0.6)
        ax_peth.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

        # Inset phase labels
        ax_peth.text(-120, ax_peth.get_ylim()[1] * 0.88, "INSPIRATION", color="#0369a1", fontsize=7.5, fontweight="bold", ha="center")
        ax_peth.text(75, ax_peth.get_ylim()[1] * 0.88, "POST-I / EXPIRATION", color="#047857", fontsize=7.5, fontweight="bold", ha="center")

        card_png_name = f"peth_ioff_card_unit_{cid:04d}.png"
        card_png_path = cards_dir / card_png_name
        fig.savefig(card_png_path, bbox_inches="tight")
        plt.close(fig)

        unit_records.append({
            "cid": cid,
            "depth": depth,
            "vrc_short": get_vrc_short(depth),
            "vrc_full": get_vrc_region(depth),
            "peak_ch": peak_ch,
            "fr": fr_hz,
            "spikes": n_spk,
            "snr": snr,
            "phase_type": phase_type,
            "phase_color": phase_color,
            "gasp_cat": gasp_cat,
            "gasp_badge_color": gasp_badge_color,
            "gasp_ratio": round(gasp_ratio, 2),
            "peak_eup": round(peak_eup, 1),
            "peak_gasp": round(peak_gasp, 1),
            "mean_eup": round(mean_eup_fr, 1),
            "mean_gasp": round(mean_gasp_fr, 1),
            "img": f"cards/{card_png_name}"
        })

    dur_render = time.time() - t0_render
    logger.info(f"Rendered {len(unit_records)} unit cards in {dur_render:.1f}s ({dur_render/max(1, len(unit_records)):.2f}s/card).")

    # Save summary dataframe
    df_peth_summary = pd.DataFrame(unit_records)
    summary_csv = out_dir / f"{sorter_key}_peth_ioff_summary.csv"
    df_peth_summary.to_csv(summary_csv, index=False)

    # -------------------------------------------------------------
    # RENDER POPULATION SUMMARY FIGURE
    # -------------------------------------------------------------
    fig_pop = plt.figure(figsize=(13.5, 9.0), dpi=200)
    gs_p = gridspec.GridSpec(2, 3, width_ratios=[1.2, 1.2, 0.9], height_ratios=[2.5, 1.0], wspace=0.28, hspace=0.25)

    # Sort population matrix by depth
    depth_order = np.argsort(df_qc["depth_um"].values)[::-1]
    sorted_depths = df_qc["depth_um"].values[depth_order]
    
    # Normalize each unit's PETH to [0, 1] for heatmap contrast
    norm_eup = pop_peth_eup[depth_order] / (np.max(pop_peth_eup[depth_order], axis=1, keepdims=True) + 1e-6)
    norm_gasp = pop_peth_gasp[depth_order] / (np.max(pop_peth_gasp[depth_order], axis=1, keepdims=True) + 1e-6)

    # 1. Eupnea Heatmap
    ax_h1 = fig_pop.add_subplot(gs_p[0, 0])
    im1 = ax_h1.imshow(norm_eup, aspect="auto", cmap="viridis", extent=[peth_centers_ms[0], peth_centers_ms[-1], sorted_depths[-1], sorted_depths[0]], origin="upper")
    ax_h1.axvline(0.0, color="#ffffff", ls="--", lw=1.2)
    for bound_y in [450, 800, 1300, 1900]:
        ax_h1.axhline(bound_y, color="#ffffff", ls=":", lw=0.8, alpha=0.55)
    ax_h1.set_title(f"A. EUPNEA Heatmap (n={n_units} units)\nVRC Longitude: Rostral Tip (0 µm) ➔ Caudal Entry", fontsize=9.5, fontweight="bold")
    ax_h1.set_ylabel("VRC Depth (µm)\n(Rostral Tip: 0 µm ➔ Caudal: ~2850 µm)", fontsize=9, fontweight="bold")
    ax_h1.tick_params(axis="both", labelsize=8)
    fig_pop.colorbar(im1, ax=ax_h1, fraction=0.046, pad=0.04, label="Norm. Rate")

    # 2. Gasping Heatmap
    ax_h2 = fig_pop.add_subplot(gs_p[0, 1])
    im2 = ax_h2.imshow(norm_gasp, aspect="auto", cmap="inferno", extent=[peth_centers_ms[0], peth_centers_ms[-1], sorted_depths[-1], sorted_depths[0]], origin="upper")
    ax_h2.axvline(0.0, color="#ffffff", ls="--", lw=1.2)
    for bound_y in [450, 800, 1300, 1900]:
        ax_h2.axhline(bound_y, color="#ffffff", ls=":", lw=0.8, alpha=0.55)
    ax_h2.set_title(f"B. GASPING Heatmap (n={n_units} units)\nVRC Longitude: Rostral Tip (0 µm) ➔ Caudal Entry", fontsize=9.5, fontweight="bold")
    ax_h2.set_ylabel("VRC Depth (µm)\n(Rostral Tip: 0 µm ➔ Caudal: ~2850 µm)", fontsize=9, fontweight="bold")
    ax_h2.tick_params(axis="both", labelsize=8)
    fig_pop.colorbar(im2, ax=ax_h2, fraction=0.046, pad=0.04, label="Norm. Rate")

    # 3. Scatter: Eupnea FR vs Gasping FR
    ax_sc = fig_pop.add_subplot(gs_p[0, 2])
    sc = ax_sc.scatter(df_peth_summary["mean_eup"], df_peth_summary["mean_gasp"], c=df_peth_summary["depth"], cmap="coolwarm", s=35, alpha=0.85, edgecolors="#1e293b", lw=0.5)
    max_lim = max(df_peth_summary["mean_eup"].max(), df_peth_summary["mean_gasp"].max()) * 1.08
    ax_sc.plot([0, max_lim], [0, max_lim], color="#94a3b8", ls="--", lw=1.2, label="1:1 Stationary")
    ax_sc.fill_between([0, max_lim], [0, max_lim * 1.5], [0, max_lim], color="#f43f5e", alpha=0.10, label="Recruited (>1.5x)")
    ax_sc.set_xlim(0, max_lim)
    ax_sc.set_ylim(0, max_lim)
    ax_sc.set_xlabel("Eupnea Rate (Hz)", fontsize=9, fontweight="bold")
    ax_sc.set_ylabel("Gasping Rate (Hz)", fontsize=9, fontweight="bold")
    ax_sc.set_title("C. Firing Rate Remodeling", fontsize=10, fontweight="bold")
    ax_sc.legend(loc="upper left", fontsize=7.5)
    ax_sc.grid(True, ls=":", alpha=0.5)
    fig_pop.colorbar(sc, ax=ax_sc, fraction=0.046, pad=0.04, label="Depth (µm)")

    # 4. Population Mean PETH
    ax_p_mean = fig_pop.add_subplot(gs_p[1, :2])
    pop_eup_mean = np.mean(pop_peth_eup, axis=0)
    pop_eup_sem = np.std(pop_peth_eup, axis=0) / np.sqrt(n_units)
    pop_gasp_mean = np.mean(pop_peth_gasp, axis=0)
    pop_gasp_sem = np.std(pop_peth_gasp, axis=0) / np.sqrt(n_units)

    ax_p_mean.plot(peth_centers_ms, pop_eup_mean, color="#0284c7", lw=2.4, label=f"Population Eupnea (Mean Peak = {np.max(pop_eup_mean):.1f} Hz)")
    ax_p_mean.fill_between(peth_centers_ms, pop_eup_mean - pop_eup_sem, pop_eup_mean + pop_eup_sem, color="#0284c7", alpha=0.22)
    ax_p_mean.plot(peth_centers_ms, pop_gasp_mean, color="#e11d48", lw=2.4, label=f"Population Gasping (Mean Peak = {np.max(pop_gasp_mean):.1f} Hz)")
    ax_p_mean.fill_between(peth_centers_ms, pop_gasp_mean - pop_gasp_sem, pop_gasp_mean + pop_gasp_sem, color="#e11d48", alpha=0.22)
    ax_p_mean.axvline(0.0, color="#64748b", ls="--", lw=1.2)
    ax_p_mean.set_xlabel("Time Relative to Inspiration Offset ($I_{off}$) [ms]", fontsize=9.5, fontweight="bold")
    ax_p_mean.set_ylabel("Pop. Firing Rate (Hz)", fontsize=9, fontweight="bold")
    ax_p_mean.set_title("D. Population-Averaged Ensemble PETH", fontsize=10, fontweight="bold")
    ax_p_mean.grid(True, ls=":", alpha=0.6)
    ax_p_mean.legend(loc="upper right", fontsize=8.5)

    # 5. Phasic Distribution Bar Chart
    ax_bar = fig_pop.add_subplot(gs_p[1, 2])
    counts_phase = df_peth_summary["phase_type"].value_counts()
    colors_bar = ["#0284c7" if "Inspiratory" in k else "#059669" if "Post-I" in k else "#d97706" if "Late" in k else "#64748b" for k in counts_phase.index]
    ax_bar.barh(counts_phase.index, counts_phase.values, color=colors_bar, edgecolor="#334155")
    ax_bar.set_xlabel("Number of Units", fontsize=8.5, fontweight="bold")
    ax_bar.set_title("E. Phenotype Distribution", fontsize=9.5, fontweight="bold")
    ax_bar.tick_params(axis="both", labelsize=7.5)
    ax_bar.grid(True, axis="x", ls=":", alpha=0.5)

    fig_pop.suptitle(f"e097 {sorter_label} — Population Inspiration Offset ($I_{{off}}$) Alignment & Gasping Remodeling (win2_anox1)", fontsize=12.5, fontweight="bold", y=0.98)
    pop_fig_path = out_dir / f"{sorter_key}_peth_ioff_population_summary.png"
    fig_pop.savefig(pop_fig_path, bbox_inches="tight")
    plt.close(fig_pop)
    logger.info(f"Saved population summary figure to: {pop_fig_path}")

    # -------------------------------------------------------------
    # BUILD INTERACTIVE HTML GALLERY EXPLORER
    # -------------------------------------------------------------
    html_content = generate_html_gallery(sorter_key, sorter_label, unit_records, pop_fig_path.name)
    gallery_html_path = out_dir / f"{sorter_key}_peth_ioff_gallery.html"
    gallery_html_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Saved interactive gallery to: {gallery_html_path}")

    return {
        "sorter_key": sorter_key,
        "sorter_label": sorter_label,
        "n_units": n_units,
        "gallery_html": gallery_html_path,
        "pop_fig": pop_fig_path,
        "summary_csv": summary_csv,
        "unit_records": unit_records
    }


def generate_html_gallery(sorter_key: str, sorter_label: str, unit_records: List[Dict[str, Any]], pop_fig_name: str) -> str:
    """Creates a high-performance, dark-theme interactive HTML gallery."""
    json_data = json.dumps(unit_records)

    n_recruited = sum(1 for u in unit_records if "Recruited" in u["gasp_cat"])
    n_suppressed = sum(1 for u in unit_records if "Suppressed" in u["gasp_cat"])
    n_insp = sum(1 for u in unit_records if "Inspiratory" in u["phase_type"])
    n_post_i = sum(1 for u in unit_records if "Post-I" in u["phase_type"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>e097 {sorter_label} - I_off Aligned PETH & Gasping Suite</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: #090d16;
      color: #f1f5f9;
      line-height: 1.5;
      padding-bottom: 60px;
    }}
    header {{
      background: linear-gradient(180deg, #0f172a 0%, #090d16 100%);
      border-bottom: 1px solid #1e293b;
      padding: 24px 32px;
    }}
    .nav-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }}
    .back-btn {{
      background: #1e293b;
      color: #38bdf8;
      text-decoration: none;
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      border: 1px solid #334155;
      transition: all 0.2s;
    }}
    .back-btn:hover {{
      background: #38bdf8;
      color: #090d16;
    }}
    h1 {{
      font-size: 22px;
      font-weight: 800;
      color: #f8fafc;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      color: #94a3b8;
      font-size: 13.5px;
      margin-top: 4px;
    }}
    .metrics-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }}
    .chip {{
      background: #1e293b;
      border: 1px solid #334155;
      padding: 6px 12px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 600;
    }}
    .chip strong {{ color: #38bdf8; }}
    .pop-banner {{
      margin: 20px 32px;
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 10px;
      padding: 16px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .pop-banner-btn {{
      background: #0284c7;
      color: #ffffff;
      padding: 8px 16px;
      border-radius: 6px;
      font-weight: 700;
      font-size: 13px;
      cursor: pointer;
      border: none;
    }}
    .pop-banner-btn:hover {{ background: #0369a1; }}
    .controls {{
      padding: 12px 32px;
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      background: #0d1322;
      border-bottom: 1px solid #1e293b;
      position: sticky;
      top: 0;
      z-index: 50;
    }}
    .btn-filter {{
      background: #1e293b;
      color: #cbd5e1;
      border: 1px solid #334155;
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 12.5px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .btn-filter.active {{
      background: #0284c7;
      color: #ffffff;
      border-color: #38bdf8;
    }}
    .search-input {{
      background: #0f172a;
      border: 1px solid #334155;
      color: #ffffff;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 13px;
      width: 220px;
      outline: none;
    }}
    .select-sort {{
      background: #0f172a;
      border: 1px solid #334155;
      color: #ffffff;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 13px;
      cursor: pointer;
      outline: none;
    }}
    .grid {{
      padding: 24px 32px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
      gap: 20px;
    }}
    .unit-card {{
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 10px;
      overflow: hidden;
      transition: transform 0.15s, border-color 0.15s;
    }}
    .unit-card:hover {{
      transform: translateY(-2px);
      border-color: #38bdf8;
    }}
    .card-head {{
      padding: 10px 14px;
      background: #141f36;
      border-bottom: 1px solid #1e293b;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .card-title {{
      font-weight: 700;
      font-size: 13.5px;
      color: #f8fafc;
    }}
    .card-badge {{
      font-size: 11px;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 12px;
      color: #ffffff;
    }}
    .card-img {{
      width: 100%;
      height: auto;
      display: block;
      cursor: pointer;
      background: #ffffff;
    }}
    .card-meta {{
      padding: 10px 14px;
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 6px;
      font-size: 11.5px;
      color: #94a3b8;
      background: #0f172a;
      border-top: 1px solid #1e293b;
    }}
    .card-meta strong {{ color: #cbd5e1; }}
    .modal {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.88);
      z-index: 100;
      justify-content: center;
      align-items: center;
      padding: 20px;
    }}
    .modal img {{
      max-width: 95vw;
      max-height: 95vh;
      border-radius: 8px;
      border: 2px solid #38bdf8;
    }}
    .ion-switch-btn {{
      background: #0f172a;
      color: #10b981;
      text-decoration: none;
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      border: 1px solid #10b981;
      transition: all 0.2s;
    }}
    .ion-switch-btn:hover {{
      background: #10b981;
      color: #090d16;
    }}
  </style>
</head>
<body>
  <header>
    <div class="nav-bar">
      <div style="display:flex;gap:12px;align-items:center;">
        <a href="../win2_anox1_sorter_comparison.html" class="back-btn">&larr; Back to win2_anox1 Comparison</a>
        <a href="../peth_ion_aligned/{sorter_key}_peth_ion_gallery.html" class="ion-switch-btn">&#x21C4; Switch to I_on Gallery</a>
      </div>
      <span style="font-size:12px;color:#64748b;">e097 &bull; Subwindow 2 (1800s - 1836s) &bull; I_off Alignment</span>
    </div>
    <h1>e097 {sorter_label} &mdash; Inspiration Offset ($I_{{off}}$) Aligned PETH &amp; Gasping Suite</h1>
    <div class="subtitle">
      All 90 breath trials aligned directly to <strong>Flow Meter Zero-Crossing ($t = 0\text{{ ms}}$)</strong>. Neuropixels probe inserted longitudinally along the Ventral Respiratory Column (VRC) from Caudal entry (~2850 µm) to Rostral tip (0 µm, near BötC / VII), matching Bush et al. (2024). Eupneic Breaths (1–80) are contrasted with the 10 Ballistic Autoresuscitative Gasps (81–90) following the 2.3s Anoxic Apnea.
    </div>
    <div class="metrics-bar">
      <div class="chip">Total Single Units: <strong>{len(unit_records)}</strong></div>
      <div class="chip">Gasp-Recruited: <strong style="color:#f43f5e;">{n_recruited}</strong></div>
      <div class="chip">Gasp-Suppressed: <strong style="color:#a855f7;">{n_suppressed}</strong></div>
      <div class="chip">Inspiratory Phasic: <strong>{n_insp}</strong></div>
      <div class="chip">Post-I / Expiratory Phasic: <strong>{n_post_i}</strong></div>
      <div class="chip">Trials: <strong>80 Eupnea + 10 Gasping</strong></div>
    </div>
  </header>

  <div class="pop-banner">
    <div>
      <h3 style="font-size:14.5px;color:#f8fafc;">Population Ensemble Summary Figure</h3>
      <p style="font-size:12px;color:#94a3b8;">Depth-sorted PETH heatmaps along VRC, rate remodeling scatter, and population average traces.</p>
    </div>
    <button class="pop-banner-btn" onclick="openModal('{pop_fig_name}')">View Population Figure &rarr;</button>
  </div>

  <div class="controls">
    <button class="btn-filter active" onclick="setFilter('all', this)">All ({len(unit_records)})</button>
    <button class="btn-filter" onclick="setFilter('recruited', this)">Gasp-Recruited ({n_recruited})</button>
    <button class="btn-filter" onclick="setFilter('suppressed', this)">Gasp-Suppressed ({n_suppressed})</button>
    <button class="btn-filter" onclick="setFilter('insp', this)">Inspiratory ({n_insp})</button>
    <button class="btn-filter" onclick="setFilter('post_i', this)">Post-I / Expiratory ({n_post_i})</button>
    <button class="btn-filter" onclick="setFilter('prebotc', this)">preBötC</button>
    <button class="btn-filter" onclick="setFilter('botc', this)">BötC</button>
    <button class="btn-filter" onclick="setFilter('rvrg', this)">rVRG</button>
    <button class="btn-filter" onclick="setFilter('cvrg', this)">cVRG</button>
    <input type="text" id="searchInput" class="search-input" placeholder="Search Cluster ID (e.g. 23, 7)..." oninput="applyFilters()">
    <select id="sortSelect" class="select-sort" onchange="applySort()">
      <option value="depth_asc">Sort: Rostral (Tip) ➔ Caudal (Entry) (Depth &uarr;)</option>
      <option value="depth_desc">Sort: Caudal (Entry) ➔ Rostral (Tip) (Depth &darr;)</option>
      <option value="ratio_desc">Sort: Gasp Fold-Change (Highest &darr;)</option>
      <option value="fr_desc">Sort: Firing Rate (Highest &darr;)</option>
      <option value="cid_asc">Sort: Cluster ID (Ascending)</option>
    </select>
  </div>

  <div class="grid" id="grid"></div>

  <div class="modal" id="modal" onclick="closeModal()">
    <img id="modalImg" src="">
  </div>

  <script>
    const items = {json_data};
    let activeFilter = 'all';

    function renderCards(list) {{
      const grid = document.getElementById('grid');
      grid.innerHTML = '';
      list.forEach(c => {{
        const el = document.createElement('div');
        el.className = 'unit-card';
        el.innerHTML = `
          <div class="card-head">
            <span class="card-title">Cluster #${{String(c.cid).padStart(3, '0')}} (${{c.depth.toFixed(1)}} µm [${{c.vrc_short}}])</span>
            <span class="card-badge" style="background:${{c.gasp_badge_color}}">${{c.gasp_cat}}</span>
          </div>
          <img src="${{c.img}}" class="card-img" loading="lazy" onclick="openModal('${{c.img}}')">
          <div class="card-meta">
            <div><strong>Phenotype:</strong> ${{c.phase_type}}</div>
            <div><strong>VRC Region:</strong> ${{c.vrc_full}}</div>
            <div><strong>Base FR:</strong> ${{c.fr.toFixed(1)}} Hz (${{c.spikes}} spk)</div>
            <div><strong>Eupnea Peak:</strong> ${{c.peak_eup}} Hz</div>
            <div><strong>Gasp Peak:</strong> ${{c.peak_gasp}} Hz (${{c.gasp_ratio}}x)</div>
          </div>
        `;
        grid.appendChild(el);
      }});
    }}

    function setFilter(filt, btn) {{
      activeFilter = filt;
      document.querySelectorAll('.btn-filter').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      applyFilters();
    }}

    function applyFilters() {{
      const q = document.getElementById('searchInput').value.trim();
      let res = items.filter(c => {{
        if (q && !String(c.cid).includes(q)) return false;
        if (activeFilter === 'recruited') return c.gasp_cat.includes('Recruited');
        if (activeFilter === 'suppressed') return c.gasp_cat.includes('Suppressed');
        if (activeFilter === 'insp') return c.phase_type.includes('Inspiratory');
        if (activeFilter === 'post_i') return c.phase_type.includes('Post-I');
        if (activeFilter === 'prebotc') return c.vrc_short === 'preBötC';
        if (activeFilter === 'botc') return c.vrc_short === 'BötC';
        if (activeFilter === 'rvrg') return c.vrc_short === 'rVRG';
        if (activeFilter === 'cvrg') return c.vrc_short === 'cVRG';
        return true;
      }});
      applySort(res);
    }}

    function applySort(list = null) {{
      if (!list) {{
        applyFilters();
        return;
      }}
      const val = document.getElementById('sortSelect').value;
      if (val === 'depth_desc') list.sort((a,b) => b.depth - a.depth);
      else if (val === 'depth_asc') list.sort((a,b) => a.depth - b.depth);
      else if (val === 'ratio_desc') list.sort((a,b) => b.gasp_ratio - a.gasp_ratio);
      else if (val === 'fr_desc') list.sort((a,b) => b.fr - a.fr);
      else if (val === 'cid_asc') list.sort((a,b) => a.cid - b.cid);
      renderCards(list);
    }}

    function openModal(src) {{
      document.getElementById('modalImg').src = src;
      document.getElementById('modal').style.display = 'flex';
    }}

    function closeModal() {{
      document.getElementById('modal').style.display = 'none';
    }}

    applySort(items);
  </script>
</body>
</html>
"""
    return html


def build_unified_dashboard(ks_res, lup_res):
    """Generates the unified win2_anox1 cross-sorter PETH and Gasping comparison dashboard."""
    out_file = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / "win2_anox1" / "win2_anox1_ioff_peth_dashboard.html"

    ks_rec = sum(1 for u in ks_res["unit_records"] if "Recruited" in u["gasp_cat"])
    lup_rec = sum(1 for u in lup_res["unit_records"] if "Recruited" in u["gasp_cat"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>e097 win2_anox1 - Dual-Sorter Inspiration Offset (I_off) & Gasping Hub</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: #0b1120;
      color: #f8fafc;
      padding: 32px;
      line-height: 1.5;
    }}
    .container {{ max-width: 1280px; margin: 0 auto; }}
    .header {{
      border-bottom: 1px solid #1e293b;
      padding-bottom: 24px;
      margin-bottom: 24px;
    }}
    .back-nav {{
      margin-bottom: 12px;
    }}
    .back-btn {{
      background: #1e293b;
      color: #38bdf8;
      text-decoration: none;
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      border: 1px solid #334155;
    }}
    .back-btn:hover {{ background: #38bdf8; color: #0b1120; }}
    h1 {{ font-size: 26px; font-weight: 800; color: #f8fafc; }}
    .sub {{ color: #94a3b8; font-size: 14px; margin-top: 6px; }}
    .cards-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 24px;
      margin-bottom: 32px;
    }}
    .hub-card {{
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 12px;
      padding: 24px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .hub-card h2 {{ font-size: 18px; margin-bottom: 8px; }}
    .stat-row {{
      display: flex;
      gap: 16px;
      margin: 16px 0;
    }}
    .stat-box {{
      background: #1f2937;
      padding: 10px 16px;
      border-radius: 8px;
      flex: 1;
    }}
    .stat-box .val {{ font-size: 20px; font-weight: 800; }}
    .stat-box .lbl {{ font-size: 11px; color: #9ca3af; text-transform: uppercase; font-weight: 600; }}
    .action-btn {{
      display: inline-block;
      text-align: center;
      background: #0284c7;
      color: #ffffff;
      padding: 10px 18px;
      border-radius: 8px;
      font-weight: 700;
      text-decoration: none;
      transition: background 0.2s;
      margin-top: 8px;
    }}
    .action-btn:hover {{ background: #0369a1; }}
    .section-title {{
      font-size: 18px;
      font-weight: 700;
      margin: 28px 0 16px;
      color: #38bdf8;
    }}
    .preview-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 24px;
    }}
    .preview-box {{
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 10px;
      overflow: hidden;
    }}
    .preview-box img {{
      width: 100%;
      height: auto;
      display: block;
      background: #ffffff;
    }}
    .preview-caption {{
      padding: 12px 16px;
      font-size: 13px;
      font-weight: 600;
      color: #cbd5e1;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="back-nav" style="display:flex;gap:12px;align-items:center;">
        <a href="win2_anox1_sorter_comparison.html" class="back-btn">&larr; Back to win2_anox1 Sorter Comparison</a>
        <a href="win2_anox1_ion_peth_dashboard.html" class="back-btn" style="border-color:#10b981;color:#10b981;">&#x21C4; View I_on Aligned Modulation Suite</a>
      </div>
      <h1>e097 win2_anox1 &mdash; Inspiration Offset ($I_{{off}}$) &amp; Gasping Analysis Hub</h1>
      <p class="sub">
        Anoxic Challenge 1 (Gas Switch) &bull; 1800.0s – 1836.0s (36s duration) &bull; 80 Eupneic Breaths + 2.3s Apnea + 10 Ballistic Autoresuscitation Gasps &bull; Longitudinal VRC probe insertion: Caudal (~2850 µm) to Rostral tip (0 µm) (Bush et al. 2024).
      </p>
    </div>

    <div class="cards-grid">
      <!-- Kilosort 4 Card -->
      <div class="hub-card">
        <div>
          <h2 style="color:#f59e0b;">Kilosort 4 (RTX 4000 Ada GPU)</h2>
          <p style="font-size:13px;color:#94a3b8;">Centroid depth tracking across 0 – 2865 µm with true physical positions.</p>
          <div class="stat-row">
            <div class="stat-box">
              <div class="val" style="color:#f59e0b;">{ks_res['n_units']}</div>
              <div class="lbl">QC Passing SUA</div>
            </div>
            <div class="stat-box">
              <div class="val" style="color:#f43f5e;">{ks_rec}</div>
              <div class="lbl">Gasp-Recruited</div>
            </div>
          </div>
        </div>
        <a href="kilosort4/peth_ioff_aligned/kilosort4_peth_ioff_gallery.html" target="_blank" class="action-btn" style="background:#d97706;">
          Open Kilosort 4 PETH &amp; Raster Explorer ({ks_res['n_units']} Units) &rarr;
        </a>
      </div>

      <!-- Lupin Card -->
      <div class="hub-card">
        <div>
          <h2 style="color:#00f5d4;">Lupin (SVD-OMP)</h2>
          <p style="font-size:13px;color:#94a3b8;">High-density template matching with localized spatial deconvolution.</p>
          <div class="stat-row">
            <div class="stat-box">
              <div class="val" style="color:#00f5d4;">{lup_res['n_units']}</div>
              <div class="lbl">QC Passing SUA</div>
            </div>
            <div class="stat-box">
              <div class="val" style="color:#f43f5e;">{lup_rec}</div>
              <div class="lbl">Gasp-Recruited</div>
            </div>
          </div>
        </div>
        <a href="lupin/peth_ioff_aligned/lupin_peth_ioff_gallery.html" target="_blank" class="action-btn" style="background:#0d9488;">
          Open Lupin PETH &amp; Raster Explorer ({lup_res['n_units']} Units) &rarr;
        </a>
      </div>
    </div>

    <div class="section-title">Ensemble Population Comparisons (I_off Aligned Heatmaps &amp; Rate Remodeling)</div>
    <div class="preview-grid">
      <div class="preview-box">
        <a href="kilosort4/peth_ioff_aligned/{ks_res['pop_fig'].name}" target="_blank">
          <img src="kilosort4/peth_ioff_aligned/{ks_res['pop_fig'].name}">
        </a>
        <div class="preview-caption">Kilosort 4: Population I_off Alignment &amp; Firing Rate Remodeling</div>
      </div>
      <div class="preview-box">
        <a href="lupin/peth_ioff_aligned/{lup_res['pop_fig'].name}" target="_blank">
          <img src="lupin/peth_ioff_aligned/{lup_res['pop_fig'].name}">
        </a>
        <div class="preview-caption">Lupin (SVD-OMP): Population I_off Alignment &amp; Firing Rate Remodeling</div>
      </div>
    </div>
  </div>
</body>
</html>
"""
    out_file.write_text(html, encoding="utf-8")
    logger.info(f"Saved master PETH hub to: {out_file}")
    return out_file


def update_sorter_comparison_page(dashboard_path: Path):
    """Adds prominent navigation buttons to win2_anox1_sorter_comparison.html."""
    comp_html_path = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / "win2_anox1" / "win2_anox1_sorter_comparison.html"
    if not comp_html_path.exists():
        return

    content = comp_html_path.read_text(encoding="utf-8")
    
    # Check if PETH buttons already added
    if "win2_anox1_ioff_peth_dashboard.html" in content:
        return

    # Add PETH button in cards grid
    target = '<div class="cards-grid">'
    addition = f"""<div style="background:linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border:1px solid #38bdf8; border-radius:10px; padding:16px 20px; margin-bottom:24px; display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h3 style="color:#38bdf8; margin:0 0 4px 0; font-size:16px;">🔬 Inspiration Offset ($I_{{off}}$) &amp; Gasping PETH Suite</h3>
            <p style="color:#94a3b8; font-size:12.5px; margin:0;">80 Eupnea Breaths vs 10 Ballistic Autoresuscitation Gasps &bull; Dual-Curve PETHs &bull; Kilosort 4 (92 units) &amp; Lupin (120 units)</p>
        </div>
        <div style="display:flex; gap:10px;">
            <a href="win2_anox1_ioff_peth_dashboard.html" class="btn" style="background:#0284c7; color:#fff;">PETH &amp; Gasping Hub &rarr;</a>
            <a href="kilosort4/peth_ioff_aligned/kilosort4_peth_ioff_gallery.html" class="btn" style="background:#d97706; color:#fff;">KS4 PETHs &rarr;</a>
            <a href="lupin/peth_ioff_aligned/lupin_peth_ioff_gallery.html" class="btn" style="background:#0d9488; color:#fff;">Lupin PETHs &rarr;</a>
        </div>
    </div>
    """
    new_content = content.replace(target, addition + "\n    " + target, 1)
    comp_html_path.write_text(new_content, encoding="utf-8")
    logger.info(f"Updated {comp_html_path.name} with PETH suite buttons.")


def main():
    t_start = time.time()
    logger.info("=" * 80)
    logger.info("GENERATING INSPIRATION OFFSET (I_off) PETH & GASPING SUITE (win2_anox1)")
    logger.info("=" * 80)

    # 1. Extract Breaths & Signal Slices
    breath_data = extract_win2_breaths()

    # 2. Compute Mean Analog Waveforms
    analog_avg = compute_analog_averages(breath_data)

    # 3. Process Kilosort 4
    ks_res = render_peth_suite_for_sorter("kilosort4", "Kilosort 4 (GPU)", breath_data, analog_avg)

    # 4. Process Lupin
    lup_res = render_peth_suite_for_sorter("lupin", "Lupin (SVD-OMP)", breath_data, analog_avg)

    # 5. Build Unified Dashboard
    dash_path = build_unified_dashboard(ks_res, lup_res)

    # 6. Update Sorter Comparison Page
    update_sorter_comparison_page(dash_path)

    dur_tot = time.time() - t_start
    logger.info(f"\nALL SUITES SUCCESSFULLY GENERATED IN {dur_tot:.1f}s!")
    logger.info(f"  Master Hub: {dash_path}")
    logger.info(f"  KS4 Gallery: {ks_res['gallery_html']}")
    logger.info(f"  Lupin Gallery: {lup_res['gallery_html']}")


if __name__ == "__main__":
    main()
