#!/usr/bin/env python3
"""
Generate Breath-Aligned Peri-Event Time Histograms (PETH), Rasters, and Jitter/Cross-Correlation
Modulation Analysis Aligned to Inspiration Onset (I_on) for e097 Subwindow 2 (win2_anox1).

Subwindow: 1800.0s - 1836.0s (30:00 - 30:36 min, duration 36.0s).
Sorters:
1. Kilosort 4 (92 QC-Passed Single Units)
2. Lupin SVD-OMP (120 QC-Passed Single Units)

Features & Methodology:
- Primary Alignment Landmark: Diaphragm EMG Onset (I_on, t = 0 ms).
- Secondary Dynamic Landmark: Expiration Onset / Flow Zero-Crossing (I_off, t = +T_I ms plotted per trial).
- Trial Stratification:
    * Eupnea Trials: Breaths 1–80 (0.0s – 25.3s, baseline challenge)
    * Prolonged Anoxic Apnea: 25.3s – 27.6s (2.3s respiratory arrest)
    * Gasping Trials: Breaths 81–90 (27.6s – 35.5s, large ballistic autoresuscitative gasps)
- Event Modulation Quantification:
    * Burst centroid / first-spike latencies relative to I_on vs I_off.
    * Trial-by-trial jitter: sigma_on vs sigma_off.
    * Normalized Locking Index: LI = (sigma_off - sigma_on) / (sigma_off + sigma_on).
    * Inspiratory duration regression slope: beta_TI = d(Lat_on) / d(T_I).
    * Continuous spike-to-event cross-correlation peak and lag for I_on vs I_off.
    * Cross-correlation ratio and PETH sharpness (Peak / FWHM).
    * Classification: I_on-Locked, I_on-Favoring, Duration-Scaler, E_on-Favoring, E_on-Locked, Tonic.
- Outputs:
    * 3-panel unit cards for all 92 KS4 units and 120 Lupin units.
    * Detailed CSV scorecards.
    * Population ensemble figures.
    * Cross-sorter synthesis figure.
    * Standalone interactive HTML galleries with filtering and sorting.
    * Master navigation dashboards.
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
from scipy import stats, signal
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
logger = logging.getLogger("PETH_ION_SUITE")

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
    valid_pairs = [(on, off) for on, off in zip(emg_proc.breath_onsets, emg_proc.breath_offsets) if on > 0.05 and on < t_vec[-1] - 0.05 and off > on]
    valid_onsets = np.array([p[0] for p in valid_pairs])
    valid_offsets = np.array([p[1] for p in valid_pairs])

    flow_res = process_flow_meter(flow_v, t_vec, valid_onsets, valid_offsets, fs=fs)
    flow_zcs = flow_res.flow_zero_crossings

    n_breaths = len(valid_onsets)
    logger.info(f"Detected {n_breaths} valid breaths in win2_anox1.")

    gasp_mask = np.zeros(n_breaths, dtype=bool)
    for i in range(n_breaths):
        if valid_onsets[i] >= 27.0:
            gasp_mask[i] = True

    logger.info(f"Eupnea trials: {np.sum(~gasp_mask)} | Gasping trials: {np.sum(gasp_mask)}")

    ti_arr = (flow_zcs - valid_onsets) * 1000.0  # ms
    logger.info(f"Mean T_I: {np.mean(ti_arr):.1f} ± {np.std(ti_arr):.1f} ms (Min: {np.min(ti_arr):.1f} ms, Max: {np.max(ti_arr):.1f} ms)")

    return {
        "fs": fs,
        "dur": dur,
        "t_vec": t_vec,
        "emg_v": emg_v,
        "emg_env": emg_proc.envelope,
        "flow_v": flow_v,
        "ecg_v": ecg_v,
        "valid_onsets": valid_onsets,
        "valid_offsets": valid_offsets,
        "flow_zcs": flow_zcs,
        "ti_arr": ti_arr,
        "gasp_mask": gasp_mask,
        "n_breaths": n_breaths
    }


def compute_analog_averages_ion(breath_data, win_pre_s=0.150, win_post_s=0.400, dt_s=0.001):
    """Computes mean and SEM traces for Flow and Diaphragm EMG aligned to I_on (Diaphragm EMG Onset)."""
    fs = breath_data["fs"]
    flow_v = breath_data["flow_v"]
    emg_env = breath_data["emg_env"]
    valid_onsets = breath_data["valid_onsets"]
    gasp_mask = breath_data["gasp_mask"]

    time_axis_ms = np.arange(-win_pre_s, win_post_s, dt_s) * 1000.0
    n_pts = len(time_axis_ms)

    flow_eup, flow_gasp = [], []
    emg_eup, emg_gasp = [], []

    step = int(fs * dt_s)

    for i, t_on in enumerate(valid_onsets):
        i0 = int((t_on - win_pre_s) * fs)
        i1 = int((t_on + win_post_s) * fs)
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
        "flow_eup_sem": np.std(flow_eup_arr, axis=0) / np.sqrt(max(1, len(flow_eup_arr))),
        "flow_gasp_mean": np.mean(flow_gasp_arr, axis=0),
        "flow_gasp_sem": np.std(flow_gasp_arr, axis=0) / np.sqrt(max(1, len(flow_gasp_arr))),
        "emg_eup_mean": np.mean(emg_eup_arr, axis=0),
        "emg_eup_sem": np.std(emg_eup_arr, axis=0) / np.sqrt(max(1, len(emg_eup_arr))),
        "emg_gasp_mean": np.mean(emg_gasp_arr, axis=0),
        "emg_gasp_sem": np.std(emg_gasp_arr, axis=0) / np.sqrt(max(1, len(emg_gasp_arr))),
    }


def render_peth_ion_suite_for_sorter(sorter_key: str, sorter_label: str, breath_data: Dict[str, Any], analog_avg: Dict[str, Any]):
    """Processes all QC-passed units for a sorter, rendering I_on cards, population plots, and HTML gallery."""
    base_dir = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / "win2_anox1" / sorter_key
    out_dir = base_dir / "peth_ion_aligned"
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

    if "passed_allen_qc" in df_metrics.columns:
        df_qc = df_metrics[df_metrics["passed_allen_qc"] == True].copy()
    else:
        df_qc = df_metrics.copy()

    # Sort units along VRC: Caudal entry (~2850+ um) to Rostral tip (0 um) (descending depth)
    df_qc = df_qc.sort_values(by="depth_um", ascending=False).reset_index(drop=True)
    n_units = len(df_qc)
    logger.info(f"Rendering I_on PETH & Jitter suite for {sorter_label}: {n_units} QC-passed single units (Bush et al. 2024 VRC column)...")

    # Parameters for PETH
    win_pre_s = 0.150   # -150 ms
    win_post_s = 0.400  # +400 ms
    dt_peth_s = 0.005   # 5 ms bins
    peth_edges_s = np.arange(-win_pre_s, win_post_s + dt_peth_s, dt_peth_s)
    peth_centers_ms = (peth_edges_s[:-1] + peth_edges_s[1:]) / 2.0 * 1000.0
    n_bins = len(peth_centers_ms)

    valid_onsets = breath_data["valid_onsets"]
    flow_zcs = breath_data["flow_zcs"]
    ti_arr = breath_data["ti_arr"]  # ms (I_off relative to I_on > 0)
    gasp_mask = breath_data["gasp_mask"]
    n_breaths = breath_data["n_breaths"]
    dur = breath_data["dur"]

    eup_indices = np.where(~gasp_mask)[0]
    gasp_indices = np.where(gasp_mask)[0]

    # Precompute binned event pulse trains for continuous cross-correlation (dt = 2 ms)
    dt_cc = 0.002
    t_cc_bins = np.arange(0, dur, dt_cc)
    e_on_binned, _ = np.histogram(valid_onsets, bins=t_cc_bins)
    e_off_binned, _ = np.histogram(flow_zcs, bins=t_cc_bins)
    cc_lags = np.arange(-int(0.15 / dt_cc), int(0.35 / dt_cc) + 1)
    cc_lag_ms = cc_lags * dt_cc * 1000.0

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

        # -------------------------------------------------------------
        # A. EXTRACT TRIAL-BY-TRIAL RASTER & JITTER LATENCIES
        # -------------------------------------------------------------
        trial_spike_times_ms = []
        trial_burst_medians_ms = []
        binned_trials = np.zeros((n_breaths, n_bins), dtype=np.float32)

        lat_on_med, lat_off_med = [], []
        lat_on_fst, lat_off_fst = [], []
        ti_sub = []

        for tr in range(n_breaths):
            t_ref_on = valid_onsets[tr]
            t_ref_off = flow_zcs[tr]
            t_next = valid_onsets[tr + 1] if tr < n_breaths - 1 else t_ref_on + 0.45

            # Spike window relative to I_on (-150ms to +400ms)
            rel_s = u_spks - t_ref_on
            in_win = rel_s[(rel_s >= -win_pre_s) & (rel_s <= win_post_s)]
            trial_spike_times_ms.append(in_win * 1000.0)

            # Histogram
            counts, _ = np.histogram(in_win, bins=peth_edges_s)
            binned_trials[tr, :] = counts / dt_peth_s  # in Hz

            # Spikes within physiological breath cycle [t_on - 50ms, min(t_next, t_on + 400ms)]
            spk_b = u_spks[(u_spks >= t_ref_on - 0.05) & (u_spks < min(t_next, t_ref_on + 0.40))]
            if len(spk_b) >= 2:
                med_t = np.median(spk_b)
                trial_burst_medians_ms.append((med_t - t_ref_on) * 1000.0)
                lat_on_med.append((med_t - t_ref_on) * 1000.0)
                lat_off_med.append((med_t - t_ref_off) * 1000.0)
                lat_on_fst.append((spk_b[0] - t_ref_on) * 1000.0)
                lat_off_fst.append((spk_b[0] - t_ref_off) * 1000.0)
                ti_sub.append(ti_arr[tr])
            else:
                trial_burst_medians_ms.append(np.nan)

        # -------------------------------------------------------------
        # B. COMPUTE JITTER & REGRESSION METRICS
        # -------------------------------------------------------------
        if len(lat_on_med) >= 12:
            sig_on = float(np.std(lat_on_med))
            sig_off = float(np.std(lat_off_med))
            li = float((sig_off - sig_on) / (sig_off + sig_on + 1e-6))
            res = stats.linregress(ti_sub, lat_on_med)
            beta_ti = float(res.slope)
            r2_ti = float(res.rvalue ** 2)
            sig_on_fst = float(np.std(lat_on_fst))
            sig_off_fst = float(np.std(lat_off_fst))
            li_fst = float((sig_off_fst - sig_on_fst) / (sig_off_fst + sig_on_fst + 1e-6))
        else:
            sig_on, sig_off, li = np.nan, np.nan, np.nan
            beta_ti, r2_ti = np.nan, np.nan
            sig_on_fst, sig_off_fst, li_fst = np.nan, np.nan, np.nan

        # -------------------------------------------------------------
        # C. CONTINUOUS SPIKE-TO-EVENT CROSS-CORRELATION
        # -------------------------------------------------------------
        s_bin, _ = np.histogram(u_spks, bins=t_cc_bins)
        denom_s = np.std(s_bin) * len(s_bin) + 1e-9

        # CC with I_on
        xc_on = signal.correlate(s_bin - np.mean(s_bin), e_on_binned - np.mean(e_on_binned), mode="full")
        mid_cc = len(xc_on) // 2
        xc_on_win = xc_on[mid_cc + cc_lags[0] : mid_cc + cc_lags[-1] + 1] / (denom_s * np.std(e_on_binned) + 1e-9)
        r_on_pk = float(np.max(xc_on_win))
        tau_on_pk = float(cc_lag_ms[np.argmax(xc_on_win)])

        # CC with I_off
        xc_off = signal.correlate(s_bin - np.mean(s_bin), e_off_binned - np.mean(e_off_binned), mode="full")
        xc_off_win = xc_off[mid_cc + cc_lags[0] : mid_cc + cc_lags[-1] + 1] / (denom_s * np.std(e_off_binned) + 1e-9)
        r_off_pk = float(np.max(xc_off_win))
        tau_off_pk = float(cc_lag_ms[np.argmax(xc_off_win)])

        cc_ratio = float((r_on_pk + 1e-5) / (r_off_pk + 1e-5))

        # -------------------------------------------------------------
        # D. PETH CURVES & MODULATION METRICS
        # -------------------------------------------------------------
        peth_eup_raw = np.mean(binned_trials[eup_indices], axis=0)
        peth_gasp_raw = np.mean(binned_trials[gasp_indices], axis=0)
        sem_eup_raw = np.std(binned_trials[eup_indices], axis=0) / np.sqrt(len(eup_indices))
        sem_gasp_raw = np.std(binned_trials[gasp_indices], axis=0) / np.sqrt(len(gasp_indices))

        peth_eup_sm = gaussian_filter1d(peth_eup_raw, sigma=2.0)
        peth_gasp_sm = gaussian_filter1d(peth_gasp_raw, sigma=2.0)
        sem_eup_sm = gaussian_filter1d(sem_eup_raw, sigma=2.0)
        sem_gasp_sm = gaussian_filter1d(sem_gasp_raw, sigma=2.0)

        pop_peth_eup[idx, :] = peth_eup_sm
        pop_peth_gasp[idx, :] = peth_gasp_sm

        peak_eup = float(np.max(peth_eup_sm))
        peak_gasp = float(np.max(peth_gasp_sm))
        mean_eup_fr = float(np.mean(peth_eup_sm))
        mean_gasp_fr = float(np.mean(peth_gasp_sm))

        gasp_ratio = float((mean_gasp_fr + 0.1) / (mean_eup_fr + 0.1))
        gasp_mod_idx = float((mean_gasp_fr - mean_eup_fr) / (mean_gasp_fr + mean_eup_fr + 1e-6))

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
        # E. ONSET VS. OFFSET MODULATION CLASSIFICATION
        # -------------------------------------------------------------
        if n_spk < 20:
            mod_class = "Low-FR / Indeterminate"
            mod_badge_color = "#64748b"
        elif r_on_pk < 0.02 and r_off_pk < 0.02 and peak_eup < 8.0:
            mod_class = "Tonic / Unmodulated"
            mod_badge_color = "#64748b"
        elif not np.isnan(li) and li > 0.06 and not np.isnan(beta_ti) and beta_ti < 0.35:
            mod_class = "I_on-Locked"
            mod_badge_color = "#10b981"  # Emerald green
        elif not np.isnan(li) and li < -0.02 and not np.isnan(beta_ti) and beta_ti > 0.60:
            mod_class = "E_on-Locked"
            mod_badge_color = "#f43f5e"  # Rose red
        elif not np.isnan(li) and li > 0.04 and not np.isnan(beta_ti) and beta_ti < 0.45:
            mod_class = "I_on-Favoring"
            mod_badge_color = "#06b6d4"  # Cyan
        elif not np.isnan(li) and (li < -0.01 or (not np.isnan(beta_ti) and beta_ti > 0.55)):
            mod_class = "E_on-Favoring"
            mod_badge_color = "#f97316"  # Amber orange
        else:
            mod_class = "Duration-Scaler"
            mod_badge_color = "#8b5cf6"  # Purple

        # -------------------------------------------------------------
        # BUILD HIGH-RESOLUTION 3-PANEL CARD (I_on Aligned)
        # -------------------------------------------------------------
        fig = plt.figure(figsize=(10.5, 9.5), dpi=180)
        gs = gridspec.GridSpec(3, 1, height_ratios=[1.2, 3.2, 1.8], hspace=0.18)

        # 1. Analog Reference Panel (Flow & EMG Aligned to I_on)
        ax_analog = fig.add_subplot(gs[0])
        ax_analog.set_facecolor("#f8fafc")
        t_ax = analog_avg["time_axis_ms"]

        ax_analog.plot(t_ax, analog_avg["flow_eup_mean"], color="#0284c7", lw=1.6, label="Eupnea Flow (V)")
        ax_analog.fill_between(t_ax, analog_avg["flow_eup_mean"] - analog_avg["flow_eup_sem"], analog_avg["flow_eup_mean"] + analog_avg["flow_eup_sem"], color="#0284c7", alpha=0.18)
        ax_analog.plot(t_ax, analog_avg["flow_gasp_mean"], color="#e11d48", lw=1.8, label="Gasp Flow (V)")
        ax_analog.fill_between(t_ax, analog_avg["flow_gasp_mean"] - analog_avg["flow_gasp_sem"], analog_avg["flow_gasp_mean"] + analog_avg["flow_gasp_sem"], color="#e11d48", alpha=0.18)

        ax_emg = ax_analog.twinx()
        ax_emg.plot(t_ax, analog_avg["emg_eup_mean"], color="#0284c7", lw=1.2, ls="--", alpha=0.75, label="Eupnea Dia. EMG")
        ax_emg.plot(t_ax, analog_avg["emg_gasp_mean"], color="#e11d48", lw=1.4, ls="--", alpha=0.75, label="Gasp Dia. EMG")
        ax_emg.set_ylabel("Dia. EMG (V)", fontsize=8.5, color="#475569", fontweight="bold")
        ax_emg.tick_params(axis="y", labelsize=8, labelcolor="#475569")

        # Alignment reference at t = 0 (I_on)
        ax_analog.axvline(0.0, color="#10b981", ls="-", lw=1.6, zorder=5, label="Insp. Onset ($I_{on}$)")
        # Mean offset indicator
        mean_ti_val = float(np.mean(ti_arr[:80]))
        ax_analog.axvline(mean_ti_val, color="#f97316", ls=":", lw=1.4, zorder=5, label=f"Mean $I_{{off}}$ (+{mean_ti_val:.0f}ms)")

        ax_analog.set_xlim(-win_pre_s * 1000.0, win_post_s * 1000.0)
        ax_analog.set_ylabel("Airflow (V)", fontsize=8.5, fontweight="bold", color="#0f172a")
        ax_analog.tick_params(axis="x", labelbottom=False)
        ax_analog.tick_params(axis="y", labelsize=8)
        ax_analog.grid(True, ls=":", alpha=0.5)

        li_str = f"LI: {li:+.2f}" if not np.isnan(li) else "LI: N/A"
        beta_str = f"β_TI: {beta_ti:+.2f}" if not np.isnan(beta_ti) else "β: N/A"
        vrc_loc = get_vrc_short(depth)
        ax_analog.set_title(
            f"e097 {sorter_label} | Unit #{cid:03d} | Depth: {depth:.1f} µm [{vrc_loc}] (Ch {peak_ch}) | FR: {fr_hz:.1f} Hz (SNR: {snr:.1f})\n"
            f"Modulation: [{mod_class}] ({li_str}, {beta_str}, Ratio: {cc_ratio:.2f}) | {gasp_cat}",
            fontsize=10.8, fontweight="bold", pad=8, color="#0f172a"
        )
        lines1, labels1 = ax_analog.get_legend_handles_labels()
        lines2, labels2 = ax_emg.get_legend_handles_labels()
        ax_analog.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7.2, framealpha=0.88, ncol=3)

        # 2. Trial-by-Trial Spike Raster Panel (Aligned to I_on)
        ax_ras = fig.add_subplot(gs[1], sharex=ax_analog)
        ax_ras.set_facecolor("#ffffff")

        ax_ras.axhspan(0.5, 80.5, color="#f0f9ff", alpha=0.9, zorder=0)  # Eupnea light blue
        ax_ras.axhspan(80.5, 90.5, color="#ffe4e6", alpha=0.9, zorder=0)  # Gasping light rose

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

        y_trials = np.arange(1, n_breaths + 1)

        # Plot dynamic Inspiration Offset curve (+T_I) across trials
        ax_ras.plot(ti_arr, y_trials, color="#ea580c", lw=1.8, ls="-", zorder=4, label="Insp. Offset / Exp. Onset ($I_{off} = +T_I$)")

        # Plot spikes
        for tr in range(n_breaths):
            spk_tr = trial_spike_times_ms[tr]
            if len(spk_tr) > 0:
                ax_ras.scatter(spk_tr, np.full_like(spk_tr, tr + 1), s=4.5, c="#0f172a", alpha=0.85, edgecolors="none", rasterized=True, zorder=3)
            # Plot median burst marker if available
            med_tr = trial_burst_medians_ms[tr]
            if not np.isnan(med_tr) and -win_pre_s * 1000.0 <= med_tr <= win_post_s * 1000.0:
                ax_ras.scatter(med_tr, tr + 1, s=12.0, c="#059669" if mod_class.startswith("I_on") else "#e11d48", marker="d", edgecolors="none", alpha=0.8, zorder=5)

        ax_ras.axvline(0.0, color="#10b981", ls="-", lw=1.6, zorder=5, label="Insp. Onset ($I_{on} = 0$)")
        ax_ras.set_ylabel("Breath Trial Number (1–90)", fontsize=9, fontweight="bold", color="#0f172a")
        ax_ras.set_ylim(0.5, n_breaths + 0.5)
        ax_ras.tick_params(axis="x", labelbottom=False)
        ax_ras.tick_params(axis="y", labelsize=8)
        ax_ras.grid(True, ls=":", alpha=0.4)
        ax_ras.legend(loc="upper right", fontsize=7.8, framealpha=0.88)

        # 3. Dual-Curve PETH Panel (Aligned to I_on)
        ax_peth = fig.add_subplot(gs[2], sharex=ax_analog)
        ax_peth.set_facecolor("#f8fafc")

        ax_peth.plot(peth_centers_ms, peth_eup_sm, color="#0284c7", lw=2.2, label=f"Eupnea (n=80 trials, Peak={peak_eup:.1f} Hz)")
        ax_peth.fill_between(peth_centers_ms, np.maximum(0, peth_eup_sm - sem_eup_sm), peth_eup_sm + sem_eup_sm, color="#0284c7", alpha=0.22)

        ax_peth.plot(peth_centers_ms, peth_gasp_sm, color="#e11d48", lw=2.4, label=f"Gasping (n=10 trials, Peak={peak_gasp:.1f} Hz, {gasp_ratio:.1f}x)")
        ax_peth.fill_between(peth_centers_ms, np.maximum(0, peth_gasp_sm - sem_gasp_sm), peth_gasp_sm + sem_gasp_sm, color="#e11d48", alpha=0.22)

        ax_peth.axvline(0.0, color="#10b981", ls="-", lw=1.6, zorder=5)
        ax_peth.axvline(mean_ti_val, color="#f97316", ls=":", lw=1.4, zorder=5)

        ax_peth.set_xlabel("Time Relative to Inspiration Onset ($I_{on}$) [ms]", fontsize=9.5, fontweight="bold", color="#0f172a")
        ax_peth.set_ylabel("Firing Rate (Hz)", fontsize=9, fontweight="bold", color="#0f172a")
        ax_peth.set_ylim(bottom=0.0)
        ax_peth.tick_params(axis="both", labelsize=8)
        ax_peth.grid(True, ls=":", alpha=0.6)
        ax_peth.legend(loc="upper right", fontsize=8.2, framealpha=0.9)

        # Modulation summary badge text box in lower left
        badge_text = (
            f"MODULATION METRICS:\n"
            f"• Driver: {mod_class} [{get_vrc_short(depth)}]\n"
            f"• Jitter: σ_on={sig_on:.1f}ms vs σ_off={sig_off:.1f}ms\n"
            f"• LI: {li:+.2f} | Slope β_TI: {beta_ti:+.2f} (R²={r2_ti:.2f})\n"
            f"• Cross-Corr: r_on={r_on_pk:.3f} vs r_off={r_off_pk:.3f} (Ratio {cc_ratio:.2f}x)"
            if not np.isnan(li) else
            f"MODULATION METRICS:\n• Driver: {mod_class} [{get_vrc_short(depth)}]\n• Insufficient spikes across trials for jitter analysis"
        )
        ax_peth.text(
            0.02, 0.95, badge_text,
            transform=ax_peth.transAxes,
            fontsize=7.8, family="monospace", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor=mod_badge_color, lw=1.4, alpha=0.92)
        )

        card_png_name = f"peth_ion_card_unit_{cid:04d}.png"
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
            "mod_class": mod_class,
            "mod_badge_color": mod_badge_color,
            "sigma_on_ms": round(sig_on, 1) if not np.isnan(sig_on) else np.nan,
            "sigma_off_ms": round(sig_off, 1) if not np.isnan(sig_off) else np.nan,
            "locking_index": round(li, 3) if not np.isnan(li) else np.nan,
            "li_first_spike": round(li_fst, 3) if not np.isnan(li_fst) else np.nan,
            "beta_ti_slope": round(beta_ti, 3) if not np.isnan(beta_ti) else np.nan,
            "r2_ti": round(r2_ti, 3) if not np.isnan(r2_ti) else np.nan,
            "r_on_pk": round(r_on_pk, 4),
            "tau_on_pk_ms": round(tau_on_pk, 1),
            "r_off_pk": round(r_off_pk, 4),
            "tau_off_pk_ms": round(tau_off_pk, 1),
            "cc_ratio": round(cc_ratio, 2),
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
    summary_csv = out_dir / f"{sorter_key}_peth_ion_and_jitter_summary.csv"
    df_peth_summary.to_csv(summary_csv, index=False)
    logger.info(f"Saved summary metrics table to {summary_csv}")

    # -------------------------------------------------------------
    # RENDER COMPREHENSIVE POPULATION SUMMARY FIGURE
    # -------------------------------------------------------------
    fig_pop = plt.figure(figsize=(16.0, 11.0), dpi=200)
    gs_p = gridspec.GridSpec(2, 3, width_ratios=[1.1, 1.1, 1.0], height_ratios=[1.3, 1.0], wspace=0.30, hspace=0.30)

    depth_order = np.argsort(df_qc["depth_um"].values)[::-1]
    sorted_depths = df_qc["depth_um"].values[depth_order]

    norm_eup = pop_peth_eup[depth_order] / (np.max(pop_peth_eup[depth_order], axis=1, keepdims=True) + 1e-6)
    norm_gasp = pop_peth_gasp[depth_order] / (np.max(pop_peth_gasp[depth_order], axis=1, keepdims=True) + 1e-6)

    # 1. Eupnea Heatmap (I_on aligned)
    ax_h1 = fig_pop.add_subplot(gs_p[0, 0])
    im1 = ax_h1.imshow(norm_eup, aspect="auto", cmap="viridis", extent=[peth_centers_ms[0], peth_centers_ms[-1], sorted_depths[-1], sorted_depths[0]], origin="upper")
    ax_h1.axvline(0.0, color="#10b981", ls="-", lw=1.4, label="$I_{on}$")
    ax_h1.axvline(mean_ti_val, color="#f97316", ls=":", lw=1.2, label="Mean $I_{off}$")
    for bound_y in [450, 800, 1300, 1900]:
        ax_h1.axhline(bound_y, color="#ffffff", ls=":", lw=0.8, alpha=0.55)
    ax_h1.set_title(f"A. EUPNEA Heatmap (N={n_units})\nVRC Longitude: Rostral Tip (0 µm) ➔ Caudal Entry", fontsize=9.5, fontweight="bold")
    ax_h1.set_ylabel("VRC Depth (µm)\n(Rostral Tip: 0 µm ➔ Caudal: ~2850 µm)", fontsize=9, fontweight="bold")
    ax_h1.set_xlabel("Time Relative to $I_{on}$ (ms)", fontsize=9, fontweight="bold")
    ax_h1.tick_params(axis="both", labelsize=8)
    fig_pop.colorbar(im1, ax=ax_h1, fraction=0.046, pad=0.04, label="Norm. Rate")

    # 2. Gasping Heatmap (I_on aligned)
    ax_h2 = fig_pop.add_subplot(gs_p[0, 1])
    im2 = ax_h2.imshow(norm_gasp, aspect="auto", cmap="magma", extent=[peth_centers_ms[0], peth_centers_ms[-1], sorted_depths[-1], sorted_depths[0]], origin="upper")
    ax_h2.axvline(0.0, color="#10b981", ls="-", lw=1.4)
    ax_h2.axvline(mean_ti_val, color="#f97316", ls=":", lw=1.2)
    for bound_y in [450, 800, 1300, 1900]:
        ax_h2.axhline(bound_y, color="#ffffff", ls=":", lw=0.8, alpha=0.55)
    ax_h2.set_title(f"B. GASPING Heatmap (N={n_units})\nAutoresuscitation Across VRC Longitude", fontsize=9.5, fontweight="bold")
    ax_h2.set_ylabel("VRC Depth (µm)\n(Rostral Tip: 0 µm ➔ Caudal: ~2850 µm)", fontsize=9, fontweight="bold")
    ax_h2.set_xlabel("Time Relative to $I_{on}$ (ms)", fontsize=9, fontweight="bold")
    ax_h2.tick_params(axis="both", labelsize=8)
    fig_pop.colorbar(im2, ax=ax_h2, fraction=0.046, pad=0.04, label="Norm. Rate")

    # 3. Jitter Scatter Plot: sigma_on vs sigma_off
    ax_sc1 = fig_pop.add_subplot(gs_p[0, 2])
    valid_df = df_peth_summary.dropna(subset=["sigma_on_ms", "sigma_off_ms"])

    palette = {
        "I_on-Locked": "#10b981",
        "I_on-Favoring": "#06b6d4",
        "Duration-Scaler": "#8b5cf6",
        "E_on-Favoring": "#f97316",
        "E_on-Locked": "#f43f5e",
        "Tonic / Unmodulated": "#64748b",
        "Low-FR / Indeterminate": "#94a3b8"
    }

    for c_name, grp in valid_df.groupby("mod_class"):
        c_col = palette.get(c_name, "#64748b")
        ax_sc1.scatter(grp["sigma_on_ms"], grp["sigma_off_ms"], color=c_col, label=f"{c_name} ({len(grp)})", s=42, alpha=0.85, edgecolors="none")

    max_lim = max(80.0, valid_df["sigma_on_ms"].max() if len(valid_df) else 80, valid_df["sigma_off_ms"].max() if len(valid_df) else 80) + 5
    ax_sc1.plot([0, max_lim], [0, max_lim], color="#94a3b8", ls="--", lw=1.2, label="Equal Jitter (y=x)")
    ax_sc1.set_xlim(0, max_lim)
    ax_sc1.set_ylim(0, max_lim)
    ax_sc1.set_xlabel(r"Insp. Onset Jitter $\sigma_{on}$ (ms)", fontsize=9.5, fontweight="bold")
    ax_sc1.set_ylabel(r"Insp. Offset Jitter $\sigma_{off}$ (ms)", fontsize=9.5, fontweight="bold")
    ax_sc1.set_title(r"C. Jitter Comparison: $\sigma_{on}$ vs $\sigma_{off}$", fontsize=10, fontweight="bold")
    ax_sc1.legend(loc="lower right", fontsize=7.2, framealpha=0.9)
    ax_sc1.grid(True, ls=":", alpha=0.5)

    # 4. Duration Regression Slope vs. Locking Index
    ax_sc2 = fig_pop.add_subplot(gs_p[1, 0])
    valid_df2 = df_peth_summary.dropna(subset=["locking_index", "beta_ti_slope"])
    for c_name, grp in valid_df2.groupby("mod_class"):
        c_col = palette.get(c_name, "#64748b")
        ax_sc2.scatter(grp["beta_ti_slope"], grp["locking_index"], color=c_col, s=45, alpha=0.85, edgecolors="none")
    ax_sc2.axhline(0.0, color="#94a3b8", ls="--", lw=1.0)
    ax_sc2.axvline(0.0, color="#10b981", ls=":", lw=1.0, label="I_on Invariant (β=0)")
    ax_sc2.axvline(1.0, color="#f43f5e", ls=":", lw=1.0, label="I_off Tracking (β=1)")
    ax_sc2.set_xlabel(r"Duration Slope $\beta_{T_I} = d(\mathrm{Lat}_{on}) / d(T_I)$", fontsize=9.5, fontweight="bold")
    ax_sc2.set_ylabel(r"Locking Index $\mathrm{LI} = (\sigma_{off} - \sigma_{on})/(\sigma_{off}+\sigma_{on})$", fontsize=9.5, fontweight="bold")
    ax_sc2.set_title(r"D. Parameter Coupling: Slope $\beta_{T_I}$ vs Locking Index", fontsize=10, fontweight="bold")
    ax_sc2.grid(True, ls=":", alpha=0.5)
    ax_sc2.legend(loc="upper right", fontsize=7.5)

    # 5. Anatomical Distribution Across VRC (Bush et al. 2024)
    ax_sc3 = fig_pop.add_subplot(gs_p[1, 1])
    ax_sc3.axhspan(0, 450, color="#f1f5f9", alpha=0.45, zorder=0)
    ax_sc3.axhspan(450, 800, color="#e0f2fe", alpha=0.45, zorder=0)
    ax_sc3.axhspan(800, 1300, color="#fef3c7", alpha=0.45, zorder=0)
    ax_sc3.axhspan(1300, 1900, color="#dcfce7", alpha=0.45, zorder=0)
    ax_sc3.axhspan(1900, 3000, color="#ede9fe", alpha=0.45, zorder=0)

    # Shading zone labels on left edge (unobscured by data and legend)
    ax_sc3.text(-0.33, 225, "Rostral / VII", fontsize=7.5, color="#64748b", fontweight="bold", va="center")
    ax_sc3.text(-0.33, 625, "BötC", fontsize=7.8, color="#0369a1", fontweight="bold", va="center")
    ax_sc3.text(-0.33, 1050, "preBötC", fontsize=7.8, color="#b45309", fontweight="bold", va="center")
    ax_sc3.text(-0.33, 1600, "rVRG", fontsize=7.8, color="#15803d", fontweight="bold", va="center")
    ax_sc3.text(-0.33, 2350, "cVRG", fontsize=7.8, color="#6d28d9", fontweight="bold", va="center")

    for c_name, grp in valid_df2.groupby("mod_class"):
        c_col = palette.get(c_name, "#64748b")
        ax_sc3.scatter(grp["locking_index"], grp["depth"], color=c_col, s=45, alpha=0.88, edgecolors="none", zorder=3)
    ax_sc3.axvline(0.0, color="#94a3b8", ls="--", lw=1.0)
    ax_sc3.set_xlim(-0.35, 0.45)
    ax_sc3.set_ylim(-50, 2950)
    ax_sc3.set_xlabel("Locking Index (LI) [>0: Onset | <0: Offset]", fontsize=9.5, fontweight="bold")
    ax_sc3.set_ylabel("VRC Depth (µm)\n(Rostral Tip: 0 µm ➔ Caudal: ~2850 µm)", fontsize=9.5, fontweight="bold")
    ax_sc3.set_title("E. Anatomical VRC Profile (Bush et al. 2024)", fontsize=10, fontweight="bold")
    ax_sc3.grid(True, ls=":", alpha=0.5)

    # 6. Classification Breakdown Donut
    ax_pie = fig_pop.add_subplot(gs_p[1, 2])
    counts = df_peth_summary["mod_class"].value_counts()
    pie_colors = [palette.get(k, "#64748b") for k in counts.index]
    wedges, texts, autotexts = ax_pie.pie(
        counts.values,
        labels=[f"{k}\n({v})" for k, v in counts.items()],
        colors=pie_colors,
        autopct='%1.1f%%',
        startangle=140,
        wedgeprops=dict(width=0.45, edgecolor='#0f172a', linewidth=1.2),
        textprops=dict(fontsize=8.0, color='#0f172a', fontweight='bold')
    )
    ax_pie.set_title(f"F. Population Phenotype Distribution (N={n_units})", fontsize=10, fontweight="bold")

    fig_pop.suptitle(
        f"e097 win2_anox1 (30:00–30:36 min) | {sorter_label} (N={n_units} Units)\n"
        r"Inspiration Onset ($I_{on}$) Alignment & Trial-by-Trial Jitter Locking Suite",
        fontsize=12.5, fontweight="bold", y=0.98
    )

    pop_png_name = f"{sorter_key}_peth_ion_population_summary.png"
    pop_png_path = out_dir / pop_png_name
    fig_pop.savefig(pop_png_path, bbox_inches="tight")
    plt.close(fig_pop)
    logger.info(f"Saved population ensemble figure to {pop_png_path}")

    # -------------------------------------------------------------
    # BUILD INTERACTIVE STANDALONE HTML EXPLORER GALLERY
    # -------------------------------------------------------------
    gallery_html_path = out_dir / f"{sorter_key}_peth_ion_gallery.html"
    build_html_gallery(
        gallery_html_path=gallery_html_path,
        sorter_label=sorter_label,
        sorter_key=sorter_key,
        unit_records=unit_records,
        pop_png_name=pop_png_name,
        n_units=n_units
    )
    logger.info(f"Generated standalone interactive HTML gallery: {gallery_html_path}")

    return {
        "df_summary": df_peth_summary,
        "pop_peth_eup": pop_peth_eup,
        "pop_peth_gasp": pop_peth_gasp,
        "peth_centers_ms": peth_centers_ms
    }


def build_html_gallery(gallery_html_path: Path, sorter_label: str, sorter_key: str, unit_records: List[Dict[str, Any]], pop_png_name: str, n_units: int):
    """Builds a rich, interactive HTML gallery for exploring I_on-aligned unit cards."""
    # Compute counts for chips and buttons
    df = pd.DataFrame(unit_records)
    n_ion_locked = int((df["mod_class"] == "I_on-Locked").sum()) if "mod_class" in df else 0
    n_ion_favor = int((df["mod_class"] == "I_on-Favoring").sum()) if "mod_class" in df else 0
    n_scaler = int((df["mod_class"] == "Duration-Scaler").sum()) if "mod_class" in df else 0
    n_eon_favor = int((df["mod_class"] == "E_on-Favoring").sum()) if "mod_class" in df else 0
    n_eon_locked = int((df["mod_class"] == "E_on-Locked").sum()) if "mod_class" in df else 0
    n_recruited = int(df["gasp_cat"].str.contains("Recruited").sum())
    n_suppressed = int(df["gasp_cat"].str.contains("Suppressed").sum())

    items_json = json.dumps(unit_records)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>e097 {sorter_label} - I_on Aligned PETH & Jitter Modulation Suite</title>
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
    .nav-links {{
      display: flex;
      gap: 10px;
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
    .ioff-switch-btn {{
      background: #1e293b;
      color: #f97316;
      text-decoration: none;
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      border: 1px solid #f97316;
      transition: all 0.2s;
    }}
    .ioff-switch-btn:hover {{
      background: #f97316;
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
      background: #10b981;
      color: #ffffff;
      padding: 8px 16px;
      border-radius: 6px;
      font-weight: 700;
      font-size: 13px;
      cursor: pointer;
      border: none;
      transition: background 0.2s;
    }}
    .pop-banner-btn:hover {{ background: #059669; }}
    .controls {{
      padding: 12px 32px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
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
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .btn-filter.active {{
      background: #10b981;
      color: #ffffff;
      border-color: #34d399;
    }}
    .search-input {{
      background: #0f172a;
      border: 1px solid #334155;
      color: #ffffff;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 13px;
      width: 200px;
      outline: none;
    }}
    .select-sort {{
      background: #0f172a;
      border: 1px solid #334155;
      color: #ffffff;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12.5px;
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
    .card-badges {{
      display: flex;
      gap: 6px;
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
  </style>
</head>
<body>
  <header>
    <div class="nav-bar">
      <div class="nav-links">
        <a href="../../win2_anox1_sorter_comparison.html" class="back-btn">&larr; Back to Sorter Comparison</a>
        <a href="../peth_ioff_aligned/{sorter_key}_peth_ioff_gallery.html" class="ioff-switch-btn">&#x21C4; Switch to I_off Gallery</a>
      </div>
      <span style="font-size:12px;color:#64748b;">e097 &bull; Subwindow 2 (1800s - 1836s) &bull; I_on Alignment</span>
    </div>
    <h1>e097 {sorter_label} &mdash; Inspiration Onset ($I_{{on}}$) Aligned PETH &amp; Burst Jitter Suite</h1>
    <div class="subtitle">
      Breaths aligned to <strong>Diaphragm EMG Onset ($t = 0\text{{ ms}}$)</strong>. Neuropixels probe inserted longitudinally along the Ventral Respiratory Column (VRC) from Caudal entry (~2850 µm) to Rostral tip (0 µm, near BötC / VII), matching Bush et al. (2024). Dynamic expiration onset ($I_{{off}} = +T_I$) is plotted trial-by-trial to quantify whether burst latencies track $I_{{on}}$ vs $I_{{off}}$.
    </div>
    <div class="metrics-bar">
      <div class="chip">Total Single Units: <strong>{n_units}</strong></div>
      <div class="chip">I_on-Locked: <strong style="color:#10b981;">{n_ion_locked}</strong></div>
      <div class="chip">I_on-Favoring: <strong style="color:#06b6d4;">{n_ion_favor}</strong></div>
      <div class="chip">Duration-Scaler: <strong style="color:#8b5cf6;">{n_scaler}</strong></div>
      <div class="chip">E_on-Favoring: <strong style="color:#f97316;">{n_eon_favor}</strong></div>
      <div class="chip">E_on-Locked: <strong style="color:#f43f5e;">{n_eon_locked}</strong></div>
      <div class="chip">Gasp-Recruited: <strong style="color:#f43f5e;">{n_recruited}</strong></div>
      <div class="chip">Gasp-Suppressed: <strong style="color:#a855f7;">{n_suppressed}</strong></div>
    </div>
  </header>

  <div class="pop-banner">
    <div>
      <h3 style="font-size:14.5px;color:#f8fafc;">Population Ensemble Summary &amp; Jitter Locking Analysis</h3>
      <p style="font-size:12px;color:#94a3b8;">Depth-sorted PETH heatmaps, $\sigma_{{on}}$ vs $\sigma_{{off}}$ jitter scatter, duration slope $\beta_{{T_I}}$, and anatomical profile.</p>
    </div>
    <button class="pop-banner-btn" onclick="openModal('{pop_png_name}')">View Population Figure &rarr;</button>
  </div>

  <div class="controls">
    <button class="btn-filter active" onclick="setFilter('all', this)">All ({n_units})</button>
    <button class="btn-filter" onclick="setFilter('ion_locked', this)">I_on-Locked ({n_ion_locked})</button>
    <button class="btn-filter" onclick="setFilter('ion_favor', this)">I_on-Favoring ({n_ion_favor})</button>
    <button class="btn-filter" onclick="setFilter('scaler', this)">Duration-Scaler ({n_scaler})</button>
    <button class="btn-filter" onclick="setFilter('eon_locked', this)">E_on-Locked ({n_eon_locked})</button>
    <button class="btn-filter" onclick="setFilter('eon_favor', this)">E_on-Favoring ({n_eon_favor})</button>
    <button class="btn-filter" onclick="setFilter('prebotc', this)">preBötC</button>
    <button class="btn-filter" onclick="setFilter('botc', this)">BötC</button>
    <button class="btn-filter" onclick="setFilter('rvrg', this)">rVRG</button>
    <button class="btn-filter" onclick="setFilter('cvrg', this)">cVRG</button>
    <button class="btn-filter" onclick="setFilter('recruited', this)">Gasp-Recruited ({n_recruited})</button>
    <button class="btn-filter" onclick="setFilter('suppressed', this)">Gasp-Suppressed ({n_suppressed})</button>
    <input type="text" id="searchInput" class="search-input" placeholder="Search Cluster ID..." oninput="applyFilters()">
    <select id="sortSelect" class="select-sort" onchange="applySort()">
      <option value="depth_asc">Sort: Rostral (Tip) ➔ Caudal (Entry) (Depth &uarr;)</option>
      <option value="depth_desc">Sort: Caudal (Entry) ➔ Rostral (Tip) (Depth &darr;)</option>
      <option value="li_desc">Sort: Locking Index (I_on locked &darr;)</option>
      <option value="li_asc">Sort: Locking Index (E_on locked &uarr;)</option>
      <option value="beta_desc">Sort: Duration Slope β (Highest &darr;)</option>
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
    const items = {items_json};
    let activeFilter = 'all';

    function renderCards(list) {{
      const grid = document.getElementById('grid');
      grid.innerHTML = '';
      list.forEach(c => {{
        const el = document.createElement('div');
        el.className = 'unit-card';
        const liTxt = (c.locking_index !== null && c.locking_index !== undefined && !isNaN(c.locking_index)) ? (c.locking_index > 0 ? '+' + c.locking_index.toFixed(2) : c.locking_index.toFixed(2)) : 'N/A';
        const betaTxt = (c.beta_ti_slope !== null && c.beta_ti_slope !== undefined && !isNaN(c.beta_ti_slope)) ? (c.beta_ti_slope > 0 ? '+' + c.beta_ti_slope.toFixed(2) : c.beta_ti_slope.toFixed(2)) : 'N/A';
        const sigOnTxt = (c.sigma_on_ms !== null && !isNaN(c.sigma_on_ms)) ? c.sigma_on_ms.toFixed(1) + ' ms' : 'N/A';
        const sigOffTxt = (c.sigma_off_ms !== null && !isNaN(c.sigma_off_ms)) ? c.sigma_off_ms.toFixed(1) + ' ms' : 'N/A';
        
        el.innerHTML = `
          <div class="card-head">
            <span class="card-title">Cluster #${{String(c.cid).padStart(3, '0')}} (${{c.depth.toFixed(1)}} µm &bull; ${{c.vrc_short}})</span>
            <div class="card-badges">
              <span class="card-badge" style="background:${{c.mod_badge_color}}">${{c.mod_class}}</span>
              <span class="card-badge" style="background:${{c.gasp_badge_color}}">${{c.gasp_cat}}</span>
            </div>
          </div>
          <img src="${{c.img}}" class="card-img" loading="lazy" onclick="openModal('${{c.img}}')">
          <div class="card-meta">
            <div><strong>VRC Region:</strong> ${{c.vrc_full}}</div>
            <div><strong>Locking Index (LI):</strong> ${{liTxt}}</div>
            <div><strong>Duration Slope (β):</strong> ${{betaTxt}}</div>
            <div><strong>Jitter σ_on / σ_off:</strong> ${{sigOnTxt}} / ${{sigOffTxt}}</div>
            <div><strong>Cross-Corr Ratio:</strong> ${{c.cc_ratio.toFixed(2)}}x</div>
            <div><strong>Gasp Fold-Change:</strong> ${{c.gasp_ratio.toFixed(2)}}x</div>
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
        if (activeFilter === 'ion_locked') return c.mod_class === 'I_on-Locked';
        if (activeFilter === 'ion_favor') return c.mod_class === 'I_on-Favoring';
        if (activeFilter === 'scaler') return c.mod_class === 'Duration-Scaler';
        if (activeFilter === 'eon_locked') return c.mod_class === 'E_on-Locked';
        if (activeFilter === 'eon_favor') return c.mod_class === 'E_on-Favoring';
        if (activeFilter === 'prebotc') return c.vrc_short === 'preBötC';
        if (activeFilter === 'botc') return c.vrc_short === 'BötC';
        if (activeFilter === 'rvrg') return c.vrc_short === 'rVRG';
        if (activeFilter === 'cvrg') return c.vrc_short === 'cVRG';
        if (activeFilter === 'recruited') return c.gasp_cat.includes('Recruited');
        if (activeFilter === 'suppressed') return c.gasp_cat.includes('Suppressed');
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
      else if (val === 'li_desc') list.sort((a,b) => (b.locking_index || -999) - (a.locking_index || -999));
      else if (val === 'li_asc') list.sort((a,b) => (a.locking_index || 999) - (b.locking_index || 999));
      else if (val === 'beta_desc') list.sort((a,b) => (b.beta_ti_slope || -999) - (a.beta_ti_slope || -999));
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
    gallery_html_path.write_text(html_content, encoding="utf-8")


def render_cross_sorter_synthesis(res_ks4: Dict[str, Any], res_lupin: Dict[str, Any]):
    """Renders a cross-sorter synthesis figure comparing Kilosort 4 vs Lupin on event locking."""
    out_dir = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / "win2_anox1"
    df_ks4 = res_ks4["df_summary"]
    df_lupin = res_lupin["df_summary"]

    fig = plt.figure(figsize=(15.0, 9.5), dpi=200)
    gs = gridspec.GridSpec(2, 2, hspace=0.28, wspace=0.24)

    palette = {
        "I_on-Locked": "#10b981",
        "I_on-Favoring": "#06b6d4",
        "Duration-Scaler": "#8b5cf6",
        "E_on-Favoring": "#f97316",
        "E_on-Locked": "#f43f5e",
        "Tonic / Unmodulated": "#64748b",
        "Low-FR / Indeterminate": "#94a3b8"
    }

    # Panel A: KS4 Locking Index vs Depth
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axhspan(0, 450, color="#f1f5f9", alpha=0.45, zorder=0)
    ax1.axhspan(450, 800, color="#e0f2fe", alpha=0.45, zorder=0)
    ax1.axhspan(800, 1300, color="#fef3c7", alpha=0.45, zorder=0)
    ax1.axhspan(1300, 1900, color="#dcfce7", alpha=0.45, zorder=0)
    ax1.axhspan(1900, 3000, color="#ede9fe", alpha=0.45, zorder=0)
    ax1.text(-0.33, 225, "Rostral / VII", fontsize=7.5, color="#64748b", fontweight="bold", va="center")
    ax1.text(-0.33, 625, "BötC", fontsize=7.8, color="#0369a1", fontweight="bold", va="center")
    ax1.text(-0.33, 1050, "preBötC", fontsize=7.8, color="#b45309", fontweight="bold", va="center")
    ax1.text(-0.33, 1600, "rVRG", fontsize=7.8, color="#15803d", fontweight="bold", va="center")
    ax1.text(-0.33, 2350, "cVRG", fontsize=7.8, color="#6d28d9", fontweight="bold", va="center")

    for c_name, grp in df_ks4.dropna(subset=["locking_index"]).groupby("mod_class"):
        c_col = palette.get(c_name, "#64748b")
        ax1.scatter(grp["locking_index"], grp["depth"], color=c_col, label=f"{c_name} ({len(grp)})", s=45, alpha=0.88, zorder=3)
    ax1.axvline(0.0, color="#94a3b8", ls="--", lw=1.2)
    ax1.set_xlim(-0.35, 0.45)
    ax1.set_ylim(-50, 2950)
    ax1.set_xlabel("Locking Index (LI)", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("VRC Depth (µm)\n(Rostral Tip: 0 µm ➔ Caudal: ~2850 µm)", fontsize=9.5, fontweight="bold")
    ax1.set_title("A. Kilosort 4 (N=92 Units): Event Locking along Rostral-Caudal VRC", fontsize=10.5, fontweight="bold")
    ax1.grid(True, ls=":", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

    # Panel B: Lupin Locking Index vs Depth
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axhspan(0, 450, color="#f1f5f9", alpha=0.45, zorder=0)
    ax2.axhspan(450, 800, color="#e0f2fe", alpha=0.45, zorder=0)
    ax2.axhspan(800, 1300, color="#fef3c7", alpha=0.45, zorder=0)
    ax2.axhspan(1300, 1900, color="#dcfce7", alpha=0.45, zorder=0)
    ax2.axhspan(1900, 3000, color="#ede9fe", alpha=0.45, zorder=0)
    ax2.text(-0.33, 225, "Rostral / VII", fontsize=7.5, color="#64748b", fontweight="bold", va="center")
    ax2.text(-0.33, 625, "BötC", fontsize=7.8, color="#0369a1", fontweight="bold", va="center")
    ax2.text(-0.33, 1050, "preBötC", fontsize=7.8, color="#b45309", fontweight="bold", va="center")
    ax2.text(-0.33, 1600, "rVRG", fontsize=7.8, color="#15803d", fontweight="bold", va="center")
    ax2.text(-0.33, 2350, "cVRG", fontsize=7.8, color="#6d28d9", fontweight="bold", va="center")

    for c_name, grp in df_lupin.dropna(subset=["locking_index"]).groupby("mod_class"):
        c_col = palette.get(c_name, "#64748b")
        ax2.scatter(grp["locking_index"], grp["depth"], color=c_col, label=f"{c_name} ({len(grp)})", s=45, alpha=0.88, zorder=3)
    ax2.axvline(0.0, color="#94a3b8", ls="--", lw=1.2)
    ax2.set_xlim(-0.35, 0.45)
    ax2.set_ylim(-50, 2950)
    ax2.set_xlabel("Locking Index (LI)", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("VRC Depth (µm)\n(Rostral Tip: 0 µm ➔ Caudal: ~2850 µm)", fontsize=9.5, fontweight="bold")
    ax2.set_title("B. Lupin SVD-OMP (N=120 Units): Event Locking along Rostral-Caudal VRC", fontsize=10.5, fontweight="bold")
    ax2.grid(True, ls=":", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

    # Panel C: Jitter Comparison (sigma_on vs sigma_off) Across Sorters
    ax3 = fig.add_subplot(gs[1, 0])
    v_ks4 = df_ks4.dropna(subset=["sigma_on_ms", "sigma_off_ms"])
    v_lupin = df_lupin.dropna(subset=["sigma_on_ms", "sigma_off_ms"])
    ax3.scatter(v_ks4["sigma_on_ms"], v_ks4["sigma_off_ms"], color="#0284c7", s=38, alpha=0.75, label=f"Kilosort 4 (N={len(v_ks4)})")
    ax3.scatter(v_lupin["sigma_on_ms"], v_lupin["sigma_off_ms"], color="#f59e0b", s=38, alpha=0.75, label=f"Lupin (N={len(v_lupin)})")
    ax3.plot([0, 95], [0, 95], color="#94a3b8", ls="--", lw=1.2, label="Equal Jitter (y=x)")
    ax3.set_xlim(0, 95)
    ax3.set_ylim(0, 95)
    ax3.set_xlabel(r"Inspiration Onset Jitter $\sigma_{on}$ (ms)", fontsize=9.5, fontweight="bold")
    ax3.set_ylabel(r"Inspiration Offset Jitter $\sigma_{off}$ (ms)", fontsize=9.5, fontweight="bold")
    ax3.set_title(r"C. Cross-Sorter Jitter Concordance ($\sigma_{on}$ vs $\sigma_{off}$)", fontsize=10.5, fontweight="bold")
    ax3.grid(True, ls=":", alpha=0.5)
    ax3.legend(loc="lower right", fontsize=8.0, framealpha=0.9)

    # Panel D: Duration Slope (beta_TI) Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    b_ks4 = df_ks4["beta_ti_slope"].dropna()
    b_lup = df_lupin["beta_ti_slope"].dropna()
    bins = np.linspace(-0.4, 1.2, 25)
    ax4.hist(b_ks4, bins=bins, color="#0284c7", alpha=0.65, label=f"KS4 (Median: {b_ks4.median():+.2f})", density=True)
    ax4.hist(b_lup, bins=bins, color="#f59e0b", alpha=0.65, label=f"Lupin (Median: {b_lup.median():+.2f})", density=True)
    ax4.axvline(0.0, color="#10b981", ls="--", lw=1.4, label=r"$I_{on}$-Locked Baseline ($\beta=0$)")
    ax4.axvline(1.0, color="#f43f5e", ls="--", lw=1.4, label=r"$I_{off}$-Locked Tracking ($\beta=1$)")
    ax4.set_xlabel(r"Inspiratory Duration Regression Slope $\beta_{T_I} = d(\mathrm{Lat}_{on}) / d(T_I)$", fontsize=9.5, fontweight="bold")
    ax4.set_ylabel("Probability Density", fontsize=9.5, fontweight="bold")
    ax4.set_title("D. Population Sensitivity to Breath Duration ($T_I$ Scaling)", fontsize=10.5, fontweight="bold")
    ax4.grid(True, ls=":", alpha=0.5)
    ax4.legend(loc="upper right", fontsize=7.8, framealpha=0.9)

    fig.suptitle(
        "e097 win2_anox1 (Anoxic Challenge 1) | Cross-Sorter Event Modulation Synthesis\n"
        "Quantitative Concordance of Inspiration Onset vs. Expiration Onset Coupling",
        fontsize=12.5, fontweight="bold", y=0.98
    )

    out_synthesis_png = out_dir / "win2_anox1_ion_ioff_modulation_synthesis.png"
    fig.savefig(out_synthesis_png, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved cross-sorter synthesis figure to {out_synthesis_png}")


def build_master_ion_dashboard():
    """Generates a master landing dashboard connecting I_on and I_off galleries for both sorters."""
    out_dir = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / "win2_anox1"
    dash_path = out_dir / "win2_anox1_ion_peth_dashboard.html"

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>e097 win2_anox1 - Inspiration Onset (I_on) Aligned Modulation Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: #090d16;
      color: #f1f5f9;
      line-height: 1.5;
      padding: 32px;
    }
    .container {
      max-width: 1200px;
      margin: 0 auto;
    }
    .nav-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
    }
    .back-btn {
      background: #1e293b;
      color: #38bdf8;
      text-decoration: none;
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      border: 1px solid #334155;
    }
    .back-btn:hover { background: #38bdf8; color: #090d16; }
    h1 {
      font-size: 26px;
      font-weight: 800;
      color: #f8fafc;
      letter-spacing: -0.02em;
    }
    .subtitle {
      color: #94a3b8;
      font-size: 14px;
      margin-top: 6px;
      margin-bottom: 24px;
    }
    .grid-2 {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 24px;
      margin-bottom: 32px;
    }
    .hub-card {
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 12px;
      padding: 24px;
      transition: all 0.2s;
    }
    .hub-card:hover {
      border-color: #10b981;
      transform: translateY(-2px);
    }
    .hub-card h2 {
      font-size: 18px;
      font-weight: 700;
      margin-bottom: 8px;
    }
    .hub-card p {
      font-size: 13px;
      color: #94a3b8;
      margin-bottom: 16px;
    }
    .badge-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
      margin-bottom: 20px;
      font-size: 12px;
    }
    .badge-pill {
      background: #1e293b;
      border: 1px solid #334155;
      padding: 6px 10px;
      border-radius: 6px;
    }
    .badge-pill strong { color: #f8fafc; }
    .hub-btn {
      display: inline-block;
      width: 100%;
      text-align: center;
      padding: 10px;
      border-radius: 8px;
      font-weight: 700;
      font-size: 14px;
      text-decoration: none;
      transition: all 0.2s;
    }
    .btn-ks4 { background: #0284c7; color: #ffffff; }
    .btn-ks4:hover { background: #0369a1; }
    .btn-lupin { background: #f59e0b; color: #090d16; }
    .btn-lupin:hover { background: #d97706; }
    .synthesis-section {
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 32px;
    }
    .synthesis-img {
      width: 100%;
      border-radius: 8px;
      margin-top: 16px;
      border: 1px solid #334155;
      cursor: pointer;
    }
    .modal {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.9);
      z-index: 100;
      justify-content: center;
      align-items: center;
      padding: 20px;
    }
    .modal img {
      max-width: 95vw;
      max-height: 95vh;
      border-radius: 8px;
      border: 2px solid #10b981;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="nav-bar">
      <a href="win2_anox1_sorter_comparison.html" class="back-btn">&larr; Back to Sorter Comparison</a>
      <a href="win2_anox1_ioff_peth_dashboard.html" class="back-btn" style="border-color:#f97316;color:#f97316;">&#x21C4; View I_off Aligned Suite</a>
    </div>

    <h1>e097 win2_anox1 &mdash; Inspiration Onset ($I_{on}$) Aligned Modulation &amp; Jitter Suite</h1>
    <div class="subtitle">
      Detailed breath-by-breath spike timing analysis aligned to <strong>Diaphragm EMG Onset ($t = 0\text{ ms}$)</strong>. 
      Contrasts Inspiration Onset ($I_{on}$) vs Expiration Onset ($I_{off}$) locking across 90 breath trials (80 Eupnea + 10 Autoresuscitative Gasps).
    </div>

    <div class="grid-2">
      <!-- Kilosort 4 Card -->
      <div class="hub-card">
        <h2 style="color:#38bdf8;">Kilosort 4 (GPU Pipeline)</h2>
        <p>92 QC-Passed Single Units. Fast automated template-matching with drift correction.</p>
        <div class="badge-grid">
          <div class="badge-pill">I_on-Locked: <strong style="color:#10b981;">37 units (40.2%)</strong></div>
          <div class="badge-pill">I_on-Favoring: <strong style="color:#06b6d4;">26 units (28.3%)</strong></div>
          <div class="badge-pill">Duration-Scaler: <strong style="color:#8b5cf6;">20 units (21.7%)</strong></div>
          <div class="badge-pill">E_on-Favoring: <strong style="color:#f97316;">5 units (5.4%)</strong></div>
          <div class="badge-pill">E_on-Locked: <strong style="color:#f43f5e;">3 units (3.3%)</strong></div>
          <div class="badge-pill">Gasp-Recruited: <strong style="color:#f43f5e;">30 units</strong></div>
        </div>
        <a href="kilosort4/peth_ion_aligned/kilosort4_peth_ion_gallery.html" class="hub-btn btn-ks4">Open Kilosort 4 I_on Gallery (92 Units) &rarr;</a>
      </div>

      <!-- Lupin Card -->
      <div class="hub-card">
        <h2 style="color:#f59e0b;">Lupin (SVD-OMP CPU Engine)</h2>
        <p>120 QC-Passed Single Units. SVD-OMP pursuer resolving overlapping spikes.</p>
        <div class="badge-grid">
          <div class="badge-pill">I_on-Locked: <strong style="color:#10b981;">39 units (32.5%)</strong></div>
          <div class="badge-pill">I_on-Favoring: <strong style="color:#06b6d4;">26 units (21.7%)</strong></div>
          <div class="badge-pill">Duration-Scaler: <strong style="color:#8b5cf6;">42 units (35.0%)</strong></div>
          <div class="badge-pill">E_on-Favoring: <strong style="color:#f97316;">7 units (5.8%)</strong></div>
          <div class="badge-pill">E_on-Locked: <strong style="color:#f43f5e;">5 units (4.2%)</strong></div>
          <div class="badge-pill">Gasp-Recruited: <strong style="color:#f43f5e;">38 units</strong></div>
        </div>
        <a href="lupin/peth_ion_aligned/lupin_peth_ion_gallery.html" class="hub-btn btn-lupin">Open Lupin I_on Gallery (120 Units) &rarr;</a>
      </div>
    </div>

    <!-- Cross-Sorter Synthesis Section -->
    <div class="synthesis-section">
      <h2 style="font-size:18px;font-weight:700;color:#f8fafc;">Cross-Sorter Event Modulation Synthesis Figure</h2>
      <p style="font-size:13px;color:#94a3b8;margin-top:4px;">
        Quantitative concordance between Kilosort 4 and Lupin SVD-OMP. Demonstrates consistent separation between $I_{on}$-locked units (tight $\sigma_{on} \approx 10\text{--}30\text{ ms}$, $\beta_{T_I} \approx 0.0$) and $E_{on}$-locked units (tight $\sigma_{off}$, $\beta_{T_I} \approx 1.0$).
      </p>
      <img src="win2_anox1_ion_ioff_modulation_synthesis.png" class="synthesis-img" onclick="openModal('win2_anox1_ion_ioff_modulation_synthesis.png')">
    </div>
  </div>

  <div class="modal" id="modal" onclick="closeModal()">
    <img id="modalImg" src="">
  </div>

  <script>
    function openModal(src) {
      document.getElementById('modalImg').src = src;
      document.getElementById('modal').style.display = 'flex';
    }
    function closeModal() {
      document.getElementById('modal').style.display = 'none';
    }
  </script>
</body>
</html>
"""
    dash_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Generated master I_on dashboard: {dash_path}")


def main():
    t_start = time.time()
    logger.info("=================================================================")
    logger.info("STARTING WIN2_ANOX1 I_ON ALIGNED PETH & JITTER MODULATION SUITE")
    logger.info("=================================================================")

    # 1. Extract breath telemetry
    breath_data = extract_win2_breaths()

    # 2. Compute analog averages aligned to I_on
    analog_avg = compute_analog_averages_ion(breath_data)

    # 3. Render Kilosort 4 suite
    logger.info("\n--- STEP 1: RENDERING KILOSORT 4 I_ON SUITE ---")
    res_ks4 = render_peth_ion_suite_for_sorter(
        sorter_key="kilosort4",
        sorter_label="Kilosort 4 (GPU)",
        breath_data=breath_data,
        analog_avg=analog_avg
    )

    # 4. Render Lupin suite
    logger.info("\n--- STEP 2: RENDERING LUPIN SVD-OMP I_ON SUITE ---")
    res_lupin = render_peth_ion_suite_for_sorter(
        sorter_key="lupin",
        sorter_label="Lupin (SVD-OMP)",
        breath_data=breath_data,
        analog_avg=analog_avg
    )

    # 5. Render cross-sorter synthesis figure
    logger.info("\n--- STEP 3: RENDERING CROSS-SORTER SYNTHESIS FIGURE ---")
    render_cross_sorter_synthesis(res_ks4, res_lupin)

    # 6. Build Master I_on Dashboard
    logger.info("\n--- STEP 4: BUILDING MASTER I_ON DASHBOARD ---")
    build_master_ion_dashboard()

    dur_tot = time.time() - t_start
    logger.info(f"\nSUCCESS! Completed full I_on suite in {dur_tot:.1f}s ({dur_tot/60.0:.2f} min).")


if __name__ == "__main__":
    main()
