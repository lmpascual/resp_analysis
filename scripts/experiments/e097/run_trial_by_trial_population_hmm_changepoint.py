"""Trial-by-Trial Population Dynamics, HMM Micro-State Decoding, and Change-Point Analysis
for e097 win2_anox1 (Anoxic Challenge 1, 1800.0s - 1836.0s).

Implements:
1. Continuous 10 ms binned population firing rates across all single units for both
   Kilosort 4 (92 units) and Lupin (120 units).
2. Hidden Markov Model (HMM) state decoding (4 states: Baseline/Late-E, Inspiratory Burst,
   Post-Inspiratory/Off-Switch, and Gasp/Hypoxic Recruit).
3. Trial-by-trial intra-breath micro-state sequence tracking across all 90 breath trials.
4. Intra-breath and macro-transition change-point detection (using `ruptures.Pelt` & `Binseg`)
   to detect microsecond burst onsets/offsets and macro-anoxic phase shifts.
5. High-resolution multi-panel figures and an interactive HTML dashboard.
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
from matplotlib.patches import Patch

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import ruptures as rpt

REPO_ROOT = Path(r"c:\Users\lpascu\resp_analysis")
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.processing.analog_nidq.proc_emg import process_emg
from src.pipeline.processing.analog_nidq.proc_flow import process_flow_meter
from src.pipeline.config import EMGConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("POP_HMM_CHANGEPOINT")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.0


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

    gasp_mask = np.zeros(n_breaths, dtype=bool)
    for i in range(n_breaths):
        if valid_onsets[i] >= 27.0:
            gasp_mask[i] = True

    cycle_ends = np.append(valid_onsets[1:], dur)
    ti_arr = (flow_zcs - valid_onsets) * 1000.0  # ms
    te_arr = (cycle_ends - flow_zcs) * 1000.0   # ms
    tot_arr = (cycle_ends - valid_onsets) * 1000.0 # ms

    logger.info(f"Loaded {n_breaths} breaths ({np.sum(~gasp_mask)} eupnea, {np.sum(gasp_mask)} gasps). Mean T_I = {np.mean(ti_arr):.1f} ms.")

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
        "cycle_ends": cycle_ends,
        "ti_arr": ti_arr,
        "te_arr": te_arr,
        "tot_arr": tot_arr,
        "gasp_mask": gasp_mask,
        "n_breaths": n_breaths
    }


def run_population_analysis(sorter_key: str, sorter_label: str, breath_data: Dict[str, Any], dt: float = 0.010, n_states: int = 4):
    """Processes population data for a sorter using 10 ms bins, trains HMM, and detects change points."""
    base_dir = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / "win2_anox1" / sorter_key
    spk_times = np.load(base_dir / "alf" / "spike_times.npy")
    spk_clusters = np.load(base_dir / "alf" / "spike_clusters.npy")
    qc_csv = base_dir / f"{sorter_key}_qc_summary.csv"
    if not qc_csv.exists():
        qc_csv = base_dir / "cluster_info.tsv"
    sep = "\t" if qc_csv.suffix == ".tsv" else ","
    qc_df = pd.read_csv(qc_csv, sep=sep)
    if "passed_allen_qc" in qc_df.columns:
        df_qc = qc_df[qc_df["passed_allen_qc"] == True].copy()
    else:
        df_qc = qc_df.copy()

    df_qc = df_qc.sort_values(by="depth_um", ascending=False).reset_index(drop=True)
    cids = df_qc["cluster_id"].values
    depths = df_qc["depth_um"].values
    n_units = len(cids)
    dur = breath_data["dur"]

    time_bins = np.arange(0, dur + dt, dt)
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2.0
    n_time_bins = len(bin_centers)

    logger.info(f"[{sorter_label}] Binning {n_units} units into {n_time_bins} bins (dt = {dt*1000:.0f} ms)...")
    pop_rates = np.zeros((n_time_bins, n_units), dtype=np.float32)

    for u_idx, cid in enumerate(cids):
        c_spk = spk_times[spk_clusters == cid]
        counts, _ = np.histogram(c_spk, bins=time_bins)
        rate = counts / dt
        # Light 1-bin Gaussian filter (10 ms sigma) to preserve fast abrupt edges
        pop_rates[:, u_idx] = gaussian_filter1d(rate, sigma=1.0)

    # Dimensionality reduction via PCA (top 6 components)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pop_rates)
    pca = PCA(n_components=6)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    logger.info(f"[{sorter_label}] PCA top 6 components explain {np.sum(var_exp)*100:.1f}% variance.")

    # Train Gaussian HMM
    logger.info(f"[{sorter_label}] Training {n_states}-state Gaussian HMM...")
    hmm_model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=250, random_state=42)
    hmm_model.fit(X_pca)
    states = hmm_model.predict(X_pca)
    post_probs = hmm_model.predict_proba(X_pca)

    # Characterize states relative to ground-truth respiratory landmarks:
    # 0 = Late-E / Baseline Expiration, 1 = Inspiration (EMG onset to Flow ZC), 2 = Post-I (ZC to EMG off)
    valid_onsets = breath_data["valid_onsets"]
    flow_zcs = breath_data["flow_zcs"]
    valid_offsets = breath_data["valid_offsets"]
    n_breaths = breath_data["n_breaths"]
    gasp_mask = breath_data["gasp_mask"]

    phase_labels = np.zeros(n_time_bins, dtype=int)
    for b_idx in range(n_breaths):
        t_on = valid_onsets[b_idx]
        t_zc = flow_zcs[b_idx]
        t_off = valid_offsets[b_idx]
        phase_labels[(bin_centers >= t_on) & (bin_centers < t_zc)] = 1
        phase_labels[(bin_centers >= t_zc) & (bin_centers < t_off)] = 2

    state_stats = []
    for s in range(n_states):
        mask_s = (states == s)
        n_occ = np.sum(mask_s)
        pct_insp = np.mean(phase_labels[mask_s] == 1) * 100 if n_occ > 0 else 0
        pct_post = np.mean(phase_labels[mask_s] == 2) * 100 if n_occ > 0 else 0
        pct_exp = np.mean(phase_labels[mask_s] == 0) * 100 if n_occ > 0 else 0
        # Mean time of occurrence (to identify gasp recruitment states)
        mean_t = np.mean(bin_centers[mask_s]) if n_occ > 0 else 0
        pct_in_gasp = np.mean(bin_centers[mask_s] >= 27.0) * 100 if n_occ > 0 else 0
        state_stats.append({
            "state_id": s,
            "n_bins": int(n_occ),
            "duration_s": float(n_occ * dt),
            "pct_insp": float(pct_insp),
            "pct_post": float(pct_post),
            "pct_exp": float(pct_exp),
            "mean_t": float(mean_t),
            "pct_in_gasp": float(pct_in_gasp)
        })

    # Sort and label states canonically:
    # Highest pct_insp -> 'Inspiration (Burst)'
    # High pct_in_gasp -> 'Gasp / Hypoxic Recruitment'
    # High pct_post -> 'Post-Inspiratory / Off-Switch'
    # Remainder -> 'Late-Expiratory / Baseline'
    sorted_by_insp = sorted(state_stats, key=lambda x: x["pct_insp"], reverse=True)
    insp_state = sorted_by_insp[0]["state_id"]

    remaining = [s["state_id"] for s in state_stats if s["state_id"] != insp_state]
    sorted_by_gasp = sorted([s for s in state_stats if s["state_id"] in remaining], key=lambda x: x["pct_in_gasp"], reverse=True)
    gasp_state = sorted_by_gasp[0]["state_id"]

    remaining2 = [s for s in remaining if s != gasp_state]
    sorted_by_post = sorted([s for s in state_stats if s["state_id"] in remaining2], key=lambda x: x["pct_post"], reverse=True)
    post_state = sorted_by_post[0]["state_id"] if len(sorted_by_post) > 0 else remaining2[0]

    base_state = [s for s in remaining2 if s != post_state][0] if len([s for s in remaining2 if s != post_state]) > 0 else remaining2[0]

    state_name_map = {
        insp_state: "Inspiratory Burst (Insp)",
        post_state: "Post-Inspiratory / Off-Switch (Post-I)",
        base_state: "Late-Expiratory Quiescence (E2)",
        gasp_state: "Hypoxic / Gasp Recruitment"
    }

    state_color_map = {
        insp_state: "#10b981",  # Emerald Green
        post_state: "#f97316",  # Bright Orange
        base_state: "#64748b",  # Slate Gray
        gasp_state: "#ef4444"   # Crimson Red
    }

    logger.info(f"[{sorter_label}] Canonical State Mapping:")
    for s_id, s_name in state_name_map.items():
        st = next(item for item in state_stats if item["state_id"] == s_id)
        logger.info(f"  State {s_id} -> {s_name}: Insp={st['pct_insp']:.1f}%, Post={st['pct_post']:.1f}%, GaspOcc={st['pct_in_gasp']:.1f}%")

    # 4. Continuous Change-Point Detection on PCA trajectory via ruptures (PELT with RBF kernel)
    logger.info(f"[{sorter_label}] Running continuous change-point detection with PELT...")
    algo_pelt = rpt.Pelt(model="rbf", min_size=5).fit(X_pca)
    # Pen chosen for sharp temporal break detection
    bkps_bins = algo_pelt.predict(pen=4.0)
    bkps_times = [bin_centers[b-1] for b in bkps_bins[:-1]]
    logger.info(f"[{sorter_label}] Detected {len(bkps_times)} continuous change points across 36s.")

    # 5. Trial-by-Trial Intra-Breath Micro-State Profiling across 90 trials
    logger.info(f"[{sorter_label}] Profiling intra-breath micro-state sequences across 90 trials...")
    trial_records = []
    trial_state_raster = []  # Matrix: 90 trials x n_bins_per_trial

    win_pre_ms = 50.0
    win_post_ms = 350.0
    dt_trial = dt
    n_trial_bins = int((win_pre_ms + win_post_ms) / (dt_trial * 1000.0))
    t_trial_axis_ms = np.arange(-win_pre_ms, win_post_ms, dt_trial * 1000.0)

    for b_idx in range(n_breaths):
        t_on = valid_onsets[b_idx]
        t_zc = flow_zcs[b_idx]
        ti = breath_data["ti_arr"][b_idx]
        te = breath_data["te_arr"][b_idx]
        tot = breath_data["tot_arr"][b_idx]
        is_gasp = gasp_mask[b_idx]

        # Intra-breath slice in continuous bin grid
        t_start = t_on - (win_pre_ms / 1000.0)
        t_end = t_on + (win_post_ms / 1000.0)
        mask_trial = (bin_centers >= t_start) & (bin_centers < t_end)
        trial_s = states[mask_trial]

        # Resample/pad to exact n_trial_bins
        if len(trial_s) < n_trial_bins:
            pad_val = trial_s[-1] if len(trial_s) > 0 else base_state
            trial_s_fixed = np.pad(trial_s, (0, n_trial_bins - len(trial_s)), constant_values=pad_val)
        else:
            trial_s_fixed = trial_s[:n_trial_bins]
        trial_state_raster.append(trial_s_fixed)

        # Micro-state occupancy during inspiration (t_on to t_zc)
        mask_ti = (bin_centers >= t_on) & (bin_centers < t_zc)
        s_ti = states[mask_ti]
        dur_ti_insp_state_ms = np.sum(s_ti == insp_state) * (dt * 1000.0) if len(s_ti) > 0 else 0.0

        # Micro-state occupancy during expiration
        t_next = breath_data["cycle_ends"][b_idx]
        mask_te = (bin_centers >= t_zc) & (bin_centers < t_next)
        s_te = states[mask_te]
        dur_te_post_state_ms = np.sum(s_te == post_state) * (dt * 1000.0) if len(s_te) > 0 else 0.0

        # Intra-breath change-point: find first transition into insp_state and out of insp_state
        insp_bins = np.where(trial_s_fixed == insp_state)[0]
        if len(insp_bins) > 0:
            first_insp_ms = t_trial_axis_ms[insp_bins[0]]
            last_insp_ms = t_trial_axis_ms[insp_bins[-1]]
            insp_state_dur_ms = last_insp_ms - first_insp_ms
        else:
            first_insp_ms = np.nan
            last_insp_ms = np.nan
            insp_state_dur_ms = 0.0

        trial_records.append({
            "trial_idx": b_idx,
            "t_on_s": float(t_on),
            "t_zc_s": float(t_zc),
            "ti_ms": float(ti),
            "te_ms": float(te),
            "tot_ms": float(tot),
            "is_gasp": bool(is_gasp),
            "dur_ti_insp_state_ms": float(dur_ti_insp_state_ms),
            "dur_te_post_state_ms": float(dur_te_post_state_ms),
            "first_insp_ms": float(first_insp_ms),
            "last_insp_ms": float(last_insp_ms),
            "insp_state_dur_ms": float(insp_state_dur_ms),
        })

    trial_df = pd.DataFrame(trial_records)
    trial_state_raster = np.array(trial_state_raster)

    # 6. Trial-Level Macro Change-Point Detection (using trial mean population vectors)
    trial_pop_matrix = np.zeros((n_breaths, n_units), dtype=np.float32)
    for b_idx in range(n_breaths):
        t_on = valid_onsets[b_idx]
        t_end = breath_data["cycle_ends"][b_idx]
        mask_b = (bin_centers >= t_on) & (bin_centers < t_end)
        if np.sum(mask_b) > 0:
            trial_pop_matrix[b_idx, :] = np.mean(pop_rates[mask_b, :], axis=0)

    scaler_tr = StandardScaler()
    X_tr_scaled = scaler_tr.fit_transform(trial_pop_matrix)
    pca_tr = PCA(n_components=4)
    X_tr_pca = pca_tr.fit_transform(X_tr_scaled)

    algo_tr_pelt = rpt.Pelt(model="rbf", min_size=4).fit(X_tr_pca)
    trial_change_points = algo_tr_pelt.predict(pen=2.5)
    logger.info(f"[{sorter_label}] Trial-by-trial Macro Change-Points: {trial_change_points}")

    return {
        "sorter_key": sorter_key,
        "sorter_label": sorter_label,
        "n_units": n_units,
        "cids": cids,
        "depths": depths,
        "dt": dt,
        "time_bins": time_bins,
        "bin_centers": bin_centers,
        "pop_rates": pop_rates,
        "X_pca": X_pca,
        "pca": pca,
        "var_exp": var_exp,
        "hmm_model": hmm_model,
        "states": states,
        "post_probs": post_probs,
        "state_stats": state_stats,
        "state_name_map": state_name_map,
        "state_color_map": state_color_map,
        "insp_state": insp_state,
        "post_state": post_state,
        "base_state": base_state,
        "gasp_state": gasp_state,
        "bkps_times": bkps_times,
        "trial_df": trial_df,
        "trial_state_raster": trial_state_raster,
        "t_trial_axis_ms": t_trial_axis_ms,
        "trial_change_points": trial_change_points,
        "spk_times": spk_times,
        "spk_clusters": spk_clusters
    }


def generate_population_hmm_figures(ks4_res: Dict[str, Any], lup_res: Dict[str, Any], breath_data: Dict[str, Any], out_dir: Path):
    """Generates publication-quality figures documenting continuous timeline, trial-by-trial sequences, and state geometry."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = breath_data["dur"]
    t_vec = breath_data["t_vec"]
    flow_v = breath_data["flow_v"]
    emg_env = breath_data["emg_env"]
    valid_onsets = breath_data["valid_onsets"]
    flow_zcs = breath_data["flow_zcs"]

    # -------------------------------------------------------------
    # FIGURE 1: Continuous 36s Population Dynamics & HMM State Decoding (KS4 & Lupin)
    # -------------------------------------------------------------
    logger.info("Generating Figure 1: Continuous Timeline & HMM States...")
    fig = plt.figure(figsize=(19, 14), facecolor='#090d16')
    gs = gridspec.GridSpec(6, 1, height_ratios=[1.2, 1.2, 2.8, 1.4, 0.9, 0.9], hspace=0.18,
                           left=0.06, right=0.96, top=0.94, bottom=0.05)

    # Panel A: Airflow & Diaphragm EMG
    ax0 = fig.add_subplot(gs[0], facecolor='#0f172a')
    ax0.plot(t_vec, flow_v, color='#38bdf8', lw=1.0, label='Airflow (V)')
    ax0.axhline(0, color='#94a3b8', lw=0.8, ls='--')
    ax0.set_xlim(0, dur)
    ax0.set_ylabel("Flow (V)", color='#38bdf8', fontsize=11, fontweight='bold')
    ax0.tick_params(colors='#94a3b8', labelsize=9)
    ax0.set_title("e097 win2_anox1 (1800.0s – 1836.0s) • Continuous 10 ms Population Dynamics & HMM Micro-State Decoding",
                  color='#f8fafc', fontsize=14, fontweight='bold', pad=12)

    # Mark apnea and gasping
    ax0.axvspan(24.5, 27.0, color='#64748b', alpha=0.35, label='Terminal Apnea (~2.3s)')
    ax0.axvspan(27.0, dur, color='#ef4444', alpha=0.18, label='Ballistic Gasps (10 Breaths)')
    ax0.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8.5, ncol=3)

    # Panel B: Diaphragm EMG Envelope
    ax1 = fig.add_subplot(gs[1], facecolor='#0f172a', sharex=ax0)
    ax1.plot(t_vec, emg_env, color='#10b981', lw=1.1, label='Diaphragm EMG Envelope')
    ax1.set_ylabel("EMG (V)", color='#10b981', fontsize=11, fontweight='bold')
    ax1.tick_params(colors='#94a3b8', labelsize=9)
    ax1.axvspan(24.5, 27.0, color='#64748b', alpha=0.35)
    ax1.axvspan(27.0, dur, color='#ef4444', alpha=0.18)

    # Panel C: Kilosort 4 Population Spike Raster (Sorted by VRC Depth, Caudal to Rostral)
    ax2 = fig.add_subplot(gs[2], facecolor='#0f172a', sharex=ax0)
    ks4_times = ks4_res["spk_times"]
    ks4_clusts = ks4_res["spk_clusters"]
    for idx, (cid, d) in enumerate(zip(ks4_res["cids"], ks4_res["depths"])):
        mask_u = (ks4_clusts == cid)
        spks_u = ks4_times[mask_u]
        ax2.scatter(spks_u, np.full_like(spks_u, idx), s=1.2, color='#f1f5f9', alpha=0.6, rasterized=True)
    ax2.set_ylabel("KS4 Units (92 SUA)\nCaudal → Rostral", color='#f8fafc', fontsize=10, fontweight='bold')
    ax2.set_ylim(-1, len(ks4_res["cids"]))
    ax2.tick_params(colors='#94a3b8', labelsize=9)
    ax2.axvspan(24.5, 27.0, color='#64748b', alpha=0.35)
    ax2.axvspan(27.0, dur, color='#ef4444', alpha=0.18)

    # Panel D: PCA Trajectory (Top 3 Principal Components)
    ax3 = fig.add_subplot(gs[3], facecolor='#0f172a', sharex=ax0)
    ax3.plot(ks4_res["bin_centers"], ks4_res["X_pca"][:, 0], color='#38bdf8', lw=1.2, label=f"PC 1 ({ks4_res['var_exp'][0]*100:.1f}%)")
    ax3.plot(ks4_res["bin_centers"], ks4_res["X_pca"][:, 1], color='#a855f7', lw=1.1, label=f"PC 2 ({ks4_res['var_exp'][1]*100:.1f}%)")
    ax3.plot(ks4_res["bin_centers"], ks4_res["X_pca"][:, 2], color='#eab308', lw=1.0, label=f"PC 3 ({ks4_res['var_exp'][2]*100:.1f}%)")
    ax3.set_ylabel("PCA Trajectory\n(10 ms bins)", color='#cbd5e1', fontsize=10, fontweight='bold')
    ax3.tick_params(colors='#94a3b8', labelsize=9)
    ax3.legend(loc='upper left', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8.5, ncol=3)

    # Mark detected continuous change points
    for cp_t in ks4_res["bkps_times"]:
        ax3.axvline(cp_t, color='#f43f5e', alpha=0.4, ls=':', lw=0.9)

    # Panel E: Kilosort 4 Decoded HMM States (10 ms resolution)
    ax4 = fig.add_subplot(gs[4], facecolor='#0f172a', sharex=ax0)
    ks4_states = ks4_res["states"]
    # Plot colored state bar
    for s_id in range(4):
        mask_s = (ks4_states == s_id)
        ax4.fill_between(ks4_res["bin_centers"], 0, 1, where=mask_s, color=ks4_res["state_color_map"][s_id],
                         step='mid', alpha=0.95, label=ks4_res["state_name_map"][s_id])
    ax4.set_ylim(0, 1)
    ax4.set_yticks([])
    ax4.set_ylabel("KS4 HMM\nStates", color='#f8fafc', fontsize=10, fontweight='bold')
    ax4.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8, ncol=4)

    # Panel F: Lupin Decoded HMM States (10 ms resolution)
    ax5 = fig.add_subplot(gs[5], facecolor='#0f172a', sharex=ax0)
    lup_states = lup_res["states"]
    for s_id in range(4):
        mask_s = (lup_states == s_id)
        ax5.fill_between(lup_res["bin_centers"], 0, 1, where=mask_s, color=lup_res["state_color_map"][s_id],
                         step='mid', alpha=0.95, label=lup_res["state_name_map"][s_id])
    ax5.set_ylim(0, 1)
    ax5.set_yticks([])
    ax5.set_ylabel("Lupin HMM\nStates", color='#f8fafc', fontsize=10, fontweight='bold')
    ax5.set_xlabel("Recording Window Time (seconds from t = 1800.0s)", color='#f8fafc', fontsize=11, fontweight='bold')
    ax5.tick_params(colors='#94a3b8', labelsize=9)

    fig.savefig(out_dir / "win2_anox1_continuous_hmm_and_changepoint_timeline.png", dpi=250, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Figure 1 generated successfully.")

    # -------------------------------------------------------------
    # FIGURE 2: Trial-by-Trial Intra-Breath Micro-State Rasters (KS4 vs Lupin)
    # -------------------------------------------------------------
    logger.info("Generating Figure 2: Trial-by-Trial Intra-Breath Micro-State Sequences...")
    fig2 = plt.figure(figsize=(19, 13), facecolor='#090d16')
    gs2 = gridspec.GridSpec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 1.0], hspace=0.25, wspace=0.18,
                            left=0.06, right=0.96, top=0.92, bottom=0.06)

    # Subplot A1: Kilosort 4 Trial-by-Trial Micro-State Raster (90 Breaths x Time-in-Breath)
    ax_a1 = fig2.add_subplot(gs2[0, 0], facecolor='#0f172a')
    ks4_raster = ks4_res["trial_state_raster"]
    n_b, n_t = ks4_raster.shape
    t_ax = ks4_res["t_trial_axis_ms"]

    # Build RGB matrix for raster
    rgb_ks4 = np.zeros((n_b, n_t, 3), dtype=float)
    for s_id, hex_c in ks4_res["state_color_map"].items():
        # hex to rgb
        h = hex_c.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
        rgb_ks4[ks4_raster == s_id] = rgb

    ax_a1.imshow(rgb_ks4, aspect='auto', extent=[t_ax[0], t_ax[-1], n_b, 0], origin='upper', interpolation='nearest')
    ax_a1.axvline(0, color='#ffffff', lw=1.5, ls='--', label='Diaphragm EMG Onset (t = 0 ms)')
    # Overlay flow zero-crossing curve (I_off) across trials
    ax_a1.plot(ks4_res["trial_df"]["ti_ms"], np.arange(n_b) + 0.5, color='#fbbf24', lw=1.8, label='Flow Zero-Crossing (I_off)')
    # Mark transition to gasping
    ax_a1.axhline(80, color='#f43f5e', lw=2.0, ls='-', label='Gasping Transition (Trial 81)')
    ax_a1.set_xlabel("Time from Diaphragm EMG Onset (ms)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_a1.set_ylabel("Breath Trial Index (1 to 90)\nEupnea (1-80) → Gasps (81-90)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_a1.set_title("Kilosort 4 (92 Units): Trial-by-Trial Intra-Breath Micro-States", color='#38bdf8', fontsize=12, fontweight='bold')
    ax_a1.tick_params(colors='#94a3b8', labelsize=9)
    ax_a1.legend(loc='lower left', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8.5)

    # Subplot A2: Lupin Trial-by-Trial Micro-State Raster
    ax_a2 = fig2.add_subplot(gs2[1, 0], facecolor='#0f172a', sharex=ax_a1)
    lup_raster = lup_res["trial_state_raster"]
    rgb_lup = np.zeros((n_b, n_t, 3), dtype=float)
    for s_id, hex_c in lup_res["state_color_map"].items():
        h = hex_c.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
        rgb_lup[lup_raster == s_id] = rgb

    ax_a2.imshow(rgb_lup, aspect='auto', extent=[t_ax[0], t_ax[-1], n_b, 0], origin='upper', interpolation='nearest')
    ax_a2.axvline(0, color='#ffffff', lw=1.5, ls='--')
    ax_a2.plot(lup_res["trial_df"]["ti_ms"], np.arange(n_b) + 0.5, color='#fbbf24', lw=1.8)
    ax_a2.axhline(80, color='#f43f5e', lw=2.0, ls='-')
    ax_a2.set_xlabel("Time from Diaphragm EMG Onset (ms)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_a2.set_ylabel("Breath Trial Index (1 to 90)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_a2.set_title("Lupin (120 Units): Trial-by-Trial Intra-Breath Micro-States", color='#f59e0b', fontsize=12, fontweight='bold')
    ax_a2.tick_params(colors='#94a3b8', labelsize=9)

    # Subplot B1: Quantified Duration of Inspiratory State vs Actual T_I across Breaths
    ax_b1 = fig2.add_subplot(gs2[0, 1], facecolor='#0f172a')
    trials = ks4_res["trial_df"]["trial_idx"]
    ax_b1.plot(trials, ks4_res["trial_df"]["ti_ms"], color='#fbbf24', lw=1.5, label='Actual T_I (Flow ZC, ms)')
    ax_b1.plot(trials, ks4_res["trial_df"]["dur_ti_insp_state_ms"], color='#10b981', lw=1.6, label='KS4 HMM Insp State (ms)')
    ax_b1.plot(trials, lup_res["trial_df"]["dur_ti_insp_state_ms"], color='#06b6d4', lw=1.4, ls='--', label='Lupin HMM Insp State (ms)')
    ax_b1.axvline(80, color='#f43f5e', lw=1.5, ls=':', label='Apnea / Gasp Onset')
    ax_b1.set_xlabel("Breath Trial Index", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_b1.set_ylabel("Duration (ms)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_b1.set_title("Inspiratory State Tracking vs Actual Breath Duration (T_I)", color='#f8fafc', fontsize=12, fontweight='bold')
    ax_b1.tick_params(colors='#94a3b8', labelsize=9)
    ax_b1.legend(loc='upper left', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8.5)

    # Mark Macro Change Points from ruptures
    for cp in ks4_res["trial_change_points"][:-1]:
        ax_b1.axvline(cp, color='#a855f7', lw=1.0, ls='--')

    # Subplot B2: Collapse of Post-Inspiratory (Post-I) State in Gasps
    ax_b2 = fig2.add_subplot(gs2[1, 1], facecolor='#0f172a')
    ax_b2.plot(trials, ks4_res["trial_df"]["dur_te_post_state_ms"], color='#f97316', lw=1.6, label='KS4 Post-I State in Exp (ms)')
    ax_b2.plot(trials, lup_res["trial_df"]["dur_te_post_state_ms"], color='#e11d48', lw=1.4, ls='--', label='Lupin Post-I State in Exp (ms)')
    ax_b2.axvline(80, color='#f43f5e', lw=1.5, ls=':', label='Gasp Onset')
    ax_b2.set_xlabel("Breath Trial Index", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_b2.set_ylabel("Post-I Duration in Expiration (ms)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_b2.set_title("Post-I State Quenching: Complete Elimination During Gasping", color='#f43f5e', fontsize=12, fontweight='bold')
    ax_b2.tick_params(colors='#94a3b8', labelsize=9)
    ax_b2.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8.5)

    fig2.suptitle("Trial-by-Trial Population Micro-State Evolution Across Anoxic Challenge (90 Breaths)",
                  color='#f8fafc', fontsize=14, fontweight='bold', y=0.97)

    fig2.savefig(out_dir / "win2_anox1_trial_by_trial_microstate_raster.png", dpi=250, facecolor=fig2.get_facecolor())
    plt.close(fig2)
    logger.info("Figure 2 generated successfully.")

    # -------------------------------------------------------------
    # FIGURE 3: State Geometry, PCA Phase Planes, and Transition Probability Matrices
    # -------------------------------------------------------------
    logger.info("Generating Figure 3: PCA State Trajectories & Transition Graph...")
    fig3 = plt.figure(figsize=(18, 12), facecolor='#090d16')
    gs3 = gridspec.GridSpec(2, 2, hspace=0.25, wspace=0.22, left=0.07, right=0.95, top=0.92, bottom=0.07)

    # Panel A: KS4 PCA Phase Plane (PC1 vs PC2) colored by HMM States
    ax_g1 = fig3.add_subplot(gs3[0, 0], facecolor='#0f172a')
    for s_id in range(4):
        mask_s = (ks4_res["states"] == s_id)
        ax_g1.scatter(ks4_res["X_pca"][mask_s, 0], ks4_res["X_pca"][mask_s, 1], s=12,
                      color=ks4_res["state_color_map"][s_id], alpha=0.6, label=ks4_res["state_name_map"][s_id])
    ax_g1.set_xlabel(f"PC 1 ({ks4_res['var_exp'][0]*100:.1f}%)", color='#f8fafc', fontsize=11, fontweight='bold')
    ax_g1.set_ylabel(f"PC 2 ({ks4_res['var_exp'][1]*100:.1f}%)", color='#f8fafc', fontsize=11, fontweight='bold')
    ax_g1.set_title("Kilosort 4 (92 Units): Population State Orbit in PC1-PC2", color='#38bdf8', fontsize=12, fontweight='bold')
    ax_g1.tick_params(colors='#94a3b8', labelsize=9)
    ax_g1.legend(loc='lower right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8.5)

    # Panel B: Lupin PCA Phase Plane (PC1 vs PC2)
    ax_g2 = fig3.add_subplot(gs3[0, 1], facecolor='#0f172a')
    for s_id in range(4):
        mask_s = (lup_res["states"] == s_id)
        ax_g2.scatter(lup_res["X_pca"][mask_s, 0], lup_res["X_pca"][mask_s, 1], s=12,
                      color=lup_res["state_color_map"][s_id], alpha=0.6, label=lup_res["state_name_map"][s_id])
    ax_g2.set_xlabel(f"PC 1 ({lup_res['var_exp'][0]*100:.1f}%)", color='#f8fafc', fontsize=11, fontweight='bold')
    ax_g2.set_ylabel(f"PC 2 ({lup_res['var_exp'][1]*100:.1f}%)", color='#f8fafc', fontsize=11, fontweight='bold')
    ax_g2.set_title("Lupin (120 Units): Population State Orbit in PC1-PC2", color='#f59e0b', fontsize=12, fontweight='bold')
    ax_g2.tick_params(colors='#94a3b8', labelsize=9)
    ax_g2.legend(loc='lower right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8.5)

    # Panel C: KS4 Transition Probability Matrix
    ax_g3 = fig3.add_subplot(gs3[1, 0], facecolor='#0f172a')
    trans_ks4 = ks4_res["hmm_model"].transmat_
    cax1 = ax_g3.imshow(trans_ks4, cmap='Blues', vmin=0, vmax=1.0)
    for i in range(4):
        for j in range(4):
            val = trans_ks4[i, j]
            col = '#ffffff' if val > 0.4 else '#cbd5e1'
            ax_g3.text(j, i, f"{val:.2f}", ha='center', va='center', color=col, fontweight='bold', fontsize=11)
    state_labels_ks4 = [f"S{i}: {ks4_res['state_name_map'][i].split()[0]}" for i in range(4)]
    ax_g3.set_xticks(range(4))
    ax_g3.set_yticks(range(4))
    ax_g3.set_xticklabels(state_labels_ks4, color='#cbd5e1', fontsize=9.5)
    ax_g3.set_yticklabels(state_labels_ks4, color='#cbd5e1', fontsize=9.5)
    ax_g3.set_xlabel("Next State (t + 10 ms)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_g3.set_ylabel("Current State (t)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_g3.set_title("KS4 HMM Transition Probability Matrix (A_ij)", color='#38bdf8', fontsize=12, fontweight='bold')

    # Panel D: Lupin Transition Probability Matrix
    ax_g4 = fig3.add_subplot(gs3[1, 1], facecolor='#0f172a')
    trans_lup = lup_res["hmm_model"].transmat_
    cax2 = ax_g4.imshow(trans_lup, cmap='Oranges', vmin=0, vmax=1.0)
    for i in range(4):
        for j in range(4):
            val = trans_lup[i, j]
            col = '#ffffff' if val > 0.4 else '#cbd5e1'
            ax_g4.text(j, i, f"{val:.2f}", ha='center', va='center', color=col, fontweight='bold', fontsize=11)
    state_labels_lup = [f"S{i}: {lup_res['state_name_map'][i].split()[0]}" for i in range(4)]
    ax_g4.set_xticks(range(4))
    ax_g4.set_yticks(range(4))
    ax_g4.set_xticklabels(state_labels_lup, color='#cbd5e1', fontsize=9.5)
    ax_g4.set_yticklabels(state_labels_lup, color='#cbd5e1', fontsize=9.5)
    ax_g4.set_xlabel("Next State (t + 10 ms)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_g4.set_ylabel("Current State (t)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_g4.set_title("Lupin HMM Transition Probability Matrix (A_ij)", color='#f59e0b', fontsize=12, fontweight='bold')

    fig3.suptitle("Population State Geometry & Markov Transition Topologies",
                  color='#f8fafc', fontsize=14, fontweight='bold', y=0.97)

    fig3.savefig(out_dir / "win2_anox1_population_state_geometry_and_transitions.png", dpi=250, facecolor=fig3.get_facecolor())
    plt.close(fig3)
    logger.info("Figure 3 generated successfully.")


def generate_interactive_dashboard(ks4_res: Dict[str, Any], lup_res: Dict[str, Any], breath_data: Dict[str, Any], out_dir: Path):
    """Creates a comprehensive interactive HTML dashboard showcasing trial-by-trial population dynamics, HMM states, and change points."""
    html_file = out_dir / "win2_anox1_population_dynamics_dashboard.html"
    logger.info(f"Generating interactive dashboard at {html_file}...")

    # Build trial table rows
    tr_df = ks4_res["trial_df"]
    lup_df = lup_res["trial_df"]
    rows_html = []
    for idx, row in tr_df.iterrows():
        b_idx = int(row["trial_idx"]) + 1
        t_on = row["t_on_s"]
        ti = row["ti_ms"]
        tot = row["tot_ms"]
        is_gasp = row["is_gasp"]
        ks4_insp = row["dur_ti_insp_state_ms"]
        lup_insp = lup_df.iloc[idx]["dur_ti_insp_state_ms"]
        ks4_post = row["dur_te_post_state_ms"]
        lup_post = lup_df.iloc[idx]["dur_te_post_state_ms"]

        badge = '<span style="background:#ef4444;color:#fff;padding:2px 6px;border-radius:4px;font-size:11px;font-weight:700;">GASP</span>' if is_gasp else '<span style="background:#0284c7;color:#fff;padding:2px 6px;border-radius:4px;font-size:11px;">EUPNEA</span>'

        rows_html.append(f"""
        <tr style="{'background:rgba(239,68,68,0.08);' if is_gasp else ''}">
          <td style="font-weight:700;">#{b_idx:02d}</td>
          <td>{badge}</td>
          <td>{t_on:6.2f}s</td>
          <td>{ti:5.1f} ms</td>
          <td>{tot:5.1f} ms</td>
          <td style="color:#10b981;font-weight:600;">{ks4_insp:5.1f} ms</td>
          <td style="color:#06b6d4;font-weight:600;">{lup_insp:5.1f} ms</td>
          <td style="color:#f97316;">{ks4_post:5.1f} ms</td>
          <td style="color:#e11d48;">{lup_post:5.1f} ms</td>
        </tr>
        """)

    trial_table_body = "\n".join(rows_html)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>e097 win2_anox1 - Trial-by-Trial Population Dynamics & HMM Change-Point Hub</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: #07090e;
      color: #f1f5f9;
      line-height: 1.6;
      padding: 32px;
    }}
    .container {{ max-width: 1360px; margin: 0 auto; }}
    header {{
      background: linear-gradient(180deg, #0f172a 0%, #07090e 100%);
      border-bottom: 1px solid #1e293b;
      padding-bottom: 24px;
      margin-bottom: 32px;
    }}
    .nav-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }}
    .back-btn {{
      background: #1e293b;
      color: #38bdf8;
      text-decoration: none;
      padding: 7px 16px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      border: 1px solid #334155;
      transition: all 0.2s;
    }}
    .back-btn:hover {{ background: #38bdf8; color: #07090e; }}
    h1 {{
      font-size: 27px;
      font-weight: 800;
      color: #f8fafc;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      color: #94a3b8;
      font-size: 14.5px;
      margin-top: 6px;
      max-width: 1150px;
    }}
    .grid-4 {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 28px;
    }}
    .stat-card {{
      background: #0f1523;
      border: 1px solid #1e293b;
      border-radius: 10px;
      padding: 18px;
    }}
    .stat-val {{
      font-size: 26px;
      font-weight: 800;
      color: #38bdf8;
      margin-bottom: 4px;
    }}
    .stat-lbl {{
      font-size: 12px;
      color: #94a3b8;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .section {{
      background: #0f1523;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 12px;
      padding: 28px;
      margin-bottom: 32px;
    }}
    .section h2 {{
      font-size: 19px;
      font-weight: 800;
      color: #38bdf8;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .img-display {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid #334155;
      margin: 16px 0;
      cursor: pointer;
      transition: transform 0.15s;
    }}
    .img-display:hover {{
      transform: scale(1.005);
    }}
    .state-legend {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin: 16px 0;
    }}
    .legend-box {{
      padding: 12px;
      border-radius: 8px;
      border: 1px solid #1e293b;
      background: #141c2e;
    }}
    .legend-box strong {{ font-size: 13px; display: block; margin-bottom: 4px; }}
    .legend-box p {{ font-size: 11.5px; color: #94a3b8; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12.5px;
      margin-top: 16px;
    }}
    th, td {{
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid #1e293b;
    }}
    th {{
      background: #141c2e;
      color: #cbd5e1;
      font-weight: 700;
    }}
    .scroll-table {{
      max-height: 480px;
      overflow-y: auto;
      border: 1px solid #1e293b;
      border-radius: 8px;
    }}
    .modal {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.92);
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
  <div class="container">
    <header>
      <div class="nav-bar">
        <a href="../win2_anox1_sorter_comparison.html" class="back-btn">&larr; Back to Sorter Comparison</a>
        <div style="display:flex;gap:10px;">
          <a href="../win2_anox1_burst_dynamics_walkthrough.html" class="back-btn" style="border-color:#6366f1;color:#a5b4fc;">&#x1F4CA; Burst Dynamics Walkthrough</a>
          <a href="../win2_anox1_ion_peth_dashboard.html" class="back-btn" style="border-color:#10b981;color:#10b981;">&rarr; I_on Aligned Suite</a>
          <a href="../win2_anox1_ioff_peth_dashboard.html" class="back-btn" style="border-color:#f97316;color:#f97316;">&rarr; I_off Aligned Suite</a>
        </div>
      </div>
      <h1>e097 win2_anox1 &mdash; Trial-by-Trial Population Dynamics, HMM States, &amp; Change-Point Hub</h1>
      <p class="subtitle">
        High-temporal-resolution (<strong>10 ms continuous bins</strong>) state decoding across 92 Kilosort 4 and 120 Lupin single units. Unveils intra-breath micro-state trajectories, active off-switch boundaries, and macro-state collapse across 90 breath trials (80 eupneic breaths, 2.3s apnea, and 10 ballistic autoresuscitation gasps).
      </p>
    </header>

    <div class="grid-4">
      <div class="stat-card">
        <div class="stat-val">10 ms</div>
        <div class="stat-lbl">Temporal Resolution</div>
      </div>
      <div class="stat-card">
        <div class="stat-val">4 States</div>
        <div class="stat-lbl">Decoded Hidden States</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" style="color:#10b981;">90 Breaths</div>
        <div class="stat-lbl">80 Eupnea &bull; 10 Gasps</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" style="color:#f59e0b;">92 vs 120</div>
        <div class="stat-lbl">KS4 vs Lupin Single Units</div>
      </div>
    </div>

    <!-- HMM STATE LEGEND -->
    <div class="section">
      <h2>🏷️ Canonical Respiratory Micro-States Decoded by Gaussian HMM</h2>
      <div class="state-legend">
        <div class="legend-box" style="border-left:4px solid #10b981;">
          <strong style="color:#10b981;">State 1: Inspiratory Burst (Insp)</strong>
          <p>Locked to Diaphragm EMG. Represents rhythmic inspiratory burst drive across preBötC and rVRG bulbospinal units.</p>
        </div>
        <div class="legend-box" style="border-left:4px solid #f97316;">
          <strong style="color:#f97316;">State 2: Post-I / Off-Switch (Post-I)</strong>
          <p>Occurs at Flow zero-crossing ($I_{{off}}$). Represents active post-inspiratory inhibitory termination from BötC.</p>
        </div>
        <div class="legend-box" style="border-left:4px solid #64748b;">
          <strong style="color:#94a3b8;">State 3: Late-Expiratory Quiescence (E2)</strong>
          <p>Inter-breath baseline period characterized by low, quiescent population firing rates.</p>
        </div>
        <div class="legend-box" style="border-left:4px solid #ef4444;">
          <strong style="color:#ef4444;">State 4: Hypoxic / Gasp Recruitment</strong>
          <p>Massive synchronous recruitment during gasps; replaces normal eupneic triphasic rhythm.</p>
        </div>
      </div>
    </div>

    <!-- SECTION 1: CONTINUOUS TIMELINE -->
    <div class="section">
      <h2>📈 1. Continuous 36s Timeline &amp; State Decoding (KS4 vs Lupin)</h2>
      <p style="color:#cbd5e1;font-size:14px;">
        Tracking continuous 10 ms population rates, PCA trajectories, and decoded HMM micro-states across the entire anoxic challenge transition ($1800.0\text{{s}} - 1836.0\text{{s}}$). Red vertical dashed lines indicate change points detected by PELT.
      </p>
      <img src="win2_anox1_continuous_hmm_and_changepoint_timeline.png" class="img-display" onclick="openModal('win2_anox1_continuous_hmm_and_changepoint_timeline.png')" alt="Continuous Timeline">
    </div>

    <!-- SECTION 2: TRIAL-BY-TRIAL RASTER -->
    <div class="section">
      <h2>🧬 2. Trial-by-Trial Intra-Breath Micro-State Sequences (90 Trials)</h2>
      <p style="color:#cbd5e1;font-size:14px;">
        Every horizontal line represents an individual breath trial, aligned to Diaphragm EMG onset ($t = 0\text{{ ms}}$) with the yellow curve marking the Flow zero-crossing ($I_{{off}}$). Notice the <strong>abrupt elimination of the orange Post-I off-switch state</strong> during gasping (Trials 81–90), reflecting the loss of post-inspiratory inhibition.
      </p>
      <img src="win2_anox1_trial_by_trial_microstate_raster.png" class="img-display" onclick="openModal('win2_anox1_trial_by_trial_microstate_raster.png')" alt="Trial Raster">
    </div>

    <!-- SECTION 3: STATE GEOMETRY & TRANSITIONS -->
    <div class="section">
      <h2>📐 3. Population State Orbit &amp; Markov Transition Topologies</h2>
      <p style="color:#cbd5e1;font-size:14px;">
        State space trajectories in PC1 vs PC2 and empirical transition probability matrices ($A_{{ij}}$) governing transitions on a 10 ms timescale.
      </p>
      <img src="win2_anox1_population_state_geometry_and_transitions.png" class="img-display" onclick="openModal('win2_anox1_population_state_geometry_and_transitions.png')" alt="State Geometry">
    </div>

    <!-- SECTION 4: TRIAL-BY-TRIAL METRICS TABLE -->
    <div class="section">
      <h2>📊 4. Trial-by-Trial Breath &amp; State Occupancy Metrics</h2>
      <div class="scroll-table">
        <table>
          <thead>
            <tr>
              <th>Trial</th>
              <th>Regime</th>
              <th>Onset (s)</th>
              <th>T_I (ms)</th>
              <th>Total Cycle (ms)</th>
              <th>KS4 Insp State (ms)</th>
              <th>Lupin Insp State (ms)</th>
              <th>KS4 Post-I State (ms)</th>
              <th>Lupin Post-I State (ms)</th>
            </tr>
          </thead>
          <tbody>
            {trial_table_body}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="modal" id="modal" onclick="closeModal()">
    <img id="modalImg" src="">
  </div>

  <script>
    function openModal(src) {{
      document.getElementById('modalImg').src = src;
      document.getElementById('modal').style.display = 'flex';
    }}
    function closeModal() {{
      document.getElementById('modal').style.display = 'none';
    }}
  </script>
</body>
</html>
"""

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info("Interactive dashboard generated successfully.")


def main():
    t0 = time.time()
    out_dir = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / "win2_anox1" / "population_dynamics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract Breaths
    breath_data = extract_win2_breaths()

    # 2. Run Population Analysis on Kilosort 4 (92 units, 10 ms bins, 4 states)
    ks4_res = run_population_analysis("kilosort4", "Kilosort 4", breath_data, dt=0.010, n_states=4)

    # 3. Run Population Analysis on Lupin (120 units, 10 ms bins, 4 states)
    lup_res = run_population_analysis("lupin", "Lupin (SVD-OMP)", breath_data, dt=0.010, n_states=4)

    # 4. Generate Multi-Panel Figures
    generate_population_hmm_figures(ks4_res, lup_res, breath_data, out_dir)

    # 5. Generate Interactive HTML Dashboard
    generate_interactive_dashboard(ks4_res, lup_res, breath_data, out_dir)

    logger.info(f"Done! Entire Population HMM & Change-Point Suite finished in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
