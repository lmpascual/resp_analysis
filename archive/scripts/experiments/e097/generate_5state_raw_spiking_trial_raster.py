"""Generates the 5-State Raw Spiking Trial-by-Trial HMM Micro-State Raster
for e097 win2_anox1 (Anoxic Challenge 1, 1800.0s - 1836.0s).

Uses:
- Raw unreduced 10 ms integer spike counts (no PCA).
- 5-State Poisson HMM.
- Both Kilosort 4 (92 units) and Lupin (120 units).
- Evaluates intra-breath micro-state sequences across all 90 trials.
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from hmmlearn import hmm

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
logger = logging.getLogger("RAW_5STATE_RASTER")

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
        "flow_v": flow_v,
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


def fit_5state_raw_poisson_hmm(sorter_key: str, sorter_label: str, breath_data: Dict[str, Any], dt: float = 0.010):
    """Fits a 5-state Poisson HMM on raw 10 ms spike counts and decodes trial-by-trial sequences."""
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

    logger.info(f"[{sorter_label}] Binning {n_units} units into {n_time_bins} 10ms bins (raw counts)...")
    counts = np.zeros((n_time_bins, n_units), dtype=np.int32)
    for u_idx, cid in enumerate(cids):
        c_spk = spk_times[spk_clusters == cid]
        counts[:, u_idx] = np.histogram(c_spk, bins=time_bins)[0][:n_time_bins]

    # Train 5-state Poisson HMM
    logger.info(f"[{sorter_label}] Fitting 5-State Poisson HMM on raw count matrix ({counts.shape})...")
    model = hmm.PoissonHMM(n_components=5, n_iter=120, random_state=42, tol=1e-3)
    model.fit(counts)
    raw_states = model.predict(counts)
    post_probs = model.predict_proba(counts)

    # Correlate states with respiratory phase landmarks
    valid_onsets = breath_data["valid_onsets"]
    flow_zcs = breath_data["flow_zcs"]
    valid_offsets = breath_data["valid_offsets"]
    n_breaths = breath_data["n_breaths"]
    gasp_mask = breath_data["gasp_mask"]

    phase_labels = np.zeros(n_time_bins, dtype=int)
    for b_idx in range(n_breaths):
        phase_labels[(bin_centers >= valid_onsets[b_idx]) & (bin_centers < flow_zcs[b_idx])] = 1  # Insp
        phase_labels[(bin_centers >= flow_zcs[b_idx]) & (bin_centers < valid_offsets[b_idx])] = 2  # Post-I

    state_stats = []
    for s in range(5):
        mask_s = (raw_states == s)
        n_occ = np.sum(mask_s)
        p_insp = np.mean(phase_labels[mask_s] == 1) * 100 if n_occ > 0 else 0
        p_post = np.mean(phase_labels[mask_s] == 2) * 100 if n_occ > 0 else 0
        p_exp = np.mean(phase_labels[mask_s] == 0) * 100 if n_occ > 0 else 0
        p_gasp = np.mean(bin_centers[mask_s] >= 27.0) * 100 if n_occ > 0 else 0
        mean_rate = np.mean(counts[mask_s, :]) / dt if n_occ > 0 else 0
        state_stats.append({
            "raw_id": s,
            "n_bins": int(n_occ),
            "pct_insp": float(p_insp),
            "pct_post": float(p_post),
            "pct_exp": float(p_exp),
            "pct_gasp": float(p_gasp),
            "mean_rate": float(mean_rate)
        })

    # Sort and establish canonical functional mapping:
    # 1. Primary Inspiratory Burst State (highest pct_insp)
    s_insp = sorted(state_stats, key=lambda x: x["pct_insp"], reverse=True)[0]["raw_id"]
    
    # 2. Gasp/Apneic Recruitment State (highest pct_gasp among remaining)
    rem1 = [x for x in state_stats if x["raw_id"] != s_insp]
    s_gasp = sorted(rem1, key=lambda x: x["pct_gasp"], reverse=True)[0]["raw_id"]

    # 3. Off-Switch / Peak Transition State (highest pct_insp + pct_post among remaining)
    rem2 = [x for x in rem1 if x["raw_id"] != s_gasp]
    s_offswitch = sorted(rem2, key=lambda x: (x["pct_insp"] + x["pct_post"]*2), reverse=True)[0]["raw_id"]

    # 4. Eupneic Expiratory Phase (remaining state with GaspOcc == 0)
    rem3 = [x for x in rem2 if x["raw_id"] != s_offswitch]
    s_exp = sorted(rem3, key=lambda x: x["mean_rate"], reverse=True)[0]["raw_id"]

    # 5. Late-Expiratory Baseline / Quiescence (lowest rate remainder)
    rem4 = [x for x in rem3 if x["raw_id"] != s_exp]
    s_base = rem4[0]["raw_id"]

    # Canonical order: 0=Insp, 1=OffSwitch/Post-I, 2=Eupnea Exp, 3=Late-E Quiescence, 4=Gasp/Apnea
    canonical_map = {
        s_insp: 0,
        s_offswitch: 1,
        s_exp: 2,
        s_base: 3,
        s_gasp: 4
    }

    # Map raw states to canonical states
    canonical_states = np.array([canonical_map[s] for s in raw_states])

    state_names = [
        "State 1: Inspiratory Burst (Insp)",
        "State 2: Peak Transition / Off-Switch",
        "State 3: Eupneic Expiratory Drive",
        "State 4: Late-Expiratory Quiescence (E2)",
        "State 5: Apneic & Gasping Regime"
    ]

    state_colors = [
        '#10b981',  # 0: Emerald Green
        '#f97316',  # 1: Bright Orange
        '#38bdf8',  # 2: Sky Blue
        '#64748b',  # 3: Slate Gray
        '#ef4444'   # 4: Crimson Red
    ]

    logger.info(f"[{sorter_label}] Canonical 5-State Mapping:")
    for raw_id, can_id in canonical_map.items():
        st = next(x for x in state_stats if x["raw_id"] == raw_id)
        logger.info(f"  Raw State {raw_id} -> Canonical {can_id} ({state_names[can_id]}): Insp={st['pct_insp']:.1f}%, Post={st['pct_post']:.1f}%, GaspOcc={st['pct_gasp']:.1f}%, Rate={st['mean_rate']:.1f}Hz")

    # 4. Extract Trial-by-Trial Micro-State Raster across all 90 Breaths
    win_pre_ms = 50.0
    win_post_ms = 350.0
    n_trial_bins = int((win_pre_ms + win_post_ms) / (dt * 1000.0))
    t_trial_axis_ms = np.arange(-win_pre_ms, win_post_ms, dt * 1000.0)

    trial_raster = []
    trial_records = []

    for b_idx in range(n_breaths):
        t_on = valid_onsets[b_idx]
        t_zc = flow_zcs[b_idx]
        ti = breath_data["ti_arr"][b_idx]
        tot = breath_data["tot_arr"][b_idx]
        is_gasp = gasp_mask[b_idx]

        # Slice in continuous bin grid
        t_start = t_on - (win_pre_ms / 1000.0)
        t_end = t_on + (win_post_ms / 1000.0)
        mask_tr = (bin_centers >= t_start) & (bin_centers < t_end)
        s_slice = canonical_states[mask_tr]

        if len(s_slice) < n_trial_bins:
            pad_val = s_slice[-1] if len(s_slice) > 0 else 3
            s_fixed = np.pad(s_slice, (0, n_trial_bins - len(s_slice)), constant_values=pad_val)
        else:
            s_fixed = s_slice[:n_trial_bins]
        trial_raster.append(s_fixed)

        # Durations of states
        mask_ti = (bin_centers >= t_on) & (bin_centers < t_zc)
        s_ti = canonical_states[mask_ti]
        dur_insp_ms = np.sum(s_ti == 0) * (dt * 1000.0) if len(s_ti) > 0 else 0.0
        dur_off_ms = np.sum(s_ti == 1) * (dt * 1000.0) if len(s_ti) > 0 else 0.0

        trial_records.append({
            "trial_idx": b_idx,
            "t_on_s": float(t_on),
            "ti_ms": float(ti),
            "tot_ms": float(tot),
            "is_gasp": bool(is_gasp),
            "dur_insp_ms": float(dur_insp_ms),
            "dur_off_ms": float(dur_off_ms)
        })

    trial_raster = np.array(trial_raster)
    trial_df = pd.DataFrame(trial_records)

    # Re-order transition matrix to canonical indices
    trans_raw = model.transmat_
    trans_can = np.zeros((5, 5), dtype=float)
    for r_i, c_i in canonical_map.items():
        for r_j, c_j in canonical_map.items():
            trans_can[c_i, c_j] = trans_raw[r_i, r_j]

    return {
        "sorter_key": sorter_key,
        "sorter_label": sorter_label,
        "n_units": n_units,
        "model": model,
        "canonical_states": canonical_states,
        "state_names": state_names,
        "state_colors": state_colors,
        "trial_raster": trial_raster,
        "trial_df": trial_df,
        "t_trial_axis_ms": t_trial_axis_ms,
        "trans_can": trans_can
    }


def generate_5state_microstate_raster_figure(ks4_res: Dict[str, Any], lup_res: Dict[str, Any], breath_data: Dict[str, Any], out_dir: Path):
    """Generates the publication-quality 5-State Trial-by-Trial HMM Micro-State Raster Figure."""
    fig = plt.figure(figsize=(20, 13), facecolor='#090d16')
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 1.0], hspace=0.25, wspace=0.18,
                           left=0.06, right=0.96, top=0.92, bottom=0.06)

    n_b, n_t = ks4_res["trial_raster"].shape
    t_ax = ks4_res["t_trial_axis_ms"]

    # --- Subplot A1: Kilosort 4 (92 Units) 5-State Trial-by-Trial Raster ---
    ax_a1 = fig.add_subplot(gs[0, 0], facecolor='#0f172a')
    rgb_ks4 = np.zeros((n_b, n_t, 3), dtype=float)
    for s_id, hex_c in enumerate(ks4_res["state_colors"]):
        h = hex_c.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
        rgb_ks4[ks4_res["trial_raster"] == s_id] = rgb

    ax_a1.imshow(rgb_ks4, aspect='auto', extent=[t_ax[0], t_ax[-1], n_b, 0], origin='upper', interpolation='nearest')
    ax_a1.axvline(0, color='#ffffff', lw=1.5, ls='--', label='Diaphragm EMG Onset (t = 0 ms)')
    ax_a1.plot(ks4_res["trial_df"]["ti_ms"], np.arange(n_b) + 0.5, color='#fbbf24', lw=1.8, label='Flow Zero-Crossing (I_off)')
    ax_a1.axhline(80, color='#f43f5e', lw=2.0, ls='-', label='Gasping Transition (Trial 81)')
    ax_a1.set_xlabel("Time from Diaphragm EMG Onset (ms)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_a1.set_ylabel("Breath Trial Index (1 to 90)\nEupnea (1-80) → Gasps (81-90)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_a1.set_title("Kilosort 4 (92 Units): 5-State Trial-by-Trial Poisson HMM Transitions (Raw Spikes)", color='#38bdf8', fontsize=12, fontweight='bold')
    ax_a1.tick_params(colors='#94a3b8', labelsize=9)
    ax_a1.legend(loc='lower left', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8.5)

    # --- Subplot A2: Lupin (120 Units) 5-State Trial-by-Trial Raster ---
    ax_a2 = fig2 = fig.add_subplot(gs[1, 0], facecolor='#0f172a', sharex=ax_a1)
    rgb_lup = np.zeros((n_b, n_t, 3), dtype=float)
    for s_id, hex_c in enumerate(lup_res["state_colors"]):
        h = hex_c.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
        rgb_lup[lup_res["trial_raster"] == s_id] = rgb

    ax_a2.imshow(rgb_lup, aspect='auto', extent=[t_ax[0], t_ax[-1], n_b, 0], origin='upper', interpolation='nearest')
    ax_a2.axvline(0, color='#ffffff', lw=1.5, ls='--')
    ax_a2.plot(lup_res["trial_df"]["ti_ms"], np.arange(n_b) + 0.5, color='#fbbf24', lw=1.8)
    ax_a2.axhline(80, color='#f43f5e', lw=2.0, ls='-')
    ax_a2.set_xlabel("Time from Diaphragm EMG Onset (ms)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_a2.set_ylabel("Breath Trial Index (1 to 90)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_a2.set_title("Lupin (120 Units): 5-State Trial-by-Trial Poisson HMM Transitions (Raw Spikes)", color='#f59e0b', fontsize=12, fontweight='bold')
    ax_a2.tick_params(colors='#94a3b8', labelsize=9)

    # --- Subplot B1: Quantified Duration of Inspiratory State vs Actual T_I across Breaths ---
    ax_b1 = fig.add_subplot(gs[0, 1], facecolor='#0f172a')
    trials = ks4_res["trial_df"]["trial_idx"]
    ax_b1.plot(trials, ks4_res["trial_df"]["ti_ms"], color='#fbbf24', lw=1.6, label='Actual T_I (Flow ZC, ms)')
    ax_b1.plot(trials, ks4_res["trial_df"]["dur_insp_ms"], color='#10b981', lw=1.6, label='KS4 State 1 (Insp Burst, ms)')
    ax_b1.plot(trials, lup_res["trial_df"]["dur_insp_ms"], color='#06b6d4', lw=1.4, ls='--', label='Lupin State 1 (Insp Burst, ms)')
    ax_b1.axvline(80, color='#f43f5e', lw=1.5, ls=':', label='Apnea / Gasp Onset')
    ax_b1.set_xlabel("Breath Trial Index", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_b1.set_ylabel("Duration (ms)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_b1.set_title("State 1 (Inspiratory Burst) Tracking vs Breath Duration (T_I)", color='#f8fafc', fontsize=12, fontweight='bold')
    ax_b1.tick_params(colors='#94a3b8', labelsize=9)
    ax_b1.legend(loc='upper left', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8.5)

    # --- Subplot B2: Transition Probability Matrix across the 5 States (KS4) ---
    ax_b2 = fig.add_subplot(gs[1, 1], facecolor='#0f172a')
    trans_can = ks4_res["trans_can"]
    im_trans = ax_b2.imshow(trans_can, cmap='Blues', vmin=0, vmax=1.0)
    for i in range(5):
        for j in range(5):
            val = trans_can[i, j]
            col = '#ffffff' if val > 0.4 else '#cbd5e1'
            ax_b2.text(j, i, f"{val:.2f}", ha='center', va='center', color=col, fontweight='bold', fontsize=10)
    labels = ["S1:Insp", "S2:OffSw", "S3:Exp", "S4:E2", "S5:Gasp"]
    ax_b2.set_xticks(range(5))
    ax_b2.set_yticks(range(5))
    ax_b2.set_xticklabels(labels, color='#cbd5e1', fontsize=9.5)
    ax_b2.set_yticklabels(labels, color='#cbd5e1', fontsize=9.5)
    ax_b2.set_xlabel("Next State (t + 10 ms)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_b2.set_ylabel("Current State (t)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_b2.set_title("5-State Markov Transition Probability Matrix (A_ij, 10 ms step)", color='#38bdf8', fontsize=12, fontweight='bold')

    fig.suptitle("e097 win2_anox1: 5-State Trial-by-Trial Population HMM Transitions (Raw Spikes, No PCA)",
                  color='#f8fafc', fontsize=14, fontweight='bold', y=0.97)

    fig_path = out_dir / "win2_anox1_trial_by_trial_microstate_raster.png"
    tmp_path = out_dir / "tmp_trial_raster.png"
    fig.savefig(str(tmp_path), dpi=250, facecolor=fig.get_facecolor())
    plt.close(fig)
    if fig_path.exists():
        try:
            fig_path.unlink()
        except Exception:
            pass
    import shutil
    shutil.move(str(tmp_path), str(fig_path))
    logger.info(f"Successfully generated and overwritten: {fig_path}")


def main():
    t0 = time.time()
    out_dir = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / "win2_anox1" / "population_dynamics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract Breaths
    breath_data = extract_win2_breaths()

    # 2. Fit 5-State Poisson HMM on Raw Spikes for Kilosort 4 (92 units)
    ks4_res = fit_5state_raw_poisson_hmm("kilosort4", "Kilosort 4", breath_data, dt=0.010)

    # 3. Fit 5-State Poisson HMM on Raw Spikes for Lupin (120 units)
    lup_res = fit_5state_raw_poisson_hmm("lupin", "Lupin (SVD-OMP)", breath_data, dt=0.010)

    # 4. Generate Figure win2_anox1_trial_by_trial_microstate_raster.png
    generate_5state_microstate_raster_figure(ks4_res, lup_res, breath_data, out_dir)

    logger.info(f"Complete! Finished in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
