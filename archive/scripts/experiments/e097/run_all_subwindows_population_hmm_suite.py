"""Comprehensive Population Dynamics & Micro-State HMM Analysis Suite
across all 7 subwindows of e097:
1. win1_baseline (1740s - 1800s, 60s) - Baseline Eupnea
2. win2_anox1 (1800s - 1836s, 36s) - Anoxic Challenge 1 (Gas Switch)
3. win3_anox2 (1836s - 1860s, 24s) - Anoxic Challenge 2 (Apnea / HR Nadir)
4. win4_gasp_early (1860s - 1950s, 90s) - Early Gasping (Autoresuscitation Surge)
5. win5_gasp_late (1950s - 2070s, 120s) - Late Gasping (Tachypneic Transition)
6. win6_flow_shift (2070s - 2136s, 66s) - Active Expiration Shift
7. win7_recovery (2136s - 2160s, 24s) - Post-Challenge Recovery

For each subwindow:
- Extracts analog NIDQ signals and segments every breath cycle (I_on, I_off).
- Fits 5-state Poisson HMM on raw 10 ms uncompressed spike counts (no PCA)
  for both Kilosort 4 and Lupin (SVD-OMP).
- Maps raw states to canonical 5 states.
- Generates high-res static trial-by-trial micro-state rasters and continuous timelines.
- Generates zoomable interactive Plotly micro-state rasters (aligned to I_off and I_on).
- Generates dedicated subwindow population dynamics dashboards.
- Updates subwindow benchmark comparison pages.
- Builds an all-in-one Master 7-Subwindow Interactive Raster Hub.
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
logger = logging.getLogger("ALL_SUBWINS_POP_HMM")

SUBWINDOWS = [
    {
        "id": "win1_baseline",
        "name": "Baseline (Pre-Anoxia)",
        "t_start": 1740.0,
        "t_end": 1800.0,
        "duration": 60.0,
        "desc": "Pure normoxic baseline eupnea with stationary limit-cycle dynamics."
    },
    {
        "id": "win2_anox1",
        "name": "Anoxic Challenge 1 (Gas Switch)",
        "t_start": 1800.0,
        "t_end": 1836.0,
        "duration": 36.0,
        "desc": "Pre-hypoxic eupnea followed by rate deceleration, terminal apnea, and initial gasps."
    },
    {
        "id": "win3_anox2",
        "name": "Anoxic Challenge 2 (Apnea / HR Nadir)",
        "t_start": 1836.0,
        "t_end": 1860.0,
        "duration": 24.0,
        "desc": "Profound respiratory depression, long inter-breath pauses, and hypoxic heart rate nadir."
    },
    {
        "id": "win4_gasp_early",
        "name": "Early Gasping (Autoresuscitation Surge)",
        "t_start": 1860.0,
        "t_end": 1950.0,
        "duration": 90.0,
        "desc": "Ballistic autoresuscitation gasps with synchronized high-amplitude inspiratory bursts."
    },
    {
        "id": "win5_gasp_late",
        "name": "Late Gasping (Tachypneic Transition)",
        "t_start": 1950.0,
        "t_end": 2070.0,
        "duration": 120.0,
        "desc": "Rapidly accelerating breathing frequency transitioning from gasps to tachypnea."
    },
    {
        "id": "win6_flow_shift",
        "name": "Active Expiration Shift",
        "t_start": 2070.0,
        "t_end": 2136.0,
        "duration": 66.0,
        "desc": "Emergence of active expiratory drive and dynamic flow profile reconfiguration."
    },
    {
        "id": "win7_recovery",
        "name": "Post-Challenge Recovery",
        "t_start": 2136.0,
        "t_end": 2160.0,
        "duration": 24.0,
        "desc": "Stabilization of eupneic rhythmicity following hypoxic challenge."
    }
]

# Canonical 5-State Consolidated Respiratory Micro-State Taxonomy
# 0: State 1: Inspiratory Drive (Insp Burst) - Green #10b981
# 1: State 2: Peak / Off-Switch Quench (I_off) - Orange #f97316
# 2: State 3: Post-Insp / Expiratory Drive (Exp) - Sky Blue #38bdf8
# 3: State 4: Pre-Insp Priming (Pre-I) - Purple #a855f7
# 4: State 5: Inter-Breath Quiescence / Silence (Pause) - Slate Gray #64748b

CONSOLIDATED_MAPPINGS = {
    # win1_baseline
    "win1_baseline_kilosort4":   {0: 0, 2: 1, 3: 2, 4: 3, 1: 4},
    "win1_baseline_lupin":       {0: 0, 4: 1, 3: 2, 2: 3, 1: 4},
    
    # win2_anox1
    "win2_anox1_kilosort4":      {0: 0, 2: 1, 4: 2, 1: 3, 3: 4},
    "win2_anox1_lupin":          {0: 0, 4: 1, 2: 2, 1: 3, 3: 4},
    
    # win3_anox2
    "win3_anox2_kilosort4":      {0: 0, 4: 1, 1: 2, 2: 3, 3: 4},
    "win3_anox2_lupin":          {3: 0, 1: 1, 4: 2, 0: 3, 2: 4},
    
    # win4_gasp_early
    "win4_gasp_early_kilosort4": {0: 0, 2: 1, 3: 2, 1: 3, 4: 4},
    "win4_gasp_early_lupin":     {0: 0, 2: 1, 3: 2, 1: 3, 4: 4},
    
    # win5_gasp_late
    "win5_gasp_late_kilosort4":  {0: 0, 1: 1, 3: 2, 2: 3, 4: 4},
    "win5_gasp_late_lupin":      {0: 0, 4: 1, 3: 2, 1: 3, 2: 4},
    
    # win6_flow_shift
    "win6_flow_shift_kilosort4": {0: 0, 2: 1, 3: 2, 4: 3, 1: 4},
    "win6_flow_shift_lupin":     {0: 0, 3: 1, 4: 2, 2: 3, 1: 4},
    
    # win7_recovery
    "win7_recovery_kilosort4":   {0: 0, 3: 1, 4: 2, 2: 3, 1: 4},
    "win7_recovery_lupin":       {0: 0, 3: 1, 4: 2, 2: 3, 1: 4}
}

STATE_NAMES = [
    "State 1: Inspiratory Drive (Insp Burst)",
    "State 2: Peak / Off-Switch Quench (I_off)",
    "State 3: Post-Insp / Expiratory Drive (Exp)",
    "State 4: Pre-Insp Priming (Pre-I)",
    "State 5: Inter-Breath Quiescence / Silence (Pause)"
]
STATE_SHORT_NAMES = ["1: Insp Drive", "2: Peak Off-Switch", "3: Exp Drive", "4: Pre-I Priming", "5: Quiescent Silence"]
STATE_COLORS = ["#10b981", "#f97316", "#38bdf8", "#a855f7", "#64748b"]


def extract_analog_and_breaths(t_start: float, dur: float, subwin_id: str) -> Dict[str, Any]:
    """Extracts analog data and segments breath cycles for a given window."""
    nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
    fs = 10000.0
    n_ch = 9
    s0 = int(t_start * fs)
    s1 = int((t_start + dur) * fs)

    logger.info(f"[{subwin_id}] Loading NIDAQ analog streams (t={t_start}s to {t_start+dur}s)...")
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

    if len(valid_onsets) > 0:
        flow_res = process_flow_meter(flow_v, t_vec, valid_onsets, valid_offsets, fs=fs)
        flow_zcs = flow_res.flow_zero_crossings
    else:
        flow_zcs = np.array([])

    n_breaths = len(valid_onsets)
    cycle_ends = np.append(valid_onsets[1:], dur) if n_breaths > 0 else np.array([])
    ti_arr = (flow_zcs - valid_onsets) * 1000.0 if n_breaths > 0 else np.array([])
    te_arr = (cycle_ends - flow_zcs) * 1000.0 if n_breaths > 0 else np.array([])
    tot_arr = (cycle_ends - valid_onsets) * 1000.0 if n_breaths > 0 else np.array([])

    # Gasps detection: in anoxia windows, flag gasps
    gasp_mask = np.zeros(n_breaths, dtype=bool)
    if subwin_id in ["win2_anox1", "win3_anox2", "win4_gasp_early"]:
        for i in range(n_breaths):
            if subwin_id == "win2_anox1" and valid_onsets[i] >= 27.0:
                gasp_mask[i] = True
            elif subwin_id in ["win3_anox2", "win4_gasp_early"]:
                gasp_mask[i] = True

    logger.info(f"[{subwin_id}] Loaded {n_breaths} breath cycles. Mean T_I = {np.mean(ti_arr) if n_breaths>0 else 0:.1f} ± {np.std(ti_arr) if n_breaths>0 else 0:.1f} ms.")

    return {
        "subwin_id": subwin_id,
        "fs": fs,
        "dur": dur,
        "t_start": t_start,
        "t_vec": t_vec,
        "emg_v": emg_v,
        "emg_env": emg_proc.envelope,
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


def fit_subwin_poisson_hmm(subwin_id: str, sorter_key: str, sorter_label: str, breath_data: Dict[str, Any], dt: float = 0.010, n_states: int = 5) -> Dict[str, Any]:
    """Fits 5-state Poisson HMM on raw 10 ms unreduced spike counts and computes alignments."""
    base_dir = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / subwin_id / sorter_key
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
    if len(df_qc) > 120:
        df_qc = df_qc.head(120)

    cids = df_qc["cluster_id"].values
    depths = df_qc["depth_um"].values
    n_units = len(cids)
    dur = breath_data["dur"]

    time_bins = np.arange(0, dur + dt, dt)
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2.0
    n_time_bins = len(bin_centers)

    counts = np.zeros((n_time_bins, n_units), dtype=np.int32)
    for u_idx, cid in enumerate(cids):
        c_spk = spk_times[spk_clusters == cid]
        counts[:, u_idx] = np.histogram(c_spk, bins=time_bins)[0][:n_time_bins]

    pop_dir = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / subwin_id / "population_dynamics"
    pop_dir.mkdir(parents=True, exist_ok=True)
    npy_states = pop_dir / f"{sorter_key}_canonical_states.npy"
    npy_consolidated = pop_dir / f"{sorter_key}_consolidated_states.npy"

    valid_onsets = breath_data["valid_onsets"]
    flow_zcs = breath_data["flow_zcs"]
    valid_offsets = breath_data["valid_offsets"]
    n_breaths = breath_data["n_breaths"]

    ds_key = f"{subwin_id}_{sorter_key}"
    cons_map = CONSOLIDATED_MAPPINGS.get(ds_key, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4})

    if npy_consolidated.exists():
        logger.info(f"[{subwin_id} - {sorter_label}] Loading cached consolidated states from {npy_consolidated}...")
        canonical_states = np.load(npy_consolidated)
        state_stats = []
    elif npy_states.exists():
        logger.info(f"[{subwin_id} - {sorter_label}] Remapping cached canonical states to consolidated taxonomy...")
        old_canonical = np.load(npy_states)
        canonical_states = np.array([cons_map[s] for s in old_canonical])
        np.save(npy_consolidated, canonical_states)
        state_stats = []
    else:
        logger.info(f"[{subwin_id} - {sorter_label}] Fitting {n_states}-state Poisson HMM on raw counts {counts.shape}...")
        model = hmm.PoissonHMM(n_components=n_states, n_iter=100, random_state=42, tol=1e-3)
        model.fit(counts)
        raw_states = model.predict(counts)

        # Compute phase labels for canonical mapping
        phase_labels = np.zeros(n_time_bins, dtype=int)
        for b_idx in range(n_breaths):
            t_on = valid_onsets[b_idx]
            t_zc = flow_zcs[b_idx]
            t_off = valid_offsets[b_idx]
            t_cend = breath_data["cycle_ends"][b_idx]

            mask_insp = (bin_centers >= t_on) & (bin_centers < t_zc)
            phase_labels[mask_insp] = 1
            mask_post = (bin_centers >= t_zc) & (bin_centers < min(t_off + 0.08, t_cend))
            phase_labels[mask_post] = 2

        # Map raw states to canonical 5 states
        state_stats = []
        for s in range(n_states):
            idx_s = np.where(raw_states == s)[0]
            if len(idx_s) == 0:
                pct_insp, pct_post, mean_rate = 0.0, 0.0, 0.0
            else:
                pct_insp = np.mean(phase_labels[idx_s] == 1) * 100.0
                pct_post = np.mean(phase_labels[idx_s] == 2) * 100.0
                mean_rate = np.mean(counts[idx_s]) / dt
            state_stats.append({
                "raw_id": s,
                "pct_insp": pct_insp,
                "pct_post": pct_post,
                "mean_rate": mean_rate
            })

        sorted_by_insp = sorted(state_stats, key=lambda x: (x["pct_insp"], x["mean_rate"]), reverse=True)
        c_s0 = sorted_by_insp[0]["raw_id"]  # State 1: Insp Burst
        c_s1 = sorted_by_insp[1]["raw_id"]  # State 2: Peak / Off-switch

        rem = [x for x in state_stats if x["raw_id"] not in [c_s0, c_s1]]
        rem_by_rate = sorted(rem, key=lambda x: x["mean_rate"], reverse=True)
        c_s2 = rem_by_rate[0]["raw_id"]  # State 3: Eupneic Exp Drive
        c_s3 = rem_by_rate[-1]["raw_id"] # State 4: Late-E Baseline
        c_s4 = [x["raw_id"] for x in rem if x["raw_id"] not in [c_s2, c_s3]][0] # State 5: Quiescence / Gasp

        canonical_map = {c_s0: 0, c_s1: 1, c_s2: 2, c_s3: 3, c_s4: 4}
        old_canonical = np.array([canonical_map[s] for s in raw_states])
        np.save(npy_states, old_canonical)
        canonical_states = np.array([cons_map[s] for s in old_canonical])
        np.save(npy_consolidated, canonical_states)

    # 1. ALIGNMENT BY INSPIRATION OFFSET (I_off)
    ioff_pre_ms = 250.0
    ioff_post_ms = 350.0
    ioff_axis_ms = np.arange(-ioff_pre_ms, ioff_post_ms, dt * 1000.0)
    n_ioff_bins = len(ioff_axis_ms)

    raster_ioff = []
    flow_ioff_trials = []
    emg_ioff_trials = []
    trial_records_ioff = []

    fs_nidq = breath_data["fs"]
    flow_v = breath_data["flow_v"]
    emg_env = breath_data["emg_env"]

    for b_idx in range(n_breaths):
        t_zc = flow_zcs[b_idx]
        t_on = valid_onsets[b_idx]
        ti = breath_data["ti_arr"][b_idx]
        tot = breath_data["tot_arr"][b_idx]
        is_gasp = bool(breath_data["gasp_mask"][b_idx])

        t_start = t_zc - (ioff_pre_ms / 1000.0)
        t_end = t_zc + (ioff_post_ms / 1000.0)
        mask_tr = (bin_centers >= t_start) & (bin_centers < t_end)
        s_slice = canonical_states[mask_tr]

        if len(s_slice) < n_ioff_bins:
            pad_val = s_slice[-1] if len(s_slice) > 0 else 3
            s_fixed = np.pad(s_slice, (0, n_ioff_bins - len(s_slice)), constant_values=pad_val)
        else:
            s_fixed = s_slice[:n_ioff_bins]
        raster_ioff.append(s_fixed)

        bin_eval_times = t_zc + (ioff_axis_ms / 1000.0)
        idx_eval = np.clip((bin_eval_times * fs_nidq).astype(int), 0, len(flow_v) - 1)
        flow_ioff_trials.append(flow_v[idx_eval])
        emg_ioff_trials.append(emg_env[idx_eval])

        trial_records_ioff.append({
            "trial_idx": b_idx,
            "trial_num": b_idx + 1,
            "is_gasp": is_gasp,
            "regime": "Gasping" if is_gasp else ("Apnea/Trans" if (subwin_id=="win2_anox1" and 24.0 <= t_on < 27.0) else "Eupnea"),
            "t_on_s": float(t_on),
            "t_zc_s": float(t_zc),
            "ti_ms": float(ti),
            "tot_ms": float(tot),
            "ion_relative_ms": float(-ti)
        })

    raster_ioff = np.array(raster_ioff) if len(raster_ioff) > 0 else np.zeros((0, n_ioff_bins), dtype=int)
    flow_ioff_mean = np.mean(flow_ioff_trials, axis=0).tolist() if len(flow_ioff_trials) > 0 else np.zeros(n_ioff_bins).tolist()
    emg_ioff_mean = np.mean(emg_ioff_trials, axis=0).tolist() if len(emg_ioff_trials) > 0 else np.zeros(n_ioff_bins).tolist()

    prob_ioff = []
    for k in range(n_states):
        if len(raster_ioff) > 0:
            prob_k = np.mean(raster_ioff == k, axis=0) * 100.0
            prob_ioff.append(prob_k.tolist())
        else:
            prob_ioff.append(np.zeros(n_ioff_bins).tolist())

    # 2. ALIGNMENT BY INSPIRATION ONSET (I_on)
    ion_pre_ms = 100.0
    ion_post_ms = 500.0
    ion_axis_ms = np.arange(-ion_pre_ms, ion_post_ms, dt * 1000.0)
    n_ion_bins = len(ion_axis_ms)

    raster_ion = []
    flow_ion_trials = []
    emg_ion_trials = []
    trial_records_ion = []

    for b_idx in range(n_breaths):
        t_on = valid_onsets[b_idx]
        t_zc = flow_zcs[b_idx]
        ti = breath_data["ti_arr"][b_idx]
        tot = breath_data["tot_arr"][b_idx]
        is_gasp = bool(breath_data["gasp_mask"][b_idx])

        t_start = t_on - (ion_pre_ms / 1000.0)
        t_end = t_on + (ion_post_ms / 1000.0)
        mask_tr = (bin_centers >= t_start) & (bin_centers < t_end)
        s_slice = canonical_states[mask_tr]

        if len(s_slice) < n_ion_bins:
            pad_val = s_slice[-1] if len(s_slice) > 0 else 3
            s_fixed = np.pad(s_slice, (0, n_ion_bins - len(s_slice)), constant_values=pad_val)
        else:
            s_fixed = s_slice[:n_ion_bins]
        raster_ion.append(s_fixed)

        bin_eval_times = t_on + (ion_axis_ms / 1000.0)
        idx_eval = np.clip((bin_eval_times * fs_nidq).astype(int), 0, len(flow_v) - 1)
        flow_ion_trials.append(flow_v[idx_eval])
        emg_ion_trials.append(emg_env[idx_eval])

        mask_ti = (bin_centers >= t_on) & (bin_centers < t_zc)
        s_ti = canonical_states[mask_ti]
        dur_insp_ms = np.sum(s_ti == 0) * (dt * 1000.0) if len(s_ti) > 0 else 0.0

        trial_records_ion.append({
            "trial_idx": b_idx,
            "trial_num": b_idx + 1,
            "is_gasp": is_gasp,
            "regime": "Gasping" if is_gasp else ("Apnea/Trans" if (subwin_id=="win2_anox1" and 24.0 <= t_on < 27.0) else "Eupnea"),
            "t_on_s": float(t_on),
            "t_zc_s": float(t_zc),
            "ti_ms": float(ti),
            "tot_ms": float(tot),
            "dur_insp_ms": float(dur_insp_ms),
            "ioff_relative_ms": float(ti)
        })

    raster_ion = np.array(raster_ion) if len(raster_ion) > 0 else np.zeros((0, n_ion_bins), dtype=int)
    flow_ion_mean = np.mean(flow_ion_trials, axis=0).tolist() if len(flow_ion_trials) > 0 else np.zeros(n_ion_bins).tolist()
    emg_ion_mean = np.mean(emg_ion_trials, axis=0).tolist() if len(emg_ion_trials) > 0 else np.zeros(n_ion_bins).tolist()

    prob_ion = []
    for k in range(n_states):
        if len(raster_ion) > 0:
            prob_k = np.mean(raster_ion == k, axis=0) * 100.0
            prob_ion.append(prob_k.tolist())
        else:
            prob_ion.append(np.zeros(n_ion_bins).tolist())

    return {
        "subwin_id": subwin_id,
        "sorter_key": sorter_key,
        "sorter_label": sorter_label,
        "n_units": n_units,
        "n_breaths": n_breaths,
        "counts": counts,
        "bin_centers": bin_centers,
        "canonical_states": canonical_states,
        "state_stats": state_stats,
        "ioff": {
            "axis_ms": ioff_axis_ms.tolist(),
            "raster": [r.tolist() for r in raster_ioff],
            "prob": prob_ioff,
            "trials": trial_records_ioff,
            "flow_mean": flow_ioff_mean,
            "emg_mean": emg_ioff_mean
        },
        "ion": {
            "axis_ms": ion_axis_ms.tolist(),
            "raster": [r.tolist() for r in raster_ion],
            "prob": prob_ion,
            "trials": trial_records_ion,
            "flow_mean": flow_ion_mean,
            "emg_mean": emg_ion_mean
        }
    }


def generate_subwin_static_plots(subwin_id: str, subwin_info: Dict[str, Any], ks4_res: Dict[str, Any], lup_res: Dict[str, Any], breath_data: Dict[str, Any], out_dir: Path):
    """Generates high-resolution trial raster and continuous timeline PNGs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_b = breath_data["n_breaths"]
    t_ax = np.array(ks4_res["ion"]["axis_ms"])
    raster_ks4 = np.array(ks4_res["ion"]["raster"])
    raster_lup = np.array(lup_res["ion"]["raster"])

    # 1. Trial-by-Trial Micro-State Raster
    logger.info(f"[{subwin_id}] Generating trial-by-trial micro-state raster PNG...")
    fig = plt.figure(figsize=(20, 13), facecolor='#090d16')
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 1.0], hspace=0.25, wspace=0.18,
                           left=0.06, right=0.96, top=0.92, bottom=0.06)

    # Subplot A1: KS4 Trials
    ax_a1 = fig.add_subplot(gs[0, 0], facecolor='#0f172a')
    if n_b > 0 and len(raster_ks4) > 0:
        rgb_ks4 = np.zeros((n_b, len(t_ax), 3), dtype=float)
        for s_id, hex_c in enumerate(STATE_COLORS):
            h = hex_c.lstrip('#')
            rgb = tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
            rgb_ks4[raster_ks4 == s_id] = rgb

        ax_a1.imshow(rgb_ks4, aspect='auto', extent=[t_ax[0], t_ax[-1], n_b, 0], origin='upper', interpolation='nearest')
        ax_a1.axvline(0, color='#ffffff', lw=1.5, ls='--', label='Diaphragm EMG Onset (I_on = 0 ms)')
        ax_a1.plot(breath_data["ti_arr"], np.arange(n_b) + 0.5, color='#fbbf24', lw=1.6, label='Flow Zero-Crossing (I_off)')
    ax_a1.set_xlabel("Time from Diaphragm EMG Onset (ms)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_a1.set_ylabel(f"Breath Trial Index (1 to {n_b})\n{subwin_info['name']}", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_a1.set_title(f"Kilosort 4 ({ks4_res['n_units']} Units): Micro-State Transitions (Raw Poisson Spikes, No PCA)", color='#38bdf8', fontsize=12, fontweight='bold')
    ax_a1.tick_params(colors='#94a3b8', labelsize=9)
    ax_a1.legend(loc='lower left', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8.5)

    # Subplot A2: Lupin Trials
    ax_a2 = fig.add_subplot(gs[1, 0], facecolor='#0f172a', sharex=ax_a1)
    if n_b > 0 and len(raster_lup) > 0:
        rgb_lup = np.zeros((n_b, len(t_ax), 3), dtype=float)
        for s_id, hex_c in enumerate(STATE_COLORS):
            h = hex_c.lstrip('#')
            rgb = tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
            rgb_lup[raster_lup == s_id] = rgb

        ax_a2.imshow(rgb_lup, aspect='auto', extent=[t_ax[0], t_ax[-1], n_b, 0], origin='upper', interpolation='nearest')
        ax_a2.axvline(0, color='#ffffff', lw=1.5, ls='--')
        ax_a2.plot(breath_data["ti_arr"], np.arange(n_b) + 0.5, color='#fbbf24', lw=1.6)
    ax_a2.set_xlabel("Time from Diaphragm EMG Onset (ms)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_a2.set_ylabel(f"Breath Trial Index (1 to {n_b})", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_a2.set_title(f"Lupin ({lup_res['n_units']} Units): Micro-State Transitions (Raw Poisson Spikes, No PCA)", color='#f59e0b', fontsize=12, fontweight='bold')
    ax_a2.tick_params(colors='#94a3b8', labelsize=9)

    # Subplot B1: Duration of Inspiratory State vs Actual T_I
    ax_b1 = fig.add_subplot(gs[0, 1], facecolor='#0f172a')
    if n_b > 0:
        trials = np.arange(1, n_b + 1)
        dur_ks4 = [t["dur_insp_ms"] for t in ks4_res["ion"]["trials"]]
        dur_lup = [t["dur_insp_ms"] for t in lup_res["ion"]["trials"]]
        ax_b1.plot(trials, breath_data["ti_arr"], color='#fbbf24', lw=1.3, label='Actual T_I (Flow ZC, ms)')
        ax_b1.plot(trials, dur_ks4, color='#10b981', lw=1.4, label='KS4 State 1 (Insp Burst, ms)')
        ax_b1.plot(trials, dur_lup, color='#06b6d4', lw=1.2, ls='--', label='Lupin State 1 (Insp Burst, ms)')
    ax_b1.set_xlabel(f"Breath Trial Index (1 to {n_b})", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_b1.set_ylabel("Duration (ms)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_b1.set_title(f"Inspiratory Burst Duration Tracking Across {n_b} Breaths", color='#f8fafc', fontsize=12, fontweight='bold')
    ax_b1.tick_params(colors='#94a3b8', labelsize=9)
    ax_b1.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8.5)

    # Subplot B2: State Occupancy (%) Across Breath Time
    ax_b2 = fig.add_subplot(gs[1, 1], facecolor='#0f172a', sharex=ax_a1)
    for s_id in range(5):
        ax_b2.plot(t_ax, ks4_res["ion"]["prob"][s_id], color=STATE_COLORS[s_id], lw=1.8, label=f"KS4 {STATE_SHORT_NAMES[s_id]}")
        ax_b2.plot(t_ax, lup_res["ion"]["prob"][s_id], color=STATE_COLORS[s_id], lw=1.3, ls='--', alpha=0.7)
    ax_b2.axvline(0, color='#ffffff', lw=1.5, ls='--')
    ax_b2.set_xlabel("Time from Diaphragm EMG Onset (ms)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_b2.set_ylabel("State Occupancy (%)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_b2.set_title("Population State Occupancy PETH (Solid: KS4, Dashed: Lupin)", color='#f8fafc', fontsize=12, fontweight='bold')
    ax_b2.tick_params(colors='#94a3b8', labelsize=9)
    ax_b2.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=7.5, ncol=2)

    fig.suptitle(f"e097 {subwin_id} ({subwin_info['name']}) — Raw Spiking Poisson HMM Population Micro-States",
                 color='#f8fafc', fontsize=15, fontweight='bold', y=0.98)
    raster_png = out_dir / f"{subwin_id}_trial_by_trial_microstate_raster.png"
    try:
        plt.savefig(raster_png, dpi=200, facecolor='#090d16')
        logger.info(f"Saved: {raster_png}")
    except Exception as e:
        logger.warning(f"Could not overwrite {raster_png}: {e}")
    plt.close()

    # 2. Continuous Timeline Figure
    logger.info(f"[{subwin_id}] Generating continuous timeline figure...")
    dur = breath_data["dur"]
    t_nidq = breath_data["t_vec"]
    flow_v = breath_data["flow_v"]
    emg_env = breath_data["emg_env"]
    bin_centers = ks4_res["bin_centers"]
    states_ks4 = ks4_res["canonical_states"]
    states_lup = lup_res["canonical_states"]

    fig = plt.figure(figsize=(20, 11), facecolor='#090d16')
    gs = gridspec.GridSpec(4, 1, height_ratios=[1.2, 1.0, 1.3, 1.3], hspace=0.18,
                           left=0.06, right=0.96, top=0.93, bottom=0.06)

    # Panel 1: Airflow
    ax1 = fig.add_subplot(gs[0, 0], facecolor='#0f172a')
    ax1.plot(t_nidq, flow_v, color='#10b981', lw=0.9, label='Airflow (V)')
    ax1.axhline(0, color='#334155', lw=1.0, ls=':')
    ax1.set_ylabel("Flow (V)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax1.set_title(f"Continuous Recording Timeline: {subwin_info['name']} (Duration: {dur:.1f}s)", color='#38bdf8', fontsize=12, fontweight='bold')
    ax1.tick_params(colors='#94a3b8', labelsize=9)

    # Panel 2: Diaphragm EMG
    ax2 = fig.add_subplot(gs[1, 0], facecolor='#0f172a', sharex=ax1)
    ax2.plot(t_nidq, emg_env, color='#ec4899', lw=1.0, label='Diaphragm EMG Envelope (V)')
    for on in breath_data["valid_onsets"]:
        ax2.axvline(on, color='#ffffff', lw=0.5, alpha=0.3)
    ax2.set_ylabel("EMG Env (V)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax2.tick_params(colors='#94a3b8', labelsize=9)

    # Panel 3: KS4 HMM States
    ax3 = fig.add_subplot(gs[2, 0], facecolor='#0f172a', sharex=ax1)
    state_img_ks4 = np.zeros((1, len(bin_centers), 3), dtype=float)
    for s_id, hex_c in enumerate(STATE_COLORS):
        h = hex_c.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
        state_img_ks4[0, states_ks4 == s_id] = rgb
    ax3.imshow(state_img_ks4, aspect='auto', extent=[0, dur, 0, 1], interpolation='nearest')
    ax3.set_ylabel("KS4 HMM", color='#f8fafc', fontsize=10, fontweight='bold')
    ax3.set_yticks([])
    ax3.tick_params(colors='#94a3b8', labelsize=9)

    # Panel 4: Lupin HMM States
    ax4 = fig.add_subplot(gs[3, 0], facecolor='#0f172a', sharex=ax1)
    state_img_lup = np.zeros((1, len(bin_centers), 3), dtype=float)
    for s_id, hex_c in enumerate(STATE_COLORS):
        h = hex_c.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
        state_img_lup[0, states_lup == s_id] = rgb
    ax4.imshow(state_img_lup, aspect='auto', extent=[0, dur, 0, 1], interpolation='nearest')
    ax4.set_ylabel("Lupin HMM", color='#f8fafc', fontsize=10, fontweight='bold')
    ax4.set_xlabel(f"Time in Subwindow (seconds, t = {subwin_info['t_start']}s to {subwin_info['t_end']}s)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax4.set_yticks([])
    ax4.tick_params(colors='#94a3b8', labelsize=9)

    timeline_png = out_dir / f"{subwin_id}_continuous_hmm_timeline.png"
    try:
        plt.savefig(timeline_png, dpi=200, facecolor='#090d16')
        logger.info(f"Saved: {timeline_png}")
    except Exception as e:
        logger.warning(f"Could not overwrite {timeline_png}: {e}")
    plt.close()


def generate_subwindow_dashboard_html(subwin_id: str, subwin_info: Dict[str, Any], ks4_res: Dict[str, Any], lup_res: Dict[str, Any], breath_data: Dict[str, Any], out_path: Path):
    """Generates the rich dashboard HTML for a given subwindow."""
    n_b = breath_data["n_breaths"]
    mean_ti = np.mean(breath_data["ti_arr"]) if n_b > 0 else 0.0
    std_ti = np.std(breath_data["ti_arr"]) if n_b > 0 else 0.0
    mean_tot = np.mean(breath_data["tot_arr"]) if n_b > 0 else 0.0
    freq_hz = (1000.0 / mean_tot) if mean_tot > 0 else 0.0

    # Build trial table rows
    table_rows = []
    trials_ks4 = ks4_res["ion"]["trials"]
    trials_lup = lup_res["ion"]["trials"]
    for i in range(min(n_b, 60)):
        tk = trials_ks4[i]
        tl = trials_lup[i]
        badge_cls = "background:#ef4444;color:#ffffff;" if tk["is_gasp"] else "background:#0f766e;color:#5eead4;"
        table_rows.append(f"""
        <tr>
          <td>#{tk['trial_num']}</td>
          <td><span style="padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;{badge_cls}">{tk['regime']}</span></td>
          <td>{tk['t_on_s']:.3f}</td>
          <td>{tk['ti_ms']:.1f}</td>
          <td>{tk['tot_ms']:.1f}</td>
          <td><b style="color:#10b981;">{tk['dur_insp_ms']:.1f}</b></td>
          <td><b style="color:#38bdf8;">{tl['dur_insp_ms']:.1f}</b></td>
        </tr>""")
    table_rows_str = "".join(table_rows)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>e097 {subwin_id} - Population Dynamics &amp; HMM Hub</title>
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
      font-family: 'JetBrains Mono', monospace;
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
    .img-display:hover {{ transform: scale(1.005); }}
    .state-legend {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
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
    th {{ background: #141c2e; color: #cbd5e1; font-weight: 700; }}
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
        <a href="../{subwin_id}_sorter_comparison.html" class="back-btn">&larr; Back to {subwin_id} Sorter Benchmark</a>
        <div style="display:flex;gap:10px;">
          <a href="../../all_subwindows_sorting_dashboard.html" class="back-btn">🌐 Master Subwindows Dashboard</a>
          <a href="../population_dynamics/{subwin_id}_interactive_microstate_raster.html" class="back-btn" style="border-color:#38bdf8;color:#38bdf8;">🔬 Open Interactive Raster &rarr;</a>
        </div>
      </div>
      <h1>e097 {subwin_id} &mdash; {subwin_info['name']} Population Dynamics &amp; HMM Hub</h1>
      <p class="subtitle">
        {subwin_info['desc']} (<strong>{subwin_info['duration']}s duration, {n_b} breath cycles</strong>, {subwin_info['t_start']}s – {subwin_info['t_end']}s).
      </p>
    </header>

    <div class="grid-4">
      <div class="stat-card">
        <div class="stat-val">10 ms</div>
        <div class="stat-lbl">Temporal Resolution</div>
      </div>
      <div class="stat-card">
        <div class="stat-val">{n_b} Breaths</div>
        <div class="stat-lbl">Analyzed Cycles ({freq_hz:.2f} Hz)</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" style="color:#10b981;">{mean_ti:.1f} ms</div>
        <div class="stat-lbl">Mean T_I (± {std_ti:.1f} ms)</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" style="color:#f59e0b;">No PCA</div>
        <div class="stat-lbl">Raw Poisson Spike Counts</div>
      </div>
    </div>

    <!-- HMM STATE LEGEND -->
    <div class="section">
      <h2>🏷️ Canonical Respiratory Micro-States Decoded by 5-State Poisson HMM</h2>
      <div class="state-legend">
        <div class="legend-box" style="border-left:4px solid #10b981;">
          <strong style="color:#10b981;">State 1: Insp Drive (Insp Burst)</strong>
          <p>Locked to Diaphragm EMG (&gt;90% in Insp). High-rate synchronized inspiratory drive.</p>
        </div>
        <div class="legend-box" style="border-left:4px solid #f97316;">
          <strong style="color:#f97316;">State 2: Peak / Off-Switch Quench</strong>
          <p>Active at Flow zero-crossing ($I_{{off}}$). Active termination of inspiration.</p>
        </div>
        <div class="legend-box" style="border-left:4px solid #38bdf8;">
          <strong style="color:#38bdf8;">State 3: Post-Insp / Exp Drive</strong>
          <p>Post-inspiratory expiratory phase and active expiration drive.</p>
        </div>
        <div class="legend-box" style="border-left:4px solid #a855f7;">
          <strong style="color:#c084fc;">State 4: Pre-Insp Priming (Pre-I)</strong>
          <p>Pre-inspiratory priming and early ramp preceding Diaphragm onset ($I_{{on}}$).</p>
        </div>
        <div class="legend-box" style="border-left:4px solid #64748b;">
          <strong style="color:#94a3b8;">State 5: Inter-Breath Quiescence / Silence</strong>
          <p>Quiescent interval and silence during inter-breath gaps, apneas, and gasps.</p>
        </div>
      </div>
    </div>

    <!-- SECTION 1: TRIAL-BY-TRIAL RASTER -->
    <div class="section">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <h2>🧬 1. Trial-by-Trial Micro-State Transitions ({n_b} Breaths, Raw Spikes)</h2>
        <a href="{subwin_id}_interactive_microstate_raster.html" target="_blank" class="back-btn" style="background:#0284c7;border-color:#38bdf8;color:#ffffff;font-size:13px;padding:8px 16px;">
          🔬 Open Zoomable Interactive Raster (I_off / I_on Aligned) &rarr;
        </a>
      </div>
      <p style="color:#cbd5e1;font-size:14px;">
        All breath trials aligned to Diaphragm EMG Onset ($t = 0\text{{ ms}}$) with Flow zero-crossing ($I_{{off}}$) in gold. Demonstrates intra-breath micro-state trajectories without PCA distortion.
      </p>
      <img src="{subwin_id}_trial_by_trial_microstate_raster.png" class="img-display" onclick="openModal('{subwin_id}_trial_by_trial_microstate_raster.png')" alt="Trial Raster">
    </div>

    <!-- SECTION 2: CONTINUOUS TIMELINE -->
    <div class="section">
      <h2>📈 2. Continuous {subwin_info['duration']}s Timeline &amp; State Decoding</h2>
      <p style="color:#cbd5e1;font-size:14px;">
        Continuous recording showing Airflow, Diaphragm EMG Envelope, and continuous decoded HMM states on 10 ms bins across both Kilosort 4 and Lupin.
      </p>
      <img src="{subwin_id}_continuous_hmm_timeline.png" class="img-display" onclick="openModal('{subwin_id}_continuous_hmm_timeline.png')" alt="Continuous Timeline">
    </div>

    <!-- SECTION 3: TRIAL TABLE -->
    <div class="section">
      <h2>📊 3. Single-Breath Metrics (First {min(n_b, 60)} of {n_b} Breaths)</h2>
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
            </tr>
          </thead>
          <tbody>
            {table_rows_str}
          </tbody>
        </table>
      </div>
    </div>

  </div>

  <div id="imageModal" class="modal" onclick="closeModal()">
    <img id="modalImg" src="">
  </div>

  <script>
    function openModal(src) {{
      document.getElementById('modalImg').src = src;
      document.getElementById('imageModal').style.display = 'flex';
    }}
    function closeModal() {{
      document.getElementById('imageModal').style.display = 'none';
    }}
  </script>
</body>
</html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Dashboard saved to {out_path}")


def update_sorter_comparison_card(subwin_id: str, n_breaths: int, mean_ti: float):
    """Updates the subwindow's sorter comparison benchmark page to add a Population Dynamics card."""
    comp_file = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / subwin_id / f"{subwin_id}_sorter_comparison.html"
    if not comp_file.exists():
        return

    content = comp_file.read_text(encoding="utf-8")
    if "Population Dynamics" in content:
        return  # already has card

    # Update cards-grid to 4 columns and insert card
    if "grid-template-columns: repeat(3, 1fr);" in content:
        content = content.replace("grid-template-columns: repeat(3, 1fr);", "grid-template-columns: repeat(4, 1fr);")

    card_html = f"""        <div class="card" style="border: 1px solid rgba(56, 189, 248, 0.4); background: #0c1829;">
            <h3 style="color:#38bdf8;">Population Dynamics &amp; HMM</h3>
            <div class="metric-val" style="color:#38bdf8;">{n_breaths} Breaths</div>
            <div class="metric-sub">Raw Spiking (No PCA) | 10ms Micro-States | Mean T_I: {mean_ti:.1f}ms</div>
            <div style="margin-top: 12px; display: flex; gap: 8px;">
                <a href="population_dynamics/{subwin_id}_population_dynamics_dashboard.html" target="_blank" class="btn" style="background:#0284c7;border-color:#38bdf8;color:#ffffff;">⚡ Open HMM Hub &rarr;</a>
            </div>
        </div>
    </div>"""

    content = content.replace("    </div>\n\n    <div class=\"table-container\">", f"{card_html}\n\n    <div class=\"table-container\">")
    comp_file.write_text(content, encoding="utf-8")
    logger.info(f"Updated benchmark page: {comp_file}")


def build_master_interactive_raster_html(all_data: Dict[str, Any], output_path: Path):
    """Builds the comprehensive Master 7-Subwindow Interactive Raster HTML with tabs."""
    json_str = json.dumps(all_data)

    subwin_buttons = []
    for idx, sw in enumerate(SUBWINDOWS):
        wid = sw["id"]
        wname = sw["name"]
        active_cls = "active" if idx == 0 else ""
        subwin_buttons.append(f'<button class="toggle-btn {active_cls}" id="btn-{wid}" onclick="setSubwindow(\'{wid}\')">{wname}</button>')
    subwin_btns_str = "\n".join(subwin_buttons)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>e097 Master Multi-Subwindow Population Micro-State Raster Hub</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: #07090e;
      color: #f1f5f9;
      line-height: 1.5;
      padding: 24px 32px;
    }}
    .container {{ max-width: 1540px; margin: 0 auto; }}
    header {{
      background: linear-gradient(180deg, #0f172a 0%, #07090e 100%);
      border-bottom: 1px solid #1e293b;
      padding-bottom: 20px;
      margin-bottom: 24px;
    }}
    .nav-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
    }}
    .back-btn {{
      background: #1e293b;
      color: #38bdf8;
      text-decoration: none;
      padding: 7px 14px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      border: 1px solid #334155;
      transition: all 0.2s;
    }}
    .back-btn:hover {{ background: #38bdf8; color: #07090e; }}
    h1 {{
      font-size: 26px;
      font-weight: 800;
      color: #f8fafc;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      color: #94a3b8;
      font-size: 14px;
      margin-top: 4px;
      max-width: 1300px;
    }}
    /* CONTROL PANEL */
    .controls-panel {{
      background: #0f1523;
      border: 1px solid #1e293b;
      border-radius: 12px;
      padding: 18px 24px;
      margin-bottom: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .subwin-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      background: #090d16;
      padding: 6px;
      border-radius: 8px;
      border: 1px solid #1e293b;
    }}
    .controls-row {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 20px;
    }}
    .control-group {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .control-label {{
      font-size: 11.5px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #94a3b8;
    }}
    .btn-group {{
      display: flex;
      gap: 6px;
      background: #090d16;
      padding: 4px;
      border-radius: 8px;
      border: 1px solid #1e293b;
    }}
    .toggle-btn {{
      padding: 8px 12px;
      background: transparent;
      border: none;
      color: #94a3b8;
      font-size: 12px;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
      text-align: center;
    }}
    .toggle-btn:hover {{ color: #f8fafc; background: rgba(255,255,255,0.05); }}
    .toggle-btn.active {{
      background: #0284c7;
      color: #ffffff;
      box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }}
    .toggle-btn.active.gold {{ background: #d97706; }}
    .toggle-btn.active.purple {{ background: #7c3aed; }}
    /* SUMMARY STATS */
    .stats-row {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }}
    .stat-card {{
      background: #0f1523;
      border: 1px solid #1e293b;
      border-radius: 10px;
      padding: 14px 18px;
    }}
    .stat-val {{
      font-size: 24px;
      font-weight: 800;
      color: #38bdf8;
      font-family: 'JetBrains Mono', monospace;
    }}
    .stat-lbl {{
      font-size: 11px;
      color: #94a3b8;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    /* STATE LEGEND BAR */
    .legend-bar {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 12px;
      margin-bottom: 20px;
    }}
    .legend-pill {{
      background: #0f1523;
      border: 1px solid #1e293b;
      border-radius: 8px;
      padding: 10px 14px;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .legend-color {{
      width: 14px;
      height: 14px;
      border-radius: 4px;
      flex-shrink: 0;
    }}
    .legend-text strong {{
      font-size: 12px;
      display: block;
      color: #f8fafc;
    }}
    .legend-text span {{
      font-size: 11px;
      color: #94a3b8;
    }}
    /* PLOT CONTAINERS */
    .plot-card {{
      background: #0f1523;
      border: 1px solid #1e293b;
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 24px;
    }}
    .plot-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }}
    .plot-title {{
      font-size: 16px;
      font-weight: 700;
      color: #f8fafc;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .zoom-helpers {{
      display: flex;
      gap: 8px;
    }}
    .zoom-btn {{
      background: #1e293b;
      color: #cbd5e1;
      border: 1px solid #334155;
      padding: 5px 10px;
      border-radius: 6px;
      font-size: 11.5px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .zoom-btn:hover {{
      background: #38bdf8;
      color: #07090e;
      border-color: #38bdf8;
    }}
    #rasterPlot {{ width: 100%; height: 560px; }}
    #probPlot {{ width: 100%; height: 260px; }}
    #analogPlot {{ width: 100%; height: 210px; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="nav-bar">
        <a href="all_subwindows_sorting_dashboard.html" class="back-btn">&larr; Back to Master Subwindows Hub</a>
        <div style="display:flex;gap:10px;">
          <a href="master_anoxia_transition_dashboard.html" class="back-btn" style="border-color:#ef4444;color:#fca5a5;">⚡ Anoxia Transition Hub</a>
        </div>
      </div>
      <h1>e097 &mdash; Master Multi-Subwindow Population Micro-State Raster Hub</h1>
      <p class="subtitle">
        Seamlessly explore, zoom, and inspect population micro-state dynamics across all 7 physiological recording stages (from normoxic baseline eupnea to anoxia, apnea, autoresuscitation gasping, active expiration, and recovery).
      </p>
    </header>

    <!-- MASTER CONTROLS PANEL -->
    <div class="controls-panel">
      <div>
        <div class="control-label" style="margin-bottom:6px;">1. Select Subwindow / Recording Stage (win1 to win7)</div>
        <div class="subwin-bar" id="subwin-bar">
          {subwin_btns_str}
        </div>
      </div>

      <div class="controls-row">
        <!-- ALIGNMENT -->
        <div class="control-group">
          <div class="control-label">2. Breath Alignment Reference</div>
          <div class="btn-group">
            <button class="toggle-btn active gold" id="btn-ioff" onclick="setAlignment('ioff')">⭐ Inspiration Offset (I_off)</button>
            <button class="toggle-btn" id="btn-ion" onclick="setAlignment('ion')">Inspiration Onset (I_on)</button>
          </div>
        </div>

        <!-- SORTER -->
        <div class="control-group">
          <div class="control-label">3. Spike Sorting Model</div>
          <div class="btn-group">
            <button class="toggle-btn active" id="btn-ks4" onclick="setSorter('kilosort4')">Kilosort 4 (GPU)</button>
            <button class="toggle-btn" id="btn-lupin" onclick="setSorter('lupin')">Lupin (SVD-OMP)</button>
          </div>
        </div>

        <!-- SORT ORDER -->
        <div class="control-group">
          <div class="control-label">4. Trial Sorting Order</div>
          <div class="btn-group">
            <button class="toggle-btn active purple" id="btn-sort-chrono" onclick="setSortOrder('chrono')">Chronological (1 &rarr; N)</button>
            <button class="toggle-btn" id="btn-sort-ti" onclick="setSortOrder('ti')">Sorted by T_I Duration</button>
          </div>
        </div>
      </div>
    </div>

    <!-- STATS ROW -->
    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-lbl">Active Breaths in Stage</div>
        <div class="stat-val" id="stat-nbreaths">187 Breaths</div>
      </div>
      <div class="stat-card">
        <div class="stat-lbl">Mean Insp Duration (T_I)</div>
        <div class="stat-val" id="stat-ti" style="color:#10b981;">120.7 ms</div>
      </div>
      <div class="stat-card">
        <div class="stat-lbl">Alignment Event</div>
        <div class="stat-val" id="stat-align" style="color:#fbbf24;">I_off (t = 0 ms)</div>
      </div>
      <div class="stat-card">
        <div class="stat-lbl">Sorting Algorithm / Units</div>
        <div class="stat-val" id="stat-units" style="color:#38bdf8;">KS4 / 120 Units</div>
      </div>
    </div>

    <!-- HMM STATE LEGEND -->
    <div class="legend-bar">
      <div class="legend-pill" style="border-left: 3px solid #10b981;">
        <div class="legend-color" style="background: #10b981;"></div>
        <div class="legend-text">
          <strong>State 1: Insp Burst</strong>
          <span>High-rate sync drive (&gt;95% in Insp)</span>
        </div>
      </div>
      <div class="legend-pill" style="border-left: 3px solid #f97316;">
        <div class="legend-color" style="background: #f97316;"></div>
        <div class="legend-text">
          <strong>State 2: Peak / Off-Switch</strong>
          <span>Flow zero-crossing quench ($I_{{off}}$)</span>
        </div>
      </div>
      <div class="legend-pill" style="border-left: 3px solid #38bdf8;">
        <div class="legend-color" style="background: #38bdf8;"></div>
        <div class="legend-text">
          <strong>State 3: Exp Drive</strong>
          <span>Post-inspiratory / Active Exp phase</span>
        </div>
      </div>
      <div class="legend-pill" style="border-left: 3px solid #a855f7;">
        <div class="legend-color" style="background: #a855f7;"></div>
        <div class="legend-text">
          <strong>State 4: Pre-I Priming</strong>
          <span>Pre-inspiratory priming preceding I_on</span>
        </div>
      </div>
      <div class="legend-pill" style="border-left: 3px solid #64748b;">
        <div class="legend-color" style="background: #64748b;"></div>
        <div class="legend-text">
          <strong>State 5: Quiescent Silence</strong>
          <span>Inter-breath silence, pauses &amp; apneas</span>
        </div>
      </div>
    </div>

    <!-- MAIN PLOT CARD 1: TRIAL-BY-TRIAL RASTER -->
    <div class="plot-card">
      <div class="plot-header">
        <div class="plot-title">
          <span>🧬 1. Interactive Micro-State Trial-by-Trial Raster</span>
          <span style="font-size:12px;color:#94a3b8;font-weight:400;" id="raster-subtitle">Loading...</span>
        </div>
        <div class="zoom-helpers">
          <button class="zoom-btn" onclick="zoomTime(-150, 50)">🔍 Zoom Off-Switch Boundary (-150 to +50ms)</button>
          <button class="zoom-btn" onclick="zoomTime(-200, 150)">🔍 Zoom Full Burst (-200 to +150ms)</button>
          <button class="zoom-btn" onclick="zoomTrials(1, 30)">Breaths 1–30</button>
          <button class="zoom-btn" onclick="resetZoom()">↺ Reset Zoom</button>
        </div>
      </div>
      <div id="rasterPlot"></div>
    </div>

    <!-- MAIN PLOT CARD 2: POPULATION STATE OCCUPANCY PROBABILITY PETH -->
    <div class="plot-card">
      <div class="plot-header">
        <div class="plot-title">
          <span>📊 2. Population State Occupancy Probability PETH (%)</span>
          <span style="font-size:12px;color:#94a3b8;font-weight:400;">Synchronized across all breath trials</span>
        </div>
      </div>
      <div id="probPlot"></div>
    </div>

    <!-- MAIN PLOT CARD 3: AVERAGE AIRFLOW & DIAPHRAGM EMG WAVEFORMS -->
    <div class="plot-card">
      <div class="plot-header">
        <div class="plot-title">
          <span>🫁 3. Average Physiological Waveforms (Airflow &amp; Diaphragm EMG Envelope)</span>
          <span style="font-size:12px;color:#94a3b8;font-weight:400;">Physiological alignment validation</span>
        </div>
      </div>
      <div id="analogPlot"></div>
    </div>

  </div>

  <script>
    const DATA = {json_str};

    let currentSubwin = 'win1_baseline';
    let currentAlign = 'ioff';
    let currentSorter = 'kilosort4';
    let currentSortOrder = 'chrono';

    const STATE_COLORS = ["#10b981", "#f97316", "#38bdf8", "#a855f7", "#64748b"];
    const STATE_NAMES = [
      "State 1: Insp Drive",
      "State 2: Peak Off-Switch",
      "State 3: Exp Drive",
      "State 4: Pre-I Priming",
      "State 5: Quiescent Silence"
    ];

    const customColorscale = [
      [0.0, '#10b981'], [0.19, '#10b981'],
      [0.2, '#f97316'], [0.39, '#f97316'],
      [0.4, '#38bdf8'], [0.59, '#38bdf8'],
      [0.6, '#a855f7'], [0.79, '#a855f7'],
      [0.8, '#64748b'], [1.00, '#64748b']
    ];

    const subwins = ['win1_baseline', 'win2_anox1', 'win3_anox2', 'win4_gasp_early', 'win5_gasp_late', 'win6_flow_shift', 'win7_recovery'];

    function setSubwindow(sub) {{
      currentSubwin = sub;
      subwins.forEach(w => {{
        const btn = document.getElementById(`btn-${{w}}`);
        if (btn) btn.classList.toggle('active', w === sub);
      }});
      renderAll();
    }}

    function setAlignment(align) {{
      currentAlign = align;
      document.getElementById('btn-ioff').classList.toggle('active', align === 'ioff');
      document.getElementById('btn-ion').classList.toggle('active', align === 'ion');
      renderAll();
    }}

    function setSorter(sorter) {{
      currentSorter = sorter;
      document.getElementById('btn-ks4').classList.toggle('active', sorter === 'kilosort4');
      document.getElementById('btn-lupin').classList.toggle('active', sorter === 'lupin');
      renderAll();
    }}

    function setSortOrder(order) {{
      currentSortOrder = order;
      document.getElementById('btn-sort-chrono').classList.toggle('active', order === 'chrono');
      document.getElementById('btn-sort-ti').classList.toggle('active', order === 'ti');
      renderAll();
    }}

    function renderAll() {{
      const key = `${{currentSubwin}}_${{currentSorter}}`;
      const bundle = DATA[key];
      if (!bundle) return;

      const alignData = bundle[currentAlign];
      const timeAxis = alignData.axis_ms;
      let rawRaster = alignData.raster;
      let trials = alignData.trials;
      const nBreaths = bundle.n_breaths;

      // Update stat cards
      document.getElementById('stat-nbreaths').textContent = `${{nBreaths}} Breaths`;
      const meanTi = nBreaths > 0 ? (trials.reduce((acc, t) => acc + t.ti_ms, 0) / nBreaths).toFixed(1) : '0.0';
      document.getElementById('stat-ti').textContent = `${{meanTi}} ms`;
      document.getElementById('stat-align').textContent = currentAlign === 'ioff' ? '⭐ I_off (t = 0 ms)' : 'I_on (t = 0 ms)';
      document.getElementById('stat-units').textContent = `${{currentSorter === 'kilosort4' ? 'KS4' : 'Lupin'}} / ${{bundle.n_units}} Units`;

      document.getElementById('raster-subtitle').textContent =
        `(${{bundle.subwin_id}} | ${{nBreaths}} Breaths | ${{currentSorter === 'kilosort4' ? 'Kilosort 4' : 'Lupin'}} | Aligned to ${{currentAlign === 'ioff' ? 'Inspiration Offset (I_off)' : 'Inspiration Onset (I_on)'}})`;

      if (nBreaths === 0) {{
        Plotly.purge('rasterPlot');
        Plotly.purge('probPlot');
        Plotly.purge('analogPlot');
        return;
      }}

      // Handle sort order
      let rowIndices = Array.from({{ length: nBreaths }}, (_, i) => i);
      if (currentSortOrder === 'ti') {{
        rowIndices.sort((a, b) => trials[a].ti_ms - trials[b].ti_ms);
      }}

      const sortedZ = [];
      const hoverText = [];
      const markerCurveX = [];
      const markerCurveY = [];

      for (let r = 0; r < nBreaths; r++) {{
        const origIdx = rowIndices[r];
        const tr = trials[origIdx];
        sortedZ.push(rawRaster[origIdx]);

        const rowHover = [];
        for (let c = 0; c < timeAxis.length; c++) {{
          const stateVal = rawRaster[origIdx][c];
          const stName = STATE_NAMES[stateVal];
          const tMs = timeAxis[c];
          rowHover.push(
            `<b>Trial #${{tr.trial_num}}</b> (${{tr.regime}})<br>` +
            `Time: <b>${{tMs >= 0 ? '+' : ''}}${{tMs.toFixed(1)}} ms</b> rel to ${{currentAlign.toUpperCase()}}<br>` +
            `State: <b>${{stName}}</b><br>` +
            `Diaphragm T_I: <b>${{tr.ti_ms.toFixed(1)}} ms</b> | T_tot: <b>${{tr.tot_ms.toFixed(1)}} ms</b>`
          );
        }}
        hoverText.push(rowHover);

        if (currentAlign === 'ioff') {{
          markerCurveX.push(tr.ion_relative_ms);
        }} else {{
          markerCurveX.push(tr.ioff_relative_ms);
        }}
        markerCurveY.push(r + 1);
      }}

      const yTrials = Array.from({{ length: nBreaths }}, (_, i) => i + 1);

      // 1. RASTER PLOT
      const traceHeatmap = {{
        x: timeAxis,
        y: yTrials,
        z: sortedZ,
        type: 'heatmap',
        colorscale: customColorscale,
        zmin: 0,
        zmax: 4,
        showscale: false,
        hoverinfo: 'text',
        text: hoverText
      }};

      const markerLabel = currentAlign === 'ioff' ? 'Diaphragm Onset (I_on = -T_I)' : 'Flow Zero-Crossing (I_off = +T_I)';
      const markerColor = currentAlign === 'ioff' ? '#34d399' : '#fbbf24';

      const traceMarker = {{
        x: markerCurveX,
        y: markerCurveY,
        mode: 'lines+markers',
        line: {{ color: markerColor, width: 2 }},
        marker: {{ size: 4, color: markerColor }},
        name: markerLabel,
        hoverinfo: 'skip'
      }};

      const rasterLayout = {{
        margin: {{ l: 65, r: 30, t: 30, bottom: 45 }},
        paper_bgcolor: '#0f1523',
        plot_bgcolor: '#090d16',
        xaxis: {{
          title: {{ text: `Time from ${{currentAlign === 'ioff' ? 'Inspiration Offset (I_off)' : 'Inspiration Onset (I_on)'}} (ms)`, font: {{ color: '#f8fafc', size: 12, weight: 700 }} }},
          tickcolor: '#334155',
          tickfont: {{ color: '#94a3b8', size: 11 }},
          gridcolor: '#1e293b',
          zeroline: true,
          zerolinecolor: '#ffffff',
          zerolinewidth: 2
        }},
        yaxis: {{
          title: {{ text: `Breath Trial Index (${{currentSortOrder === 'chrono' ? 'Chronological 1 to ' + nBreaths : 'Sorted by T_I Duration'}})`, font: {{ color: '#f8fafc', size: 12, weight: 700 }} }},
          tickcolor: '#334155',
          tickfont: {{ color: '#94a3b8', size: 11 }},
          gridcolor: '#1e293b',
          autorange: 'reversed'
        }},
        shapes: [
          {{
            type: 'line',
            x0: 0, x1: 0,
            y0: 0.5, y1: nBreaths + 0.5,
            line: {{ color: '#ffffff', width: 2, dash: 'dash' }}
          }}
        ],
        showlegend: true,
        legend: {{
          x: 0.01, y: 0.02,
          bgcolor: 'rgba(15, 23, 42, 0.9)',
          bordercolor: '#334155',
          borderwidth: 1,
          font: {{ color: '#f8fafc', size: 11 }}
        }}
      }};

      Plotly.react('rasterPlot', [traceHeatmap, traceMarker], rasterLayout, {{ responsive: true }});

      // 2. OCCUPANCY PROBABILITY PETH PLOT
      const probTraces = [];
      for (let k = 0; k < 5; k++) {{
        probTraces.push({{
          x: timeAxis,
          y: alignData.prob[k],
          name: STATE_NAMES[k],
          mode: 'lines',
          line: {{ color: STATE_COLORS[k], width: 2.2 }},
          hoverinfo: 'x+y+name'
        }});
      }}

      const probLayout = {{
        margin: {{ l: 65, r: 30, t: 15, bottom: 40 }},
        paper_bgcolor: '#0f1523',
        plot_bgcolor: '#090d16',
        xaxis: {{
          title: {{ text: `Time from ${{currentAlign === 'ioff' ? 'I_off' : 'I_on'}} (ms)`, font: {{ color: '#f8fafc', size: 11 }} }},
          tickcolor: '#334155',
          tickfont: {{ color: '#94a3b8', size: 10 }},
          gridcolor: '#1e293b',
          zeroline: true,
          zerolinecolor: '#ffffff',
          zerolinewidth: 1.5
        }},
        yaxis: {{
          title: {{ text: 'Occupancy (%)', font: {{ color: '#f8fafc', size: 11, weight: 700 }} }},
          tickcolor: '#334155',
          tickfont: {{ color: '#94a3b8', size: 10 }},
          gridcolor: '#1e293b',
          range: [0, 102]
        }},
        shapes: [
          {{
            type: 'line',
            x0: 0, x1: 0,
            y0: 0, y1: 100,
            line: {{ color: '#ffffff', width: 1.5, dash: 'dash' }}
          }}
        ],
        showlegend: true,
        legend: {{
          orientation: 'h',
          x: 0.05, y: 1.18,
          font: {{ color: '#cbd5e1', size: 10.5 }}
        }}
      }};

      Plotly.react('probPlot', probTraces, probLayout, {{ responsive: true }});

      // 3. ANALOG WAVEFORMS PLOT (AIRFLOW & EMG)
      const traceFlow = {{
        x: timeAxis,
        y: alignData.flow_mean,
        name: 'Mean Airflow (V)',
        mode: 'lines',
        line: {{ color: '#10b981', width: 2.0 }}
      }};

      const traceEMG = {{
        x: timeAxis,
        y: alignData.emg_mean,
        name: 'Mean Diaphragm EMG Env (V)',
        yaxis: 'y2',
        mode: 'lines',
        line: {{ color: '#ec4899', width: 2.0 }}
      }};

      const analogLayout = {{
        margin: {{ l: 65, r: 65, t: 15, bottom: 40 }},
        paper_bgcolor: '#0f1523',
        plot_bgcolor: '#090d16',
        xaxis: {{
          title: {{ text: `Time from ${{currentAlign === 'ioff' ? 'I_off' : 'I_on'}} (ms)`, font: {{ color: '#f8fafc', size: 11 }} }},
          tickcolor: '#334155',
          tickfont: {{ color: '#94a3b8', size: 10 }},
          gridcolor: '#1e293b',
          zeroline: true,
          zerolinecolor: '#ffffff',
          zerolinewidth: 1.5
        }},
        yaxis: {{
          title: {{ text: 'Airflow (V)', font: {{ color: '#10b981', size: 11, weight: 700 }} }},
          tickfont: {{ color: '#10b981', size: 10 }},
          gridcolor: '#1e293b'
        }},
        yaxis2: {{
          title: {{ text: 'EMG Env (V)', font: {{ color: '#ec4899', size: 11, weight: 700 }} }},
          tickfont: {{ color: '#ec4899', size: 10 }},
          overlaying: 'y',
          side: 'right'
        }},
        shapes: [
          {{
            type: 'line',
            x0: 0, x1: 0,
            y0: -0.5, y1: 0.5,
            line: {{ color: '#ffffff', width: 1.5, dash: 'dash' }}
          }}
        ],
        showlegend: true,
        legend: {{
          orientation: 'h',
          x: 0.3, y: 1.2,
          font: {{ color: '#cbd5e1', size: 10.5 }}
        }}
      }};

      Plotly.react('analogPlot', [traceFlow, traceEMG], analogLayout, {{ responsive: true }});
    }}

    function zoomTime(minT, maxT) {{
      Plotly.relayout('rasterPlot', {{ 'xaxis.range': [minT, maxT] }});
      Plotly.relayout('probPlot', {{ 'xaxis.range': [minT, maxT] }});
      Plotly.relayout('analogPlot', {{ 'xaxis.range': [minT, maxT] }});
    }}

    function zoomTrials(minTrial, maxTrial) {{
      Plotly.relayout('rasterPlot', {{ 'yaxis.range': [maxTrial + 0.5, minTrial - 0.5] }});
    }}

    function resetZoom() {{
      Plotly.relayout('rasterPlot', {{ 'xaxis.autorange': true, 'yaxis.autorange': 'reversed' }});
      Plotly.relayout('probPlot', {{ 'xaxis.autorange': true, 'yaxis.autorange': true }});
      Plotly.relayout('analogPlot', {{ 'xaxis.autorange': true, 'yaxis.autorange': true }});
    }}

    window.addEventListener('DOMContentLoaded', () => {{
      renderAll();
    }});
  </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Master Interactive Dashboard written to {output_path} ({len(html)} bytes).")


def clean_for_json(res: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "subwin_id": str(res["subwin_id"]),
        "sorter_key": str(res["sorter_key"]),
        "sorter_label": str(res["sorter_label"]),
        "n_units": int(res["n_units"]),
        "n_breaths": int(res["n_breaths"]),
        "ioff": {
            "axis_ms": [float(x) for x in res["ioff"]["axis_ms"]],
            "raster": [[int(val) for val in row] for row in res["ioff"]["raster"]],
            "prob": [[float(val) for val in row] for row in res["ioff"]["prob"]],
            "trials": res["ioff"]["trials"],
            "flow_mean": [float(x) for x in res["ioff"]["flow_mean"]],
            "emg_mean": [float(x) for x in res["ioff"]["emg_mean"]]
        },
        "ion": {
            "axis_ms": [float(x) for x in res["ion"]["axis_ms"]],
            "raster": [[int(val) for val in row] for row in res["ion"]["raster"]],
            "prob": [[float(val) for val in row] for row in res["ion"]["prob"]],
            "trials": res["ion"]["trials"],
            "flow_mean": [float(x) for x in res["ion"]["flow_mean"]],
            "emg_mean": [float(x) for x in res["ion"]["emg_mean"]]
        }
    }


def main():
    t_start_all = time.time()
    logger.info("==========================================================================")
    logger.info("   Starting Full Population HMM & Micro-State Suite for All 7 Subwindows   ")
    logger.info("==========================================================================")

    master_payload = {}

    for sw in SUBWINDOWS:
        subwin_id = sw["id"]
        t_sub_start = time.time()
        logger.info(f"\n>>> Processing Subwindow: {subwin_id} ({sw['name']}) <<<")

        breath_data = extract_analog_and_breaths(sw["t_start"], sw["duration"], subwin_id)

        # Fit Kilosort 4
        ks4_res = fit_subwin_poisson_hmm(subwin_id, "kilosort4", "Kilosort 4 (GPU)", breath_data)
        master_payload[f"{subwin_id}_kilosort4"] = clean_for_json(ks4_res)

        # Fit Lupin
        lup_res = fit_subwin_poisson_hmm(subwin_id, "lupin", "Lupin (SVD-OMP)", breath_data)
        master_payload[f"{subwin_id}_lupin"] = clean_for_json(lup_res)

        # Output directory
        pop_dir = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / subwin_id / "population_dynamics"

        # Generate static plots
        generate_subwin_static_plots(subwin_id, sw, ks4_res, lup_res, breath_data, pop_dir)

        # Generate subwindow dashboard HTML
        dash_path = pop_dir / f"{subwin_id}_population_dynamics_dashboard.html"
        generate_subwindow_dashboard_html(subwin_id, sw, ks4_res, lup_res, breath_data, dash_path)

        # Update sorter benchmark page
        mean_ti = np.mean(breath_data["ti_arr"]) if breath_data["n_breaths"] > 0 else 0.0
        update_sorter_comparison_card(subwin_id, breath_data["n_breaths"], mean_ti)

        logger.info(f"Finished {subwin_id} in {time.time() - t_sub_start:.2f}s.")

    # Generate Master All-Subwindows Interactive Hub
    master_html_path = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / "master_interactive_population_microstate_raster.html"
    logger.info("\n>>> Generating Master 7-Subwindow Interactive Raster Hub <<<")
    build_master_interactive_raster_html(master_payload, master_html_path)

    # Also write individual interactive rasters for each subwindow using the master payload
    for sw in SUBWINDOWS:
        subwin_id = sw["id"]
        pop_dir = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / subwin_id / "population_dynamics"
        sub_html = pop_dir / f"{subwin_id}_interactive_microstate_raster.html"
        # Can write standalone or copy master
        with open(sub_html, "w", encoding="utf-8") as f:
            f.write(master_html_path.read_text(encoding="utf-8").replace("let currentSubwin = 'win1_baseline';", f"let currentSubwin = '{subwin_id}';"))
        logger.info(f"Wrote {sub_html}")

    total_time = time.time() - t_start_all
    logger.info(f"\n==========================================================================")
    logger.info(f"   ALL 7 SUBWINDOWS COMPLETED IN {total_time:.2f}s ({total_time/60.0:.2f} min)   ")
    logger.info(f"==========================================================================")


if __name__ == "__main__":
    main()
