#!/usr/bin/env python3
"""
Multi-State Poisson HMM Engine (6-State and 7-State) on Intact 420.0s Window (e097):
- Fits 6-State and 7-State Poisson HMMs on raw 10 ms unreduced spike counts (120 top QC units)
- Canonical State Mapping & Resolving Physiological Taxonomies
- Parallelized Subsampling Dropout Experiment (N in {120, 80, 50, 30, 15, 8, 3}, 8 draws each, 8 cores)
- Single-Neuron Mutual Information Attribution, Gini Inequality, Lorenz Curves
- Parallelized Targeted Driver Ablations (Top 10/20%, Random 10/20%, Bottom 10/20%)
- Ensemble Dimensionality (Participation Ratio, PCA, Co-Modulation)
- Trial-by-Trial Microstate Rasters aligned to both I_off and I_on across all 959 breaths
- Generates JSON outputs:
    outputs/e097/intact_window/kilosort4_intact_6state_analysis_results.json
    outputs/e097/intact_window/kilosort4_intact_7state_analysis_results.json
"""
import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import sem
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
from hmmlearn import hmm
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.e097.run_intact_window_dual_sorting_and_hmm_pipeline import (
    INTACT_T_START, INTACT_DURATION, extract_intact_breaths, load_intact_spikes_and_qc
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ANALYZE_MULTI_STATE_HMM")

# Canonical State Taxonomy for 6-State Model:
CANONICAL_6STATE_NAMES = [
    "State 1: Insp Burst (Peak Drive)",
    "State 2: Post-Insp Deceleration (E1 / Post-I)",
    "State 3: Peak / Off-Switch Quench (I_off)",
    "State 4: Late Expiratory Drive (E2)",
    "State 5: Pre-Insp Priming (Pre-I)",
    "State 6: Inter-Breath Pause (Quiescence)"
]

CANONICAL_6STATE_SHORT = [
    "Insp Burst",
    "Post-I (E1)",
    "Off-Switch (I_off)",
    "Late Exp (E2)",
    "Pre-I Priming",
    "Pause / Quiescence"
]

CANONICAL_6STATE_COLORS = [
    "#10b981", # Emerald
    "#06b6d4", # Cyan
    "#f97316", # Orange
    "#38bdf8", # Sky blue
    "#a855f7", # Purple
    "#64748b"  # Slate grey
]

# Canonical State Taxonomy for 7-State Model:
CANONICAL_7STATE_NAMES = [
    "State 1: Insp Burst (Peak Drive)",
    "State 2: Early Off-Switch Transition",
    "State 3: Off-Switch Quench / Post-I Deceleration",
    "State 4: Early Expiratory Plateau (E1)",
    "State 5: Late Expiratory Drive (E2)",
    "State 6: Pre-Insp Priming (Pre-I)",
    "State 7: Inter-Breath Pause (Quiescence)"
]

CANONICAL_7STATE_SHORT = [
    "Insp Burst",
    "Early Off-Switch",
    "Quench / Post-I",
    "Early Exp (E1)",
    "Late Exp (E2)",
    "Pre-I Priming",
    "Pause / Quiescence"
]

CANONICAL_7STATE_COLORS = [
    "#10b981", # Emerald
    "#eab308", # Yellow / Amber
    "#f97316", # Orange
    "#06b6d4", # Cyan
    "#38bdf8", # Sky blue
    "#a855f7", # Purple
    "#64748b"  # Slate grey
]

def align_states_hungarian(pred: np.ndarray, true: np.ndarray, n_states: int) -> Tuple[np.ndarray, float, float, np.ndarray, float]:
    """Bijective Hungarian state alignment against ground-truth intact trajectory."""
    cost_matrix = np.zeros((n_states, n_states), dtype=np.int32)
    for i in range(n_states):
        for j in range(n_states):
            cost_matrix[i, j] = -np.sum((pred == i) & (true == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    mapped_pred = np.array([mapping.get(s, s) for s in pred])
    ari = adjusted_rand_score(true, mapped_pred)
    nmi = normalized_mutual_info_score(true, mapped_pred)
    f1_per_state = f1_score(true, mapped_pred, average=None, labels=list(range(n_states)))
    macro_f1 = f1_score(true, mapped_pred, average="macro")
    return mapped_pred, ari, nmi, f1_per_state, macro_f1


def fit_canonical_multi_state_hmm(spike_mat: np.ndarray, n_states: int, dt: float = 0.010, breath_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Fits Poisson HMM with n_states and resolves canonical physiological state mapping."""
    logger.info(f"Fitting {n_states}-state Poisson HMM on raw spike counts (shape={spike_mat.shape})...")
    model = hmm.PoissonHMM(n_components=n_states, n_iter=120, tol=1e-3, random_state=42, verbose=False)
    model.fit(spike_mat)

    state_seq = model.predict(spike_mat)
    post_probs = model.predict_proba(spike_mat)
    rates = model.lambdas_ # (n_states, n_units)
    mean_state_rates = rates.mean(axis=1) / dt # Hz
    n_bins = spike_mat.shape[0]

    # Resolve canonical state mapping based on phase and transitions
    valid_on = breath_data["valid_onsets"]
    valid_off = breath_data["valid_offsets"]

    on_bins = []
    for t in valid_on:
        b = int(t / dt)
        if 0 <= b < n_bins:
            on_bins.extend(range(max(0, b - 5), min(n_bins, b + 5)))
    on_bins = np.unique(on_bins)

    off_bins = []
    for t in valid_off:
        b = int(t / dt)
        if 0 <= b < n_bins:
            off_bins.extend(range(max(0, b - 5), min(n_bins, b + 5)))
    off_bins = np.unique(off_bins)

    on_prefs = [np.mean(state_seq[on_bins] == s) * 100.0 for s in range(n_states)]
    off_prefs = [np.mean(state_seq[off_bins] == s) * 100.0 for s in range(n_states)]
    trans = model.transmat_

    if n_states == 6:
        # 6-State Canonical Mapping:
        # 0: Insp Peak (highest rate)
        # 1: Post-I / E1 (transitions from Insp, moderate off_pref)
        # 2: Off-Switch Quench (high off_pref, transitions from Post-I)
        # 3: Late Exp E2 (transitions from Off-switch, high on_pref)
        # 4: Pre-I (transitions into Insp, highest on_pref with high rate)
        # 5: Pause (lowest rate)
        s_insp = int(np.argmax(mean_state_rates))
        s_pause = int(np.argmin(mean_state_rates))
        remaining = [s for s in range(6) if s not in (s_insp, s_pause)]

        # Pre-I has highest transition into Insp
        pre_i_cands = sorted(remaining, key=lambda s: trans[s, s_insp], reverse=True)
        s_pre_i = pre_i_cands[0]
        remaining = [s for s in remaining if s != s_pre_i]

        # Post-I has highest transition from Insp
        post_i_cands = sorted(remaining, key=lambda s: trans[s_insp, s], reverse=True)
        s_post_i = post_i_cands[0]
        remaining = [s for s in remaining if s != s_post_i]

        # Off-switch has higher off_pref than remaining late exp
        off_cands = sorted(remaining, key=lambda s: off_prefs[s], reverse=True)
        s_off = off_cands[0]
        s_late_exp = off_cands[1]

        raw_to_canon = {
            s_insp: 0,
            s_post_i: 1,
            s_off: 2,
            s_late_exp: 3,
            s_pre_i: 4,
            s_pause: 5
        }
        names = CANONICAL_6STATE_NAMES

    elif n_states == 7:
        # 7-State Canonical Mapping:
        # 0: Insp Peak (highest rate)
        # 1: Early Off-Switch (transitions from Insp, rate ~20Hz)
        # 2: Quench / Post-I (transitions from Early Off-Switch, high off_pref)
        # 3: Early Exp E1 (moderate rate, transitions from Quench)
        # 4: Late Exp E2 (high on_pref)
        # 5: Pre-I (transitions into Insp, high on_pref)
        # 6: Pause (lowest rate)
        s_insp = int(np.argmax(mean_state_rates))
        s_pause = int(np.argmin(mean_state_rates))
        remaining = [s for s in range(7) if s not in (s_insp, s_pause)]

        # Pre-I transitions into Insp
        pre_i_cands = sorted(remaining, key=lambda s: trans[s, s_insp], reverse=True)
        s_pre_i = pre_i_cands[0]
        remaining = [s for s in remaining if s != s_pre_i]

        # Early Off-Switch transitions from Insp
        off_init_cands = sorted(remaining, key=lambda s: trans[s_insp, s], reverse=True)
        s_off_init = off_init_cands[0]
        remaining = [s for s in remaining if s != s_off_init]

        # Quench transitions from Early Off-Switch
        quench_cands = sorted(remaining, key=lambda s: trans[s_off_init, s], reverse=True)
        s_quench = quench_cands[0]
        remaining = [s for s in remaining if s != s_quench]

        # Late Exp has higher on_pref than Early Exp
        late_cands = sorted(remaining, key=lambda s: on_prefs[s], reverse=True)
        s_late_exp = late_cands[0]
        s_early_exp = late_cands[1]

        raw_to_canon = {
            s_insp: 0,
            s_off_init: 1,
            s_quench: 2,
            s_early_exp: 3,
            s_late_exp: 4,
            s_pre_i: 5,
            s_pause: 6
        }
        names = CANONICAL_7STATE_NAMES

    canon_state_seq = np.array([raw_to_canon[s] for s in state_seq])
    canon_post_probs = np.zeros_like(post_probs)
    for raw_idx, canon_idx in raw_to_canon.items():
        canon_post_probs[:, canon_idx] = post_probs[:, raw_idx]

    canon_rates = np.zeros_like(rates)
    for raw_idx, canon_idx in raw_to_canon.items():
        canon_rates[canon_idx, :] = rates[raw_idx, :]

    logger.info(f"Canonical {n_states}-state mapping resolved:")
    for c_i, name in enumerate(names):
        occ = float(np.mean(canon_state_seq == c_i) * 100.0)
        rate = float(canon_rates[c_i].mean() / dt)
        logger.info(f"  {name}: {occ:.1f}% occupancy, rate = {rate:.1f} Hz")

    return {
        "model": model,
        "dt": dt,
        "n_states": n_states,
        "spike_mat": spike_mat,
        "state_seq": canon_state_seq,
        "post_probs": canon_post_probs,
        "canon_rates": canon_rates,
        "raw_to_canon": raw_to_canon,
        "names": names
    }


def _fit_one_subsample(seed: int, N: int, total_units: int, spike_mat: np.ndarray, true_states: np.ndarray, n_states: int) -> Tuple[float, float, float, np.ndarray]:
    """Helper for parallel execution of a single subsample draw."""
    rng = np.random.RandomState(42 + seed * 17)
    sub_units = rng.choice(total_units, size=N, replace=False)
    sub_mat = spike_mat[:, sub_units]
    sub_model = hmm.PoissonHMM(n_components=n_states, n_iter=50, tol=1e-3, random_state=seed, verbose=False)
    sub_model.fit(sub_mat)
    pred_raw = sub_model.predict(sub_mat)
    _, ari, nmi, f1_per_st, mf1 = align_states_hungarian(pred_raw, true_states, n_states=n_states)
    return ari, nmi, mf1, f1_per_st


def run_parallel_subsampling(spike_mat: np.ndarray, true_states: np.ndarray, n_states: int, n_draws: int = 8, n_jobs: int = 8) -> Dict[str, Any]:
    """Evaluates subsampling robustness across N in {120, 80, 50, 30, 15, 8, 3} using joblib parallelization."""
    total_units = spike_mat.shape[1]
    tiers = [u for u in [120, 80, 50, 30, 15, 8, 3] if u <= total_units]
    if total_units not in tiers:
        tiers.insert(0, total_units)

    results_by_size = []
    logger.info(f"Executing Parallel Subsampling Dropout for {n_states}-state model across population sizes...")

    for N in tiers:
        t0 = time.time()
        logger.info(f"  Testing Subsample Size N = {N} ({n_draws} draws on {n_jobs} cores)...")

        if N == total_units:
            results_by_size.append({
                "N": N,
                "fraction": 1.0,
                "ari_mean": 1.0,
                "ari_std": 0.0,
                "ari_sem": 0.0,
                "macro_f1_mean": 1.0,
                "macro_f1_sem": 0.0,
                "nmi_mean": 1.0,
                "state_f1_mean": [1.0] * n_states,
                "state_f1_sem": [0.0] * n_states
            })
            continue

        raw_results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_one_subsample)(seed, N, total_units, spike_mat, true_states, n_states)
            for seed in range(n_draws)
        )

        ari_draws = [r[0] for r in raw_results]
        nmi_draws = [r[1] for r in raw_results]
        macro_f1_draws = [r[2] for r in raw_results]
        state_f1_arr = np.array([r[3] for r in raw_results])

        dt_s = time.time() - t0
        logger.info(f"    Finished N={N} in {dt_s:.1f}s: ARI = {np.mean(ari_draws):.3f} ± {sem(ari_draws):.3f}")

        results_by_size.append({
            "N": N,
            "fraction": float(N / total_units),
            "ari_mean": float(np.mean(ari_draws)),
            "ari_std": float(np.std(ari_draws)),
            "ari_sem": float(sem(ari_draws) if len(ari_draws) > 1 else 0.0),
            "macro_f1_mean": float(np.mean(macro_f1_draws)),
            "macro_f1_sem": float(sem(macro_f1_draws) if len(macro_f1_draws) > 1 else 0.0),
            "nmi_mean": float(np.mean(nmi_draws)),
            "state_f1_mean": [float(x) for x in np.mean(state_f1_arr, axis=0)],
            "state_f1_sem": [float(x) for x in (sem(state_f1_arr, axis=0) if len(ari_draws) > 1 else np.zeros(n_states))]
        })

    n_star_80 = total_units
    n_star_90 = total_units
    for res in sorted(results_by_size, key=lambda x: x["N"]):
        if res["ari_mean"] >= 0.80 and n_star_80 == total_units:
            n_star_80 = res["N"]
        if res["ari_mean"] >= 0.90 and n_star_90 == total_units:
            n_star_90 = res["N"]

    return {
        "results_by_size": results_by_size,
        "n_star_80": n_star_80,
        "n_star_90": n_star_90
    }


def _fit_ablation_model(keep_idx: np.ndarray, spike_mat: np.ndarray, true_states: np.ndarray, n_states: int, seed: int) -> Tuple[float, float]:
    sub_mat = spike_mat[:, keep_idx]
    m = hmm.PoissonHMM(n_components=n_states, n_iter=60, tol=1e-3, random_state=seed, verbose=False)
    m.fit(sub_mat)
    p = m.predict(sub_mat)
    _, ari, _, _, f1 = align_states_hungarian(p, true_states, n_states=n_states)
    return float(ari), float(f1)


def run_parallel_attribution_and_ablations(spike_mat: np.ndarray, true_states: np.ndarray, depths: np.ndarray, n_states: int, dt: float = 0.010, n_jobs: int = 8) -> Dict[str, Any]:
    """Single-neuron mutual information attribution, Lorenz inequality, and parallelized targeted ablations."""
    total_units = spike_mat.shape[1]
    logger.info(f"Computing Single-Neuron Mutual Information Attribution for {n_states}-state model...")

    unit_stats = []
    mi_list = []
    for u_idx in range(total_units):
        u_spk = spike_mat[:, u_idx]
        mean_fr = float(np.mean(u_spk) / dt)
        mi_bits = float(normalized_mutual_info_score(u_spk, true_states))
        mi_list.append(mi_bits)
        unit_stats.append({
            "unit_idx": int(u_idx),
            "depth_um": float(depths[u_idx]) if u_idx < len(depths) else 0.0,
            "mean_fr": mean_fr,
            "mi_bits": mi_bits
        })

    mi_arr = np.array(mi_list)
    mi_sorted = np.sort(mi_arr)
    n = len(mi_sorted)
    index = np.arange(1, n + 1)
    gini = float((np.sum((2 * index - n - 1) * mi_sorted)) / (n * np.sum(mi_sorted) + 1e-12))

    cum_units = np.linspace(0, 1, n + 1).tolist()
    cum_info = (np.concatenate(([0.0], np.cumsum(mi_sorted))) / (np.sum(mi_sorted) + 1e-12)).tolist()

    sort_indices = np.argsort(mi_arr)
    top_indices = sort_indices[::-1]
    bottom_indices = sort_indices

    ablation_results = []
    for drop_frac in [0.10, 0.20]:
        k_drop = max(1, int(round(drop_frac * total_units)))
        logger.info(f"  Testing Ablation: Dropping {k_drop} units ({int(drop_frac*100)}%)...")

        keep_no_top = np.setdiff1d(np.arange(total_units), top_indices[:k_drop])
        keep_no_bot = np.setdiff1d(np.arange(total_units), bottom_indices[:k_drop])

        # Random sets (5 draws)
        rand_sets = []
        for r_seed in range(5):
            rng = np.random.RandomState(100 + r_seed)
            drop_r = rng.choice(total_units, size=k_drop, replace=False)
            rand_sets.append(np.setdiff1d(np.arange(total_units), drop_r))

        # Parallelize all 7 fits (Top, Bottom, 5 Random)
        tasks = [
            (keep_no_top, 42),
            (keep_no_bot, 42),
            *[(r_idx, r_seed) for r_seed, r_idx in enumerate(rand_sets)]
        ]

        fit_outs = Parallel(n_jobs=min(len(tasks), n_jobs))(
            delayed(_fit_ablation_model)(keep_idx, spike_mat, true_states, n_states, s_val)
            for keep_idx, s_val in tasks
        )

        ari_top, f1_top = fit_outs[0]
        ari_bot, f1_bot = fit_outs[1]
        rand_aris = [fo[0] for fo in fit_outs[2:]]
        rand_f1s = [fo[1] for fo in fit_outs[2:]]

        ablation_results.append({
            "frac": drop_frac,
            "k_dropped": k_drop,
            "ari_no_top": float(ari_top),
            "f1_no_top": float(f1_top),
            "ari_no_bottom": float(ari_bot),
            "f1_no_bottom": float(f1_bot),
            "ari_no_random_mean": float(np.mean(rand_aris)),
            "ari_no_random_sem": float(sem(rand_aris) if len(rand_aris) > 1 else 0.0),
            "f1_no_random_mean": float(np.mean(rand_f1s)),
            "f1_no_random_sem": float(sem(rand_f1s) if len(rand_f1s) > 1 else 0.0)
        })

    return {
        "unit_stats": unit_stats,
        "gini_mi": float(gini),
        "lorenz_cum_units": cum_units,
        "lorenz_cum_info": cum_info,
        "ablation_results": ablation_results
    }


def run_multi_state_coordination(spike_mat: np.ndarray, true_states: np.ndarray, dt: float = 0.010) -> Dict[str, Any]:
    """Computes Participation Ratio, PCA variance spectrum, and transition co-modulation fractions."""
    total_units = spike_mat.shape[1]
    box_pts = int(0.050 / dt)
    smoothed = np.zeros_like(spike_mat, dtype=np.float32)
    for u in range(total_units):
        smoothed[:, u] = np.convolve(spike_mat[:, u], np.ones(box_pts)/box_pts, mode='same')

    cov = np.cov(smoothed, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    eigvals = eigvals[eigvals > 0]
    total_var = np.sum(eigvals)

    pr = float((total_var ** 2) / np.sum(eigvals ** 2))
    pr_frac = float(pr / total_units)

    top1_pct = float(eigvals[0] / total_var * 100.0)
    top3_pct = float(np.sum(eigvals[:3]) / total_var * 100.0)
    top5_pct = float(np.sum(eigvals[:5]) / total_var * 100.0)

    transitions = np.where(np.diff(true_states) != 0)[0]
    w_bins = int(0.050 / dt)
    comod_fractions = []
    upmod_fractions = []
    downmod_fractions = []
    unit_stds = np.std(smoothed, axis=0) + 1e-6

    for t_idx in transitions:
        if t_idx < w_bins or t_idx > len(true_states) - w_bins:
            continue
        pre = smoothed[t_idx - w_bins : t_idx, :]
        post = smoothed[t_idx : t_idx + w_bins, :]
        diff_z = (np.mean(post, axis=0) - np.mean(pre, axis=0)) / unit_stds

        n_up = np.sum(diff_z > 2.0)
        n_down = np.sum(diff_z < -2.0)
        comod_fractions.append((n_up + n_down) / total_units)
        upmod_fractions.append(n_up / total_units)
        downmod_fractions.append(n_down / total_units)

    return {
        "participation_ratio": pr,
        "pr_fraction": pr_frac,
        "top1_pct": top1_pct,
        "top3_pct": top3_pct,
        "top5_pct": top5_pct,
        "mean_comod_fraction": float(np.mean(comod_fractions)) if comod_fractions else 0.0,
        "mean_upmod_fraction": float(np.mean(upmod_fractions)) if upmod_fractions else 0.0,
        "mean_downmod_fraction": float(np.mean(downmod_fractions)) if downmod_fractions else 0.0,
        "n_transitions": len(transitions)
    }


def extract_trial_rasters(true_states: np.ndarray, breath_data: Dict[str, Any], dt: float = 0.010) -> Dict[str, Any]:
    valid_onsets = breath_data["valid_onsets"]
    valid_offsets = breath_data["valid_offsets"]
    n_breaths = breath_data["n_breaths"]

    t_pre_off = 0.200
    t_post_off = 0.600
    win_len_off = int((t_pre_off + t_post_off) / dt)
    raster_off = np.full((n_breaths, win_len_off), -1, dtype=np.int32)

    t_pre_on = 0.200
    t_post_on = 0.600
    win_len_on = int((t_pre_on + t_post_on) / dt)
    raster_on = np.full((n_breaths, win_len_on), -1, dtype=np.int32)

    for b_idx in range(n_breaths):
        off_t = valid_offsets[b_idx]
        b0_off = int((off_t - t_pre_off) / dt)
        b1_off = b0_off + win_len_off
        if b0_off >= 0 and b1_off <= len(true_states):
            raster_off[b_idx, :] = true_states[b0_off:b1_off]

        on_t = valid_onsets[b_idx]
        b0_on = int((on_t - t_pre_on) / dt)
        b1_on = b0_on + win_len_on
        if b0_on >= 0 and b1_on <= len(true_states):
            raster_on[b_idx, :] = true_states[b0_on:b1_on]

    return {
        "raster_off": raster_off.tolist(),
        "raster_on": raster_on.tolist(),
        "breath_stages": breath_data["breath_stages"].tolist(),
        "ti_arr": breath_data["ti_arr"].tolist(),
        "te_arr": breath_data["te_arr"].tolist(),
        "tot_arr": breath_data["tot_arr"].tolist()
    }


def run_pipeline_for_state_order(k: int, spike_mat: np.ndarray, depths: np.ndarray, df_qc: pd.DataFrame, breath_data: Dict[str, Any], out_base: Path):
    logger.info(f"\n========================================================")
    logger.info(f"   STARTING COMPLETE ANALYSIS FOR K = {k} STATES")
    logger.info(f"========================================================")
    t_start = time.time()

    # 1. Fit Canonical Model
    hmm_res = fit_canonical_multi_state_hmm(spike_mat, n_states=k, dt=0.010, breath_data=breath_data)

    # 2. Parallel Subsampling Dropout
    sub_res = run_parallel_subsampling(spike_mat, hmm_res["state_seq"], n_states=k, n_draws=8, n_jobs=8)

    # 3. Parallel Single-Neuron Attribution & Ablations
    attrib_res = run_parallel_attribution_and_ablations(spike_mat, hmm_res["state_seq"], depths, n_states=k, dt=0.010, n_jobs=8)

    # 4. Coordination Analysis
    coord_res = run_multi_state_coordination(spike_mat, hmm_res["state_seq"], dt=0.010)

    # 5. Extract Trial-by-Trial Rasters across all 959 breaths
    rasters = extract_trial_rasters(hmm_res["state_seq"], breath_data, dt=0.010)

    # Compile Results
    results = {
        "n_states": k,
        "canonical_names": hmm_res["names"],
        "n_units": len(df_qc),
        "duration_s": INTACT_DURATION,
        "n_breaths": breath_data["n_breaths"],
        "subsampling": sub_res,
        "attribution": attrib_res,
        "coordination": coord_res,
        "mean_state_rates_hz": (hmm_res["canon_rates"].mean(axis=1) / 0.010).tolist(),
        "state_occupancy_pct": [float(np.mean(hmm_res["state_seq"] == s) * 100.0) for s in range(k)],
        "trial_rasters": rasters
    }

    out_json = out_base / f"kilosort4_intact_{k}state_analysis_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - t_start
    logger.info(f"Successfully finished complete {k}-state pipeline in {total_time:.1f}s ({total_time/60:.2f} min).")
    logger.info(f"Saved results to: {out_json}")
    return results


def main():
    out_base = REPO_ROOT / "outputs" / "e097" / "intact_window"
    ks4_dir = out_base / "kilosort4"
    nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")

    logger.info("Loading spikes and extracting breaths...")
    spk_times, spk_clusters, depths, df_qc = load_intact_spikes_and_qc(ks4_dir, max_units=120)
    breath_data = extract_intact_breaths(nidq_file, start_t=INTACT_T_START, dur=INTACT_DURATION)

    dt = 0.010
    time_bins = np.arange(0, INTACT_DURATION + dt, dt)
    n_bins = len(time_bins) - 1
    cids = df_qc["cluster_id"].values
    n_units = len(cids)

    logger.info(f"Discretizing spikes into 10 ms bins ({n_bins} bins, {n_units} units)...")
    spike_mat = np.zeros((n_bins, n_units), dtype=np.int32)
    for u_idx, cid in enumerate(cids):
        ts = spk_times[spk_clusters == cid]
        ts_valid = ts[(ts >= 0) & (ts <= INTACT_DURATION)]
        counts, _ = np.histogram(ts_valid, bins=time_bins)
        spike_mat[:, u_idx] = counts

    # Execute 6-State Pipeline
    res6 = run_pipeline_for_state_order(6, spike_mat, depths, df_qc, breath_data, out_base)

    # Execute 7-State Pipeline
    res7 = run_pipeline_for_state_order(7, spike_mat, depths, df_qc, breath_data, out_base)

    logger.info("\nALL MULTI-STATE PIPELINES (6-State & 7-State) COMPLETED SUCCESSFULLY!")

if __name__ == '__main__':
    main()
