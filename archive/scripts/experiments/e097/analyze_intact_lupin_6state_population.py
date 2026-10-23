#!/usr/bin/env python3
"""
Lupin 6-State Poisson HMM Pipeline & Cross-Sorter State Agreement on Intact 420.0s Window (e097):
- Loads 120 top QC single units sorted by Lupin (SpyKING-CIRCUS 2 + OMP-SVD)
- Fits canonical 6-state Poisson HMM on raw 10 ms unreduced spike counts (42,000 bins)
- Parallelized Subsampling Dropout (N in {120, 80, 50, 30, 15, 8, 3}, 8 draws each on 8 cores)
- Single-Neuron Mutual Information Attribution, Lorenz Curves, Gini Inequality
- Targeted Driver Ablations (Drop 10% and Drop 20%)
- Coordination & Dimensionality (Participation Ratio, PCA, Transition Co-Modulation)
- Aligned Trial-by-Trial Rasters across all 959 continuous breaths
- Cross-Sorter State Agreement against Kilosort 4 6-State Sequence (ARI, NMI, Confusion Matrix)
- Saves outputs:
    outputs/e097/intact_window/lupin_intact_6state_analysis_results.json
    outputs/e097/intact_window/cross_sorter_6state_agreement.json
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score, confusion_matrix
from hmmlearn import hmm
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.e097.run_intact_window_dual_sorting_and_hmm_pipeline import (
    INTACT_T_START, INTACT_DURATION, extract_intact_breaths, load_intact_spikes_and_qc
)
from scripts.experiments.e097.analyze_intact_kilosort4_multi_state_population import (
    CANONICAL_6STATE_NAMES, CANONICAL_6STATE_SHORT, CANONICAL_6STATE_COLORS,
    align_states_hungarian, fit_canonical_multi_state_hmm, run_parallel_subsampling,
    run_parallel_attribution_and_ablations, run_multi_state_coordination, extract_trial_rasters
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ANALYZE_LUPIN_6STATE")


def compute_cross_sorter_agreement(lupin_seq: np.ndarray, ks4_seq: np.ndarray, n_states: int = 6) -> Dict[str, Any]:
    """Computes Hungarian alignment, ARI, NMI, and per-state concordance between Lupin and KS4 state sequences."""
    logger.info("Computing cross-sorter 6-state sequence agreement (Lupin vs. Kilosort 4)...")
    mapped_lupin, ari, nmi, f1_per_state, macro_f1 = align_states_hungarian(lupin_seq, ks4_seq, n_states=n_states)
    conf = confusion_matrix(ks4_seq, mapped_lupin, labels=list(range(n_states)))
    conf_norm = conf.astype(np.float64) / (conf.sum(axis=1, keepdims=True) + 1e-12)

    agreement_pct = float(np.mean(mapped_lupin == ks4_seq) * 100.0)
    logger.info(f"Cross-Sorter Agreement: Overall Concordance = {agreement_pct:.2f}%, ARI = {ari:.4f}, NMI = {nmi:.4f}, Macro F1 = {macro_f1:.4f}")

    return {
        "overall_concordance_pct": agreement_pct,
        "ari": float(ari),
        "nmi": float(nmi),
        "macro_f1": float(macro_f1),
        "f1_per_state": [float(x) for x in f1_per_state],
        "confusion_matrix": conf.tolist(),
        "normalized_confusion_matrix": conf_norm.tolist()
    }


def main():
    out_base = REPO_ROOT / "outputs" / "e097" / "intact_window"
    lupin_dir = out_base / "lupin"
    ks4_dir = out_base / "kilosort4"
    nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")

    logger.info("Loading Lupin intact spikes and QC metadata...")
    spk_times, spk_clusters, depths, df_qc = load_intact_spikes_and_qc(lupin_dir, max_units=120)
    logger.info(f"Loaded {len(df_qc)} top QC Lupin units across intact 420s recording.")

    logger.info("Extracting continuous breaths from NIDQ analog stream...")
    breath_data = extract_intact_breaths(nidq_file, start_t=INTACT_T_START, dur=INTACT_DURATION)

    dt = 0.010
    time_bins = np.arange(0, INTACT_DURATION + dt, dt)
    n_bins = len(time_bins) - 1
    cids = df_qc["cluster_id"].values
    n_units = len(cids)

    logger.info(f"Discretizing Lupin spikes into 10 ms bins ({n_bins} bins, {n_units} units)...")
    spike_mat = np.zeros((n_bins, n_units), dtype=np.int32)
    for u_idx, cid in enumerate(cids):
        ts = spk_times[spk_clusters == cid]
        ts_valid = ts[(ts >= 0) & (ts <= INTACT_DURATION)]
        counts, _ = np.histogram(ts_valid, bins=time_bins)
        spike_mat[:, u_idx] = counts

    # 1. Fit Canonical 6-State Poisson HMM
    logger.info("\n========================================================")
    logger.info("   FITTING LUPIN 6-STATE CANONICAL POISSON HMM")
    logger.info("========================================================")
    t0 = time.time()
    hmm_res = fit_canonical_multi_state_hmm(spike_mat, n_states=6, dt=0.010, breath_data=breath_data)

    # 2. Parallel Subsampling Dropout Experiment
    sub_res = run_parallel_subsampling(spike_mat, hmm_res["state_seq"], n_states=6, n_draws=8, n_jobs=8)

    # 3. Parallel Single-Neuron Attribution & Targeted Ablations
    attrib_res = run_parallel_attribution_and_ablations(spike_mat, hmm_res["state_seq"], depths, n_states=6, dt=0.010, n_jobs=8)

    # 4. Coordination Analysis
    coord_res = run_multi_state_coordination(spike_mat, hmm_res["state_seq"], dt=0.010)

    # 5. Extract Trial-by-Trial Rasters across all 959 breaths
    rasters = extract_trial_rasters(hmm_res["state_seq"], breath_data, dt=0.010)

    # Compile Lupin 6-state results
    lupin_results = {
        "sorter": "Lupin",
        "n_states": 6,
        "canonical_names": hmm_res["names"],
        "n_units": len(df_qc),
        "duration_s": INTACT_DURATION,
        "n_breaths": breath_data["n_breaths"],
        "subsampling": sub_res,
        "attribution": attrib_res,
        "coordination": coord_res,
        "mean_state_rates_hz": (hmm_res["canon_rates"].mean(axis=1) / 0.010).tolist(),
        "state_occupancy_pct": [float(np.mean(hmm_res["state_seq"] == s) * 100.0) for s in range(6)],
        "trial_rasters": rasters,
        "state_seq": hmm_res["state_seq"].tolist()
    }

    out_lupin_json = out_base / "lupin_intact_6state_analysis_results.json"
    with open(out_lupin_json, "w", encoding="utf-8") as f:
        json.dump(lupin_results, f, indent=2)
    logger.info(f"Saved Lupin 6-state results to {out_lupin_json} ({time.time() - t0:.1f}s)")

    # 6. Cross-Sorter State Agreement against Kilosort 4
    ks4_json_path = out_base / "kilosort4_intact_6state_analysis_results.json"
    if ks4_json_path.exists():
        with open(ks4_json_path, "r", encoding="utf-8") as f:
            ks4_data = json.load(f)
        
        # Load KS4 state sequence if available or fit/predict
        if "state_seq" in ks4_data:
            ks4_seq = np.array(ks4_data["state_seq"])
        else:
            # Load KS4 spike mat and predict with KS4 model or reload
            logger.info("Loading KS4 spikes to reproduce identical 6-state sequence...")
            spk_t_ks4, spk_c_ks4, _, df_qc_ks4 = load_intact_spikes_and_qc(ks4_dir, max_units=120)
            cids_ks4 = df_qc_ks4["cluster_id"].values
            mat_ks4 = np.zeros((n_bins, len(cids_ks4)), dtype=np.int32)
            for u_idx, cid in enumerate(cids_ks4):
                ts = spk_t_ks4[spk_c_ks4 == cid]
                ts_v = ts[(ts >= 0) & (ts <= INTACT_DURATION)]
                c_v, _ = np.histogram(ts_v, bins=time_bins)
                mat_ks4[:, u_idx] = c_v
            
            ks4_fit = fit_canonical_multi_state_hmm(mat_ks4, n_states=6, dt=0.010, breath_data=breath_data)
            ks4_seq = ks4_fit["state_seq"]

        agreement = compute_cross_sorter_agreement(hmm_res["state_seq"], ks4_seq, n_states=6)
        out_agr_json = out_base / "cross_sorter_6state_agreement.json"
        with open(out_agr_json, "w", encoding="utf-8") as f:
            json.dump(agreement, f, indent=2)
        logger.info(f"Saved cross-sorter 6-state agreement to {out_agr_json}")

    logger.info("\nLUPIN 6-STATE PIPELINE COMPLETED SUCCESSFULLY!")


if __name__ == '__main__':
    main()
