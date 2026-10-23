#!/usr/bin/env python3
"""
Dedicated analysis engine for Intact Window Kilosort 4 (1740s - 2160s / 420s):
- Fits 5-State Poisson HMM on raw 10 ms unreduced spike counts (120 top QC units)
- Resolves Canonical State Taxonomy:
    State 1: Insp Burst (#10b981)
    State 2: Peak / Off-Switch Quench (#f97316)
    State 3: Post-Insp / Expiratory Drive (#38bdf8)
    State 4: Pre-Insp Priming (#a855f7)
    State 5: Inter-Breath Pause (#64748b)
- Performs Drop-Out Robustness (N in {120, 80, 50, 30, 15, 8, 3}, 8 draws each)
- Performs Single-Neuron Attribution, Gini Inequality, Lorenz Curves
- Performs Targeted Ablations (Top 10/20%, Random 10/20%, Bottom 10/20%)
- Performs Ensemble Dimensionality (Participation Ratio, PCA, Co-Modulation)
- Extracts Trial-by-Trial Microstate Rasters aligned to both I_off and I_on across all 959 breaths
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

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.e097.run_intact_window_dual_sorting_and_hmm_pipeline import (
    INTACT_T_START, INTACT_DURATION, STAGES, CANONICAL_STATE_NAMES, CANONICAL_COLORS,
    extract_intact_breaths, load_intact_spikes_and_qc, fit_intact_poisson_hmm,
    run_intact_subsampling_experiment, run_intact_single_neuron_attribution,
    run_intact_coordination_analysis
)
from scripts.visualization.plot_interactive_sorter_raster import build_interactive_sorter_raster

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ANALYZE_INTACT_KS4")

def main():
    out_base = REPO_ROOT / "outputs" / "e097" / "intact_window"
    ks4_dir = out_base / "kilosort4"
    alf_dir = ks4_dir / "alf"
    nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")

    if not (alf_dir / "spike_times.npy").exists():
        logger.error(f"Kilosort 4 ALF files not found at {alf_dir}!")
        return

    logger.info("Loading Kilosort 4 intact spikes and QC metadata...")
    spk_times, spk_clusters, depths, df_qc = load_intact_spikes_and_qc(ks4_dir, max_units=120)
    logger.info(f"Loaded {len(df_qc)} top QC units across intact 420s recording.")

    # 1. Build interactive raster if not present
    raster_html = ks4_dir / "kilosort4_interactive_raster.html"
    if not raster_html.exists():
        logger.info(f"Building interactive raster: {raster_html}...")
        try:
            build_interactive_sorter_raster(
                sorter_dir=ks4_dir,
                sorter_name="kilosort4",
                start_t_s=INTACT_T_START
            )
        except Exception as e:
            logger.warning(f"Failed to build interactive sorter raster: {e}")

    # 2. Extract breaths across intact 420s window
    breath_data = extract_intact_breaths(nidq_file, start_t=INTACT_T_START, dur=INTACT_DURATION)

    # 3. Fit 5-State Poisson HMM
    hmm_res = fit_intact_poisson_hmm(spk_times, spk_clusters, df_qc, dur=INTACT_DURATION, dt=0.010, n_states=5)

    # 4. Run Subsampling Dropout Experiment
    sub_res = run_intact_subsampling_experiment(hmm_res["spike_mat"], hmm_res["state_seq"], n_draws=8)

    # 5. Run Single-Neuron Attribution & Ablations
    attrib_res = run_intact_single_neuron_attribution(hmm_res["spike_mat"], hmm_res["state_seq"], depths, dt=0.010)

    # 6. Run Coordination Analysis
    coord_res = run_intact_coordination_analysis(hmm_res["spike_mat"], hmm_res["state_seq"], dt=0.010)

    # 7. Trial-by-Trial Microstate Raster Extraction (aligned to I_off and I_on)
    logger.info("Extracting trial-by-trial microstate rasters across all breaths...")
    dt = hmm_res["dt"]
    valid_onsets = breath_data["valid_onsets"]
    valid_offsets = breath_data["valid_offsets"]
    n_breaths = breath_data["n_breaths"]
    state_seq = hmm_res["state_seq"]

    # Align to I_off: [-200 ms, +600 ms]
    t_pre_off = 0.200
    t_post_off = 0.600
    win_len_off = int((t_pre_off + t_post_off) / dt)
    raster_off = np.full((n_breaths, win_len_off), -1, dtype=np.int32)

    # Align to I_on: [-200 ms, +600 ms]
    t_pre_on = 0.200
    t_post_on = 0.600
    win_len_on = int((t_pre_on + t_post_on) / dt)
    raster_on = np.full((n_breaths, win_len_on), -1, dtype=np.int32)

    for b_idx in range(n_breaths):
        # I_off alignment
        off_t = valid_offsets[b_idx]
        b0_off = int((off_t - t_pre_off) / dt)
        b1_off = b0_off + win_len_off
        if b0_off >= 0 and b1_off <= len(state_seq):
            raster_off[b_idx, :] = state_seq[b0_off:b1_off]

        # I_on alignment
        on_t = valid_onsets[b_idx]
        b0_on = int((on_t - t_pre_on) / dt)
        b1_on = b0_on + win_len_on
        if b0_on >= 0 and b1_on <= len(state_seq):
            raster_on[b_idx, :] = state_seq[b0_on:b1_on]

    # Save complete JSON results
    ks4_results = {
        "n_units": len(df_qc),
        "duration_s": INTACT_DURATION,
        "n_breaths": n_breaths,
        "subsampling": sub_res,
        "attribution": attrib_res,
        "coordination": coord_res,
        "mean_state_rates_hz": (hmm_res["canon_rates"].mean(axis=1) / dt).tolist(),
        "state_occupancy_pct": [float(np.mean(state_seq == s) * 100.0) for s in range(5)],
        "trial_rasters": {
            "raster_off": raster_off.tolist(),
            "raster_on": raster_on.tolist(),
            "breath_stages": breath_data["breath_stages"].tolist(),
            "ti_arr": breath_data["ti_arr"].tolist(),
            "te_arr": breath_data["te_arr"].tolist(),
            "tot_arr": breath_data["tot_arr"].tolist()
        }
    }

    out_json = out_base / "kilosort4_intact_analysis_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ks4_results, f, indent=2)
    logger.info(f"Saved Kilosort 4 intact analysis results to {out_json}")

if __name__ == '__main__':
    main()
