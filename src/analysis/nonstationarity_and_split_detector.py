#!/usr/bin/env python3
"""
Nonstationarity Quantification & Cross-Cluster Split-Pair Discovery Engine

Identifies potential spike sorting split artifacts caused by uncorrected or residual
probe motion vs. true physiological nonstationarities.

Key Capabilities:
1. Combinatorial spatial neighbor analysis (depth difference <= 75 um).
2. Temporal Complementarity Index (TC): Detects abrupt on/off reciprocity across epochs.
3. Cross-Cluster Trial-to-Trial Schreiber Similarity: Compares trial spike trains of Cluster A
   during its active epoch with Cluster B during its active epoch using Gaussian smoothing.
4. Phase Tuning & PETH Congruence: Verifies identical respiratory phase locking.
5. Composite Split-Candidate Scoring & Classification.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.analysis.epoch_spiking_statistics import convolve_trials_gaussian, safe_matmul

def compute_cross_cluster_trial_similarity(
    trial_counts_a: np.ndarray,
    trial_counts_b: np.ndarray,
    dt_s: float = 0.002,
    sigma_s: float = 0.025
) -> Tuple[float, float, float]:
    """
    Computes cross-cluster trial similarity between active trials of Cluster A
    and active trials of Cluster B using smoothed Gaussian rate vectors.
    
    Returns:
    - mean_schreiber_cross: Average cosine correlation across cross-cluster trial pairs in [0, 1].
    - peth_correlation: Pearson r between the trial-averaged PETH profiles.
    - van_rossum_cross: Average Euclidean distance across cross-cluster trial pairs.
    """
    if len(trial_counts_a) == 0 or len(trial_counts_b) == 0:
        return 0.0, 0.0, 999.0

    # Smooth trials
    sm_a = convolve_trials_gaussian(trial_counts_a, dt_s, sigma_s)
    sm_b = convolve_trials_gaussian(trial_counts_b, dt_s, sigma_s)

    # Norms
    norms_a = np.sqrt(np.sum(sm_a ** 2, axis=1) * dt_s)
    norms_b = np.sqrt(np.sum(sm_b ** 2, axis=1) * dt_s)

    # Inner products (Gram cross matrix)
    G = safe_matmul(sm_a, sm_b.T) * dt_s

    # Schreiber similarity
    outer_norm = np.outer(norms_a, norms_b)
    valid_mask = outer_norm > 1e-12
    R_cross = np.zeros_like(G, dtype=np.float32)
    np.divide(G, outer_norm, out=R_cross, where=valid_mask)
    R_cross = np.clip(R_cross, 0.0, 1.0)

    # Filter to pairs where both trials spiked
    active_pairs = (norms_a[:, None] > 1e-6) & (norms_b[None, :] > 1e-6)
    if np.sum(active_pairs) > 0:
        mean_schreiber_cross = float(np.mean(R_cross[active_pairs]))
    else:
        mean_schreiber_cross = 0.0

    # van Rossum cross distance
    norms_sq_a = np.sum(sm_a ** 2, axis=1) * dt_s
    norms_sq_b = np.sum(sm_b ** 2, axis=1) * dt_s
    D_sq = norms_sq_a[:, None] + norms_sq_b[None, :] - 2.0 * G
    D_sq = np.maximum(D_sq, 0.0)
    D_cross = np.sqrt(D_sq)
    van_rossum_cross = float(np.mean(D_cross))

    # PETH correlation (Signal correlation)
    peth_a = np.mean(trial_counts_a / dt_s, axis=0)
    peth_b = np.mean(trial_counts_b / dt_s, axis=0)
    if np.std(peth_a) > 1e-6 and np.std(peth_b) > 1e-6:
        r_signal, _ = pearsonr(peth_a, peth_b)
        if np.isnan(r_signal):
            r_signal = 0.0
    else:
        r_signal = 0.0

    return float(mean_schreiber_cross), float(r_signal), float(van_rossum_cross)

def evaluate_cross_cluster_split_candidates(
    cluster_profiles: Optional[Dict[int, Dict[str, Any]]] = None,
    cluster_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
    max_depth_delta_um: float = 75.0,
    dt_s: float = 0.002,
    **kwargs
) -> pd.DataFrame:
    """
    Evaluates all spatially proximate cluster pairs (|depth_a - depth_b| <= max_depth_delta_um)
    to discover potential probe motion-induced split units.
    """
    if cluster_profiles is None:
        cluster_profiles = kwargs.get("profiles", {})
    if cluster_metadata is None:
        cluster_metadata = kwargs.get("metadata", {})
    candidate_records = []
    cids = sorted(list(cluster_profiles.keys()))
    n_cids = len(cids)

    for i in range(n_cids):
        cid_a = cids[i]
        meta_a = cluster_metadata.get(cid_a, {})
        depth_a = meta_a.get("depth_um", 0.0)
        prof_a = cluster_profiles[cid_a]
        stats_a = prof_a["epoch_stats"]

        for j in range(i + 1, n_cids):
            cid_b = cids[j]
            meta_b = cluster_metadata.get(cid_b, {})
            depth_b = meta_b.get("depth_um", 0.0)
            delta_y = abs(depth_a - depth_b)

            if delta_y > max_depth_delta_um:
                continue

            prof_b = cluster_profiles[cid_b]
            stats_b = prof_b["epoch_stats"]

            # Firing rates in Epoch 1 (pre-laser) and Epoch 4 (post-laser 3)
            fr_a_ep1 = stats_a[1]["mean_fr_hz"]
            fr_a_ep4 = stats_a[4]["mean_fr_hz"]
            fr_b_ep1 = stats_b[1]["mean_fr_hz"]
            fr_b_ep4 = stats_b[4]["mean_fr_hz"]

            # Rate contrasts
            contrast_a = prof_a["rate_contrast_1_4"]
            contrast_b = prof_b["rate_contrast_1_4"]

            # Check if one is primarily active in Ep 1 and the other in Ep 4
            # Case 1: A active in Ep 1, B active in Ep 4
            # Case 2: B active in Ep 1, A active in Ep 4
            is_case_1 = (fr_a_ep1 > fr_a_ep4) and (fr_b_ep4 > fr_b_ep1)
            is_case_2 = (fr_b_ep1 > fr_b_ep4) and (fr_a_ep4 > fr_a_ep1)

            if not (is_case_1 or is_case_2):
                # Both increased or both decreased -> not complementary split
                tc_score = 0.0
            else:
                # Quantify reciprocal transition complementarity
                # High score when contrast_a and contrast_b have opposite signs and large separation
                contrast_diff = abs(contrast_a - contrast_b) / 2.0  # in [0, 1]
                if is_case_1:
                    overlap_penalty = (fr_a_ep4 * fr_b_ep1) / ((fr_a_ep1 * fr_b_ep4) + 1e-6)
                else:
                    overlap_penalty = (fr_b_ep4 * fr_a_ep1) / ((fr_b_ep1 * fr_a_ep4) + 1e-6)
                tc_score = float(np.clip(contrast_diff * (1.0 - min(1.0, overlap_penalty)), 0.0, 1.0))

            # Select trials from the dominant active epoch of each cluster
            if is_case_1 or (fr_a_ep1 >= fr_a_ep4 and fr_b_ep4 >= fr_b_ep1):
                trials_a = stats_a[1]["_trial_counts"]
                trials_b = stats_b[4]["_trial_counts"]
                active_epoch_a, active_epoch_b = 1, 4
            elif is_case_2 or (fr_b_ep1 >= fr_b_ep4 and fr_a_ep4 >= fr_a_ep1):
                trials_a = stats_a[4]["_trial_counts"]
                trials_b = stats_b[1]["_trial_counts"]
                active_epoch_a, active_epoch_b = 4, 1
            else:
                trials_a = stats_a[1]["_trial_counts"]
                trials_b = stats_b[1]["_trial_counts"]
                active_epoch_a, active_epoch_b = 1, 1

            # Cross-cluster trial Schreiber similarity and signal correlation
            schreiber_cross, r_signal, vr_cross = compute_cross_cluster_trial_similarity(
                trials_a, trials_b, dt_s=dt_s, sigma_s=0.025
            )

            # Preferred phase match
            pref_phase_a = stats_a[active_epoch_a]["preferred_phase_rad"]
            pref_phase_b = stats_b[active_epoch_b]["preferred_phase_rad"]
            if pref_phase_a is not None and pref_phase_b is not None:
                phase_diff = abs((pref_phase_a - pref_phase_b + np.pi) % (2.0 * np.pi) - np.pi)
                phase_alignment_score = float(np.cos(phase_diff))  # in [-1, 1]
            else:
                phase_alignment_score = 0.0

            # Spatial distance weight (Gaussian decay with sigma = 30 um)
            w_dist = float(np.exp(-0.5 * (delta_y / 30.0) ** 2))

            # Composite Split Score
            # High when: strong complementarity, high trial similarity, matching phase tuning, close distance
            s_split = float(
                tc_score *
                max(0.0, schreiber_cross) *
                max(0.0, r_signal) *
                max(0.0, phase_alignment_score) *
                w_dist
            )

            # Confidence Level
            if s_split >= 0.25:
                confidence = "High (Probable Split Artifact)"
            elif s_split >= 0.10:
                confidence = "Moderate (Possible Split / Recruited Neighbor)"
            elif s_split >= 0.03:
                confidence = "Low (Weak Complementarity)"
            else:
                confidence = "Unlikely (Independent Neurons)"

            candidate_records.append({
                "cluster_a": cid_a,
                "cluster_b": cid_b,
                "depth_a_um": depth_a,
                "depth_b_um": depth_b,
                "delta_y_um": delta_y,
                "fr_a_ep1": fr_a_ep1,
                "fr_a_ep4": fr_a_ep4,
                "fr_b_ep1": fr_b_ep1,
                "fr_b_ep4": fr_b_ep4,
                "contrast_a_1_4": contrast_a,
                "contrast_b_1_4": contrast_b,
                "temporal_complementarity": tc_score,
                "cross_cluster_schreiber": schreiber_cross,
                "cross_cluster_signal_corr": r_signal,
                "van_rossum_cross": vr_cross,
                "phase_alignment_score": phase_alignment_score,
                "spatial_weight": w_dist,
                "composite_split_score": s_split,
                "split_confidence": confidence,
                "active_epoch_a": active_epoch_a,
                "active_epoch_b": active_epoch_b,
                "archetype_a": stats_a[active_epoch_a]["archetype"],
                "archetype_b": stats_b[active_epoch_b]["archetype"]
            })

    df = pd.DataFrame(candidate_records)
    if len(df) > 0:
        df = df.sort_values(by="composite_split_score", ascending=False).reset_index(drop=True)
    return df
