#!/usr/bin/env python3
"""
Epoch Spiking Statistics & Trial-to-Trial Spike Train Similarity Engine

Implements comprehensive physiological characterization across experimental epochs:
1. Rate & Dynamic Metrics: Mean firing rate, peak PETH rate, spikes per breath trial.
2. Phase Tuning & Archetypes: Circular preferred phase, vector strength (resultant length),
   modulation depth, and respiratory archetype classification.
3. Regularity & Bursting: Fano Factor, CV of ISI, burst fraction (< 10 ms).
4. Three Literature-Grounded Trial-to-Trial Similarity & Correlation Methods:
   a. Schreiber Spike Timing Reliability (Schreiber et al. 2003, Neurocomputing):
      - Gaussian smoothing windows (sigma = 10 ms fine temporal precision, sigma = 25 ms burst envelope).
      - Circular phase-domain Schreiber reliability (sigma_phi = 0.15 rad approx 8.6 deg).
      - Full M x M pairwise correlation matrix and within-epoch average reliability.
   b. van Rossum Spike Distance Metric (van Rossum 2001, Neural Computation):
      - Filtered Euclidean metric distance sensitive to timing and absolute rate differences.
   c. Signal vs. Noise Correlation (Cohen & Kohn 2011, Nat. Neurosci.):
      - Signal correlation (similarity of trial-averaged tuning curves).
      - Noise correlation (breath-by-breath trial spike count covariation around the mean).
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import scipy.signal as signal
from scipy.stats import pearsonr
import torch

def safe_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Safe 2D matrix multiplication using PyTorch to prevent OpenBLAS crashes on Windows."""
    t_a = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    t_b = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32))
    return (t_a @ t_b).numpy()

def safe_spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    """Safe Spearman rank correlation without calling OpenBLAS np.corrcoef."""
    n = len(x)
    if n < 3:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= np.mean(rx)
    ry -= np.mean(ry)
    denom = np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(rx * ry) / denom)

@dataclass
class EpochDefinition:
    epoch_id: int
    name: str
    t_start: float
    t_end: float
    description: str

DEFAULT_EPOCHS = [
    EpochDefinition(1, "Epoch 1 (Pre-Laser Baseline)", 0.0, 60.0, "Baseline eupneic breathing prior to Laser Train 1"),
    EpochDefinition(2, "Epoch 2 (Inter-Train 1)", 62.0, 119.7, "Steady breathing between Laser Train 1 and Train 2"),
    EpochDefinition(3, "Epoch 3 (Inter-Train 2)", 122.0, 164.3, "Steady breathing between Laser Train 2 and Train 3"),
    EpochDefinition(4, "Epoch 4 (Post-Laser 3 Recovery)", 166.5, 226.24, "Recovery breathing following Laser Train 3 / gasp")
]

def convolve_trials_gaussian(
    trial_counts: np.ndarray,
    dt_s: float,
    sigma_s: float
) -> np.ndarray:
    """
    Convolves 2D trial binned counts (n_trials x n_timebins) with a Gaussian kernel.
    Returns smoothed firing rate traces (n_trials x n_timebins) in spikes/sec.
    """
    n_trials, n_bins = trial_counts.shape
    if n_trials == 0 or n_bins == 0:
        return np.zeros_like(trial_counts, dtype=np.float32)

    half_win_bins = int(np.ceil(4.0 * sigma_s / dt_s))
    t_kern = np.arange(-half_win_bins, half_win_bins + 1) * dt_s
    kernel = np.exp(-0.5 * (t_kern / sigma_s) ** 2)
    kernel /= (np.sum(kernel) * dt_s)  # Integral = 1.0, so convolution gives spikes/sec

    # Apply 1D convolution along axis 1
    smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same") * dt_s, axis=1, arr=trial_counts)
    return smoothed.astype(np.float32)

def compute_schreiber_and_van_rossum(
    smoothed_trials: np.ndarray,
    dt_s: float
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Computes Schreiber cosine correlation matrix and van Rossum distance matrix
    from smoothed trial rate vectors (n_trials x n_timebins).
    
    References:
    - Schreiber et al. (2003), Neurocomputing
    - van Rossum (2001), Neural Computation
    
    Returns:
    - mean_schreiber: Scalar average non-diagonal pairwise cosine correlation in [0, 1].
    - mean_van_rossum: Scalar average non-diagonal Euclidean distance.
    - R_matrix: (n_trials x n_trials) Schreiber correlation matrix.
    - D_matrix: (n_trials x n_trials) van Rossum distance matrix.
    """
    n_trials = smoothed_trials.shape[0]
    if n_trials < 2:
        return 0.0, 0.0, np.ones((n_trials, n_trials), dtype=np.float32), np.zeros((n_trials, n_trials), dtype=np.float32)

    # Compute trial L2 norms
    # norm_i = sqrt( sum_t (s_i(t)^2) * dt )
    norms_sq = np.sum(smoothed_trials ** 2, axis=1) * dt_s
    norms = np.sqrt(norms_sq)

    # Gram matrix of dot products: G_ij = integral s_i(t) * s_j(t) dt
    G = safe_matmul(smoothed_trials, smoothed_trials.T) * dt_s

    # Schreiber Cosine Similarity: R_ij = G_ij / (norm_i * norm_j)
    norm_outer = np.outer(norms, norms)
    valid_mask = norm_outer > 1e-12

    R_matrix = np.zeros((n_trials, n_trials), dtype=np.float32)
    np.divide(G, norm_outer, out=R_matrix, where=valid_mask)
    R_matrix = np.clip(R_matrix, 0.0, 1.0)
    np.fill_diagonal(R_matrix, 1.0)

    # van Rossum Distance: D_ij^2 = norms_sq[i] + norms_sq[j] - 2 * G_ij
    D_sq = norms_sq[:, None] + norms_sq[None, :] - 2.0 * G
    D_sq = np.maximum(D_sq, 0.0)
    D_matrix = np.sqrt(D_sq).astype(np.float32)
    np.fill_diagonal(D_matrix, 0.0)

    # Extract non-diagonal pairwise values
    triu_idx = np.triu_indices(n_trials, k=1)
    # Only average over pairs where at least one trial had spikes (or both)
    # If both had 0 spikes, similarity is undefined/0
    pair_valid = (norms[triu_idx[0]] > 1e-6) & (norms[triu_idx[1]] > 1e-6)

    if np.sum(pair_valid) > 0:
        mean_schreiber = float(np.mean(R_matrix[triu_idx][pair_valid]))
    else:
        mean_schreiber = 0.0

    mean_van_rossum = float(np.mean(D_matrix[triu_idx]))

    return mean_schreiber, mean_van_rossum, R_matrix, D_matrix

def compute_phase_domain_schreiber(
    trial_spike_phases: List[np.ndarray],
    n_phase_bins: int = 64,
    sigma_phi: float = 0.15
) -> Tuple[float, np.ndarray]:
    """
    Computes Schreiber trial-to-trial similarity in normalized circular phase space [0, 2pi].
    Removes the confound of respiratory cycle period variability.
    
    sigma_phi is in radians (0.15 rad approx 8.6 degrees).
    """
    n_trials = len(trial_spike_phases)
    if n_trials < 2:
        return 0.0, np.ones((n_trials, n_trials), dtype=np.float32)

    phase_bins = np.linspace(0.0, 2.0 * np.pi, n_phase_bins + 1)
    phase_centers = 0.5 * (phase_bins[:-1] + phase_bins[1:])
    d_phi = phase_bins[1] - phase_bins[0]

    # Circular Gaussian kernel
    k_phi = np.arange(-n_phase_bins // 2, n_phase_bins // 2) * d_phi
    kernel = np.exp(-0.5 * (k_phi / sigma_phi) ** 2)
    kernel /= np.sum(kernel)

    smoothed_phase_trials = np.zeros((n_trials, n_phase_bins), dtype=np.float32)
    for i, phases in enumerate(trial_spike_phases):
        if len(phases) == 0:
            continue
        counts, _ = np.histogram(phases % (2.0 * np.pi), bins=phase_bins)
        # Circular convolution
        conv = signal.convolve(np.tile(counts, 3), kernel, mode="same")
        smoothed_phase_trials[i, :] = conv[n_phase_bins : 2 * n_phase_bins]

    norms = np.linalg.norm(smoothed_phase_trials, axis=1)
    norm_outer = np.outer(norms, norms)
    valid_mask = norm_outer > 1e-12

    G = safe_matmul(smoothed_phase_trials, smoothed_phase_trials.T)
    R_phase = np.zeros((n_trials, n_trials), dtype=np.float32)
    np.divide(G, norm_outer, out=R_phase, where=valid_mask)
    R_phase = np.clip(R_phase, 0.0, 1.0)
    np.fill_diagonal(R_phase, 1.0)

    triu_idx = np.triu_indices(n_trials, k=1)
    pair_valid = (norms[triu_idx[0]] > 1e-6) & (norms[triu_idx[1]] > 1e-6)

    if np.sum(pair_valid) > 0:
        mean_r_phase = float(np.mean(R_phase[triu_idx][pair_valid]))
    else:
        mean_r_phase = 0.0

    return mean_r_phase, R_phase

def compute_signal_and_noise_correlations(
    trial_counts_half1: np.ndarray,
    trial_counts_half2: np.ndarray
) -> Tuple[float, float]:
    """
    Computes Signal Correlation (PETH profile match) and Noise Correlation (trial-by-trial
    spike count covariation) between two trial blocks (e.g. halves of an epoch or across epochs).
    Reference: Cohen & Kohn (2011), Nature Neuroscience.
    """
    peth1 = np.mean(trial_counts_half1, axis=0) if len(trial_counts_half1) > 0 else np.zeros(trial_counts_half1.shape[1])
    peth2 = np.mean(trial_counts_half2, axis=0) if len(trial_counts_half2) > 0 else np.zeros(trial_counts_half2.shape[1])

    # Signal correlation: Pearson r of mean PETHs
    std1, std2 = np.std(peth1), np.std(peth2)
    if std1 > 1e-6 and std2 > 1e-6:
        r_signal, _ = pearsonr(peth1, peth2)
        if np.isnan(r_signal):
            r_signal = 0.0
    else:
        r_signal = 0.0

    # Noise correlation: correlation between total trial spike counts across corresponding trials
    n1 = np.sum(trial_counts_half1, axis=1)
    n2 = np.sum(trial_counts_half2, axis=1)
    min_len = min(len(n1), len(n2))
    if min_len >= 5 and np.std(n1[:min_len]) > 1e-6 and np.std(n2[:min_len]) > 1e-6:
        r_noise, _ = pearsonr(n1[:min_len], n2[:min_len])
        if np.isnan(r_noise):
            r_noise = 0.0
    else:
        r_noise = 0.0

    return float(r_signal), float(r_noise)

def calculate_single_epoch_statistics(
    unit_spikes: np.ndarray,
    epoch_def: EpochDefinition,
    all_onsets: np.ndarray,
    all_offsets: np.ndarray,
    time_axis_s: np.ndarray,
    dt_s: float
) -> Dict[str, Any]:
    """
    Calculates comprehensive spiking statistics for a single unit within a specific experimental epoch.
    """
    t_start, t_end = epoch_def.t_start, epoch_def.t_end
    epoch_dur = t_end - t_start

    # Spikes in epoch
    spk_epoch = unit_spikes[(unit_spikes >= t_start) & (unit_spikes < t_end)]
    n_spikes = len(spk_epoch)
    mean_fr = n_spikes / epoch_dur if epoch_dur > 0 else 0.0

    # Breath trials within this epoch
    epoch_breath_idx = np.where((all_onsets >= t_start) & (all_onsets < t_end))[0]
    n_breaths = len(epoch_breath_idx)

    win_pre_s = -time_axis_s[0]
    win_post_s = time_axis_s[-1]
    n_timebins = len(time_axis_s)

    trial_counts = np.zeros((n_breaths, n_timebins), dtype=np.float32)
    trial_spike_phases: List[np.ndarray] = []
    trial_counts_per_breath = np.zeros(n_breaths, dtype=np.float32)

    all_epoch_phases = []

    for b_i, global_idx in enumerate(epoch_breath_idx):
        on = all_onsets[global_idx]
        off = all_offsets[global_idx]
        next_on = all_onsets[global_idx + 1] if global_idx + 1 < len(all_onsets) else on + (off - on) * 3.0

        rel_spks = unit_spikes - on
        in_win = rel_spks[(rel_spks >= -win_pre_s) & (rel_spks <= win_post_s)]
        counts, _ = np.histogram(in_win, bins=np.append(time_axis_s, time_axis_s[-1] + dt_s))
        trial_counts[b_i, :] = counts
        trial_counts_per_breath[b_i] = len(in_win)

        # Map spikes in [on, next_on] to phase phi in [0, 2pi]
        breath_spks = unit_spikes[(unit_spikes >= on) & (unit_spikes < next_on)]
        phases = []
        for s in breath_spks:
            if s < off:
                # Inspiration [0, pi]
                phi = (s - on) / (off - on + 1e-6) * np.pi
            else:
                # Expiration [pi, 2pi]
                phi = np.pi + (s - off) / (next_on - off + 1e-6) * np.pi
            phases.append(phi)
            all_epoch_phases.append(phi)
        trial_spike_phases.append(np.array(phases, dtype=np.float32))

    # PETH & Modulation
    peth = np.mean(trial_counts / dt_s, axis=0) if n_breaths > 0 else np.zeros(n_timebins, dtype=np.float32)
    peak_peth_fr = float(np.max(peth)) if len(peth) > 0 else 0.0
    min_peth_fr = float(np.min(peth)) if len(peth) > 0 else 0.0
    modulation_depth = (peak_peth_fr - min_peth_fr) / (mean_fr + 1.0)

    # Phase Tuning
    if len(all_epoch_phases) >= 5:
        cos_sum = np.sum(np.cos(all_epoch_phases))
        sin_sum = np.sum(np.sin(all_epoch_phases))
        pref_phase_rad = float(np.arctan2(sin_sum, cos_sum) % (2.0 * np.pi))
        vector_strength = float(np.sqrt(cos_sum**2 + sin_sum**2) / len(all_epoch_phases))
    else:
        pref_phase_rad = np.nan
        vector_strength = 0.0

    # Archetype Classification
    if mean_fr < 0.2:
        archetype = "Silent/Dormant"
    elif vector_strength > 0.18 and modulation_depth > 1.2:
        # Check if peak is during inspiration [0, pi] or expiration [pi, 2pi]
        if not np.isnan(pref_phase_rad) and pref_phase_rad < np.pi:
            archetype = "Inspiratory Phasic"
        else:
            archetype = "Expiratory Phasic"
    elif vector_strength > 0.10:
        archetype = "Phase-Modulated"
    elif mean_fr > 5.0:
        archetype = "Tonic Regular"
    else:
        archetype = "Sparse Irregular"

    # Regularity & Bursting
    if n_breaths > 2 and np.mean(trial_counts_per_breath) > 0.05:
        fano_factor = float(np.var(trial_counts_per_breath) / (np.mean(trial_counts_per_breath) + 1e-6))
    else:
        fano_factor = 1.0

    if len(spk_epoch) > 3:
        isis = np.diff(spk_epoch)
        cv_isi = float(np.std(isis) / (np.mean(isis) + 1e-6))
        burst_fraction = float(np.mean(isis < 0.010))  # < 10 ms
    else:
        cv_isi = 0.0
        burst_fraction = 0.0

    # --- THREE TRIAL-TO-TRIAL METHODS ---
    # 1. Schreiber Reliability (Dual Windows)
    # sigma = 10 ms (fine temporal precision)
    smoothed_10ms = convolve_trials_gaussian(trial_counts, dt_s, sigma_s=0.010)
    schreiber_10ms, van_rossum_10ms, R_10ms, D_10ms = compute_schreiber_and_van_rossum(smoothed_10ms, dt_s)

    # sigma = 25 ms (burst envelope)
    smoothed_25ms = convolve_trials_gaussian(trial_counts, dt_s, sigma_s=0.025)
    schreiber_25ms, van_rossum_25ms, R_25ms, D_25ms = compute_schreiber_and_van_rossum(smoothed_25ms, dt_s)

    # Phase-domain circular Schreiber
    schreiber_phase, R_phase = compute_phase_domain_schreiber(trial_spike_phases, n_phase_bins=64, sigma_phi=0.15)

    # Within-epoch stationarity: Compare first half vs second half of the epoch
    if n_breaths >= 6:
        h_mid = n_breaths // 2
        fr_h1 = np.mean(trial_counts_per_breath[:h_mid])
        fr_h2 = np.mean(trial_counts_per_breath[h_mid:])
        stationarity_index = float(1.0 - abs(fr_h1 - fr_h2) / (fr_h1 + fr_h2 + 1e-6))

        # Trial drift correlation (Spearman rho of trial spike count vs breath index)
        rho_drift = safe_spearmanr(np.arange(n_breaths), trial_counts_per_breath)
        if np.isnan(rho_drift):
            rho_drift = 0.0

        # Signal and noise correlation between halves
        r_signal_halves, r_noise_halves = compute_signal_and_noise_correlations(
            trial_counts[:h_mid], trial_counts[h_mid:2 * h_mid]
        )
    else:
        stationarity_index = 1.0
        rho_drift = 0.0
        r_signal_halves = 1.0
        r_noise_halves = 0.0

    active_breath_fraction = float(np.mean(trial_counts_per_breath > 0)) if n_breaths > 0 else 0.0

    return {
        "epoch_id": epoch_def.epoch_id,
        "epoch_name": epoch_def.name,
        "n_spikes": n_spikes,
        "n_breaths": n_breaths,
        "mean_fr_hz": float(mean_fr),
        "peak_peth_fr_hz": float(peak_peth_fr),
        "min_peth_fr_hz": float(min_peth_fr),
        "modulation_depth": float(modulation_depth),
        "preferred_phase_rad": float(pref_phase_rad) if not np.isnan(pref_phase_rad) else None,
        "vector_strength": float(vector_strength),
        "archetype": archetype,
        "fano_factor": float(fano_factor),
        "cv_isi": float(cv_isi),
        "burst_fraction": float(burst_fraction),
        "active_breath_fraction": float(active_breath_fraction),
        # Method 1: Schreiber Reliability
        "schreiber_reliability_10ms": float(schreiber_10ms),
        "schreiber_reliability_25ms": float(schreiber_25ms),
        "schreiber_reliability_phase": float(schreiber_phase),
        # Method 2: van Rossum Distance
        "van_rossum_dist_10ms": float(van_rossum_10ms),
        "van_rossum_dist_25ms": float(van_rossum_25ms),
        # Method 3: Signal & Noise Correlations
        "within_epoch_signal_corr": float(r_signal_halves),
        "within_epoch_noise_corr": float(r_noise_halves),
        # Within-epoch nonstationarity
        "stationarity_index": float(stationarity_index),
        "drift_rho": float(rho_drift),
        # Raw matrices and PETH cached for cross-cluster and cross-epoch comparisons
        "_peth": peth,
        "_smoothed_25ms": smoothed_25ms,
        "_trial_counts": trial_counts,
        "_R_25ms": R_25ms,
        "_trial_spike_phases": trial_spike_phases
    }

def calculate_cluster_multi_epoch_profile(
    unit_spikes: np.ndarray,
    all_onsets: np.ndarray,
    all_offsets: np.ndarray,
    epochs: List[EpochDefinition] = DEFAULT_EPOCHS,
    time_axis_s: Optional[np.ndarray] = None,
    dt_s: float = 0.002
) -> Dict[str, Any]:
    """
    Computes 4-epoch spiking statistics and cross-epoch transition dynamics for a cluster.
    """
    if time_axis_s is None:
        time_axis_s = np.arange(-0.150, 0.350 + dt_s, dt_s)

    epoch_stats = {}
    for ep in epochs:
        epoch_stats[ep.epoch_id] = calculate_single_epoch_statistics(
            unit_spikes=unit_spikes,
            epoch_def=ep,
            all_onsets=all_onsets,
            all_offsets=all_offsets,
            time_axis_s=time_axis_s,
            dt_s=dt_s
        )

    # Cross-Epoch Comparisons (Epoch 1 vs Epoch 4; Epoch 1 vs Epoch 2, etc.)
    # 1. Rate Contrast: Delta FR = (FR_b - FR_a) / (FR_b + FR_a + eps) in [-1, +1]
    fr_ep1 = epoch_stats[1]["mean_fr_hz"]
    fr_ep4 = epoch_stats[4]["mean_fr_hz"]
    rate_contrast_1_4 = float((fr_ep4 - fr_ep1) / (fr_ep4 + fr_ep1 + 1e-6))

    fr_ep2 = epoch_stats[2]["mean_fr_hz"]
    rate_contrast_1_2 = float((fr_ep2 - fr_ep1) / (fr_ep2 + fr_ep1 + 1e-6))

    # 2. PETH Correlation (Signal Correlation across epochs)
    peth_1 = epoch_stats[1]["_peth"]
    peth_4 = epoch_stats[4]["_peth"]
    if np.std(peth_1) > 1e-6 and np.std(peth_4) > 1e-6:
        r_peth_1_4, _ = pearsonr(peth_1, peth_4)
        if np.isnan(r_peth_1_4):
            r_peth_1_4 = 0.0
    else:
        r_peth_1_4 = 0.0

    # 3. Cross-Epoch Schreiber Trial Similarity
    sm_1 = epoch_stats[1]["_smoothed_25ms"]
    sm_4 = epoch_stats[4]["_smoothed_25ms"]
    if len(sm_1) > 0 and len(sm_4) > 0:
        norm_1 = np.sqrt(np.sum(sm_1 ** 2, axis=1) * dt_s)
        norm_4 = np.sqrt(np.sum(sm_4 ** 2, axis=1) * dt_s)
        G_cross = safe_matmul(sm_1, sm_4.T) * dt_s
        outer_norm = np.outer(norm_1, norm_4)
        valid = outer_norm > 1e-12
        R_cross = np.zeros_like(G_cross)
        np.divide(G_cross, outer_norm, out=R_cross, where=valid)
        pair_valid = (norm_1[:, None] > 1e-6) & (norm_4[None, :] > 1e-6)
        schreiber_cross_1_4 = float(np.mean(R_cross[pair_valid])) if np.sum(pair_valid) > 0 else 0.0
    else:
        schreiber_cross_1_4 = 0.0

    # Classification of unit trajectory across stimulation
    if fr_ep1 < 0.5 and fr_ep4 >= 3.0:
        trajectory_type = "Recruited (Silent -> Active)"
    elif fr_ep1 >= 3.0 and fr_ep4 < 0.5:
        trajectory_type = "Quenched (Active -> Silent)"
    elif fr_ep1 >= 2.0 and fr_ep4 >= 2.0:
        if abs(rate_contrast_1_4) < 0.25 and r_peth_1_4 > 0.6:
            trajectory_type = "Stationary Conserved"
        else:
            trajectory_type = "Modulated / Altered"
    else:
        trajectory_type = "Low Firing / Sporadic"

    return {
        "epoch_stats": epoch_stats,
        "rate_contrast_1_4": rate_contrast_1_4,
        "rate_contrast_1_2": rate_contrast_1_2,
        "signal_correlation_1_4": float(r_peth_1_4),
        "schreiber_cross_1_4": float(schreiber_cross_1_4),
        "trajectory_type": trajectory_type
    }
