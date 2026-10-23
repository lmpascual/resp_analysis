"""
Conservative Single-Channel blech_clust Spike Sorting & Coincidence Processor.

Optogenetic Laser Artifact Removal Strategy:
  1. Linearly interpolate the raw voltage across each laser pulse window
     (onset - pre_pad to offset + post_pad). This prevents step discontinuities
     at blank edges from creating filtfilt ringing artifacts.
  2. Bandpass filter the interpolated signal.
  3. Zero the filtered signal inside blank windows + ringing margin to remove
     any residual edge artifacts.
  4. Estimate sigma_n only from clean non-blanked segments so the detection
     threshold is not biased by artifact or zeros.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import logging
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


def make_laser_blank_mask(
    n_samples: int,
    fs: float,
    laser_onsets: np.ndarray,
    laser_offsets: np.ndarray,
    pre_pad_ms: float = 1.0,
    post_pad_ms: float = 2.0
) -> np.ndarray:
    """
    Build a boolean mask (True = clean signal, False = blanked) of length n_samples.
    """
    mask = np.ones(n_samples, dtype=bool)
    pre_s = int(pre_pad_ms / 1000.0 * fs)
    post_s = int(post_pad_ms / 1000.0 * fs)
    for t_on, t_off in zip(laser_onsets, laser_offsets):
        i0 = max(0, int(t_on * fs) - pre_s)
        i1 = min(n_samples, int(t_off * fs) + post_s)
        mask[i0:i1] = False
    return mask


def interpolate_blanked_signal(raw: np.ndarray, clean_mask: np.ndarray) -> np.ndarray:
    """
    Replace blanked samples with linear interpolation between the last clean sample
    before and first clean sample after each blank window. This prevents step-edge
    discontinuities from creating ringing when passed to filtfilt.
    """
    out = raw.copy().astype(np.float32)
    n = len(out)
    # Find contiguous False (blanked) runs
    diff = np.diff(clean_mask.astype(np.int8), prepend=1, append=1)
    starts = np.where(diff == -1)[0]  # first blanked sample
    ends   = np.where(diff ==  1)[0]  # first clean sample after blank

    for s, e in zip(starts, ends):
        i0 = max(0, s - 1)    # last clean sample before blank
        i1 = min(n - 1, e)    # first clean sample after blank
        v0 = out[i0]
        v1 = out[i1]
        length = i1 - i0
        if length > 0:
            out[i0:i1 + 1] = np.linspace(v0, v1, length + 1, dtype=np.float32)
    return out


def filter_ap_channel(
    signal: np.ndarray,
    fs: float = 30000.0,
    f_low: float = 300.0,
    f_high: float = 3000.0
) -> np.ndarray:
    """4th-order Butterworth bandpass filter for AP channel spikes (300-3000 Hz)."""
    nyq = 0.5 * fs
    b, a = butter(4, [f_low / nyq, f_high / nyq], btype='band')
    return filtfilt(b, a, signal)


def filter_and_blank(
    raw: np.ndarray,
    fs: float,
    clean_mask: np.ndarray,
    ringing_pad_ms: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline:
      1. Interpolate across blanked windows to avoid filter edge ringing.
      2. Bandpass filter.
      3. Zero the filtered signal in blanked windows + ringing_pad margin.
      4. Return (filtered_signal, expanded_blank_mask).
    """
    # Step 1 & 2
    interp = interpolate_blanked_signal(raw, clean_mask)
    filt = filter_ap_channel(interp, fs=fs)

    # Step 3: expand blank mask by ringing_pad to catch filter edge artifacts
    rpad = int(ringing_pad_ms / 1000.0 * fs)
    from scipy.ndimage import binary_dilation
    expanded_mask = binary_dilation(~clean_mask, iterations=rpad)
    filt[expanded_mask] = 0.0

    # Return the inverted expanded mask as "clean" for threshold estimation
    return filt, ~expanded_mask


def detect_spike_snippets(
    filtered_signal: np.ndarray,
    fs: float = 30000.0,
    thresh_sd: float = 4.0,
    pre_ms: float = 0.5,
    post_ms: float = 1.0,
    clean_mask: Optional[np.ndarray] = None,
    laser_onsets: Optional[np.ndarray] = None,
    laser_offsets: Optional[np.ndarray] = None,
    laser_pre_pad_ms: float = 1.0,
    laser_post_pad_ms: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect negative-going spikes.
    - sigma_n is estimated ONLY on clean (non-blanked) samples to avoid bias.
    - Spike candidates falling inside blanked windows are excluded.
    """
    # Threshold from clean signal only
    if clean_mask is not None and np.any(clean_mask):
        sigma_n = np.median(np.abs(filtered_signal[clean_mask])) / 0.6745
    else:
        sigma_n = np.median(np.abs(filtered_signal)) / 0.6745
    thresh = thresh_sd * sigma_n

    n_pre = int(pre_ms * fs / 1000.0)
    n_post = int(post_ms * fs / 1000.0)
    snippet_len = n_pre + n_post
    refractory_samples = int(1.0 * fs / 1000.0)

    crossings = np.where(filtered_signal < -thresh)[0]
    if len(crossings) == 0:
        return np.array([]), np.array([])

    # Exclude crossings that fall in blanked windows
    if clean_mask is not None:
        crossings = crossings[clean_mask[crossings]]
    if len(crossings) == 0:
        return np.array([]), np.array([])

    spike_idx = []
    i = 0
    while i < len(crossings):
        idx = crossings[i]
        win_end = min(idx + refractory_samples, len(filtered_signal))
        local_min = idx + np.argmin(filtered_signal[idx:win_end])

        if local_min - n_pre >= 0 and local_min + n_post < len(filtered_signal):
            # Only keep if peak itself is in clean signal
            if clean_mask is None or clean_mask[local_min]:
                spike_idx.append(local_min)
            while i < len(crossings) and crossings[i] <= local_min + refractory_samples:
                i += 1
        else:
            i += 1

    spike_idx = np.array(spike_idx, dtype=int)
    if len(spike_idx) == 0:
        return np.array([]), np.array([])

    # Extract snippets
    snippets, valid_idx = [], []
    for idx in spike_idx:
        snip = filtered_signal[idx - n_pre: idx + n_post]
        if len(snip) == snippet_len:
            snippets.append(snip)
            valid_idx.append(idx)

    return np.array(valid_idx), np.array(snippets)


def cluster_single_channel_gmm(
    snippets: np.ndarray,
    n_components: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCA (top 3 PCs + peak-to-trough) + GMM clustering.
    Returns the cluster with the highest mean peak-to-trough amplitude.
    """
    if len(snippets) < 20:
        return np.zeros(len(snippets), dtype=int), np.ones(len(snippets), dtype=bool)

    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(snippets)
    ptp_features = np.ptp(snippets, axis=1, keepdims=True)
    features = np.hstack([pca_features, ptp_features])

    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    labels = gmm.fit_predict(features)

    cluster_amps = []
    for k in range(n_components):
        mask_k = (labels == k)
        if np.sum(mask_k) > 0:
            mean_snip = np.mean(snippets[mask_k], axis=0)
            cluster_amps.append((k, np.ptp(mean_snip)))
        else:
            cluster_amps.append((k, 0.0))

    best_cluster_id = max(cluster_amps, key=lambda x: x[1])[0]
    best_mask = (labels == best_cluster_id)
    return labels, best_mask


def cross_check_spike_coincidence(
    unit_spiketrains: Dict[int, np.ndarray],
    unit_amplitudes: Dict[int, float],
    window_ms: float = 0.5,
    match_thresh: float = 0.80
) -> Tuple[Dict[int, np.ndarray], List[Dict[str, Any]]]:
    """
    Pairwise sub-millisecond spike coincidence deduplication.
    If coincidence ratio >= match_thresh, retains unit with larger peak amplitude.
    """
    unit_ids = sorted(list(unit_spiketrains.keys()))
    win_s = window_ms / 1000.0

    coincidence_report = []
    duplicate_units = set()

    for i in range(len(unit_ids)):
        u1 = unit_ids[i]
        if u1 in duplicate_units:
            continue
        spikes1 = unit_spiketrains[u1]
        if len(spikes1) == 0:
            continue

        for j in range(i + 1, len(unit_ids)):
            u2 = unit_ids[j]
            if u2 in duplicate_units:
                continue
            spikes2 = unit_spiketrains[u2]
            if len(spikes2) == 0:
                continue

            idx_left = np.searchsorted(spikes2, spikes1 - win_s)
            idx_right = np.searchsorted(spikes2, spikes1 + win_s)
            n_coincident = int(np.sum(idx_right > idx_left))
            overlap_ratio = (2.0 * n_coincident) / (len(spikes1) + len(spikes2))

            coincidence_report.append({
                'unit1': u1,
                'unit2': u2,
                'n_spikes1': len(spikes1),
                'n_spikes2': len(spikes2),
                'n_coincident': n_coincident,
                'coincidence_ratio': float(overlap_ratio),
                'is_duplicate': bool(overlap_ratio >= match_thresh)
            })

            if overlap_ratio >= match_thresh:
                amp1 = unit_amplitudes.get(u1, 0.0)
                amp2 = unit_amplitudes.get(u2, 0.0)
                discard_u = u2 if amp1 >= amp2 else u1
                keep_u = u1 if amp1 >= amp2 else u2
                duplicate_units.add(discard_u)
                logger.info(
                    f"  [Coincidence Match {overlap_ratio*100:.1f}%] Unit {discard_u} is "
                    f"duplicate of Unit {keep_u}. Retaining Unit {keep_u} (amp {max(amp1, amp2):.1f} uV)."
                )
                if discard_u == u1:
                    break

    deduplicated = {u: spk for u, spk in unit_spiketrains.items() if u not in duplicate_units}
    return deduplicated, coincidence_report
