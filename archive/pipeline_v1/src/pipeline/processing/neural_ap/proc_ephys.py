"""
Neuropixel ephys processing module.
Handles unit spike alignment to opto pulses / respiratory breath onsets, PSTHs, and depth firing rate stationarity.
"""
import numpy as np
from typing import Dict, Any, List
from src.pipeline.config import EphysConfig
from src.pipeline.models import EphysData, PulseEvents, EMGProcessed

def compute_respiratory_modulation(ephys: EphysData, emg: EMGProcessed, config: EphysConfig) -> Dict[int, Dict[str, Any]]:
    """
    Compute unit spike alignment relative to diaphragm breath onsets (PSTH / modulation index).
    """
    breath_onsets = emg.breath_onsets
    results = {}
    if len(breath_onsets) == 0:
        return results

    bin_width_s = config.bin_size_ms / 1000.0
    window_s = 0.5  # [-0.5s to +0.5s relative to breath onset]
    bins = np.arange(-window_s, window_s + bin_width_s, bin_width_s)

    for uid in ephys.unit_ids:
        spikes = ephys.unit_spiketimes[uid]
        psth_counts = np.zeros(len(bins) - 1)
        
        for onset in breath_onsets:
            rel_spikes = spikes - onset
            in_win = rel_spikes[(rel_spikes >= -window_s) & (rel_spikes <= window_s)]
            counts, _ = np.histogram(in_win, bins=bins)
            psth_counts += counts

        total_trials = len(breath_onsets)
        firing_rate_hz = psth_counts / (total_trials * bin_width_s)
        
        # Calculate modulation depth
        max_rate = np.max(firing_rate_hz) if len(firing_rate_hz) > 0 else 0
        min_rate = np.min(firing_rate_hz) if len(firing_rate_hz) > 0 else 0
        mean_rate = np.mean(firing_rate_hz) if len(firing_rate_hz) > 0 else 1e-6
        mod_index = (max_rate - min_rate) / (max_rate + min_rate + 1e-6)

        results[int(uid)] = {
            "depth_um": ephys.unit_depths.get(uid, 0.0),
            "mean_rate_hz": float(mean_rate),
            "mod_index": float(mod_index),
            "psth_bins": (bins[:-1] + bins[1:]) / 2.0,
            "psth_rate_hz": firing_rate_hz
        }

    return results
