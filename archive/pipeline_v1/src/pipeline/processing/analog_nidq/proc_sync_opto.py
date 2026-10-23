"""
Sync & Opto Laser Pulse Extraction Module.
Detects rising and falling edges for time synchronization and laser stimulation pulses.
"""
import numpy as np
from src.pipeline.config import SyncOptoConfig
from src.pipeline.models import PulseEvents

def extract_pulse_events(raw_signal: np.ndarray, time_vector: np.ndarray, config: SyncOptoConfig) -> PulseEvents:
    """
    Extract rising edge onsets, falling edge offsets, and pulse durations.
    """
    threshold = config.pulse_threshold_v
    binary = (raw_signal >= threshold).astype(int)
    diff = np.diff(binary, prepend=0)

    rising_idx = np.where(diff == 1)[0]
    falling_idx = np.where(diff == -1)[0]

    # Ensure matching array lengths
    if len(falling_idx) > 0 and len(rising_idx) > 0:
        if falling_idx[0] < rising_idx[0]:
            falling_idx = falling_idx[1:]
    min_len = min(len(rising_idx), len(falling_idx))
    rising_idx = rising_idx[:min_len]
    falling_idx = falling_idx[:min_len]

    onsets = time_vector[rising_idx]
    offsets = time_vector[falling_idx]
    durations = offsets - onsets

    # Filter out spurious noise pulses below min duration
    valid = durations >= (config.min_pulse_width_ms / 1000.0)
    
    return PulseEvents(
        onsets=onsets[valid],
        offsets=offsets[valid],
        durations=durations[valid]
    )
