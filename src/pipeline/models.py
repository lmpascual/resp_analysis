"""
Data models representing raw data signals, processed feature metrics, and events.
"""
from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional, Dict, Any

@dataclass
class AnalogData:
    time_vector: np.ndarray  # Seconds
    emg_raw: np.ndarray      # Volts
    ecg_raw: np.ndarray      # Volts
    sync_raw: np.ndarray     # Volts / Digital
    opto_raw: np.ndarray     # Volts / Digital
    fs: float                # Hz
    all_channels: Dict[int, np.ndarray] = field(default_factory=dict)

@dataclass
class EMGProcessed:
    filtered: np.ndarray
    envelope: np.ndarray
    breath_onsets: np.ndarray       # Times in seconds
    breath_offsets: np.ndarray      # Times in seconds
    peak_amplitudes: np.ndarray     # Envelope peak value per breath
    instantaneous_rr: np.ndarray    # Respiratory rate (bpm)
    snr_time_vector: np.ndarray     # Windows across time
    snr_db: np.ndarray              # SNR in dB over time
    apnea_events: List[Dict[str, float]] = field(default_factory=list)

@dataclass
class ECGProcessed:
    filtered: np.ndarray
    r_peaks: np.ndarray            # Times in seconds
    instantaneous_hr: np.ndarray   # Heart rate (bpm)
    rr_intervals_ms: np.ndarray    # R-R interval times (ms)
    hrv_sdnn: float                # Standard deviation of R-R intervals (ms)

@dataclass
class PulseEvents:
    onsets: np.ndarray      # Pulse start times (s)
    offsets: np.ndarray     # Pulse end times (s)
    durations: np.ndarray   # Durations (s)

@dataclass
class EphysData:
    unit_ids: np.ndarray             # Array of cluster IDs
    unit_spiketimes: Dict[int, np.ndarray] # Map unit_id -> array of spike times (s)
    unit_depths: Dict[int, float]   # Map unit_id -> depth in um
    unit_quality: Dict[int, str]    # Map unit_id -> 'good' / 'mua' / 'noise'

@dataclass
class SessionResults:
    animal_id: str
    session_id: str
    analog_raw: AnalogData
    emg: EMGProcessed
    ecg: ECGProcessed
    sync: PulseEvents
    opto: PulseEvents
    ephys: Optional[EphysData] = None
