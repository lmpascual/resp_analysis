"""
SpikeGLX NIDQ analog and digital data loader.
Reads multi-channel raw binary files (*.nidq.bin) and extracts time vectors and channel arrays.
"""
from pathlib import Path
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
from ..models import AnalogData

logger = logging.getLogger(__name__)

def read_nidq_bin(bin_path: Path, meta_path: Optional[Path] = None,
                  num_channels: int = 8, fs: float = 10000.0,
                  dtype: np.dtype = np.int16, gain: float = 1.0) -> AnalogData:
    """
    Read SpikeGLX NIDQ binary file (*.nidq.bin).
    If file doesn't exist, logs warning and raises FileNotFoundError.
    """
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f"NIDQ file not found: {bin_path}")

    # Read binary
    raw_flat = np.fromfile(bin_path, dtype=dtype)
    num_samples = len(raw_flat) // num_channels
    data_2d = raw_flat[:num_samples * num_channels].reshape((num_samples, num_channels)).T  # shape: (num_channels, num_samples)

    # Convert to Volts (mV) using gain
    volts_2d = data_2d.astype(np.float32) * (1.0 / gain)

    time_vec = np.arange(num_samples, dtype=np.float64) / fs

    emg_raw = volts_2d[0] if num_channels > 0 else np.zeros(num_samples)
    ecg_raw = volts_2d[1] if num_channels > 1 else np.zeros(num_samples)
    sync_raw = volts_2d[2] if num_channels > 2 else np.zeros(num_samples)
    opto_raw = volts_2d[3] if num_channels > 3 else np.zeros(num_samples)

    return AnalogData(
        time_vector=time_vec,
        emg_raw=emg_raw,
        ecg_raw=ecg_raw,
        sync_raw=sync_raw,
        opto_raw=opto_raw,
        fs=fs
    )

def generate_synthetic_nidq(duration_s: float = 60.0, fs: float = 10000.0) -> AnalogData:
    """
    Generate synthetic NIDQ analog signals for testing pipeline processors.
    Includes simulated EMG bursts, ECG R-peaks, 1 Hz sync TTLs, and 5 Hz laser opto pulses.
    """
    num_samples = int(duration_s * fs)
    t = np.arange(num_samples, dtype=np.float64) / fs

    # 1. Diaphragm EMG: 2.5 Hz breathing rate (bursts every 0.4s) + noise
    emg_base = np.random.normal(0, 0.05, num_samples)
    breath_freq = 2.5
    burst_mask = (np.sin(2 * np.pi * breath_freq * t) > 0.6).astype(float)
    emg_bursts = np.random.normal(0, 0.4, num_samples) * burst_mask
    emg_raw = emg_base + emg_bursts

    # 2. ECG: ~ 8 Hz heart rate (480 bpm mouse heart rate)
    ecg_raw = np.random.normal(0, 0.02, num_samples)
    hr_freq = 8.0
    r_peak_locs = np.where(np.diff((np.sin(2 * np.pi * hr_freq * t) > 0.98).astype(int)) == 1)[0]
    for loc in r_peak_locs:
        if loc + 50 < num_samples:
            ecg_raw[loc:loc+50] += np.hanning(50) * 1.5

    # 3. Sync: 1 Hz square pulse TTL
    sync_raw = ((np.sin(2 * np.pi * 1.0 * t) > 0.0).astype(float)) * 3.3

    # 4. Opto Laser: 20 Hz stimulation trains during seconds 20 to 40
    opto_raw = np.zeros(num_samples)
    stim_mask = (t >= 20.0) & (t <= 40.0)
    opto_raw[stim_mask] = ((np.sin(2 * np.pi * 20.0 * t[stim_mask]) > 0.5).astype(float)) * 5.0

    return AnalogData(
        time_vector=t,
        emg_raw=emg_raw,
        ecg_raw=ecg_raw,
        sync_raw=sync_raw,
        opto_raw=opto_raw,
        fs=fs
    )
