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

def parse_nidq_meta(meta_path: Any) -> Dict[str, Any]:
    """Parse SpikeGLX *.nidq.meta file into a key-value dictionary."""
    meta = {}
    if meta_path is None:
        return meta
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return meta
    with open(meta_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                meta[k] = v.strip()
    return meta

def read_nidq_bin(bin_path: Path, meta_path: Optional[Path] = None,
                  num_channels: int = 8, fs: float = 10000.0,
                  emg_chan: int = 0, ecg_chan: int = 4,
                  sync_chan: int = 7, opto_chan: int = 3,
                  dtype: np.dtype = np.int16) -> AnalogData:
    """
    Read SpikeGLX NIDQ binary file (*.nidq.bin).
    Auto-detects *.nidq.meta if meta_path is not specified.
    """
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f"NIDQ file not found: {bin_path}")

    if meta_path is None:
        possible_meta = bin_path.with_suffix('.meta')
        if not possible_meta.exists():
            possible_meta = bin_path.parent / (bin_path.stem + '.meta')
        if possible_meta.exists():
            meta_path = possible_meta

    if meta_path and meta_path.exists():
        meta = parse_nidq_meta(meta_path)
        num_channels = int(meta.get('nSavedChans', num_channels))
        fs = float(meta.get('niSampRate', fs))
        range_max = float(meta.get('niAiRangeMax', 10.0))
        max_int = float(meta.get('niMaxInt', 32768))
        volts_per_count = range_max / max_int
    else:
        volts_per_count = 10.0 / 32768.0

    raw_flat = np.fromfile(bin_path, dtype=dtype)
    num_samples = len(raw_flat) // num_channels
    data_2d = raw_flat[:num_samples * num_channels].reshape((num_samples, num_channels)).T

    volts_2d = data_2d.astype(np.float32) * volts_per_count

    time_vec = np.arange(num_samples, dtype=np.float64) / fs

    emg_raw = volts_2d[emg_chan] if emg_chan < num_channels else np.zeros(num_samples)
    ecg_raw = volts_2d[ecg_chan] if ecg_chan < num_channels else np.zeros(num_samples)
    sync_raw = volts_2d[sync_chan] if sync_chan < num_channels else np.zeros(num_samples)
    opto_raw = volts_2d[opto_chan] if opto_chan < num_channels else np.zeros(num_samples)

    all_chans = {ch: volts_2d[ch] for ch in range(num_channels)}

    return AnalogData(
        time_vector=time_vec,
        emg_raw=emg_raw,
        ecg_raw=ecg_raw,
        sync_raw=sync_raw,
        opto_raw=opto_raw,
        fs=fs,
        all_channels=all_chans
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
