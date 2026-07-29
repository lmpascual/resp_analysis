"""
ECG processing module.
Handles bandpass filtering, R-peak detection, heart rate tachogram, and HRV (SDNN) computation.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from ..config import ECGConfig
from ..models import ECGProcessed

def process_ecg(raw_signal: np.ndarray, time_vector: np.ndarray, config: ECGConfig) -> ECGProcessed:
    """
    Process ECG signal:
    1. Bandpass filter (5-100 Hz)
    2. R-peak detection
    3. Instantaneous Heart Rate (bpm) & HRV (SDNN)
    """
    if not config.enabled or config.channel_index < 0:
        return ECGProcessed(
            filtered=np.zeros_like(raw_signal),
            r_peaks=np.array([]),
            instantaneous_hr=np.array([]),
            rr_intervals_ms=np.array([]),
            hrv_sdnn=0.0
        )

    fs = config.fs
    nyq = 0.5 * fs
    low = config.highpass_cutoff / nyq
    high = min(config.lowpass_cutoff / nyq, 0.99)
    b, a = butter(3, [low, high], btype='bandpass')
    filtered = filtfilt(b, a, raw_signal)

    # Threshold for R-peak detection
    mean_val = np.mean(filtered)
    std_val = np.std(filtered)
    peak_threshold = mean_val + config.r_peak_threshold_sd * std_val
    min_dist_samples = int((config.min_rr_interval_ms / 1000.0) * fs)

    peaks, _ = find_peaks(filtered, height=peak_threshold, distance=min_dist_samples)
    r_peaks = time_vector[peaks]

    if len(r_peaks) > 1:
        rr_sec = np.diff(r_peaks)
        rr_ms = rr_sec * 1000.0
        instantaneous_hr = 60.0 / rr_sec
        hrv_sdnn = float(np.std(rr_ms))
    else:
        instantaneous_hr = np.array([])
        rr_ms = np.array([])
        hrv_sdnn = 0.0

    return ECGProcessed(
        filtered=filtered,
        r_peaks=r_peaks,
        instantaneous_hr=instantaneous_hr,
        rr_intervals_ms=rr_ms,
        hrv_sdnn=hrv_sdnn
    )
