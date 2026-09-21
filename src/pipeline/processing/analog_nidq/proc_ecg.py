"""
ECG processing module.
Handles bandpass filtering, Pan-Tompkins derivative QRS energy transform,
adaptive local R-peak detection, physiological outlier filtering (250-750 bpm for mouse),
heart rate tachogram, and HRV (SDNN) computation.
"""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
from typing import Tuple
from src.pipeline.config import ECGConfig
from src.pipeline.models import ECGProcessed

def process_ecg(raw_signal: np.ndarray, time_vector: np.ndarray, config: ECGConfig) -> ECGProcessed:
    """
    Process ECG signal using Pan-Tompkins QRS Derivative Energy transform:
    1. Bandpass filter (10-100 Hz) + Notch filter (60 Hz)
    2. Compute derivative energy: (d/dt ECG)^2 with moving integration window
    3. Detect QRS energy peaks (invariant to positive/negative R-peak polarity)
    4. Apply physiological filtering (250-750 bpm for mouse) to remove double detections & missed beats
    5. Compute Instantaneous Heart Rate (bpm) & HRV (SDNN)
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

    # 1. Bandpass filter (10-100 Hz)
    low = max(10.0 / nyq, 0.001)
    high = min(config.lowpass_cutoff / nyq, 0.99)
    b, a = butter(3, [low, high], btype='bandpass')
    filtered = filtfilt(b, a, raw_signal)

    # Notch filter
    if config.notch_freq > 0 and config.notch_freq < nyq:
        w0 = config.notch_freq / nyq
        b_notch, a_notch = iirnotch(w0, Q=30)
        filtered = filtfilt(b_notch, a_notch, filtered)

    # 2. Pan-Tompkins QRS Derivative Energy Transform: (d/dt ECG)^2
    diff_ecg = np.diff(filtered, prepend=filtered[0]) * fs
    sq_ecg = diff_ecg ** 2
    win_int = max(1, int(0.02 * fs))  # 20 ms integration window
    kernel = np.ones(win_int) / win_int
    ecg_energy = np.convolve(sq_ecg, kernel, mode='same')

    # 3. Adaptive local window thresholding on QRS energy
    win_samples = int(10.0 * fs)
    num_wins = max(1, len(ecg_energy) // win_samples)
    min_dist_samples = int((config.min_rr_interval_ms / 1000.0) * fs)

    r_peaks_idx = []
    for w in range(num_wins):
        idx_s = w * win_samples
        idx_e = (w + 1) * win_samples if w < num_wins - 1 else len(ecg_energy)
        seg = ecg_energy[idx_s:idx_e]
        std_s = np.std(seg)
        if std_s > 1e-6:
            thresh = np.mean(seg) + 1.2 * std_s
            p_seg, _ = find_peaks(seg, height=thresh, distance=min_dist_samples)
            r_peaks_idx.extend(idx_s + p_seg)

    r_peaks = time_vector[r_peaks_idx] if len(r_peaks_idx) > 0 else np.array([])

    if len(r_peaks) > 1:
        rr_sec = np.diff(r_peaks)
        hr_inst = 60.0 / rr_sec
        hr_times = r_peaks[1:]

        # 4. Physiological Outlier Filtering (50 - 750 bpm for mouse to preserve opto-induced bradycardia)
        valid_mask = (hr_inst >= 50.0) & (hr_inst <= 750.0)
        if np.any(valid_mask):
            hr_inst = hr_inst[valid_mask]
            hr_times = hr_times[valid_mask]
            r_peaks = np.concatenate(([r_peaks[0]], hr_times))
            rr_sec = np.diff(r_peaks)
            rr_ms = rr_sec * 1000.0
            hrv_sdnn = float(np.std(rr_ms))
        else:
            rr_ms = np.array([])
            hrv_sdnn = 0.0
    else:
        instantaneous_hr = np.array([])
        rr_ms = np.array([])
        hrv_sdnn = 0.0

    return ECGProcessed(
        filtered=filtered,
        r_peaks=r_peaks,
        instantaneous_hr=hr_inst,
        rr_intervals_ms=rr_ms,
        hrv_sdnn=hrv_sdnn
    )
