"""
Diaphragm EMG processing module.
Handles bandpass filtering, notch filtering, envelope extraction, breath peak segmentation,
apnea detection, and time-resolved SNR stationarity computation.
"""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
from typing import Tuple, List, Dict
from ..config import EMGConfig
from ..models import EMGProcessed

def process_emg(raw_signal: np.ndarray, time_vector: np.ndarray, config: EMGConfig) -> EMGProcessed:
    """
    Process diaphragm EMG voltage traces.
    1. Bandpass filter (default 300-3000 Hz)
    2. Notch filter (60 Hz powerline)
    3. Rectification & moving RMS/Hilbert envelope smoothing
    4. Breath peak detection & instantaneous respiratory rate
    5. Time-resolved SNR computation across moving windows to verify stationarity
    """
    fs = config.fs
    
    # 1. Bandpass filter
    nyq = 0.5 * fs
    low = config.highpass_cutoff / nyq
    high = min(config.lowpass_cutoff / nyq, 0.99)
    b, a = butter(3, [low, high], btype='bandpass')
    filtered = filtfilt(b, a, raw_signal)

    # 2. Notch filter for powerline noise if applicable
    if config.notch_freq > 0 and config.notch_freq < nyq:
        w0 = config.notch_freq / nyq
        b_notch, a_notch = iirnotch(w0, Q=30)
        filtered = filtfilt(b_notch, a_notch, filtered)

    # 3. Rectification & Smoothing Envelope
    rectified = np.abs(filtered) if config.rectify else filtered
    window_samples = int((config.smoothing_window_ms / 1000.0) * fs)
    if window_samples > 1:
        kernel = np.ones(window_samples) / window_samples
        envelope = np.convolve(rectified, kernel, mode='same')
    else:
        envelope = rectified

    # 4. Breath Peak Detection
    baseline_median = np.median(envelope)
    baseline_std = np.std(envelope)
    height_threshold = baseline_median + config.peak_threshold_sd * baseline_std
    min_dist_samples = int((config.min_inter_breath_interval_ms / 1000.0) * fs)

    peaks, properties = find_peaks(envelope, height=height_threshold, distance=min_dist_samples)
    breath_onsets = time_vector[peaks]
    peak_amplitudes = envelope[peaks]

    # Instantaneous Respiratory Rate (breaths per minute)
    if len(breath_onsets) > 1:
        ibi = np.diff(breath_onsets)  # Inter-breath interval in seconds
        instantaneous_rr = 60.0 / ibi
    else:
        instantaneous_rr = np.array([])

    # 5. Apnea Event Detection
    apneas = []
    if len(breath_onsets) > 1:
        intervals = np.diff(breath_onsets)
        apnea_indices = np.where(intervals >= config.apnea_threshold_s)[0]
        for idx in apnea_indices:
            apneas.append({
                "start_s": float(breath_onsets[idx]),
                "duration_s": float(intervals[idx]),
                "end_s": float(breath_onsets[idx + 1])
            })

    # 6. Time-resolved SNR Stationarity Evaluation
    # Partition recording into 10-second non-overlapping windows
    win_sec = 10.0
    win_len = int(win_sec * fs)
    num_windows = len(raw_signal) // win_len
    
    snr_times = []
    snr_vals = []
    for w in range(num_windows):
        segment_env = envelope[w*win_len : (w+1)*win_len]
        signal_pwr = np.percentile(segment_env, 95) ** 2
        noise_pwr = np.percentile(segment_env, 10) ** 2 + 1e-12
        snr_db = 10.0 * np.log10(signal_pwr / noise_pwr)
        
        snr_times.append((w + 0.5) * win_sec)
        snr_vals.append(float(snr_db))

    return EMGProcessed(
        filtered=filtered,
        envelope=envelope,
        breath_onsets=breath_onsets,
        breath_offsets=breath_onsets + 0.1,  # Approx offset
        peak_amplitudes=peak_amplitudes,
        instantaneous_rr=instantaneous_rr,
        snr_time_vector=np.array(snr_times),
        snr_db=np.array(snr_vals),
        apnea_events=apneas
    )
