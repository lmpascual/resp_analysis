"""
Diaphragm EMG processing module matching official cibrrig preprocessing standards.
1. Bandpass filter (300 - 5000 Hz, 2nd order Butterworth SOS)
2. Notch filter (60 Hz powerline)
3. Full-wave rectification
4. 50 ms Median filtering + 10 ms Savitzky-Golay polynomial smoothing
5. Burst segmentation using cibrrig burst_stats_dia (rel_height=0.95, prominence=1.0)
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, find_peaks, peak_widths, savgol_filter
from scipy.ndimage import median_filter
import sklearn.preprocessing
from typing import Tuple, List, Dict
from src.pipeline.config import EMGConfig
from src.pipeline.models import EMGProcessed


def process_emg(raw_signal: np.ndarray, time_vector: np.ndarray, config: EMGConfig) -> EMGProcessed:
    """
    Process diaphragm EMG voltage traces strictly using cibrrig standards:
    1. Bandpass filter (300-5000 Hz)
    2. Notch filter (60 Hz powerline)
    3. Full-wave rectification (|x|)
    4. 50 ms Median filter + 10 ms Savitzky-Golay smoothing envelope
    5. Onset & Offset extraction using peak_widths(rel_height=0.95)
    """
    fs = float(config.fs)
    nyq = 0.5 * fs
    
    # 1. Bandpass filter (300 - 5000 Hz, 2nd order SOS)
    high_cut = min(5000.0, nyq * 0.95)
    low_cut = 300.0
    sos = butter(2, [low_cut / nyq, high_cut / nyq], btype='bandpass', output='sos')
    filtered = sosfiltfilt(sos, raw_signal)

    # 2. Notch filter for 60 Hz powerline noise
    if config.notch_freq > 0 and config.notch_freq < nyq:
        w0 = config.notch_freq / nyq
        b_notch, a_notch = iirnotch(w0, Q=30)
        from scipy.signal import filtfilt
        filtered = filtfilt(b_notch, a_notch, filtered)

    # 3. Rectification
    rectified = np.abs(filtered)

    # 4. cibrrig 2-Stage Envelope Smoothing: 50 ms Median Filter + 10 ms SavGol
    med_win_samples = int(0.050 * fs)
    if med_win_samples % 2 == 0:
        med_win_samples += 1
    med_win_samples = max(3, med_win_samples)
    
    env_median = median_filter(rectified, size=med_win_samples)

    savgol_win_samples = int(0.010 * fs)
    if savgol_win_samples % 2 == 0:
        savgol_win_samples += 1
    savgol_win_samples = max(3, savgol_win_samples)
    
    envelope = savgol_filter(env_median, window_length=savgol_win_samples, polyorder=1)

    # 5. cibrrig burst_stats_dia Detection
    # Standardize envelope (z-score)
    scl = sklearn.preprocessing.StandardScaler(with_mean=False)
    env_scaled = scl.fit_transform(envelope[:, np.newaxis]).ravel()

    min_dist_samples = int((config.min_inter_breath_interval_ms / 1000.0) * fs)
    min_width_samples = int(0.025 * fs)  # 25 ms minimum width

    # Find peaks with prominence threshold (z-score prominence >= 1.0)
    peaks, properties = find_peaks(
        env_scaled,
        prominence=1.0,
        distance=min_dist_samples,
        width=min_width_samples
    )

    if len(peaks) > 0:
        # Calculate true diaphragm onsets (lips) and offsets (rips) using cibrrig rel_height=0.95
        rel_height = 0.95
        widths_res = peak_widths(env_scaled, peaks, rel_height=rel_height)
        lips = np.clip(np.round(widths_res[2]).astype(int), 0, len(time_vector) - 1)
        rips = np.clip(np.round(widths_res[3]).astype(int), 0, len(time_vector) - 1)

        # Remove zero-duration bursts
        valid = rips > lips
        peaks = peaks[valid]
        lips = lips[valid]
        rips = rips[valid]

        # Clean overlapping breaths / sighs (cibrrig filter logic)
        if len(lips) > 1:
            idx = np.where(np.diff(lips) < 0)[0]
            to_rm = idx[rips[idx + 1] > rips[idx]] if len(idx) > 0 else np.array([], dtype=int)
            if len(to_rm) > 0:
                peaks = np.delete(peaks, to_rm)
                lips = np.delete(lips, to_rm)
                rips = np.delete(rips, to_rm)

            # Keep longer of overlapping bursts
            if len(lips) > 1:
                postBI = lips[1:] - rips[:-1]
                idx_ov = np.where(postBI < 0)[0]
                if len(idx_ov) > 0:
                    dur_first = rips[idx_ov] - lips[idx_ov]
                    dur_second = rips[idx_ov + 1] - lips[idx_ov + 1]
                    temp = np.vstack([dur_first, dur_second]).T
                    to_rm_ov = idx_ov + np.argmin(temp, 1)
                    peaks = np.delete(peaks, to_rm_ov)
                    lips = np.delete(lips, to_rm_ov)
                    rips = np.delete(rips, to_rm_ov)

        breath_onsets = time_vector[lips]
        breath_offsets = time_vector[rips]
        peak_amplitudes = envelope[peaks]
    else:
        breath_onsets = np.array([])
        breath_offsets = np.array([])
        peak_amplitudes = np.array([])

    # Instantaneous Respiratory Rate (breaths per minute)
    if len(breath_onsets) > 1:
        ibi = np.diff(breath_onsets)  # Inter-breath interval in seconds
        instantaneous_rr = 60.0 / ibi
    else:
        instantaneous_rr = np.array([])

    # 6. Apnea Event Detection
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

    # 7. Time-resolved SNR Stationarity Evaluation
    win_sec = 10.0
    win_len = int(win_sec * fs)
    num_windows = len(raw_signal) // win_len if win_len > 0 else 0
    
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
        breath_offsets=breath_offsets,
        peak_amplitudes=peak_amplitudes,
        instantaneous_rr=instantaneous_rr,
        snr_time_vector=np.array(snr_times),
        snr_db=np.array(snr_vals),
        apnea_events=apneas
    )
