#!/usr/bin/env python3
"""
Diaphragm EMG Burst Segmentation & Transition Trial Classification Processor

Extracts high-fidelity respiratory landmarks directly from Diaphragm EMG (Channel 0):
1. Bandpass filter (300 - 4500 Hz Butterworth SOS) + 60 Hz Notch filter.
2. Rectification & 25 ms Median Filter + 10 ms Savitzky-Golay envelope smoothing.
3. Adaptive burst peak detection capturing eupneic breaths, in-laser breakthrough breaths,
   and post-laser autoresuscitation gasps.
4. Diaphragm Inspiratory Onset (I_on) and Inspiratory Offset (I_off) at 80% relative depth.
5. Trial classification:
   - PRE_EUPNEA: baseline rhythmic breathing before laser arrest.
   - LASER_BREAKTHROUGH: rhythmic motor unit bursts occurring during optogenetic laser train.
   - DIAPHRAGM_GASP: augmented diaphragm bursts during post-laser autoresuscitation / apnea.
   - POST_EUPNEA: recovery rhythmic breathing.
6. Oral vs. Nasal flow coupling evaluation (detecting open-mouth gasps with flat nasal flow).
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, find_peaks, peak_widths, savgol_filter
from scipy.ndimage import median_filter


@dataclass
class TransitionTrial:
    tr_id: int
    row_idx: int
    t_on: float         # Diaphragm I_on (s)
    t_off: float        # Diaphragm I_off (s) -> Master alignment landmark (t=0)
    t_peak: float       # Peak Diaphragm burst envelope (s)
    t_dur_ms: float     # Inspiratory duration T_I (ms)
    emg_peak_v: float   # Peak EMG envelope amplitude (V)
    flow_min_v: float   # Peak inspiratory nasal airflow (V)
    flow_max_v: float   # Peak expiratory nasal airflow (V)
    flow_amp_v: float   # Flow peak-to-peak amplitude (V)
    is_oral: bool       # True if breath involves open-mouth breathing (nasal flow decoupled)
    state: str          # PRE_EUPNEA, LASER_BREAKTHROUGH, DIAPHRAGM_GASP, POST_EUPNEA
    state_color: str    # Hex background color for raster band
    badge_color: str    # Hex badge/bar color


@dataclass
class TrainTransitionData:
    train_key: str
    train_name: str
    t_laser_on: float
    t_laser_off: float
    laser_dur: float
    t_win_start: float
    t_win_end: float
    trials: List[TransitionTrial]
    sub_t_ds: np.ndarray
    sub_emg_ds: np.ndarray
    sub_flow_ds: np.ndarray
    sub_emg_rms_ds: np.ndarray


def extract_diaphragm_envelope(
    raw_emg: np.ndarray,
    fs: float = 10000.0,
    low_cut: float = 300.0,
    high_cut: float = 4500.0,
    notch_freq: float = 60.0,
    med_win_ms: float = 25.0,
    savgol_win_ms: float = 10.0
) -> np.ndarray:
    """
    Computes smoothed envelope preserving rapid falling edges for I_off detection.
    """
    nyq = 0.5 * fs
    sos = butter(2, [low_cut / nyq, min(high_cut, nyq * 0.95) / nyq], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, raw_emg)

    if 0.0 < notch_freq < nyq:
        b_notch, a_notch = iirnotch(notch_freq / nyq, Q=30)
        from scipy.signal import filtfilt
        filtered = filtfilt(b_notch, a_notch, filtered)

    rectified = np.abs(filtered)

    med_win = int((med_win_ms / 1000.0) * fs) | 1
    env_median = median_filter(rectified, size=med_win)

    savgol_win = int((savgol_win_ms / 1000.0) * fs) | 1
    envelope = savgol_filter(env_median, window_length=savgol_win, polyorder=1)

    return envelope.astype(np.float32)


def segment_diaphragm_breaths(
    envelope: np.ndarray,
    time_vector: np.ndarray,
    fs: float = 10000.0,
    rel_height: float = 0.80
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Segments all diaphragm breaths and extracts:
    - t_on: Inspiratory onsets (s)
    - t_off: Inspiratory offsets / burst termination (s)
    - t_pk: Peak burst times (s)
    - pks: Peak sample indices
    """
    # Estimate eupneic baseline statistics from quiet eupnea window (50s to 59s)
    idx_base0 = int(50.0 * fs)
    idx_base1 = int(min(len(envelope), 59.0 * fs))
    eup_env = envelope[idx_base0:idx_base1]
    eup_med = float(np.median(eup_env))
    eup_sd = float(np.std(eup_env))

    # Detect peaks: minimum distance 120ms (allows in-laser breakthrough breaths up to ~8Hz)
    min_dist_samples = int(0.120 * fs)
    min_width_samples = int(0.020 * fs)
    pks, _ = find_peaks(
        envelope,
        height=eup_med + 1.2 * eup_sd,
        distance=min_dist_samples,
        prominence=0.8 * eup_sd,
        width=min_width_samples
    )

    # Compute burst widths at rel_height (80% drop from prominence towards baseline)
    widths_res = peak_widths(envelope, pks, rel_height=rel_height)
    lips = np.clip(np.round(widths_res[2]).astype(int), 0, len(time_vector) - 1)
    rips = np.clip(np.round(widths_res[3]).astype(int), 0, len(time_vector) - 1)

    valid = rips > lips
    pks = pks[valid]
    lips = lips[valid]
    rips = rips[valid]

    t_on = time_vector[lips]
    t_off = time_vector[rips]
    t_pk = time_vector[pks]

    return t_on, t_off, t_pk, pks


def build_train_transition_data(
    train_key: str,
    train_name: str,
    t_laser_on: float,
    laser_dur: float,
    t_win_pre: float,
    t_win_post: float,
    t_gasp_end: float,
    time_vector: np.ndarray,
    emg_v: np.ndarray,
    flow_v: np.ndarray,
    envelope: np.ndarray,
    t_on_all: np.ndarray,
    t_off_all: np.ndarray,
    t_pk_all: np.ndarray,
    fs: float = 10000.0,
    decim: int = 5
) -> TrainTransitionData:
    """
    Builds segmented, classified transition trials for a specific laser train.
    """
    t_laser_off = t_laser_on + laser_dur
    t_win_start = t_laser_on - t_win_pre
    t_win_end = t_laser_on + t_win_post

    # Find breaths within window
    m = np.where((t_on_all >= t_win_start) & (t_on_all <= t_win_end))[0]

    trials: List[TransitionTrial] = []
    for row_idx, tr_idx in enumerate(m):
        ton = float(t_on_all[tr_idx])
        toff = float(t_off_all[tr_idx])
        tpk = float(t_pk_all[tr_idx])
        dur_ms = (toff - ton) * 1000.0
        emg_peak = float(envelope[int(tpk * fs)])

        # Measure local nasal flow
        i0 = max(0, int((ton - 0.05) * fs))
        i1 = min(len(flow_v), int((toff + 0.15) * fs))
        f_min = float(np.min(flow_v[i0:i1])) if i1 > i0 else 0.0
        f_max = float(np.max(flow_v[i0:i1])) if i1 > i0 else 0.0
        f_amp = f_max - f_min

        # Physiological State Classification
        # 1. Pre-laser eupnea
        if ton < t_laser_on and toff <= t_laser_on + 0.08:
            state = "PRE_EUPNEA"
            state_col = "#f0fdf4"   # Soft mint green
            badge_col = "#16a34a"   # Forest green
            is_oral = False

        # 2. In-laser breakthrough breaths
        elif t_laser_on <= tpk <= t_laser_off:
            state = "LASER_BREAKTHROUGH"
            state_col = "#ffedd5"   # Warm amber/orange
            badge_col = "#f97316"   # Orange
            is_oral = (f_amp < 0.8)

        # 3. Autoresuscitation gasps post-laser
        elif t_laser_off < tpk <= t_gasp_end:
            state = "DIAPHRAGM_GASP"
            state_col = "#ffe4e6"   # Rose/pink
            badge_col = "#e11d48"   # Bold rose
            # In mice, open-mouth gasping decouples nasal flow
            is_oral = (f_amp < 1.0)

        # 4. Post-laser resumption of eupnea
        else:
            state = "POST_EUPNEA"
            state_col = "#f0f9ff"   # Pale sky blue
            badge_col = "#0284c7"   # Cerulean blue
            is_oral = False

        trials.append(TransitionTrial(
            tr_id=int(tr_idx),
            row_idx=int(row_idx),
            t_on=ton,
            t_off=toff,
            t_peak=tpk,
            t_dur_ms=dur_ms,
            emg_peak_v=emg_peak,
            flow_min_v=f_min,
            flow_max_v=f_max,
            flow_amp_v=f_amp,
            is_oral=is_oral,
            state=state,
            state_color=state_col,
            badge_color=badge_col
        ))

    # Downsampled telemetry slice for high-speed plotting
    idx0 = int(max(0, (t_win_start - 0.5) * fs))
    idx1 = int(min(len(time_vector), (t_win_end + 0.5) * fs))
    sub_t = time_vector[idx0:idx1]
    sub_emg = emg_v[idx0:idx1]
    sub_flow = flow_v[idx0:idx1]

    w_rms = int(0.040 * fs)
    sub_emg_sq = sub_emg ** 2
    sub_emg_rms = np.sqrt(np.convolve(sub_emg_sq, np.ones(w_rms) / w_rms, mode="same"))

    return TrainTransitionData(
        train_key=train_key,
        train_name=train_name,
        t_laser_on=t_laser_on,
        t_laser_off=t_laser_off,
        laser_dur=laser_dur,
        t_win_start=t_win_start,
        t_win_end=t_win_end,
        trials=trials,
        sub_t_ds=sub_t[::decim],
        sub_emg_ds=sub_emg[::decim],
        sub_flow_ds=sub_flow[::decim],
        sub_emg_rms_ds=sub_emg_rms[::decim]
    )
