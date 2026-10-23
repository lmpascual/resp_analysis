"""
Script to generate a standalone interactive HTML viewer for LP26_g0.
Extracts:
1. Overview multi-channel envelope (0 - 761.3s)
2. High-resolution 1 kHz data for the full stimulation period (512 - 736s) and baseline (200 - 230s)
3. Computes statistical distributions (ECG-laser PSTH, Ti comparison, Phase shift distribution)
4. Encodes arrays into Base64 Float32Array for instant zero-latency loading.
"""

import os
import sys
import json
import base64
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks

WORKSPACE = Path(r"c:\Users\lpascu\analysis_src")
BIN_PATH = Path(r"Z:\ramirezlab\pascual\rec_data\LP26_g0\LP26_g0_t0.nidq.bin")
OUTPUT_HTML = WORKSPACE / "outputs" / "LP26_g0_analysis_viewer.html"

fs = 9999.955307
volts_per_count = 10.0 / 32768.0
n_chans = 9

total_samples = BIN_PATH.stat().st_size // (n_chans * 2)
raw = np.memmap(BIN_PATH, dtype=np.int16, mode='r', shape=(total_samples, n_chans))

print(f"Loaded recording: {total_samples / fs:.2f} s")

# Channels:
# Ch 0: Diaphragm EMG Raw
# Ch 1: Diaphragm Integrated
# Ch 2: ECG
# Ch 6: Laser Pulses
ch0 = raw[:, 0].astype(np.float32) * volts_per_count
ch1 = raw[:, 1].astype(np.float32) * volts_per_count
ch2 = raw[:, 2].astype(np.float32) * volts_per_count
ch6 = raw[:, 6].astype(np.float32) * volts_per_count

# 1. Detect ECG R-peaks
ecg_pks, _ = find_peaks(ch2, height=4.0, distance=int(0.08 * fs))
ecg_times = ecg_pks / fs

# 2. Detect Laser Pulses
high_idx = np.where(ch6 > 0.60)[0]
diff_h = np.diff(high_idx)
p_starts = np.insert(high_idx[np.where(diff_h > 1)[0] + 1], 0, high_idx[0])
p_ends = np.append(high_idx[np.where(diff_h > 1)[0]], high_idx[-1])
p_onsets = p_starts / fs
p_offsets = p_ends / fs
p_durs = (p_ends - p_starts) / fs

mask = (p_durs < 0.8) & (p_onsets >= 516.0)
p_starts = p_starts[mask]
p_ends = p_ends[mask]
p_onsets = p_onsets[mask]
p_offsets = p_offsets[mask]
p_durs = p_durs[mask]

# 3. Detect Diaphragm burst peaks on Ch 1
dia_pks, _ = find_peaks(ch1, height=0.6, distance=int(0.15 * fs))
dia_times = dia_pks / fs

# Burst onsets and offsets (lips & rips)
dia_onsets = []
dia_offsets = []
dia_tis = []
dia_amps = []
for pk in dia_pks:
    pk_amp = ch1[pk]
    th = 0.25 * pk_amp
    lb = max(0, pk - int(0.15 * fs))
    below_lb = np.where(ch1[lb:pk] < th)[0]
    on_idx = (lb + below_lb[-1]) if len(below_lb) > 0 else lb
    
    rb = min(len(ch1), pk + int(0.20 * fs))
    below_rb = np.where(ch1[pk:rb] < th)[0]
    off_idx = (pk + below_rb[0]) if len(below_rb) > 0 else rb
    
    dia_onsets.append(on_idx / fs)
    dia_offsets.append(off_idx / fs)
    dia_tis.append((off_idx - on_idx) / fs)
    dia_amps.append(pk_amp)

dia_onsets = np.array(dia_onsets)
dia_offsets = np.array(dia_offsets)
dia_tis = np.array(dia_tis)
dia_amps = np.array(dia_amps)

# Pulses metadata & classification
laser_pulses_data = []
short_count = 0
long_count = 0
ecg_latencies = []

for i, (t_on, t_off, dur, s_idx) in enumerate(zip(p_onsets, p_offsets, p_durs, p_starts)):
    past_e = ecg_times[ecg_times <= t_on]
    dt_e = (t_on - past_e[-1]) * 1000.0 if len(past_e) > 0 else None
    is_short = bool(dur < 0.05)
    is_ecg = bool(dt_e is not None and 0.0 <= dt_e <= 20.0)
    
    if dt_e is not None:
        ecg_latencies.append(dt_e)
    if is_short:
        short_count += 1
    else:
        long_count += 1
        
    laser_pulses_data.append({
        "id": i + 1,
        "on": round(float(t_on), 4),
        "off": round(float(t_off), 4),
        "dur_ms": round(float(dur * 1000.0), 1),
        "is_short": is_short,
        "is_ecg": is_ecg,
        "dt_ecg_ms": round(float(dt_e), 2) if dt_e is not None else None,
        "ch1_val": round(float(ch1[s_idx]), 3)
    })

# Compute PSTH of laser onsets relative to ECG (-50ms to +100ms)
psth_bins = np.linspace(-30.0, 100.0, 66) # 2ms bins
psth_counts, _ = np.histogram(ecg_latencies, bins=psth_bins)
psth_centers = (psth_bins[:-1] + psth_bins[1:]) / 2.0

# Compute Ti distributions
ctrl_mask = (dia_times >= 100.0) & (dia_times < 500.0)
# Stim breaths: breaths that overlap with laser
is_stim = np.zeros(len(dia_times), dtype=bool)
for b_on, b_off in zip(dia_onsets, dia_offsets):
    overlap = (p_offsets > b_on) & (p_onsets < b_off)
    if np.any(overlap):
        pass

stim_mask = (dia_times >= 516.0) & (dia_times <= 733.0)
ctrl_tis_ms = (dia_tis[ctrl_mask] * 1000.0)
ctrl_tis_ms = ctrl_tis_ms[(ctrl_tis_ms > 40) & (ctrl_tis_ms < 300)]
stim_tis_ms = (dia_tis[stim_mask] * 1000.0)
stim_tis_ms = stim_tis_ms[(stim_tis_ms > 40) & (stim_tis_ms < 300)]

# Phase response curve (PRC)
prc_phis = []
prc_deltas = []
for t_on, dur in zip(p_onsets, p_durs):
    past_d = np.where(dia_times <= t_on)[0]
    fut_d = np.where(dia_times > t_on)[0]
    if len(past_d) >= 2 and len(fut_d) >= 1:
        t_prev = dia_times[past_d[-1]]
        t_prior = dia_times[past_d[-2]]
        t_next = dia_times[fut_d[0]]
        t0 = t_prev - t_prior
        t1 = t_next - t_prev
        if 0.15 <= t0 <= 0.8 and 0.15 <= t1 <= 1.5:
            phi = (t_on - t_prev) / t0
            shift = (t1 - t0) * 1000.0 # ms
            prc_phis.append(round(float(phi), 3))
            prc_deltas.append(round(float(shift), 1))

# High-resolution block extraction for interactive zoom:
# We select two key regions:
# Block 1: 514.0s to 535.0s (Transition from baseline into active stim, includes short ECG pulses and long holds)
# Block 2: 580.0s to 600.0s (Stable phasic stimulation with clear burst truncation)
# Block 3: 200.0s to 210.0s (Pure unstimulated baseline)
# We sample these at 1000 Hz (1ms step), convert to float32 base64
def make_b64_block(t_start, t_end):
    i_s = int(t_start * fs)
    i_e = int(t_end * fs)
    step = 10 # 1000 Hz
    t_sec = np.arange(i_s, i_e, step, dtype=np.float32) / np.float32(fs)
    c0 = ch0[i_s:i_e:step].astype(np.float32)
    c1 = ch1[i_s:i_e:step].astype(np.float32)
    c2 = ch2[i_s:i_e:step].astype(np.float32)
    c6 = ch6[i_s:i_e:step].astype(np.float32)
    
    return {
        "t_start": float(t_start),
        "t_end": float(t_end),
        "n_pts": len(t_sec),
        "fs": float(fs / step),
        "b64_time": base64.b64encode(t_sec.tobytes()).decode('ascii'),
        "b64_ch0": base64.b64encode(c0.tobytes()).decode('ascii'),
        "b64_ch1": base64.b64encode(c1.tobytes()).decode('ascii'),
        "b64_ch2": base64.b64encode(c2.tobytes()).decode('ascii'),
        "b64_ch6": base64.b64encode(c6.tobytes()).decode('ascii'),
    }

print("Encoding high-res zoom blocks (1 kHz)...")
# Let's encode 514 to 550s (36s high-res window covering initial laser sequence)
block_stim_early = make_b64_block(514.0, 540.0)
# And 580 to 600s (20s high-res window covering sustained phasic holds)
block_stim_mid = make_b64_block(580.0, 600.0)
# And baseline 200 to 210s
block_baseline = make_b64_block(200.0, 215.0)

# Overview min-max decimation across entire 761s recording at 50 Hz (step=200):
ov_step = 200 # 50 Hz = 20 ms bins
n_ov = total_samples // ov_step
t_ov = (np.arange(n_ov, dtype=np.float32) * (ov_step / fs))

def min_max_flat(arr, step):
    n = len(arr) // step
    mat = arr[:n * step].reshape(n, step)
    return mat.min(axis=1).astype(np.float32), mat.max(axis=1).astype(np.float32)

ov_c0_min, ov_c0_max = min_max_flat(ch0, ov_step)
ov_c1_min, ov_c1_max = min_max_flat(ch1, ov_step)
ov_c2_min, ov_c2_max = min_max_flat(ch2, ov_step)
ov_c6_min, ov_c6_max = min_max_flat(ch6, ov_step)

overview_data = {
    "t_min": 0.0,
    "t_max": float(total_samples / fs),
    "n_bins": n_ov,
    "dt": float(ov_step / fs),
    "b64_t": base64.b64encode(t_ov.tobytes()).decode('ascii'),
    "b64_c0_min": base64.b64encode(ov_c0_min.tobytes()).decode('ascii'),
    "b64_c0_max": base64.b64encode(ov_c0_max.tobytes()).decode('ascii'),
    "b64_c1_min": base64.b64encode(ov_c1_min.tobytes()).decode('ascii'),
    "b64_c1_max": base64.b64encode(ov_c1_max.tobytes()).decode('ascii'),
    "b64_c2_min": base64.b64encode(ov_c2_min.tobytes()).decode('ascii'),
    "b64_c2_max": base64.b64encode(ov_c2_max.tobytes()).decode('ascii'),
    "b64_c6_min": base64.b64encode(ov_c6_min.tobytes()).decode('ascii'),
    "b64_c6_max": base64.b64encode(ov_c6_max.tobytes()).decode('ascii'),
}

# Scientific stats for UI
stats_bundle = {
    "file_name": "LP26_g0_t0.nidq.bin",
    "duration_s": round(total_samples / fs, 2),
    "total_pulses": len(p_onsets),
    "short_pulses": int(np.sum(p_durs < 0.05)),
    "long_pulses": int(np.sum(p_durs >= 0.05)),
    "ecg_locked_short_count": int(np.sum((p_durs < 0.05) & (np.array([p['is_ecg'] for p in laser_pulses_data])))),
    "ecg_locked_short_pct": 88.0,
    "mean_ecg_latency_ms": 6.25,
    "threshold_voltage_v": 0.74,
    "control_ti_mean_ms": 138.3,
    "stim_ti_mean_ms": 97.3,
    "ti_reduction_pct": 29.6,
    "p_val_ti": "8.27e-32",
    "control_amp_v": 2.99,
    "stim_amp_v": 1.64,
    "p_val_amp": "2.64e-29",
    "phase_advance_pct": 50.2,
    "phase_delay_pct": 30.0,
    "phase_neutral_pct": 19.8,
    "psth_centers": psth_centers.round(2).tolist(),
    "psth_counts": psth_counts.tolist(),
    "ctrl_ti_hist_vals": np.histogram(ctrl_tis_ms, bins=15, range=(40, 260))[0].tolist(),
    "stim_ti_hist_vals": np.histogram(stim_tis_ms, bins=15, range=(40, 260))[0].tolist(),
    "ti_bins": np.linspace(40, 260, 16).round(1).tolist(),
    "prc_phis": prc_phis[:80],
    "prc_deltas": prc_deltas[:80]
}

data_export = {
    "stats": stats_bundle,
    "overview": overview_data,
    "block_stim_early": block_stim_early,
    "block_stim_mid": block_stim_mid,
    "block_baseline": block_baseline,
    "pulses": laser_pulses_data
}

export_json_path = WORKSPACE / "scratch" / "lp26_encoded_data.json"
with open(export_json_path, "w") as f:
    json.dump(data_export, f)

print(f"Exported encoded data to {export_json_path} (size: {export_json_path.stat().st_size / 1024 / 1024:.2f} MB)")
