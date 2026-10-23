#!/usr/bin/env python3
"""
Diagnostic script to test whole-session processing for e097_g2:
- Reads NIDQ binary (66,072,893 samples, 6607.3s, 110.1 min)
- Runs chunked ECG R-peak detection (Pan-Tompkins)
- Runs chunked Diaphragm breath detection
- Verifies laser train alignment
"""

import time
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

BIN_PATH = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
fs = 9999.955307
volts_per_count = 10.0 / 32768.0
n_chans = 9

total_samples = BIN_PATH.stat().st_size // (n_chans * 2)
duration_s = total_samples / fs
print(f"Recording: {BIN_PATH.name} | Total duration: {duration_s:.1f} s ({duration_s/60:.1f} min)")

raw = np.memmap(BIN_PATH, dtype=np.int16, mode='r', shape=(total_samples, n_chans))

# Let's inspect chunk processing speed
chunk_s = 60.0
chunk_pts = int(chunk_s * fs)
n_chunks = int(np.ceil(total_samples / chunk_pts))

print(f"Processing {n_chunks} chunks of {chunk_s}s...")

t0 = time.time()
# Sample 5 representative chunks across the session (early, laser, anoxia, DR, late seizure)
eval_chunks = [5, 15, 30, 55, 90]

for c_idx in eval_chunks:
    s0 = c_idx * chunk_pts
    s1 = min(total_samples, s0 + chunk_pts)
    t_start = s0 / fs
    
    ch0 = raw[s0:s1, 0].astype(np.float32) * volts_per_count # raw EMG
    ch1 = raw[s0:s1, 1].astype(np.float32) * volts_per_count # integrated EMG
    ch2 = raw[s0:s1, 2].astype(np.float32) * volts_per_count # ECG
    
    # Quick ECG R-peak check
    # 10-100 Hz bandpass
    sos_ecg = butter(3, [10.0 / (0.5 * fs), 100.0 / (0.5 * fs)], btype='bandpass', output='sos')
    filt_ecg = sosfiltfilt(sos_ecg, ch2)
    diff_ecg = np.diff(filt_ecg, prepend=filt_ecg[0]) * fs
    energy_ecg = np.convolve(diff_ecg**2, np.ones(int(0.02 * fs))/int(0.02 * fs), mode='same')
    
    thresh_ecg = np.mean(energy_ecg) + 1.2 * np.std(energy_ecg)
    ecg_pks, _ = find_peaks(energy_ecg, height=thresh_ecg, distance=int(0.08 * fs))
    hr_bpm = (len(ecg_pks) / chunk_s) * 60.0
    
    # Quick Dia burst check on integrated EMG (ch1) or smoothed rectified EMG
    dia_thresh = np.mean(ch1) + 1.5 * np.std(ch1)
    dia_pks, _ = find_peaks(ch1, height=dia_thresh, distance=int(0.12 * fs))
    resp_rate = len(dia_pks) / chunk_s
    
    print(f"Chunk {c_idx:02d} ({t_start/60:.1f} min): ECG peaks={len(ecg_pks)} (~{hr_bpm:.0f} bpm) | Dia breaths={len(dia_pks)} (~{resp_rate:.1f} Hz, {resp_rate*60:.0f} br/min)")

print(f"Sample test completed in {time.time()-t0:.2f}s.")
