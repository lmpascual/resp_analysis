import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_ecg import process_ecg
from src.pipeline.config import ECGConfig

meta_path = Path(r'Z:\ramirezlab\pascual\rec_data\e086_g0\e086_g0_t0.nidq.meta')
bin_path = Path(r'Z:\ramirezlab\pascual\rec_data\e086_g0\e086_g0_t0.nidq.bin')

analog_data = read_nidq_bin(bin_path, meta_path)
t_full = analog_data.time_vector
fs = analog_data.fs

# Crop window t >= 2985.0 s
win_start = 2985.0
idx_win = np.where(t_full >= win_start)[0]
t = t_full[idx_win]
ecg_raw = analog_data.all_channels[1][idx_win]

ecg_config = ECGConfig(enabled=True, channel_index=1, fs=fs, highpass_cutoff=10.0, lowpass_cutoff=100.0,
                       notch_freq=60.0, r_peak_threshold_sd=1.2, min_rr_interval_ms=80.0)
ecg_proc = process_ecg(ecg_raw, t, ecg_config)
r_peaks = ecg_proc.r_peaks

trial1_onset_2nd = 4087.702
trial2_onset_2nd = 4216.340

print("=== Trial 1 ECG Inspection (t=4082.7 to 4097.7s) ===")
in_t1 = (r_peaks >= trial1_onset_2nd - 2.0) & (r_peaks <= trial1_onset_2nd + 8.0)
print(f"R-peaks detected near Trial 1 onset (-2s to +8s): {r_peaks[in_t1]}")
print(f"Count: {len(r_peaks[in_t1])}")
if len(r_peaks[in_t1]) > 1:
    print(f"Inter-beat intervals (ms): {np.diff(r_peaks[in_t1])*1000.0}")
    print(f"Instantaneous HR (bpm): {60.0 / np.diff(r_peaks[in_t1])}")

print("\n=== Trial 2 ECG Inspection (t=4211.3 to 4226.3s) ===")
in_t2 = (r_peaks >= trial2_onset_2nd - 2.0) & (r_peaks <= trial2_onset_2nd + 8.0)
print(f"R-peaks detected near Trial 2 onset (-2s to +8s): {r_peaks[in_t2]}")
print(f"Count: {len(r_peaks[in_t2])}")
if len(r_peaks[in_t2]) > 1:
    print(f"Inter-beat intervals (ms): {np.diff(r_peaks[in_t2])*1000.0}")
    print(f"Instantaneous HR (bpm): {60.0 / np.diff(r_peaks[in_t2])}")
