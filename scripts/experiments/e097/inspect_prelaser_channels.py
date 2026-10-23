"""
Inspect all NIDQ channels during the 60 seconds leading up to the first laser stim (t=796.8s to 856.8s).
Identify EMG (respiratory bursts), ECG (QRS peaks), Flow, etc.
"""
from pathlib import Path
import numpy as np

bin_path = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
fs = 9999.955307
volts_per_count = 10.0 / 32768.0

t_end = 856.827
t_start = t_end - 60.0

s_start = int(t_start * fs)
s_end = int(t_end * fs)

data = np.memmap(bin_path, dtype=np.int16, mode='r', shape=(66072893, 9))
win_data = data[s_start:s_end, :] * volts_per_count
t_vec = np.arange(s_start, s_end) / fs

print(f"60s Pre-laser window: {t_start:.3f}s to {t_end:.3f}s ({s_end - s_start} samples)")
print("=" * 70)
for ch in range(9):
    col = win_data[:, ch]
    print(f"Chan {ch}: min={col.min():.4f}V, max={col.max():.4f}V, mean={col.mean():.4f}V, std={col.std():.4f}V")

# Check power in typical frequency bands:
# Respiratory/flow: 1-5 Hz
# Cardiac/ECG: 5-15 Hz (or R-peaks)
# Diaphragm EMG: 300-3000 Hz
import scipy.signal as signal

print("\nRMS power in EMG band (300 - 3000 Hz):")
for ch in [0, 1, 2, 5]:
    col = win_data[:, ch]
    sos = signal.butter(4, [300, 3000], btype='bandpass', fs=fs, output='sos')
    filt = signal.sosfilt(sos, col)
    rms = np.sqrt(np.mean(filt**2))
    print(f"  Chan {ch} EMG band RMS: {rms:.5f} V")

print("\nRMS power in ECG band (10 - 100 Hz):")
for ch in [0, 1, 2, 5]:
    col = win_data[:, ch]
    sos = signal.butter(4, [10, 100], btype='bandpass', fs=fs, output='sos')
    filt = signal.sosfilt(sos, col)
    rms = np.sqrt(np.mean(filt**2))
    print(f"  Chan {ch} ECG band RMS: {rms:.5f} V")

print("\nRMS power in Flow/slow band (0.5 - 10 Hz):")
for ch in [0, 1, 2, 5]:
    col = win_data[:, ch]
    sos = signal.butter(4, [0.5, 10], btype='bandpass', fs=fs, output='sos')
    filt = signal.sosfilt(sos, col - col.mean())
    rms = np.sqrt(np.mean(filt**2))
    print(f"  Chan {ch} Flow band RMS: {rms:.5f} V")

