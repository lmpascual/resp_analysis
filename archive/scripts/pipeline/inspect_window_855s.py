"""
Investigate all channels and extract exact pulse trains in Chan 6.
Also inspect Chan 7 around stim times to see if there is any laser signal or if it is purely sync.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

bin_path = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
meta_path = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.meta")

meta = {}
with open(meta_path, 'r') as f:
    for line in f:
        if '=' in line:
            k, v = line.strip().split('=', 1)
            meta[k] = v

n_chans = int(meta.get('nSavedChans', 9))
fs = float(meta.get('niSampRate', 10000.0))
range_max = float(meta.get('niAiRangeMax', 10.0))
max_int = float(meta.get('niMaxInt', 32768))
volts_per_count = range_max / max_int

file_bytes = bin_path.stat().st_size
total_samples = file_bytes // (n_chans * 2)

data = np.memmap(bin_path, dtype=np.int16, mode='r', shape=(total_samples, n_chans))

# Let's inspect a 10-second window around the first stim at t = 856s
s_idx = int(855.0 * fs)
e_idx = int(865.0 * fs)
win_data = data[s_idx:e_idx, :] * volts_per_count
t_win = np.arange(s_idx, e_idx) / fs

print("Window t=855 to 865s channel stats:")
for ch in range(n_chans):
    col = win_data[:, ch]
    print(f"Chan {ch}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}, std={col.std():.4f}")

# Let's check Chan 7: what is its frequency or waveform?
ch7_slice = win_data[:, 7]
print(f"Chan 7 unique values or std: min={ch7_slice.min():.4f}, max={ch7_slice.max():.4f}, std={ch7_slice.std():.4f}")

# Let's check if Chan 7 has any pulses or if it is a 1 Hz square wave:
# In meta: syncNiChan=7, syncNiThresh=1.1, syncSourcePeriod=1
# But Chan 7 is between 1.23V and 1.36V - let's see why it's around 1.3V
print(f"Chan 7 first 50 values: {ch7_slice[:50]}")

