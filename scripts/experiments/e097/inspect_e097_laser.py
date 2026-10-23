"""
Inspect NIDQ channels for e097_g2 to identify laser pulses and channel assignment.
"""
from pathlib import Path
import numpy as np

bin_path = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
meta_path = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.meta")

# Read metadata
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
duration_s = total_samples / fs

print(f"File: {bin_path.name}")
print(f"Channels: {n_chans}, Fs: {fs} Hz, Total samples: {total_samples}, Duration: {duration_s:.2f} s ({duration_s/60:.2f} min)")

# Memory map binary file
data = np.memmap(bin_path, dtype=np.int16, mode='r', shape=(total_samples, n_chans))

# Quick scan of channel statistics by subsampling every 100 samples (~100 Hz)
sub_step = 100
sub_data = data[::sub_step, :] * volts_per_count

print("\n--- Subsampled channel stats (every 100 samples) ---")
for ch in range(n_chans):
    col = sub_data[:, ch]
    print(f"Chan {ch}: min={col.min():.3f}V, max={col.max():.3f}V, mean={col.mean():.3f}V, std={col.std():.3f}V")

