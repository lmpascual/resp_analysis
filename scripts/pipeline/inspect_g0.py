"""
Inspect e097_g0 to see where left laser and right laser signals were recorded.
"""
from pathlib import Path
import numpy as np

bin_path = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g0\e097_g0_t0.nidq.bin")
meta_path = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g0\e097_g0_t0.nidq.meta")

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

print(f"e097_g0: channels={n_chans}, Fs={fs}, duration={total_samples/fs:.1f}s ({total_samples/fs/60:.2f} min)")

data = np.memmap(bin_path, dtype=np.int16, mode='r', shape=(total_samples, n_chans))

# Scan channels 5, 6, 7 and others for pulses > 0.5V
for ch in range(n_chans):
    col = data[:, ch] * volts_per_count
    above = np.where(col > 0.5)[0]
    if len(above) > 0:
        t_first = above[0] / fs
        t_last = above[-1] / fs
        print(f"Chan {ch}: min={col.min():.3f}V, max={col.max():.3f}V, has {len(above)} samples > 0.5V (from {t_first:.1f}s to {t_last:.1f}s)")
    else:
        print(f"Chan {ch}: min={col.min():.3f}V, max={col.max():.3f}V, mean={col.mean():.3f}V (NO samples > 0.5V)")

