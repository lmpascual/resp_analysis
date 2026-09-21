"""
Comprehensive check of:
1. Channel 8 digital lines (bits 0-7) across the entire session.
2. Channel 7 during laser train 1 vs baseline.
3. Pulse waveform shape on Channel 6 (rise time, fall time, step levels).
4. Check if other channels (0,1,2,3,4,5) have any stimulus artifacts.
"""
from pathlib import Path
import numpy as np

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

# 1. Check all digital bits in channel 8 across entire file in chunks
print("=== 1. Digital channel 8 (XD0) bits check ===")
all_unique_xd0 = set()
chunk_size = 5_000_000
for c in range(int(np.ceil(total_samples / chunk_size))):
    s = c * chunk_size
    e = min((c + 1) * chunk_size, total_samples)
    u = np.unique(data[s:e, 8])
    all_unique_xd0.update(u.tolist())

print(f"Unique values in channel 8 (XD0): {sorted(list(all_unique_xd0))}")
for val in sorted(list(all_unique_xd0)):
    # print binary representation of each 16-bit word
    print(f"  Value {val}: binary = {val:016b} (active bits: {[b for b in range(16) if (val & (1 << b)) > 0]})")

# 2. Check Channel 7 around Train 1 (856.8 s)
print("\n=== 2. Channel 7 during laser Train 1 (t=856s to 859s) ===")
s_t1 = int(856.0 * fs)
e_t1 = int(859.0 * fs)
ch7_t1 = data[s_t1:e_t1, 7] * volts_per_count
ch6_t1 = data[s_t1:e_t1, 6] * volts_per_count

print(f"Chan 7 in Train 1 window: min={ch7_t1.min():.5f}V, max={ch7_t1.max():.5f}V, mean={ch7_t1.mean():.5f}V, std={ch7_t1.std():.5f}V")
# Correlation between ch6 and ch7
corr = np.corrcoef(ch6_t1, ch7_t1)[0, 1]
print(f"Pearson correlation between Chan 6 and Chan 7 during Train 1: {corr:.5f}")

# 3. Waveform of Channel 6 pulses
print("\n=== 3. Channel 6 pulse details ===")
# First pulse in train 1
p1_s = int(856.827 * fs) - 20
p1_e = p1_s + 100
p1_v = data[p1_s:p1_e, 6] * volts_per_count
print(f"First pulse values around onset (sample step = 0.1 ms):")
print(np.round(p1_v[15:35], 4))

