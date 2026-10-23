"""
Detailed analysis of channels 6 and 7 (and any other pulse channels) in e097_g2_t0.nidq.bin.
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

chunk_size = 2_000_000 # 200 seconds per chunk
n_chunks = int(np.ceil(total_samples / chunk_size))

ch6_maxes = []
ch7_maxes = []
ch6_mins = []
ch7_mins = []

# Also let's track threshold crossings for ch6 and ch7
# For ch6, baseline is ~0.018V, let's check > 0.2V
# For ch7, baseline is ~1.338V, let's check deviations from 1.338V, e.g. > 2.0V or < 0.5V
ch6_above_thresh = []
ch7_diff_thresh = []

print(f"Scanning {total_samples} samples ({total_samples/fs:.1f} s) in {n_chunks} chunks...")

for c in range(n_chunks):
    start_idx = c * chunk_size
    end_idx = min((c + 1) * chunk_size, total_samples)
    
    # Read chunk
    chunk_raw = data[start_idx:end_idx, :]
    
    ch6 = chunk_raw[:, 6] * volts_per_count
    ch7 = chunk_raw[:, 7] * volts_per_count
    
    ch6_maxes.append(ch6.max())
    ch7_maxes.append(ch7.max())
    ch6_mins.append(ch6.min())
    ch7_mins.append(ch7.min())
    
    # Check if ch6 has pulses > 0.2V
    idx6 = np.where(ch6 > 0.2)[0]
    if len(idx6) > 0:
        ch6_above_thresh.append((start_idx + idx6[0], start_idx + idx6[-1], len(idx6), ch6.max()))
        
    # Check if ch7 has significant deviations (> 2.0V or < 0.5V or std > 0.1)
    idx7 = np.where((ch7 > 2.0) | (ch7 < 0.5))[0]
    if len(idx7) > 0:
        ch7_diff_thresh.append((start_idx + idx7[0], start_idx + idx7[-1], len(idx7), ch7.min(), ch7.max()))

print(f"\nOverall Chan 6 min={min(ch6_mins):.4f}V, max={max(ch6_maxes):.4f}V")
print(f"Overall Chan 7 min={min(ch7_mins):.4f}V, max={max(ch7_maxes):.4f}V")

print(f"\nChan 6 chunks with values > 0.2V: {len(ch6_above_thresh)}")
for start_s, end_s, count, mx in ch6_above_thresh:
    print(f"  t = {start_s/fs:.2f}s ({start_s/fs/60:.2f}m) to {end_s/fs:.2f}s ({end_s/fs/60:.2f}m): count={count}, max={mx:.3f}V")

print(f"\nChan 7 chunks with values outside [0.5V, 2.0V]: {len(ch7_diff_thresh)}")
for start_s, end_s, count, mn, mx in ch7_diff_thresh:
    print(f"  t = {start_s/fs:.2f}s ({start_s/fs/60:.2f}m) to {end_s/fs:.2f}s ({end_s/fs/60:.2f}m): count={count}, min={mn:.3f}V, max={mx:.3f}V")

