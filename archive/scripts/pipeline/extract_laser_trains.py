"""
Extract every pulse and pulse train in Channel 6, and check all channels during stim epochs.
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

# Scan Channel 6 in chunks and find all rising edges above threshold
threshold = 0.5 # volts (baseline is ~0.01V, pulse peak is ~1.04V)
thresh_counts = int(threshold / volts_per_count)

chunk_size = 2_000_000
n_chunks = int(np.ceil(total_samples / chunk_size))

rising_edges = []
falling_edges = []
pulse_peaks = []

prev_val = 0

for c in range(n_chunks):
    start_idx = c * chunk_size
    end_idx = min((c + 1) * chunk_size, total_samples)
    
    raw_ch6 = data[start_idx:end_idx, 6]
    
    # Prepend the previous sample to detect edges crossing chunk boundary
    concat = np.empty(len(raw_ch6) + 1, dtype=np.int16)
    concat[0] = prev_val
    concat[1:] = raw_ch6
    prev_val = raw_ch6[-1]
    
    above = concat >= thresh_counts
    diff = np.diff(above.astype(np.int8))
    
    r_idx = np.where(diff == 1)[0] # sample index relative to start_idx
    f_idx = np.where(diff == -1)[0]
    
    for r in r_idx:
        rising_edges.append(start_idx + r)
    for f in f_idx:
        falling_edges.append(start_idx + f)

rising_edges = np.array(rising_edges)
falling_edges = np.array(falling_edges)

print(f"Total pulses detected: {len(rising_edges)} rising edges, {len(falling_edges)} falling edges")

# Match rising and falling edges
durations_ms = []
peak_volts = []
for i in range(min(len(rising_edges), len(falling_edges))):
    r = rising_edges[i]
    f = falling_edges[i]
    if f > r:
        dur_ms = (f - r) / fs * 1000.0
        durations_ms.append(dur_ms)
        pk = data[r:f+1, 6].max() * volts_per_count
        peak_volts.append(pk)

durations_ms = np.array(durations_ms)
peak_volts = np.array(peak_volts)

print(f"Pulse durations: mean = {durations_ms.mean():.2f} ms, min = {durations_ms.min():.2f} ms, max = {durations_ms.max():.2f} ms, std = {durations_ms.std():.3f} ms")
print(f"Pulse peak amplitudes: mean = {peak_volts.mean():.3f} V, min = {peak_volts.min():.3f} V, max = {peak_volts.max():.3f} V, std = {peak_volts.std():.4f} V")

# Group into pulse trains (gap between trains > 1.0 s)
isi_s = np.diff(rising_edges) / fs
train_break_indices = np.where(isi_s > 1.0)[0]

train_starts = [rising_edges[0]]
train_ends = []

for b in train_break_indices:
    train_ends.append(rising_edges[b])
    train_starts.append(rising_edges[b + 1])
train_ends.append(rising_edges[-1])

print(f"\nTotal stimulus trains detected: {len(train_starts)}")
print("=" * 80)
print(f"{'Train #':<8}{'Start (s)':<12}{'Start (mm:ss)':<15}{'Duration (s)':<14}{'Pulses':<10}{'Freq (Hz)':<12}{'Pulse Width':<12}")
print("=" * 80)

for idx, (s, e) in enumerate(zip(train_starts, train_ends)):
    p_in_train = np.where((rising_edges >= s) & (rising_edges <= e))[0]
    n_p = len(p_in_train)
    t_start_s = s / fs
    m, sec = divmod(t_start_s, 60)
    
    # Train duration: from first pulse rising to last pulse falling
    last_fall = falling_edges[np.where(falling_edges >= e)[0][0]]
    train_dur = (last_fall - s) / fs
    
    if n_p > 1:
        pulse_isis = np.diff(rising_edges[p_in_train]) / fs
        freq = 1.0 / np.mean(pulse_isis)
    else:
        freq = 0.0
        
    pw_mean = np.mean(durations_ms[p_in_train])
    
    print(f"{idx+1:<8}{t_start_s:<12.3f}{f'{int(m):02d}:{sec:06.3f}':<15}{train_dur:<14.3f}{n_p:<10}{freq:<12.1f}{pw_mean:<12.2f} ms")

