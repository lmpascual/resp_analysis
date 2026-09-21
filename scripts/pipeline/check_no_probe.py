"""
Scan e097_no_probe_g0 to see which channels recorded Left vs Right laser!
"""
from pathlib import Path
import numpy as np

bin_path = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_no_probe_g0\e097_no_probe_g0_t0.nidq.bin")
fs = 10000.0
volts_per_count = 10.0 / 32768.0
total_samples = 386188686 // (9 * 2)

data = np.memmap(bin_path, dtype=np.int16, mode='r', shape=(total_samples, 9))

# Check Channel 6 pulses
ch6 = data[:, 6] * volts_per_count
ch7 = data[:, 7] * volts_per_count

diff6 = np.diff((ch6 > 0.5).astype(np.int8))
r6 = np.where(diff6 == 1)[0]

print(f"Total pulses on Ch6 in e097_no_probe_g0: {len(r6)}")

if len(r6) > 0:
    isi_s = np.diff(r6) / fs
    breaks = np.where(isi_s > 1.0)[0]
    train_starts = [r6[0]] + list(r6[breaks + 1])
    train_ends = list(r6[breaks]) + [r6[-1]]
    
    print(f"Found {len(train_starts)} trains on Ch6:")
    for i, (s, e) in enumerate(zip(train_starts, train_ends)):
        n_p = len(np.where((r6 >= s) & (r6 <= e))[0])
        t_s = s / fs
        m, sec = divmod(t_s, 60)
        print(f"  Train {i+1}: t={t_s:.1f}s ({int(m):02d}:{sec:04.1f}) | {n_p} pulses | max={ch6[s:e+50].max():.3f}V")

print(f"\nChannel 7 stats in e097_no_probe_g0: min={ch7.min():.3f}V, max={ch7.max():.3f}V, mean={ch7.mean():.3f}V, std={ch7.std():.3f}V")

