"""
Check rising and falling edges of individual pulses in e097_g2 Channel 6 for any dual-edge, stepping, or jitter.
"""
from pathlib import Path
import numpy as np

bin_path = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
fs = 9999.955307
volts_per_count = 10.0 / 32768.0

data = np.memmap(bin_path, dtype=np.int16, mode='r', shape=(66072893, 9))

# Let's inspect 10 consecutive pulses from Train 1 (starting at sample 8568270)
t_start = int(856.827 * fs)
ch6_chunk = data[t_start : t_start + int(0.6 * fs), 6] * volts_per_count
t_chunk = (np.arange(len(ch6_chunk))) / fs * 1000.0 # ms

# Find pulses
thresh = 0.5
diff = np.diff((ch6_chunk > thresh).astype(np.int8))
r_indices = np.where(diff == 1)[0]
f_indices = np.where(diff == -1)[0]

print(f"Inspecting first {len(r_indices)} pulses of Train 1:")
for i in range(min(5, len(r_indices))):
    r = r_indices[i]
    f = f_indices[i]
    # Look at 10 samples around rising edge
    snip_r = ch6_chunk[max(0, r-3) : r+6]
    snip_f = ch6_chunk[f-3 : f+6]
    print(f"\nPulse {i+1}:")
    print(f"  Rising edge samples (V): {np.round(snip_r, 3)}")
    print(f"  Falling edge samples (V): {np.round(snip_f, 3)}")
    print(f"  Exact width (samples): {f - r} ({ (f - r) / fs * 1000.0 :.2f} ms)")

