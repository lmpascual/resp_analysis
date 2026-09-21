"""
Triggered average and waveform inspection across all channels around laser pulse onsets.
"""
from pathlib import Path
import numpy as np

bin_path = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
meta_path = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.meta")

fs = 9999.955307
volts_per_count = 10.0 / 32768.0
total_samples = 66072893
n_chans = 9

data = np.memmap(bin_path, dtype=np.int16, mode='r', shape=(total_samples, n_chans))

# Detect rising edges on ch6
train1_start = int(856.0 * fs)
train1_end = int(859.0 * fs)

ch6_t1 = data[train1_start:train1_end, 6] * volts_per_count
diff6 = np.diff((ch6_t1 > 0.5).astype(np.int8))
r_edges = np.where(diff6 == 1)[0] + train1_start

print(f"Number of pulses in Train 1: {len(r_edges)}")

# Extract snippets [-2ms to +8ms] (i.e. -20 to +80 samples) around each pulse onset
pre_samp = 20
post_samp = 80
window_len = pre_samp + post_samp

snippets = np.zeros((len(r_edges), window_len, n_chans))

for i, r in enumerate(r_edges):
    snip = data[r - pre_samp : r + post_samp, :] * volts_per_count
    snippets[i] = snip

mean_snip = np.mean(snippets, axis=0) # shape (window_len, n_chans)
t_axis = (np.arange(window_len) - pre_samp) / fs * 1000.0 # ms

print(f"{'Time (ms)':<10}" + "".join([f"Ch{ch:<8}" for ch in range(n_chans)]))
for s_idx in range(0, window_len, 5):
    t_ms = t_axis[s_idx]
    row_vals = "".join([f"{mean_snip[s_idx, ch]:<8.4f}" for ch in range(n_chans)])
    print(f"{t_ms:<10.2f}{row_vals}")

print("\nPeak-to-peak modulation during laser pulse across all channels (mean snippet):")
for ch in range(n_chans):
    ptp = mean_snip[:, ch].max() - mean_snip[:, ch].min()
    print(f"  Ch {ch}: ptp = {ptp:.5f} V (min={mean_snip[:, ch].min():.4f}, max={mean_snip[:, ch].max():.4f})")

