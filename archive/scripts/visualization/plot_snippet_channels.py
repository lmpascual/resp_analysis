"""
Plot a 5-second snippet of channels 0, 1, 2, 5 to verify physiological modalities.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

bin_path = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
fs = 9999.955307
volts_per_count = 10.0 / 32768.0

# 5 seconds from 810s to 815s (within the 60s pre-laser window)
s_start = int(810.0 * fs)
s_end = int(815.0 * fs)

data = np.memmap(bin_path, dtype=np.int16, mode='r', shape=(66072893, 9))
snip = data[s_start:s_end, :] * volts_per_count
t = np.arange(s_start, s_end) / fs

fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
plt.subplots_adjust(hspace=0.25)

axs[0].plot(t, snip[:, 0], color='#1f77b4', lw=0.8)
axs[0].set_title("Channel 0 (High frequency bursts -> Diaphragm EMG)")
axs[0].set_ylabel("Volts")
axs[0].grid(True, alpha=0.3)

axs[1].plot(t, snip[:, 1], color='#2ca02c', lw=1.2)
axs[1].set_title("Channel 1 (Slow oscillatory breathing waveform -> Respiratory Airflow / Pleth)")
axs[1].set_ylabel("Volts")
axs[1].grid(True, alpha=0.3)

axs[2].plot(t, snip[:, 2], color='#d62728', lw=1.0)
axs[2].set_title("Channel 2 (Cardiac QRS complexes -> ECG)")
axs[2].set_ylabel("Volts")
axs[2].grid(True, alpha=0.3)

axs[3].plot(t, snip[:, 5], color='#9467bd', lw=1.0)
axs[3].set_title("Channel 5")
axs[3].set_ylabel("Volts")
axs[3].set_xlabel("Time (seconds)")
axs[3].grid(True, alpha=0.3)

out_fig = Path(r"C:\Users\lpascu\analysis_src\e097\e097_prelaser_channels_inspection.png")
plt.savefig(out_fig, dpi=200, bbox_inches='tight')
print(f"Saved to {out_fig}")

