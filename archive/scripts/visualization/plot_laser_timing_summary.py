"""
Generate a publication-quality diagnostic figure summarizing laser pulse delivery in e097_g2.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

bin_path = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
meta_path = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.meta")
out_fig = Path(r"C:\Users\lpascu\analysis_src\e097\e097_g2_laser_timing_summary.png")

fs = 9999.955307
volts_per_count = 10.0 / 32768.0
total_samples = 66072893
total_sec = total_samples / fs

data = np.memmap(bin_path, dtype=np.int16, mode='r', shape=(total_samples, 9))

# Downsampled trace for whole-session plot (100 Hz = step of 100 samples)
step = 100
t_sub = np.arange(0, total_samples, step) / 60.0 # minutes
ch6_sub = data[::step, 6] * volts_per_count
ch7_sub = data[::step, 7] * volts_per_count

fig, axs = plt.subplots(4, 1, figsize=(14, 11), gridspec_kw={'height_ratios': [1.2, 1, 1, 1]})
plt.subplots_adjust(hspace=0.45)

# Panel 1: Full session timeline
axs[0].plot(t_sub, ch6_sub, color='#2b5c8f', lw=1.2, label='Channel 6 (Laser)')
axs[0].plot(t_sub, ch7_sub, color='#888888', lw=0.8, alpha=0.7, label='Channel 7 (Sync/DC bias)')
axs[0].set_title("e097_g2 NIDQ: Whole Session Laser Stimulation Timeline (110.1 min)", fontsize=13, fontweight='bold', pad=10)
axs[0].set_xlabel("Elapsed Time (minutes)", fontsize=11)
axs[0].set_ylabel("Voltage (V)", fontsize=11)
axs[0].set_ylim(-0.2, 1.8)
axs[0].legend(loc='upper right', frameon=True)
axs[0].grid(True, alpha=0.3)

train_times_min = [14.28, 15.27, 16.02, 17.55, 18.73, 20.00, 21.02, 23.13, 62.01]
for i, tm in enumerate(train_times_min):
    axs[0].annotate(f"T{i+1}", xy=(tm, 1.15), xytext=(tm, 1.4),
                    arrowprops=dict(arrowstyle="->", color='#d95f02', lw=1.2),
                    fontsize=9, fontweight='bold', color='#d95f02', ha='center')

# Panel 2: Zoom on Train 1 vs Channel 7
t1_s = int(855.5 * fs)
t1_e = int(859.5 * fs)
t_t1 = (np.arange(t1_s, t1_e) - t1_s) / fs # seconds from 855.5s
v6_t1 = data[t1_s:t1_e, 6] * volts_per_count
v7_t1 = data[t1_s:t1_e, 7] * volts_per_count

axs[1].plot(t_t1, v6_t1, color='#2b5c8f', lw=1.2, label='Ch 6 (20 Hz, 2s stim, 40 pulses)')
axs[1].plot(t_t1, v7_t1, color='#888888', lw=1.0, alpha=0.7, label='Ch 7 (Flat DC ~1.34V)')
axs[1].set_title("Train 1 Zoom (t = 856.83 s / 14:16.8): 2.0 s duration, 20 Hz, 5 ms pulses", fontsize=11, fontweight='bold')
axs[1].set_xlabel("Time within window (seconds)", fontsize=10)
axs[1].set_ylabel("Voltage (V)", fontsize=10)
axs[1].set_ylim(-0.2, 1.6)
axs[1].legend(loc='upper right', frameon=True)
axs[1].grid(True, alpha=0.3)

# Panel 3: Zoom on 5-second Train 4
t4_s = int(1052.5 * fs)
t4_e = int(1059.0 * fs)
t_t4 = (np.arange(t4_s, t4_e) - t4_s) / fs
v6_t4 = data[t4_s:t4_e, 6] * volts_per_count

axs[2].plot(t_t4, v6_t4, color='#e7298a', lw=1.2, label='Ch 6 (20 Hz, 5s stim, 100 pulses)')
axs[2].set_title("Train 4 Zoom (t = 1053.16 s / 17:33.2): 5.0 s duration, 20 Hz, 100 pulses", fontsize=11, fontweight='bold')
axs[2].set_xlabel("Time within window (seconds)", fontsize=10)
axs[2].set_ylabel("Voltage (V)", fontsize=10)
axs[2].set_ylim(-0.2, 1.3)
axs[2].legend(loc='upper right', frameon=True)
axs[2].grid(True, alpha=0.3)

# Panel 4: Single pulse zoom showing width and peak
p_s = int(856.827 * fs) - 30
p_e = p_s + 120
t_p = (np.arange(p_s, p_e) - (int(856.827 * fs))) / fs * 1000.0 # ms relative to onset
v6_p = data[p_s:p_e, 6] * volts_per_count

axs[3].plot(t_p, v6_p, 'o-', color='#1b9e77', lw=1.5, ms=3, label='Ch 6 Single Pulse (10 kHz sampling)')
axs[3].axvspan(0, 5.0, color='#1b9e77', alpha=0.15, label='5.0 ms Pulse Width')
axs[3].set_title("Single Pulse Profile (Onset at t = 856.827 s): Width = 5.00 ms, Peak = 1.033 V", fontsize=11, fontweight='bold')
axs[3].set_xlabel("Time from pulse onset (ms)", fontsize=10)
axs[3].set_ylabel("Voltage (V)", fontsize=10)
axs[3].set_ylim(-0.1, 1.2)
axs[3].legend(loc='upper right', frameon=True)
axs[3].grid(True, alpha=0.3)

plt.savefig(out_fig, dpi=300, bbox_inches='tight')
print(f"Saved figure to {out_fig}")

