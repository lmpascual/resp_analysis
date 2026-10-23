import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, find_peaks

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.io.io_nidq import read_nidq_bin

meta_path = Path(r'Z:\ramirezlab\pascual\rec_data\e086_g0\e086_g0_t0.nidq.meta')
bin_path = Path(r'Z:\ramirezlab\pascual\rec_data\e086_g0\e086_g0_t0.nidq.bin')

analog_data = read_nidq_bin(bin_path, meta_path)
t_full = analog_data.time_vector
fs = analog_data.fs

win_start = 2985.0
idx_win = np.where(t_full >= win_start)[0]
t = t_full[idx_win]
ecg_raw = analog_data.all_channels[1][idx_win]
opto1_raw = analog_data.all_channels[4][idx_win]
opto2_raw = analog_data.all_channels[6][idx_win]

# Bandpass filter (10-100 Hz)
nyq = 0.5 * fs
b, a = butter(3, [10.0 / nyq, 100.0 / nyq], btype='bandpass')
filtered = filtfilt(b, a, ecg_raw)
b_notch, a_notch = iirnotch(60.0 / nyq, Q=30)
filtered = filtfilt(b_notch, a_notch, filtered)

# QRS energy
diff_ecg = np.diff(filtered, prepend=filtered[0]) * fs
sq_ecg = diff_ecg ** 2
win_int = int(0.02 * fs)
kernel = np.ones(win_int) / win_int
ecg_energy = np.convolve(sq_ecg, kernel, mode='same')

trial1_onset_2nd = 4087.702
trial2_onset_2nd = 4216.340

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)

# Trial 1 zoomed (4085 to 4092s)
mask1 = (t >= trial1_onset_2nd - 2.0) & (t <= trial1_onset_2nd + 5.0)
axes[0].plot(t[mask1], ecg_raw[mask1], color='crimson', label='Raw ECG (Ch 1)')
axes[0].plot(t[mask1], opto1_raw[mask1]*0.2 - 1.5, color='red', alpha=0.7, label='Laser 1 (Ch 4)')
axes[0].axvline(trial1_onset_2nd, color='black', linestyle='--', label='2nd Laser Onset')
axes[0].set_title('Trial 1: Raw ECG Waveform at Laser Onset (4085.7s to 4092.7s)')
axes[0].legend(loc='upper right')
axes[0].grid(True)

axes[1].plot(t[mask1], ecg_energy[mask1], color='purple', label='QRS Energy')
# compute threshold in this 10s segment
seg_energy = ecg_energy[mask1]
thresh1 = np.mean(seg_energy) + 1.2 * np.std(seg_energy)
axes[1].axhline(thresh1, color='orange', linestyle='--', label=f'Local Thresh (1.2 SD = {thresh1:.1f})')
axes[1].set_title('Trial 1: QRS Energy & Threshold')
axes[1].legend(loc='upper right')
axes[1].grid(True)

# Trial 2 zoomed (4214 to 4221s)
mask2 = (t >= trial2_onset_2nd - 2.0) & (t <= trial2_onset_2nd + 5.0)
axes[2].plot(t[mask2], ecg_raw[mask2], color='crimson', label='Raw ECG (Ch 1)')
axes[2].plot(t[mask2], opto1_raw[mask2]*0.2 - 1.5, color='red', alpha=0.7, label='Laser 1 (Ch 4)')
axes[2].axvline(trial2_onset_2nd, color='black', linestyle='--', label='2nd Laser Onset')
axes[2].set_title('Trial 2: Raw ECG Waveform at Laser Onset (4214.3s to 4221.3s)')
axes[2].legend(loc='upper right')
axes[2].grid(True)

axes[3].plot(t[mask2], ecg_energy[mask2], color='purple', label='QRS Energy')
seg_energy2 = ecg_energy[mask2]
thresh2 = np.mean(seg_energy2) + 1.2 * np.std(seg_energy2)
axes[3].axhline(thresh2, color='orange', linestyle='--', label=f'Local Thresh (1.2 SD = {thresh2:.1f})')
axes[3].set_title('Trial 2: QRS Energy & Threshold')
axes[3].legend(loc='upper right')
axes[3].grid(True)

plt.tight_layout()
plt.savefig(r'c:\Users\lpascu\analysis_src\output_e086\debug_laser_ecg_zoom.png', dpi=200)
print("Saved debug zoom plot to c:\\Users\\lpascu\\analysis_src\\output_e086\\debug_laser_ecg_zoom.png")
