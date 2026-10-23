"""
Scan AP band RMS across all 384 channels in e097_g2 to map where active neural units are located.
Window: 60 seconds leading up to first laser stim (t=796.827s to 856.827s).
"""
from pathlib import Path
import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap_bin = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_imec0\e097_g2_t0.imec0.ap.bin")
ap_meta = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_imec0\e097_g2_t0.imec0.ap.meta")

fs = 30000.0
n_chans = 385 # 384 neural + 1 sync

t_laser = 856.827
t_start = t_laser - 60.0
t_end = t_laser

s_start = int(t_start * fs)
s_end = int(t_end * fs)
num_samples = s_end - s_start

print(f"AP window: {t_start:.3f}s to {t_end:.3f}s | {num_samples} samples ({num_samples/fs:.1f}s)")

# We can sample 2 seconds from this window to quickly compute AP RMS across all channels
# e.g., 10 chunks of 0.2s spread across the 60s window
chunk_len = int(0.2 * fs)
n_chunks = 10
chunk_starts = np.linspace(s_start, s_end - chunk_len, n_chunks, dtype=int)

# memmap
file_bytes = ap_bin.stat().st_size
total_file_samples = file_bytes // (n_chans * 2)

data = np.memmap(ap_bin, dtype=np.int16, mode='r', shape=(total_file_samples, n_chans))

# Conversion factor: range is [-0.62, 0.62] V, 10-bit or 16-bit int (rangeMax=0.62, maxInt=512 for NP2 or 8192?)
# Let's check meta for uV per count
# imAiRangeMax=0.62, gain=100 or NP2: 0.62 / (512 * 100) or 0.62 / 8192
uV_per_count = (0.62 / 8192.0) * 1e6 # approx ~7.57 uV/count for NP2 or 1.2 uV/count

all_rms = np.zeros(384)

b, a = signal.butter(3, [300, 6000], btype='bandpass', fs=fs)

for cs in chunk_starts:
    raw = data[cs : cs + chunk_len, :384].astype(np.float32)
    # Common median reference across channels
    cmr = raw - np.median(raw, axis=1, keepdims=True)
    filt = signal.filtfilt(b, a, cmr, axis=0)
    all_rms += np.sqrt(np.mean(filt**2, axis=0))

all_rms /= n_chunks

print(f"AP RMS across 384 channels: min={all_rms.min():.2f}, max={all_rms.max():.2f}, median={all_rms.median() if hasattr(all_rms, 'median') else np.median(all_rms):.2f}")

# Find channels with high RMS / spiking activity
high_act_chans = np.where(all_rms > np.median(all_rms) * 1.5)[0]
print(f"Channels with elevated AP band RMS (> 1.5x median): {len(high_act_chans)}")
if len(high_act_chans) > 0:
    print(f"Ranges: min ch = {high_act_chans.min()}, max ch = {high_act_chans.max()}")

plt.figure(figsize=(12, 4))
plt.plot(all_rms, color='#1f77b4', lw=1.2)
plt.axhline(np.median(all_rms), color='gray', linestyle='--', label=f'Median: {np.median(all_rms):.1f}')
plt.xlabel("Channel Index (0 - 383)")
plt.ylabel("Filtered AP RMS (counts)")
plt.title(f"e097_g2 AP Band RMS Profile across all 384 Channels (t={t_start:.1f} to {t_end:.1f}s)")
plt.legend()
plt.grid(True, alpha=0.3)
out_png = Path(r"C:\Users\lpascu\analysis_src\e097\e097_ap_rms_profile.png")
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print(f"Saved AP RMS profile to {out_png}")

