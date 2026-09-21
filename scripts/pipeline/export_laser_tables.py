"""
Export laser stimulation train metadata and pulse onset times to CSV.
"""
from pathlib import Path
import numpy as np
import pandas as pd

bin_path = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
fs = 9999.955307
volts_per_count = 10.0 / 32768.0
total_samples = 66072893

data = np.memmap(bin_path, dtype=np.int16, mode='r', shape=(total_samples, 9))

# Threshold channel 6
ch6 = data[:, 6] * volts_per_count
diff = np.diff((ch6 > 0.5).astype(np.int8))
r_edges = np.where(diff == 1)[0]
f_edges = np.where(diff == -1)[0]

# Train separation (gap > 1.0 s)
isi_s = np.diff(r_edges) / fs
breaks = np.where(isi_s > 1.0)[0]

train_starts = [r_edges[0]] + list(r_edges[breaks + 1])
train_ends = list(r_edges[breaks]) + [r_edges[-1]]

log_notes = [
    "14:00 Bilateral light stim (effective, brief apnea, rapid recovery)",
    "Doing a few more (train 2)",
    "Doing a few more (train 3, before 16:30 recovery pause)",
    "17:30 A 5 second stim (severe, rapid recovery)",
    "18:40 Another 5s stim (differentiating insp-off vs pre-insp)",
    "20:00 Another 5s stim (testing AP75 expiratory neuron)",
    "Another 5s stim (train 7)",
    "Another 5s stim (train 8, before 24:00 done with stims)",
    "1:02:00 One more long laser stim before seizure (10s stim)"
]

train_rows = []
for i, (s, e) in enumerate(zip(train_starts, train_ends)):
    p_indices = np.where((r_edges >= s) & (r_edges <= e))[0]
    n_p = len(p_indices)
    t_start = s / fs
    f_last = f_edges[np.where(f_edges >= e)[0][0]]
    t_end = f_last / fs
    dur = t_end - t_start
    m, sec = divmod(t_start, 60)
    
    pulse_widths = (f_edges[p_indices] - r_edges[p_indices]) / fs * 1000.0
    p_peaks = [ch6[r_edges[p]:f_edges[p]+1].max() for p in p_indices]
    
    train_rows.append({
        "train_index": i + 1,
        "start_sample": int(s),
        "start_time_s": round(t_start, 4),
        "start_time_mm_ss": f"{int(m):02d}:{sec:06.3f}",
        "end_sample": int(f_last),
        "end_time_s": round(t_end, 4),
        "duration_s": round(dur, 4),
        "pulse_count": n_p,
        "frequency_hz": round(1.0 / np.mean(np.diff(r_edges[p_indices]) / fs), 2) if n_p > 1 else 0.0,
        "pulse_width_mean_ms": round(float(np.mean(pulse_widths)), 2),
        "pulse_peak_mean_v": round(float(np.mean(p_peaks)), 3),
        "experimental_log_context": log_notes[i]
    })

df_trains = pd.DataFrame(train_rows)
out_csv = Path(r"C:\Users\lpascu\analysis_src\e097\e097_g2_laser_trains.csv")
df_trains.to_csv(out_csv, index=False)
print(f"Saved {len(df_trains)} trains to {out_csv}")

# Also save all individual pulse onsets
pulse_rows = []
for i, (r, f) in enumerate(zip(r_edges, f_edges)):
    t_on = r / fs
    dur = (f - r) / fs * 1000.0
    # find which train it belongs to
    tr_id = 0
    for tid, (ts, te) in enumerate(zip(train_starts, train_ends)):
        if ts <= r <= te:
            tr_id = tid + 1
            break
    pulse_rows.append({
        "pulse_index": i + 1,
        "train_index": tr_id,
        "sample_onset": int(r),
        "time_onset_s": round(t_on, 5),
        "duration_ms": round(dur, 3),
        "peak_v": round(float(ch6[r:f+1].max()), 3)
    })

df_pulses = pd.DataFrame(pulse_rows)
out_pulses_csv = Path(r"C:\Users\lpascu\analysis_src\e097\e097_g2_laser_pulse_onsets.csv")
df_pulses.to_csv(out_pulses_csv, index=False)
print(f"Saved {len(df_pulses)} pulses to {out_pulses_csv}")

