import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(r'c:\Users\lpascu\resp_analysis')
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.processing.analog_nidq.proc_emg import process_emg
from src.pipeline.processing.analog_nidq.proc_flow import process_flow_meter
from src.pipeline.config import EMGConfig

nidq_file = Path(r'Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin')
fs = 10000.0
n_ch = 9
raw = np.memmap(nidq_file, dtype='int16', mode='r')

windows = [
    ('win1_baseline', 1740.0, 60.0),
    ('win2_anox1', 1800.0, 36.0),
    ('win3_anox2', 1836.0, 24.0),
    ('win4_gasp_early', 1860.0, 90.0),
    ('win5_gasp_late', 1950.0, 120.0),
    ('win6_flow_shift', 2070.0, 66.0),
    ('win7_recovery', 2136.0, 24.0),
]

for wid, t0, dur in windows:
    s0 = int(t0 * fs)
    s1 = int((t0 + dur) * fs)
    slice_data = raw[s0 * n_ch : s1 * n_ch].reshape(-1, n_ch)
    emg_v = slice_data[:, 0].astype(np.float32) * (10.0 / 32768.0)
    flow_v = slice_data[:, 5].astype(np.float32) * (10.0 / 32768.0)
    t_vec = np.arange(len(emg_v)) / fs
    emg_cfg = EMGConfig(fs=fs, highpass_cutoff=300.0, lowpass_cutoff=3000.0, smoothing_window_ms=50.0, peak_threshold_sd=2.0)
    emg_proc = process_emg(emg_v, t_vec, emg_cfg)
    valid_pairs = [(on, off) for on, off in zip(emg_proc.breath_onsets, emg_proc.breath_offsets) if on > 0.05 and on < t_vec[-1] - 0.05 and off > on]
    valid_onsets = np.array([p[0] for p in valid_pairs])
    valid_offsets = np.array([p[1] for p in valid_pairs])
    if len(valid_onsets) > 0:
        flow_res = process_flow_meter(flow_v, t_vec, valid_onsets, valid_offsets, fs=fs)
        ti_arr = (flow_res.flow_zero_crossings - valid_onsets) * 1000.0
        print(f'{wid} ({t0}-{t0+dur}s): {len(valid_onsets)} breaths. Mean T_I = {np.mean(ti_arr):.1f} +/- {np.std(ti_arr):.1f} ms.')
    else:
        print(f'{wid} ({t0}-{t0+dur}s): 0 breaths (apnea).')
