#!/usr/bin/env python3
"""
Whole-session breathing, heart-rate and laser summary for e097_g2, written to <outputs>/e097/e097_whole_session_data.json.

    python e097_session_summary.py          (npxsort or cibrrig env; ~2 min, reads the 1.2 GB NIDQ file)

anoxia_recovery.py (--breathing-json) and the optotag configs (session_json) read this file. It was written on
2026-10-06 as the data step of an interactive session dashboard (the HTML builder is in archive/scripts/experiments/
e097/build_whole_session_html.py); the milestones and laser trains are typed in from the experiment log, e097.docx.

- Processes the complete 110.1-minute NIDQ session (66,072,893 samples at 10 kHz).
- Extracts Diaphragm EMG envelope and breath onsets.
- Computes cycle-by-cycle Inter-Breath Intervals (IBI, seconds) and Respiratory Rate (BPM).
- Runs Pan-Tompkins QRS R-peak detection on ECG to compute Instantaneous Heart Rate (BPM).
- Identifies all 9 Laser Stimulation Trains and timing parameters.
- Integrates all experimental protocol details and surgical log observations from e097.docx.
"""

import sys
import time
import json
import base64
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sorting"))
from npxsort.paths import path  # noqa: E402

BIN_PATH = path("raw_data") / "e097_g2" / "e097_g2_t0.nidq.bin"
fs = 9999.955307
volts_per_count = 10.0 / 32768.0
n_chans = 9

out_dir = path("outputs") / "e097"
out_json = out_dir / "e097_whole_session_data.json"


def process_whole_session():
    t_start_all = time.time()
    total_samples = BIN_PATH.stat().st_size // (n_chans * 2)
    duration_s = total_samples / fs
    print(f"Loading {BIN_PATH.name} ({total_samples:,} samples, {duration_s:.1f} s, {duration_s/60:.1f} min)...")
    
    raw = np.memmap(BIN_PATH, dtype=np.int16, mode='r', shape=(total_samples, n_chans))
    
    # 1. Chunked Processing for ECG R-peaks & EMG Breaths
    # Chunk size: 30 seconds to maintain stable local adaptive thresholds
    chunk_s = 30.0
    chunk_pts = int(chunk_s * fs)
    n_chunks = int(np.ceil(total_samples / chunk_pts))
    
    # Filter coefficients
    # ECG: 10-100 Hz bandpass
    sos_ecg = butter(3, [10.0 / (0.5 * fs), 100.0 / (0.5 * fs)], btype='bandpass', output='sos')
    # EMG raw: 300-3000 Hz bandpass
    sos_emg = butter(2, [300.0 / (0.5 * fs), 3000.0 / (0.5 * fs)], btype='bandpass', output='sos')
    
    all_r_peaks = []
    all_breath_onsets = []
    all_breath_peaks = []
    
    # Overview downsampling: 20 Hz envelope (500 samples per bin)
    step_ov = 500
    n_bins_ov = total_samples // step_ov
    t_ov = (np.arange(n_bins_ov, dtype=np.float32) * (step_ov / fs))
    
    ov_c0_min = np.zeros(n_bins_ov, dtype=np.float32)
    ov_c0_max = np.zeros(n_bins_ov, dtype=np.float32)
    ov_c1_env = np.zeros(n_bins_ov, dtype=np.float32) # integrated EMG
    ov_c2_min = np.zeros(n_bins_ov, dtype=np.float32) # ECG min
    ov_c2_max = np.zeros(n_bins_ov, dtype=np.float32) # ECG max
    ov_c5_flow = np.zeros(n_bins_ov, dtype=np.float32) # Flow
    ov_c6_laser = np.zeros(n_bins_ov, dtype=np.float32) # Laser
    
    print(f"Executing chunked extraction across {n_chunks} blocks...")
    
    for c_idx in range(n_chunks):
        s0 = c_idx * chunk_pts
        s1 = min(total_samples, s0 + chunk_pts)
        t_offset = s0 / fs
        
        c0 = raw[s0:s1, 0].astype(np.float32) * volts_per_count
        c1 = raw[s0:s1, 1].astype(np.float32) * volts_per_count
        c2 = raw[s0:s1, 2].astype(np.float32) * volts_per_count
        c5 = raw[s0:s1, 5].astype(np.float32) * volts_per_count
        c6 = raw[s0:s1, 6].astype(np.float32) * volts_per_count
        
        # A. Fill downsampled envelopes
        b0 = s0 // step_ov
        n_bins_chunk = len(c0) // step_ov
        n_sub = n_bins_chunk * step_ov
        if n_sub > 0 and b0 + n_bins_chunk <= n_bins_ov:
            b1 = b0 + n_bins_chunk
            sub0 = c0[:n_sub].reshape(n_bins_chunk, step_ov)
            sub1 = c1[:n_sub].reshape(n_bins_chunk, step_ov)
            sub2 = c2[:n_sub].reshape(n_bins_chunk, step_ov)
            sub5 = c5[:n_sub].reshape(n_bins_chunk, step_ov)
            sub6 = c6[:n_sub].reshape(n_bins_chunk, step_ov)
            
            ov_c0_min[b0:b1] = sub0.min(axis=1)
            ov_c0_max[b0:b1] = sub0.max(axis=1)
            ov_c1_env[b0:b1] = sub1.mean(axis=1)
            ov_c2_min[b0:b1] = sub2.min(axis=1)
            ov_c2_max[b0:b1] = sub2.max(axis=1)
            ov_c5_flow[b0:b1] = sub5.mean(axis=1)
            ov_c6_laser[b0:b1] = sub6.max(axis=1)
            
        # B. ECG R-Peak Detection
        if len(c2) > int(fs * 2):
            filt_c2 = sosfiltfilt(sos_ecg, c2)
            diff_c2 = np.diff(filt_c2, prepend=filt_c2[0]) * fs
            int_win = max(1, int(0.015 * fs))
            kernel = np.ones(int_win) / int_win
            energy_c2 = np.convolve(diff_c2**2, kernel, mode='same')
            
            std_en = np.std(energy_c2)
            if std_en > 1e-5:
                thresh_ecg = np.mean(energy_c2) + 1.1 * std_en
                min_rr_dist = int(0.075 * fs) # max 800 bpm for mouse
                pks_ecg, _ = find_peaks(energy_c2, height=thresh_ecg, distance=min_rr_dist)
                all_r_peaks.extend(t_offset + pks_ecg / fs)
                
        # C. Diaphragm Breath Detection on Integrated EMG (c1) + smoothed rectified (c0)
        if len(c1) > int(fs * 2):
            # Smooth c1 slightly
            win_sm = max(3, int(0.02 * fs))
            c1_sm = np.convolve(c1, np.ones(win_sm)/win_sm, mode='same')
            std_c1 = np.std(c1_sm)
            if std_c1 > 1e-4:
                thresh_dia = np.mean(c1_sm) + 1.0 * std_c1
                min_ibi_dist = int(0.10 * fs) # max 10 Hz (600 breaths/min)
                pks_dia, _ = find_peaks(c1_sm, height=thresh_dia, distance=min_ibi_dist)
                
                # For each peak, find local onset (valley before peak)
                for pk in pks_dia:
                    lookback = max(0, pk - int(0.08 * fs))
                    if pk > lookback:
                        onset_rel = lookback + np.argmin(c1_sm[lookback:pk])
                        all_breath_onsets.append(t_offset + onset_rel / fs)
                        all_breath_peaks.append(t_offset + pk / fs)
                        
        if (c_idx + 1) % 25 == 0 or c_idx == n_chunks - 1:
            print(f"  Processed chunk {c_idx+1}/{n_chunks} ({t_offset/60:.1f} min) | Total Breaths: {len(all_breath_onsets):,} | ECG R-Peaks: {len(all_r_peaks):,}")

    all_r_peaks = np.sort(np.unique(np.array(all_r_peaks)))
    all_breath_onsets = np.sort(np.unique(np.array(all_breath_onsets)))
    
    # 2. Compute Heart Rate Tachogram
    rr_intervals_s = np.diff(all_r_peaks)
    hr_bpm_raw = 60.0 / rr_intervals_s
    hr_times = all_r_peaks[1:]
    
    # Physiological filter (50 to 800 BPM)
    valid_hr = (hr_bpm_raw >= 50.0) & (hr_bpm_raw <= 800.0)
    hr_times = hr_times[valid_hr]
    hr_bpm = hr_bpm_raw[valid_hr]
    
    # 3. Compute Inter-Breath Interval (IBI) and Respiratory Rate
    ibi_s_raw = np.diff(all_breath_onsets)
    resp_rate_bpm_raw = 60.0 / ibi_s_raw
    resp_times = all_breath_onsets[1:]
    
    # Filter spurious double-detections (< 60 ms)
    valid_resp = (ibi_s_raw >= 0.06) & (ibi_s_raw <= 30.0)
    resp_times = resp_times[valid_resp]
    ibi_s = ibi_s_raw[valid_resp]
    resp_rate_bpm = resp_rate_bpm_raw[valid_resp]
    
    print(f"Extracted {len(all_breath_onsets):,} breath onsets and {len(all_r_peaks):,} ECG heartbeats.")
    
    # 4. Laser Trains Metadata from export_laser_tables or raw detect
    laser_trains = [
        {"id": 1, "t_start": 856.827, "dur": 1.955, "pulses": 40, "freq": 20.0, "note": "14:00 Bilateral light stim (effective, brief apnea, rapid recovery)"},
        {"id": 2, "t_start": 916.491, "dur": 1.955, "pulses": 40, "freq": 20.0, "note": "Train 2 (2s 20Hz)"},
        {"id": 3, "t_start": 961.115, "dur": 1.955, "pulses": 40, "freq": 20.0, "note": "Train 3 (2s 20Hz, before 16:30 pause)"},
        {"id": 4, "t_start": 1053.155, "dur": 4.955, "pulses": 100, "freq": 20.0, "note": "17:30 A 5 second stim (severe, rapid recovery)"},
        {"id": 5, "t_start": 1124.076, "dur": 4.955, "pulses": 100, "freq": 20.0, "note": "18:40 Another 5s stim (differentiating insp-off vs pre-insp)"},
        {"id": 6, "t_start": 1200.236, "dur": 4.955, "pulses": 100, "freq": 20.0, "note": "20:00 Another 5s stim (testing AP75 expiratory neuron)"},
        {"id": 7, "t_start": 1261.012, "dur": 4.955, "pulses": 100, "freq": 20.0, "note": "Train 7 (5s 20Hz)"},
        {"id": 8, "t_start": 1387.981, "dur": 4.955, "pulses": 100, "freq": 20.0, "note": "Train 8 (5s 20Hz, before 24:00 done with stims)"},
        {"id": 9, "t_start": 3720.527, "dur": 9.955, "pulses": 200, "freq": 20.0, "note": "1:02:00 One more long laser stim before seizure (10s stim)"}
    ]

    # 5. Experimental Epoch Milestones from e097.docx
    milestones = [
        {"name": "Tissue Settling / Baseline", "t_start": 0.0, "t_end": 840.0, "badge": "BASELINE", "desc": "Pre-stimulus baseline data. Excellent VRC signal quality on shank 1."},
        {"name": "Laser Opto Epoch (Trains 1-8)", "t_start": 840.0, "t_end": 1440.0, "badge": "LASER STIM", "desc": "Bilateral light stimulation over VRC/vestibular targets. Rapid-recovery apneas."},
        {"name": "Post-Stim Recovery", "t_start": 1440.0, "t_end": 1800.0, "badge": "RECOVERY", "desc": "Recovery pause; transitioning animal to nitrogen challenge."},
        {"name": "15s Anoxia #1 (Gas Switch)", "t_start": 1800.0, "t_end": 1815.0, "badge": "ANOXIA 1", "desc": "15-second N2 challenge. Strong gasping and return of rostral gasping group."},
        {"name": "Active Expiration Shift", "t_start": 2078.0, "t_end": 2150.0, "badge": "FLOW SHIFT", "desc": "Active expiration transition in airflow waveform."},
        {"name": "15s Anoxia #2 (Gas Switch)", "t_start": 2310.0, "t_end": 2325.0, "badge": "ANOXIA 2", "desc": "Second 15s anoxia challenge with expiratory gasping dynamics."},
        {"name": "Diving Reflex #1", "t_start": 2760.0, "t_end": 2820.0, "badge": "DR 1", "desc": "Facemask water application testing diving reflex."},
        {"name": "Noise / Ground Adjustment", "t_start": 3060.0, "t_end": 3240.0, "badge": "MAINTENANCE", "desc": "Scooted and reclipped ground clips to eliminate noise dropouts."},
        {"name": "Diving Reflex #2 (Severe Apnea)", "t_start": 3330.0, "t_end": 3450.0, "badge": "DR 2", "desc": "Strong prolonged diving reflex apnea; facemask removed during stimulus."},
        {"name": "Pre-Seizure Long Laser Stim (Train 9)", "t_start": 3720.0, "t_end": 3730.0, "badge": "LASER 10S", "desc": "10-second 20Hz laser stim test prior to chemoconvulsant administration."},
        {"name": "Kainic Acid Injection (12 mg/kg)", "t_start": 3990.0, "t_end": 4050.0, "badge": "KA INJECTION", "desc": "KA delivered to induce status epilepticus and terminal respiratory arrest."},
        {"name": "Diaphragm Tonicity / Pre-Seizure", "t_start": 4603.0, "t_end": 5400.0, "badge": "SEIZURE TONICITY", "desc": "Early diaphragm tonicity emerging. Increasing amplitude and frequency."},
        {"name": "Massive Seizure Bursts & Agony", "t_start": 5400.0, "t_end": 6480.0, "badge": "STATUS EPILEPTICUS", "desc": "Huge diaphragm bursts, ECG tonicity, severe convulsive respiratory motor discharge."},
        {"name": "Terminal Apnea & Post-Mortem", "t_start": 6480.0, "t_end": 6607.0, "badge": "TERMINAL ARREST", "desc": "Full hindlimb extension, terminal apnea, irreversible arrest (1:48:00 post-start)."}
    ]

    # 6. Decimate Time-Series for Smooth Browser Rendering (1-second time bins)
    bin_s = 1.0
    t_timeline = np.arange(0, duration_s, bin_s)
    n_t = len(t_timeline)
    
    # Binned Heart Rate
    binned_hr = np.full(n_t, np.nan, dtype=np.float32)
    hr_bin_idx = np.clip((hr_times / bin_s).astype(int), 0, n_t - 1)
    for b in range(n_t):
        m = (hr_bin_idx == b)
        if np.any(m):
            binned_hr[b] = np.mean(hr_bpm[m])
            
    # Interpolate rare NaNs for smooth plotting
    valid_hr_mask = ~np.isnan(binned_hr)
    if np.any(valid_hr_mask):
        binned_hr[~valid_hr_mask] = np.interp(t_timeline[~valid_hr_mask], t_timeline[valid_hr_mask], binned_hr[valid_hr_mask])

    # Binned Respiration Rate & IBI
    binned_rr = np.full(n_t, np.nan, dtype=np.float32)
    binned_ibi = np.full(n_t, np.nan, dtype=np.float32)
    resp_bin_idx = np.clip((resp_times / bin_s).astype(int), 0, n_t - 1)
    for b in range(n_t):
        m = (resp_bin_idx == b)
        if np.any(m):
            binned_rr[b] = np.mean(resp_rate_bpm[m])
            binned_ibi[b] = np.mean(ibi_s[m])
            
    valid_rr_mask = ~np.isnan(binned_rr)
    if np.any(valid_rr_mask):
        binned_rr[~valid_rr_mask] = np.interp(t_timeline[~valid_rr_mask], t_timeline[valid_rr_mask], binned_rr[valid_rr_mask])
        binned_ibi[~valid_rr_mask] = np.interp(t_timeline[~valid_rr_mask], t_timeline[valid_rr_mask], binned_ibi[valid_rr_mask])

    # Downsample IBI scatter points for snappy plotting (max 3000 points across the whole session)
    step_scatter = max(1, len(resp_times) // 3000)
    ibi_scatter_times = [round(float(t), 2) for t in resp_times[::step_scatter]]
    ibi_scatter_vals = [round(float(v), 3) for v in ibi_s[::step_scatter]]

    # High-resolution 1kHz blocks for key epochs (Laser 1, Anoxia 1, DR 2, Seizure Burst)
    def extract_hi_res_block(t0, t1):
        s_0 = int(t0 * fs)
        s_1 = int(t1 * fs)
        step_1k = 10 # 10 kHz -> 1 kHz
        t_arr = (np.arange(s_0, s_1, step_1k) / fs).astype(np.float32)
        c0_arr = (raw[s_0:s_1:step_1k, 0].astype(np.float32) * volts_per_count)
        c1_arr = (raw[s_0:s_1:step_1k, 1].astype(np.float32) * volts_per_count)
        c2_arr = (raw[s_0:s_1:step_1k, 2].astype(np.float32) * volts_per_count)
        c6_arr = (raw[s_0:s_1:step_1k, 6].astype(np.float32) * volts_per_count)
        return {
            "t0": t0,
            "t1": t1,
            "t": [round(float(x), 3) for x in t_arr],
            "emg": [round(float(x), 4) for x in c0_arr],
            "int_emg": [round(float(x), 4) for x in c1_arr],
            "ecg": [round(float(x), 4) for x in c2_arr],
            "laser": [round(float(x), 4) for x in c6_arr]
        }
        
    print("Extracting high-resolution 1 kHz diagnostic snapshots...")
    snap_laser1 = extract_hi_res_block(850.0, 868.0) # Laser 1
    snap_anoxia1 = extract_hi_res_block(1800.0, 1835.0) # Anoxia 1
    snap_dr2 = extract_hi_res_block(3325.0, 3355.0) # DR 2
    snap_seizure = extract_hi_res_block(5420.0, 5450.0) # Seizure

    # Compile JSON bundle
    data_payload = {
        "metadata": {
            "animal_id": "e097 (e096)",
            "strain": "Vglut2:ai32",
            "weight_g": 35.8,
            "sex": "Male",
            "date": "2026-09-04",
            "surgeon": "Ryan",
            "anesthesia": "Isoflurane induction, Urethane 1.2 g/kg primary + 0.15 g/kg supp",
            "probe": "Neuropixels 2.0 (4-shank, Shank 1 active, 4500 um depth)",
            "target": "Ventral Respiratory Column (VRC) / preBötC / rVRG",
            "total_duration_s": round(duration_s, 2),
            "total_duration_min": round(duration_s / 60.0, 2),
            "total_breaths_detected": len(all_breath_onsets),
            "total_heartbeats_detected": len(all_r_peaks),
            "baseline_hr_bpm": round(float(np.median(hr_bpm[(hr_times >= 100) & (hr_times <= 800)])), 1),
            "baseline_rr_bpm": round(float(np.median(resp_rate_bpm[(resp_times >= 100) & (resp_times <= 800)])), 1),
            "baseline_ibi_s": round(float(np.median(ibi_s[(resp_times >= 100) & (resp_times <= 800)])), 3),
        },
        "laser_trains": laser_trains,
        "milestones": milestones,
        "timeline": {
            "time_s": [round(float(t), 1) for t in t_timeline],
            "time_min": [round(float(t / 60.0), 2) for t in t_timeline],
            "hr_bpm": [round(float(v), 1) for v in binned_hr],
            "resp_rate_bpm": [round(float(v), 1) for v in binned_rr],
            "ibi_s": [round(float(v), 3) for v in binned_ibi],
            "emg_envelope": [round(float(v), 3) for v in np.interp(t_timeline, t_ov, ov_c1_env)],
            "laser_state": [round(float(v), 3) for v in np.interp(t_timeline, t_ov, ov_c6_laser)]
        },
        "ibi_scatter": {
            "time_s": ibi_scatter_times,
            "time_min": [round(t / 60.0, 2) for t in ibi_scatter_times],
            "ibi_s": ibi_scatter_vals
        },
        "snapshots": {
            "laser1": snap_laser1,
            "anoxia1": snap_anoxia1,
            "dr2": snap_dr2,
            "seizure": snap_seizure
        }
    }
    
    print(f"Data payload compiled successfully in {time.time()-t_start_all:.1f}s.")
    return data_payload


if __name__ == "__main__":
    payload = process_whole_session()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"Saved {out_json}")
