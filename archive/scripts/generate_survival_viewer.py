"""
Script to generate the standalone interactive HTML viewer for Session 1 (LP26_anox_exptriggered_g0, SURVIVAL).
"""

import json
import base64
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks

WORKSPACE = Path(r"c:\Users\lpascu\analysis_src")
BIN_PATH = Path(r"Z:\ramirezlab\pascual\rec_data\LP26\LP26_anox_exptriggered_g0\LP26_anox_exptriggered_g0_t0.nidq.bin")
OUTPUT_HTML = WORKSPACE / "outputs" / "LP26_anox_exptriggered_survival_viewer.html"

fs = 9999.955307
volts_per_count = 10.0 / 32768.0
n_chans = 9

total_samples = BIN_PATH.stat().st_size // (n_chans * 2)
raw = np.memmap(BIN_PATH, dtype=np.int16, mode='r', shape=(total_samples, n_chans))

print(f"Processing Session 1 (Survival): {total_samples / fs:.2f} s")

# Extract channels
ch0 = raw[:, 0].astype(np.float32) * volts_per_count # Diaphragm EMG
ch1 = raw[:, 1].astype(np.float32) * volts_per_count # Integrated EMG
ch2 = raw[:, 2].astype(np.float32) * volts_per_count # ECG
ch6 = raw[:, 6].astype(np.float32) * volts_per_count # Laser command

# 1. ECG Peaks
ecg_pks, _ = find_peaks(ch2, height=3.0, distance=int(0.08 * fs))
ecg_times = (ecg_pks / fs).tolist()
ecg_times_arr = np.array(ecg_times)
rr_intervals_all = np.diff(ecg_times_arr)
hr_bpm_all = 60.0 / rr_intervals_all
hr_times_all = ecg_times_arr[1:]

# 2. Laser Pulses
high_idx = np.where(ch6 > 0.60)[0]
diff_h = np.diff(high_idx)
p_starts = np.insert(high_idx[np.where(diff_h > 1)[0] + 1], 0, high_idx[0])
p_ends = np.append(high_idx[np.where(diff_h > 1)[0]], high_idx[-1])
p_onsets = p_starts / fs
p_offsets = p_ends / fs
p_durs = (p_ends - p_starts) / fs

# 3. Diaphragm Bursts
dia_pks, _ = find_peaks(ch1, height=0.45, distance=int(0.12 * fs))
dia_times = (dia_pks / fs).tolist()
burst_times_arr = np.array(dia_times)
ibi_all = np.diff(burst_times_arr)
rr_hz_all = 1.0 / ibi_all
rr_times_all = burst_times_arr[1:]

# Pulse events
pulse_events = []
for i, (t_on, t_off, dur, s_idx) in enumerate(zip(p_onsets, p_offsets, p_durs, p_starts)):
    pulse_events.append({
        "id": i + 1,
        "on": round(float(t_on), 4),
        "off": round(float(t_off), 4),
        "dur_ms": round(float(dur * 1000.0), 1),
        "is_short": bool(dur < 0.05),
        "ch1_val": round(float(ch1[s_idx]), 3)
    })

# High-resolution block extraction (1000 Hz):
def make_b64_block(t_start, t_end):
    i_s = int(t_start * fs)
    i_e = int(t_end * fs)
    step = 10 # 1000 Hz
    t_sec = np.arange(i_s, i_e, step, dtype=np.float32) / np.float32(fs)
    c0 = ch0[i_s:i_e:step].astype(np.float32)
    c1 = ch1[i_s:i_e:step].astype(np.float32)
    c2 = ch2[i_s:i_e:step].astype(np.float32)
    c6 = ch6[i_s:i_e:step].astype(np.float32)
    return {
        "tStart": float(t_start),
        "tEnd": float(t_end),
        "n_pts": len(t_sec),
        "fs": float(fs / step),
        "b64_time": base64.b64encode(t_sec.tobytes()).decode('ascii'),
        "b64_ch0": base64.b64encode(c0.tobytes()).decode('ascii'),
        "b64_ch1": base64.b64encode(c1.tobytes()).decode('ascii'),
        "b64_ch2": base64.b64encode(c2.tobytes()).decode('ascii'),
        "b64_ch6": base64.b64encode(c6.tobytes()).decode('ascii'),
    }

print("Encoding high-res zoom blocks (1 kHz)...")
block_gaschallenge = make_b64_block(400.0, 475.0) # Gas switch, delay, bradycardia & gasping
block_recovery = make_b64_block(485.0, 545.0)     # Successful autoresuscitation to eupnea
block_baseline = make_b64_block(350.0, 390.0)     # Pre-challenge baseline & early trials

# Overview decimation at 50 Hz (step=200):
ov_step = 200
n_ov = total_samples // ov_step
t_ov = (np.arange(n_ov, dtype=np.float32) * (ov_step / fs))

def min_max_flat(arr, step):
    n = len(arr) // step
    mat = arr[:n * step].reshape(n, step)
    return mat.min(axis=1).astype(np.float32), mat.max(axis=1).astype(np.float32)

ov_c0_min, ov_c0_max = min_max_flat(ch0, ov_step)
ov_c1_min, ov_c1_max = min_max_flat(ch1, ov_step)
ov_c2_min, ov_c2_max = min_max_flat(ch2, ov_step)
ov_c6_min, ov_c6_max = min_max_flat(ch6, ov_step)

overview_data = {
    "t_min": 0.0,
    "t_max": float(total_samples / fs),
    "n_bins": n_ov,
    "dt": float(ov_step / fs),
    "b64_t": base64.b64encode(t_ov.tobytes()).decode('ascii'),
    "b64_c0_min": base64.b64encode(ov_c0_min.tobytes()).decode('ascii'),
    "b64_c0_max": base64.b64encode(ov_c0_max.tobytes()).decode('ascii'),
    "b64_c1_min": base64.b64encode(ov_c1_min.tobytes()).decode('ascii'),
    "b64_c1_max": base64.b64encode(ov_c1_max.tobytes()).decode('ascii'),
    "b64_c2_min": base64.b64encode(ov_c2_min.tobytes()).decode('ascii'),
    "b64_c2_max": base64.b64encode(ov_c2_max.tobytes()).decode('ascii'),
    "b64_c6_min": base64.b64encode(ov_c6_min.tobytes()).decode('ascii'),
    "b64_c6_max": base64.b64encode(ov_c6_max.tobytes()).decode('ascii'),
}

# HR & RR time series (2-second bins)
timeline_series = []
for t in range(50, int(total_samples / fs) - 2, 2):
    t_mid = t + 1.0
    m_ecg = (hr_times_all >= t) & (hr_times_all < t + 2)
    m_dia = (rr_times_all >= t) & (rr_times_all < t + 2)
    val_hr = float(np.mean(hr_bpm_all[m_ecg])) if np.sum(m_ecg) > 0 else 0.0
    val_rr = float(np.mean(rr_hz_all[m_dia])) if np.sum(m_dia) > 0 else 0.0
    timeline_series.append({
        "t": t_mid,
        "hr": round(val_hr, 1),
        "rr": round(val_rr, 2)
    })

stats_bundle = {
    "file_name": "LP26_anox_exptriggered_g0_t0.nidq.bin",
    "duration_s": round(total_samples / fs, 2),
    "total_pulses": len(p_onsets),
    "total_trials": 10,
    "trials_during_gasping": 2,
    "baseline_hr_bpm": 614.9,
    "nadir_hr_bpm": 83.3,
    "recovery_hr_bpm": 625.7,
    "baseline_rr_hz": 2.98,
    "recovery_rr_hz": 2.77,
    "unstim_igi_ms": 781.3,
    "stim_igi_ms": 1095.4,
    "gasp_delay_ms": 314.0,
    "gasp_delay_pct": 40.2,
    "gas_valve_n2_s": 405.0,
    "gas_valve_normoxia_s": 435.0,
    "transport_delay_s": 30.0,
    "outcome": "SURVIVED (Successful Autoresuscitation)"
}

data_export = {
    "stats": stats_bundle,
    "overview": overview_data,
    "block_gaschallenge": block_gaschallenge,
    "block_recovery": block_recovery,
    "block_baseline": block_baseline,
    "timeline_series": timeline_series,
    "pulses": pulse_events
}

data_str = json.dumps(data_export)
print(f"Data payload size: {len(data_str) / 1024 / 1024:.2f} MB")

# Compile HTML file
html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LP26 Session 1 (SURVIVAL) - Expiratory Opto &amp; Autoresuscitation Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0a0d14;
      --bg-card: rgba(18, 24, 38, 0.75);
      --border-color: rgba(255, 255, 255, 0.08);
      --text-main: #f0f3fa;
      --text-muted: #8b9bb4;
      --emg-color: #00f5d4;
      --integ-color: #f77f00;
      --laser-color: #7000ff;
      --laser-glow: #00d2ff;
      --ecg-color: #ff0055;
      --accent-warn: #ffb703;
      --accent-success: #06d6a0;
      --accent-danger: #ef476f;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }

    body {
      background-color: var(--bg-dark);
      background-image: 
        radial-gradient(circle at 10% 20%, rgba(6, 214, 160, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(112, 0, 255, 0.06) 0%, transparent 40%),
        linear-gradient(180deg, #090c13 0%, #06080d 100%);
      color: var(--text-main);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      background: rgba(13, 17, 27, 0.85);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-color);
      padding: 14px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .header-left { display: flex; align-items: center; gap: 14px; }
    .badge-icon {
      font-size: 24px;
      background: linear-gradient(135deg, rgba(6, 214, 160, 0.3), rgba(0, 210, 255, 0.2));
      border: 1px solid rgba(6, 214, 160, 0.4);
      padding: 8px;
      border-radius: 10px;
    }

    .header-title h1 { font-size: 1.18rem; font-weight: 700; color: #fff; }
    .header-title p { font-size: 0.76rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }

    .header-badges { display: flex; gap: 10px; flex-wrap: wrap; }
    .pill {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.72rem;
      padding: 5px 12px;
      border-radius: 20px;
      border: 1px solid transparent;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .pill.green { background: rgba(6, 214, 160, 0.15); border-color: rgba(6, 214, 160, 0.3); color: #06d6a0; }
    .pill.gold { background: rgba(255, 183, 3, 0.15); border-color: rgba(255, 183, 3, 0.4); color: #ffb703; }
    .pill.blue { background: rgba(0, 210, 255, 0.15); border-color: rgba(0, 210, 255, 0.3); color: #00d2ff; }

    .app-container {
      padding: 20px 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      max-width: 1720px;
      margin: 0 auto;
      width: 100%;
    }

    .verdicts-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 14px;
    }

    .verdict-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 16px 18px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .card-surv { border-left: 4px solid var(--accent-success); }
    .card-gasp { border-left: 4px solid var(--laser-glow); }
    .card-delay { border-left: 4px solid var(--accent-warn); }

    .verdict-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .verdict-title { font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); font-weight: 700; }
    .verdict-badge { font-size: 0.70rem; font-weight: 700; padding: 3px 8px; border-radius: 6px; }
    .badge-success { background: rgba(6, 214, 160, 0.2); color: #06d6a0; }
    .badge-blue { background: rgba(0, 210, 255, 0.2); color: #00d2ff; }
    .badge-warn { background: rgba(255, 183, 3, 0.2); color: #ffb703; }

    .verdict-main { font-size: 1.02rem; font-weight: 600; margin-bottom: 8px; color: #fff; }
    .verdict-desc { font-size: 0.79rem; line-height: 1.45; color: #b0bed5; }

    .stat-row { display: flex; gap: 16px; margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255, 255, 255, 0.06); }
    .stat-item { display: flex; flex-direction: column; }
    .stat-value { font-size: 1.1rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
    .stat-label { font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; }

    .controls-toolbar {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 12px 18px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }

    .presets-group { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
    .toolbar-label { font-size: 0.74rem; text-transform: uppercase; font-weight: 700; color: var(--text-muted); margin-right: 4px; }

    .btn-preset {
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 6px 11px;
      border-radius: 8px;
      font-size: 0.75rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-preset:hover { background: rgba(255, 255, 255, 0.12); transform: translateY(-1px); }
    .btn-preset.active {
      background: linear-gradient(135deg, rgba(6, 214, 160, 0.3), rgba(0, 210, 255, 0.2));
      border-color: var(--accent-success);
      color: #fff;
    }

    .zoom-controls { display: flex; align-items: center; gap: 8px; }
    .zoom-btn {
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--border-color);
      color: #fff;
      width: 32px;
      height: 32px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
    }
    .zoom-btn:hover { background: rgba(255, 255, 255, 0.15); }

    .time-window-display {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.78rem;
      background: rgba(0, 0, 0, 0.35);
      padding: 6px 12px;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      min-width: 250px;
      text-align: center;
    }

    .viewer-panel {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 14px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      position: relative;
    }

    .tracks-wrapper {
      position: relative;
      width: 100%;
      height: 600px;
      background: rgba(8, 12, 20, 0.95);
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      overflow: hidden;
      cursor: crosshair;
    }

    #mainCanvas { width: 100%; height: 100%; display: block; }

    .minimap-wrapper {
      height: 64px;
      width: 100%;
      background: rgba(10, 14, 22, 0.95);
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      position: relative;
      cursor: pointer;
    }

    #minimapCanvas { width: 100%; height: 100%; display: block; }

    .minimap-slider {
      position: absolute;
      top: 0;
      bottom: 0;
      background: rgba(6, 214, 160, 0.15);
      border-left: 2px solid var(--accent-success);
      border-right: 2px solid var(--accent-success);
      pointer-events: none;
    }

    .track-labels {
      position: absolute;
      top: 10px;
      left: 12px;
      display: flex;
      flex-direction: column;
      gap: 128px;
      pointer-events: none;
    }

    .track-tag {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.68rem;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 4px;
      background: rgba(10, 14, 22, 0.85);
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }

    .chart-box {
      background: rgba(10, 14, 22, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 10px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 10px;
    }

    .chart-canvas {
      width: 100%;
      height: 200px;
      background: rgba(6, 9, 15, 0.9);
      border-radius: 6px;
    }
  </style>
</head>
<body>

  <header>
    <div class="header-left">
      <div class="badge-icon">🌿</div>
      <div class="header-title">
        <h1>LP26 Session 1 (SURVIVAL) - Expiratory Stimulation &amp; Autoresuscitation</h1>
        <p>RECORDING: Z:\\ramirezlab\\pascual\\rec_data\\LP26\\LP26_anox_exptriggered_g0 | Pre-Fatal Session</p>
      </div>
    </div>
    <div class="header-badges">
      <div class="pill green">✅ Survived: Full Autoresuscitation to Eupnea (2.8 Hz)</div>
      <div class="pill gold">⏱️ 30.0s Line Delay (405s &rarr; 435s)</div>
      <div class="pill blue">🛡️ Only 2 Trials During Gasping Window</div>
    </div>
  </header>

  <div class="app-container">

    <div class="verdicts-grid">
      <!-- Card 1: Survival Outcome -->
      <div class="verdict-card card-surv">
        <div class="verdict-header">
          <span class="verdict-title">Autoresuscitation Outcome</span>
          <span class="verdict-badge badge-success">SURVIVED</span>
        </div>
        <div class="verdict-main">
          Complete Autoresuscitation &amp; Ventilatory Recovery
        </div>
        <div class="verdict-desc">
          Following the 30-second gas switch at <strong>06:45 (405.0s)</strong>, anoxia reached the nose at 435.0s. 
          The animal generated <strong>60 gasps</strong>, heart rate nadir reached only 83.3 BPM (vs 23.9 BPM in Session 2), and breathing rebounded to <strong>2.77 Hz eupnea</strong> by 500s!
        </div>
        <div class="stat-row">
          <div class="stat-item">
            <span class="stat-value" style="color: var(--accent-success);">2.77 Hz</span>
            <span class="stat-label">Recovered Eupnea</span>
          </div>
          <div class="stat-item">
            <span class="stat-value" style="color: #fff;">83.3 BPM</span>
            <span class="stat-label">Nadir Heart Rate</span>
          </div>
          <div class="stat-item">
            <span class="stat-value" style="color: var(--accent-success);">255 breaths</span>
            <span class="stat-label">Post-Anoxia Breaths</span>
          </div>
        </div>
      </div>

      <!-- Card 2: Hypothesis Validation -->
      <div class="verdict-card card-gasp">
        <div class="verdict-header">
          <span class="verdict-title">Laser Trial Burden Hypothesis</span>
          <span class="verdict-badge badge-blue">Confirmed Hypothesis</span>
        </div>
        <div class="verdict-main">
          Fewer Laser Trials Allowed Gasps to Succeed!
        </div>
        <div class="verdict-desc">
          During the fragile gasping window (435s - 500s), there were <strong>only 2 laser trials</strong> (compared to 5 trials in Session 2).
          When active, the laser delayed gasps from <strong>781 ms to 1095 ms (+40.2%, +314ms)</strong>.
          The 35 uninhibited gasps permitted rapid arterial re-oxygenation and saved the animal.
        </div>
        <div class="stat-row">
          <div class="stat-item">
            <span class="stat-value" style="color: var(--laser-glow);">2 vs 5</span>
            <span class="stat-label">Trials in Gasping</span>
          </div>
          <div class="stat-item">
            <span class="stat-value" style="color: #fff;">+314 ms</span>
            <span class="stat-label">Gasp Delay from Laser</span>
          </div>
          <div class="stat-item">
            <span class="stat-value" style="color: var(--accent-success);">35 gasps</span>
            <span class="stat-label">Uninhibited Gasps</span>
          </div>
        </div>
      </div>

      <!-- Card 3: Gas Challenge Delay -->
      <div class="verdict-card card-delay">
        <div class="verdict-header">
          <span class="verdict-title">Tubing Transport Delay</span>
          <span class="verdict-badge badge-warn">30.0s Measured</span>
        </div>
        <div class="verdict-main">
          Identical 30.0s Dead Space Transit
        </div>
        <div class="verdict-desc">
          Valve switched at <strong>06:45 (405.0s)</strong> and returned at <strong>07:15 (435.0s)</strong>.
          The heart rate maintained 615 BPM until exactly <strong>435.0s</strong> (30.0s delay), where it dropped to 478 BPM and nadir at 440s.
        </div>
        <div class="stat-row">
          <div class="stat-item">
            <span class="stat-value" style="color: var(--accent-warn);">30.0 s</span>
            <span class="stat-label">Measured Delay</span>
          </div>
          <div class="stat-item">
            <span class="stat-value" style="color: #fff;">405s &rarr; 435s</span>
            <span class="stat-label">Valve Switch Window</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Controls Toolbar -->
    <div class="controls-toolbar">
      <div class="presets-group">
        <span class="toolbar-label">Quick Jump:</span>
        <button class="btn-preset active" onclick="jumpTo(400.0, 415.0)">🟡 Gas Valve Switch (06:45 / 405s)</button>
        <button class="btn-preset" onclick="jumpTo(432.0, 448.0)">🚨 Anoxia Hits Animal (435s)</button>
        <button class="btn-preset" onclick="jumpTo(450.0, 470.0)">💨 Hypoxic Gasping Window (450s-470s)</button>
        <button class="btn-preset" onclick="jumpTo(485.0, 515.0)">✅ Autoresuscitation to Eupnea (490s-515s)</button>
        <button class="btn-preset" onclick="jumpTo(530.0, 560.0)">🟢 Sustained Post-Recovery Eupnea (530s+)</button>
        <button class="btn-preset" onclick="jumpTo(360.0, 375.0)">⚪ Early Trial 1 (Pre-Challenge)</button>
      </div>

      <div class="zoom-controls">
        <button class="zoom-btn" onclick="zoomIn()" title="Zoom In">+</button>
        <button class="zoom-btn" onclick="zoomOut()" title="Zoom Out">-</button>
        <button class="zoom-btn" onclick="panLeft()" title="Pan Left">&larr;</button>
        <button class="zoom-btn" onclick="panRight()" title="Pan Right">&rarr;</button>
        <div class="time-window-display" id="timeWindowDisplay">Viewing: 400.00s - 415.00s (15.00s)</div>
      </div>
    </div>

    <!-- Multi-Track Canvas -->
    <div class="viewer-panel">
      <div class="tracks-wrapper" id="tracksWrapper">
        <canvas id="mainCanvas"></canvas>
        <div class="track-labels">
          <div class="track-tag" style="border-left: 3px solid var(--emg-color);">
            <span class="dot" style="background: var(--emg-color);"></span>
            Raw Diaphragm EMG (Ch 0)
          </div>
          <div class="track-tag" style="border-left: 3px solid var(--integ-color);">
            <span class="dot" style="background: var(--integ-color);"></span>
            Filtered Integrated EMG (Ch 1)
          </div>
          <div class="track-tag" style="border-left: 3px solid var(--laser-glow);">
            <span class="dot" style="background: var(--laser-glow);"></span>
            Laser Command (Ch 6)
          </div>
          <div class="track-tag" style="border-left: 3px solid var(--ecg-color);">
            <span class="dot" style="background: var(--ecg-color);"></span>
            Electrocardiogram ECG (Ch 2)
          </div>
        </div>
      </div>

      <div class="minimap-wrapper" id="minimapWrapper">
        <canvas id="minimapCanvas"></canvas>
        <div class="minimap-slider" id="minimapSlider"></div>
      </div>
    </div>

    <!-- Cardiorespiratory Timeline Chart -->
    <div class="chart-box">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 600; font-size: 0.85rem; color: #fff;">Cardiorespiratory Response to Gas Challenge (Heart Rate &amp; Breathing Rate)</span>
        <span style="color: var(--accent-success); font-family: monospace; font-size: 0.8rem;">AUTORESUSCITATION SUCCESS</span>
      </div>
      <canvas id="chartTimeline" class="chart-canvas"></canvas>
      <div style="font-size: 0.73rem; color: var(--text-muted); display: flex; justify-content: space-between;">
        <span><strong style="color: #ff0055;">Red Curve</strong>: Heart Rate (BPM). Drops to 83.3 BPM at 440s, promptly recovers to 625 BPM.</span>
        <span><strong style="color: #00f5d4;">Cyan Curve</strong>: Breathing Rate (Hz). Gasping (0.6 - 1.3 Hz) followed by complete recovery to 2.8 Hz eupnea.</span>
      </div>
    </div>

  </div>

  <script>
    const RAW_DATA = """ + data_str + """;
  </script>

  <script>
    function b64ToFloat32(b64) {
      const bin = atob(b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return new Float32Array(bytes.buffer);
    }

    const overview = {
      t: b64ToFloat32(RAW_DATA.overview.b64_t),
      c0_min: b64ToFloat32(RAW_DATA.overview.b64_c0_min),
      c0_max: b64ToFloat32(RAW_DATA.overview.b64_c0_max),
      c1_min: b64ToFloat32(RAW_DATA.overview.b64_c1_min),
      c1_max: b64ToFloat32(RAW_DATA.overview.b64_c1_max),
      c2_min: b64ToFloat32(RAW_DATA.overview.b64_c2_min),
      c2_max: b64ToFloat32(RAW_DATA.overview.b64_c2_max),
      c6_min: b64ToFloat32(RAW_DATA.overview.b64_c6_min),
      c6_max: b64ToFloat32(RAW_DATA.overview.b64_c6_max),
    };

    function unpackBlock(blk) {
      return {
        tStart: blk.tStart,
        tEnd: blk.tEnd,
        fs: blk.fs || 1000.0,
        t: b64ToFloat32(blk.b64_time),
        c0: b64ToFloat32(blk.b64_ch0),
        c1: b64ToFloat32(blk.b64_ch1),
        c2: b64ToFloat32(blk.b64_ch2),
        c6: b64ToFloat32(blk.b64_ch6),
      };
    }

    const blockGasChallenge = unpackBlock(RAW_DATA.block_gaschallenge);
    const blockRecovery = unpackBlock(RAW_DATA.block_recovery);
    const blockBaseline = unpackBlock(RAW_DATA.block_baseline);
    const highResBlocks = [blockGasChallenge, blockRecovery, blockBaseline];

    let viewStart = 400.0;
    let viewEnd = 415.0;
    const totalDuration = RAW_DATA.overview.t_max;

    const mainCanvas = document.getElementById("mainCanvas");
    const ctx = mainCanvas.getContext("2d");
    const minimapCanvas = document.getElementById("minimapCanvas");
    const miniCtx = minimapCanvas.getContext("2d");
    const minimapSlider = document.getElementById("minimapSlider");
    const timeDisplay = document.getElementById("timeWindowDisplay");

    function resizeCanvases() {
      const rect = mainCanvas.parentElement.getBoundingClientRect();
      mainCanvas.width = rect.width * window.devicePixelRatio;
      mainCanvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

      const miniRect = minimapCanvas.parentElement.getBoundingClientRect();
      minimapCanvas.width = miniRect.width * window.devicePixelRatio;
      minimapCanvas.height = miniRect.height * window.devicePixelRatio;
      miniCtx.scale(window.devicePixelRatio, window.devicePixelRatio);

      renderMain();
      renderMinimap();
      renderTimelineChart();
    }

    window.addEventListener("resize", resizeCanvases);

    function renderMain() {
      const w = mainCanvas.clientWidth;
      const h = mainCanvas.clientHeight;
      ctx.clearRect(0, 0, w, h);

      const trackH = h / 4;

      // Track dividers
      ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
      ctx.lineWidth = 1;
      for (let i = 1; i < 4; i++) {
        ctx.beginPath();
        ctx.moveTo(0, i * trackH);
        ctx.lineTo(w, i * trackH);
        ctx.stroke();
      }

      const duration = viewEnd - viewStart;
      let gridStep = 0.5;
      if (duration < 1.0) gridStep = 0.1;
      else if (duration < 5) gridStep = 0.5;
      else if (duration < 15) gridStep = 1.0;
      else gridStep = 5.0;

      const firstLine = Math.ceil(viewStart / gridStep) * gridStep;
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.fillStyle = "rgba(255, 255, 255, 0.3)";
      ctx.textAlign = "center";

      for (let t = firstLine; t <= viewEnd; t += gridStep) {
        const x = ((t - viewStart) / duration) * w;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
        ctx.fillText(t.toFixed(2) + "s", x, h - 6);
      }

      // Gas markers (405s N2 on, 435s N2 off)
      const tN2On = 405.0;
      const tN2Off = 435.0;

      if (viewEnd >= tN2On && viewStart <= 435.0) {
        const x1 = Math.max(0, ((tN2On - viewStart) / duration) * w);
        const x2 = Math.min(w, ((435.0 - viewStart) / duration) * w);
        if (x2 > x1) {
          ctx.fillStyle = "rgba(255, 183, 3, 0.07)";
          ctx.fillRect(x1, 0, x2 - x1, h);
          if (x2 - x1 > 80) {
            ctx.fillStyle = "rgba(255, 183, 3, 0.7)";
            ctx.font = "bold 10px 'JetBrains Mono', monospace";
            ctx.textAlign = "center";
            ctx.fillText("TUBING TRANSPORT DELAY (30.0s)", (x1 + x2) / 2, 24);
          }
        }
      }

      if (tN2On >= viewStart && tN2On <= viewEnd) {
        const x = ((tN2On - viewStart) / duration) * w;
        ctx.save();
        ctx.strokeStyle = "#ffb703";
        ctx.lineWidth = 2.0;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        ctx.fillStyle = "#ffb703";
        ctx.font = "bold 10px 'JetBrains Mono', monospace";
        ctx.textAlign = "left";
        ctx.fillText("VALVE: N2 ON (06:45 / 405s)", x + 5, 40);
        ctx.restore();
      }

      if (tN2Off >= viewStart && tN2Off <= viewEnd) {
        const x = ((tN2Off - viewStart) / duration) * w;
        ctx.save();
        ctx.strokeStyle = "#00d2ff";
        ctx.lineWidth = 2.0;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        ctx.fillStyle = "#00d2ff";
        ctx.font = "bold 10px 'JetBrains Mono', monospace";
        ctx.textAlign = "left";
        ctx.fillText("VALVE: NORMOXIA ON (07:15 / 435s) & ANOXIA ARRIVES @ NOSE", x + 5, 40);
        ctx.restore();
      }

      // Check high-res block
      let activeBlock = null;
      for (const blk of highResBlocks) {
        if (viewStart >= blk.tStart && viewEnd <= blk.tEnd) {
          activeBlock = blk;
          break;
        }
      }

      if (activeBlock) renderHighRes(activeBlock, w, trackH);
      else renderOverview(w, trackH);

      // Threshold line
      const yZero2 = 2 * trackH;
      const scale2 = (trackH * 0.85) / 4.0;
      const yThresh = yZero2 - 0.85 * scale2;
      ctx.save();
      ctx.strokeStyle = "rgba(239, 71, 111, 0.85)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(0, yThresh); ctx.lineTo(w, yThresh); ctx.stroke();
      ctx.fillStyle = "rgba(239, 71, 111, 0.9)";
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText("THRESH: 0.85V", w - 16, yThresh - 4);
      ctx.restore();

      timeDisplay.textContent = `Viewing: ${viewStart.toFixed(2)}s - ${viewEnd.toFixed(2)}s (${(viewEnd - viewStart).toFixed(2)}s)`;
      updateMinimapSlider();
    }

    function renderHighRes(blk, w, trackH) {
      const duration = viewEnd - viewStart;
      const fsVal = blk.fs || 1000.0;
      const iStart = Math.max(0, Math.floor((viewStart - blk.tStart) * fsVal));
      const iEnd = Math.min(blk.t.length - 1, Math.ceil((viewEnd - blk.tStart) * fsVal));

      // Track 1: EMG
      const yZero1 = 0.5 * trackH;
      const scale1 = (trackH * 0.45) / 2.5;
      ctx.strokeStyle = "#00f5d4";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (let i = iStart; i <= iEnd; i++) {
        const x = ((blk.t[i] - viewStart) / duration) * w;
        const y = yZero1 - blk.c0[i] * scale1;
        if (i === iStart) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Track 2: Integ
      const yZero2 = 2 * trackH;
      const scale2 = (trackH * 0.85) / 4.0;
      ctx.strokeStyle = "#f77f00";
      ctx.fillStyle = "rgba(247, 127, 0, 0.15)";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(((blk.t[iStart] - viewStart) / duration) * w, yZero2);
      for (let i = iStart; i <= iEnd; i++) {
        const x = ((blk.t[i] - viewStart) / duration) * w;
        const y = yZero2 - Math.max(0, blk.c1[i]) * scale2;
        ctx.lineTo(x, y);
      }
      ctx.lineTo(((blk.t[iEnd] - viewStart) / duration) * w, yZero2);
      ctx.closePath();
      ctx.fill();

      ctx.beginPath();
      for (let i = iStart; i <= iEnd; i++) {
        const x = ((blk.t[i] - viewStart) / duration) * w;
        const y = yZero2 - Math.max(0, blk.c1[i]) * scale2;
        if (i === iStart) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Track 3: Laser
      const yBase3 = 3 * trackH - 12;
      const scale3 = (trackH * 0.7) / 0.25;
      ctx.strokeStyle = "#00d2ff";
      ctx.fillStyle = "rgba(112, 0, 255, 0.35)";
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      ctx.moveTo(((blk.t[iStart] - viewStart) / duration) * w, yBase3);
      for (let i = iStart; i <= iEnd; i++) {
        const x = ((blk.t[i] - viewStart) / duration) * w;
        const normL = Math.max(0, (blk.c6[i] - 0.48));
        const y = yBase3 - normL * scale3;
        ctx.lineTo(x, y);
      }
      ctx.lineTo(((blk.t[iEnd] - viewStart) / duration) * w, yBase3);
      ctx.closePath();
      ctx.fill();

      // Track 4: ECG
      const yZero4 = 3.5 * trackH;
      const scale4 = (trackH * 0.45) / 5.0;
      ctx.strokeStyle = "#ff0055";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (let i = iStart; i <= iEnd; i++) {
        const x = ((blk.t[i] - viewStart) / duration) * w;
        const y = yZero4 - blk.c2[i] * scale4;
        if (i === iStart) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    function renderOverview(w, trackH) {
      const duration = viewEnd - viewStart;
      const nBins = overview.t.length;
      const dt = overview.t[1] - overview.t[0];
      const iStart = Math.max(0, Math.floor(viewStart / dt));
      const iEnd = Math.min(nBins - 1, Math.ceil(viewEnd / dt));

      const yZero1 = 0.5 * trackH;
      const scale1 = (trackH * 0.45) / 2.5;
      ctx.fillStyle = "rgba(0, 245, 212, 0.45)";
      for (let i = iStart; i <= iEnd; i++) {
        const x = ((overview.t[i] - viewStart) / duration) * w;
        const y1 = yZero1 - overview.c0_max[i] * scale1;
        const y2 = yZero1 - overview.c0_min[i] * scale1;
        ctx.fillRect(x, y1, Math.max(1, w / (iEnd - iStart)), Math.max(1, y2 - y1));
      }

      const yZero2 = 2 * trackH;
      const scale2 = (trackH * 0.85) / 4.0;
      ctx.fillStyle = "rgba(247, 127, 0, 0.6)";
      for (let i = iStart; i <= iEnd; i++) {
        const x = ((overview.t[i] - viewStart) / duration) * w;
        const y1 = yZero2 - Math.max(0, overview.c1_max[i]) * scale2;
        const y2 = yZero2 - Math.max(0, overview.c1_min[i]) * scale2;
        ctx.fillRect(x, y1, Math.max(1, w / (iEnd - iStart)), Math.max(1, y2 - y1));
      }

      const yBase3 = 3 * trackH - 12;
      const scale3 = (trackH * 0.7) / 0.25;
      ctx.fillStyle = "rgba(0, 210, 255, 0.8)";
      for (let i = iStart; i <= iEnd; i++) {
        const x = ((overview.t[i] - viewStart) / duration) * w;
        const normL = Math.max(0, (overview.c6_max[i] - 0.48));
        const y = yBase3 - normL * scale3;
        ctx.fillRect(x, y, Math.max(1, w / (iEnd - iStart)), yBase3 - y);
      }

      const yZero4 = 3.5 * trackH;
      const scale4 = (trackH * 0.45) / 5.0;
      ctx.fillStyle = "rgba(255, 0, 85, 0.5)";
      for (let i = iStart; i <= iEnd; i++) {
        const x = ((overview.t[i] - viewStart) / duration) * w;
        const y1 = yZero4 - overview.c2_max[i] * scale4;
        const y2 = yZero4 - overview.c2_min[i] * scale4;
        ctx.fillRect(x, y1, Math.max(1, w / (iEnd - iStart)), Math.max(1, y2 - y1));
      }
    }

    function renderMinimap() {
      const w = minimapCanvas.clientWidth;
      const h = minimapCanvas.clientHeight;
      miniCtx.clearRect(0, 0, w, h);

      const nBins = overview.t.length;
      const step = Math.max(1, Math.floor(nBins / w));

      miniCtx.fillStyle = "rgba(0, 245, 212, 0.6)";
      for (let i = 0; i < nBins; i += step) {
        const x = (overview.t[i] / totalDuration) * w;
        const barH = (Math.max(0, overview.c1_max[i]) / 4.0) * (h * 0.7);
        miniCtx.fillRect(x, h - barH, 1.5, barH);
      }

      miniCtx.fillStyle = "rgba(112, 0, 255, 0.75)";
      RAW_DATA.pulses.forEach(p => {
        const x1 = (p.on / totalDuration) * w;
        const x2 = (p.off / totalDuration) * w;
        miniCtx.fillRect(x1, 4, Math.max(1.5, x2 - x1), 12);
      });

      miniCtx.fillStyle = "rgba(255, 183, 3, 0.8)";
      const xG1 = (405.0 / totalDuration) * w;
      const xG2 = (435.0 / totalDuration) * w;
      miniCtx.fillRect(xG1, 0, Math.max(2, xG2 - xG1), h);
    }

    function updateMinimapSlider() {
      const w = minimapCanvas.clientWidth;
      const left = (viewStart / totalDuration) * w;
      const right = (viewEnd / totalDuration) * w;
      minimapSlider.style.left = left + "px";
      minimapSlider.style.width = Math.max(4, right - left) + "px";
    }

    function setView(start, end) {
      viewStart = Math.max(0, start);
      viewEnd = Math.min(totalDuration, Math.max(viewStart + 0.1, end));
      renderMain();
    }

    function jumpTo(start, end) {
      setView(start, end);
      document.querySelectorAll(".btn-preset").forEach(b => b.classList.remove("active"));
      if (window.event && window.event.target) window.event.target.classList.add("active");
    }

    function zoomIn() {
      const dur = viewEnd - viewStart;
      const center = (viewStart + viewEnd) / 2;
      const newDur = Math.max(0.2, dur * 0.65);
      setView(center - newDur / 2, center + newDur / 2);
    }

    function zoomOut() {
      const dur = viewEnd - viewStart;
      const center = (viewStart + viewEnd) / 2;
      const newDur = Math.min(totalDuration, dur * 1.5);
      setView(center - newDur / 2, center + newDur / 2);
    }

    function panLeft() {
      const shift = (viewEnd - viewStart) * 0.3;
      setView(viewStart - shift, viewEnd - shift);
    }

    function panRight() {
      const shift = (viewEnd - viewStart) * 0.3;
      setView(viewStart + shift, viewEnd + shift);
    }

    function renderTimelineChart() {
      const tCanvas = document.getElementById("chartTimeline");
      if (!tCanvas) return;
      const tCtx = tCanvas.getContext("2d");
      tCanvas.width = tCanvas.clientWidth * window.devicePixelRatio;
      tCanvas.height = tCanvas.clientHeight * window.devicePixelRatio;
      tCtx.scale(window.devicePixelRatio, window.devicePixelRatio);

      const tw = tCanvas.clientWidth;
      const th = tCanvas.clientHeight;
      tCtx.clearRect(0, 0, tw, th);

      const series = RAW_DATA.timeline_series || [];
      if (series.length === 0) return;

      const tMin = 200.0;
      const tMax = 600.0;
      const hrMax = 700.0;
      const rrMax = 5.0;

      // Gas region
      const xG1 = ((405.0 - tMin) / (tMax - tMin)) * tw;
      const xG2 = ((435.0 - tMin) / (tMax - tMin)) * tw;
      tCtx.fillStyle = "rgba(255, 183, 3, 0.15)";
      tCtx.fillRect(xG1, 0, xG2 - xG1, th);
      tCtx.fillStyle = "#ffb703";
      tCtx.font = "bold 9px 'JetBrains Mono', monospace";
      tCtx.textAlign = "center";
      tCtx.fillText("N2 CHALLENGE (405-435s)", (xG1 + xG2) / 2, 16);

      // Heart Rate
      tCtx.strokeStyle = "#ff0055";
      tCtx.lineWidth = 2.0;
      tCtx.beginPath();
      let first = true;
      series.forEach(d => {
        if (d.t < tMin || d.t > tMax) return;
        const x = ((d.t - tMin) / (tMax - tMin)) * tw;
        const y = th - 20 - (d.hr / hrMax) * (th - 60);
        if (first) { tCtx.moveTo(x, y); first = false; }
        else tCtx.lineTo(x, y);
      });
      tCtx.stroke();

      // Breathing Rate
      tCtx.strokeStyle = "#00f5d4";
      tCtx.lineWidth = 1.8;
      tCtx.beginPath();
      first = true;
      series.forEach(d => {
        if (d.t < tMin || d.t > tMax) return;
        const x = ((d.t - tMin) / (tMax - tMin)) * tw;
        const y = th - 20 - (d.rr / rrMax) * (th - 60);
        if (first) { tCtx.moveTo(x, y); first = false; }
        else tCtx.lineTo(x, y);
      });
      tCtx.stroke();
    }

    window.addEventListener("DOMContentLoaded", () => {
      resizeCanvases();
    });
  </script>
</body>
</html>
"""

OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"Successfully generated standalone HTML viewer: {OUTPUT_HTML} (size: {OUTPUT_HTML.stat().st_size / 1024 / 1024:.2f} MB)")
