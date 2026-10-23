#!/usr/bin/env python3
"""
Generate High-Performance Interactive Zoomable HTML Plot of the Full 420.0s Intact Window
Animal: e097 (1740.0s - 2160.0s)
Synchronized Traces:
  1. Airflow Trace (V) + Peak Inspiration (Slope Inversion dFlow/dt = 0) + Diaphragm I_off Projection
  2. Integrated Diaphragm EMG (Envelope, a.u.) + Inspiration-Onset (I_on) & Inspiration-Off (I_off)
  3. Raw Diaphragm EMG (V) + Sub-Millisecond Action Potential Peaks
"""

import os
import sys
import json
import base64
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.e097.run_intact_window_dual_sorting_and_hmm_pipeline import (
    INTACT_T_START, INTACT_DURATION, STAGES, extract_intact_breaths
)
from src.pipeline.processing.analog_nidq.proc_flow import process_flow_meter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("GENERATE_RAW_EMG_FLOW_PLOT")

def min_max_time_ordered(t: np.ndarray, y: np.ndarray, bin_size: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Peak-preserving min-max decimation with strictly monotonic time ordering.
    Preserves true positive and negative spikes for high-frequency neurophysiological signals.
    """
    n_bins = len(y) // bin_size
    y_cut = y[:n_bins * bin_size].reshape(n_bins, bin_size)
    
    idx_min = np.argmin(y_cut, axis=1)
    idx_max = np.argmax(y_cut, axis=1)
    
    bin_offsets = np.arange(n_bins, dtype=np.int64) * bin_size
    first_is_min = idx_min <= idx_max
    
    first_idx = np.where(first_is_min, idx_min, idx_max) + bin_offsets
    second_idx = np.where(first_is_min, idx_max, idx_min) + bin_offsets
    
    out_idx = np.empty(n_bins * 2, dtype=np.int64)
    out_idx[0::2] = first_idx
    out_idx[1::2] = second_idx
    
    return t[out_idx], y[out_idx]

def encode_float32_b64(arr: np.ndarray) -> str:
    """Encodes float32 array into base64 for fast client-side Float32Array parsing."""
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")

def main():
    t0_all = time.time()
    out_dir = REPO_ROOT / "outputs" / "e097" / "intact_window"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = out_dir / "intact_nidq_cache.npz"
    nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
    
    logger.info(f"Extracting NIDQ signals with updated Diaphragm I_off and Peak Inspiration logic...")
    data = extract_intact_breaths(nidq_file, start_t=INTACT_T_START, dur=INTACT_DURATION)
    t_vec = data["t_vec"]
    emg_v = data["emg_v"]
    emg_env = data["emg_env"]
    flow_v = data["flow_v"]
    valid_onsets = data["valid_onsets"]
    valid_offsets = data["valid_offsets"]
    peak_insp_flow = data["peak_insp_flow"]
    flow_zcs = data["flow_zcs"]
    ti_arr = data["ti_arr"]  # Inspiration duration from integrated diaphragm: valid_offsets - valid_onsets
    te_arr = data["te_arr"]  # Expiration duration: cycle_ends - valid_offsets
    
    # Save cache for fast future re-runs
    np.savez_compressed(
        cache_file,
        t_vec=t_vec,
        emg_v=emg_v,
        emg_env=emg_env,
        flow_v=flow_v,
        valid_onsets=valid_onsets,
        valid_offsets=valid_offsets,
        peak_insp_flow=peak_insp_flow,
        flow_zcs=flow_zcs,
        ti_arr=ti_arr,
        te_arr=te_arr
    )
    logger.info(f"Saved updated NIDQ cache to {cache_file}")

    # Load shift analysis summary if available
    shift_json_path = out_dir / "post_gasping_offswitch_shift_analysis.json"
    shift_summary = {}
    if shift_json_path.exists():
        with open(shift_json_path, "r", encoding="utf-8") as f:
            shift_summary = json.load(f).get("summary_by_epoch", {})

    logger.info("Performing decimation for WebGL high-performance rendering...")
    # 1. Raw EMG: Peak-preserving min-max decimation at bin_size=10 (1,000 bins/sec -> 840,000 points)
    t_raw_dec, emg_raw_dec = min_max_time_ordered(t_vec, emg_v, bin_size=10)
    logger.info(f"Raw EMG decimated to {len(emg_raw_dec)} points (0.5 ms resolution).")

    # 2. Flow: Downsample by factor of 10 (1,000 Hz -> 420,000 points)
    flow_step = 10
    t_flow_dec = t_vec[::flow_step]
    flow_dec = flow_v[::flow_step]
    logger.info(f"Flow decimated to {len(flow_dec)} points (1.0 ms resolution).")

    # 3. Integrated EMG: Downsample by factor of 10 (1,000 Hz -> 420,000 points)
    emg_env_dec = emg_env[::flow_step]
    logger.info(f"Integrated EMG decimated to {len(emg_env_dec)} points (1.0 ms resolution).")

    # Exact landmark values for plotting directly on curves:
    # 1. Peak inspiration: airflow value at the trough (slope inversion from negative to positive)
    peak_insp_flow_vals = np.interp(peak_insp_flow, t_flow_dec, flow_dec).astype(np.float32)
    # 2. Diaphragm I_off projected onto airflow trace
    flow_at_ioff_vals = np.interp(valid_offsets, t_flow_dec, flow_dec).astype(np.float32)
    # 3. Integrated EMG at onset and offset
    onset_env_vals = np.interp(valid_onsets, t_flow_dec, emg_env_dec).astype(np.float32)
    offset_env_vals = np.interp(valid_offsets, t_flow_dec, emg_env_dec).astype(np.float32)
    # 4. Flow zero crossings (legacy)
    zc_flow_vals = np.zeros_like(flow_zcs, dtype=np.float32)

    logger.info("Encoding arrays to Base64 Float32Array buffers...")
    b64_t_raw = encode_float32_b64(t_raw_dec)
    b64_raw_emg = encode_float32_b64(emg_raw_dec)
    
    b64_t_flow = encode_float32_b64(t_flow_dec)
    b64_flow = encode_float32_b64(flow_dec)
    
    b64_emg_env = encode_float32_b64(emg_env_dec)

    # Encode landmarks
    b64_peak_insp_flow = encode_float32_b64(peak_insp_flow)
    b64_peak_insp_flow_vals = encode_float32_b64(peak_insp_flow_vals)

    b64_flow_at_ioff_vals = encode_float32_b64(flow_at_ioff_vals)

    b64_onsets = encode_float32_b64(valid_onsets)
    b64_onset_env_vals = encode_float32_b64(onset_env_vals)

    b64_offsets = encode_float32_b64(valid_offsets)
    b64_offset_env_vals = encode_float32_b64(offset_env_vals)

    b64_flow_zcs = encode_float32_b64(flow_zcs)
    b64_zc_flow_vals = encode_float32_b64(zc_flow_vals)

    # Format stages for JS injection
    stages_data = []
    for s in STAGES:
        rel_t0 = s["t0"] - INTACT_T_START
        rel_t1 = s["t1"] - INTACT_T_START
        stages_data.append({
            "id": s["id"],
            "name": s["name"],
            "abs_t0": s["t0"],
            "abs_t1": s["t1"],
            "rel_t0": rel_t0,
            "rel_t1": rel_t1,
            "color": s["color"]
        })

    # Time to peak inspiration from onset
    t_peak_rel = (peak_insp_flow - valid_onsets) * 1000.0

    # Epoch Summary cards data
    epoch_cards_html = ""
    epoch_colors = {
        "Pre-Gasping": "#10b981",
        "During-Gasping": "#f59e0b",
        "Post-Gasping": "#a855f7"
    }
    for ep_name, ep_col in epoch_colors.items():
        if ep_name in shift_summary:
            info = shift_summary[ep_name]
            delta = info["mean_delta_emg_flow_ms"]
            sign = "+" if delta >= 0 else ""
            epoch_cards_html += f"""
            <div class="stat-card" style="border-top: 3px solid {ep_col};">
              <div class="stat-title" style="color: {ep_col};">{ep_name}</div>
              <div class="stat-value">{info['n_breaths']} <span style="font-size:12px; font-weight:normal; color:#94a3b8;">breaths</span></div>
              <div class="stat-detail">Diaphragm T<sub>I</sub> (I<sub>on</sub>&rarr;I<sub>off</sub>): <b style="color:#f8fafc;">{info['mean_ti_emg_ms']:.1f} ± {info['std_ti_emg_ms']:.1f} ms</b></div>
              <div class="stat-detail">Flow T<sub>I</sub> (Flow ZC): <span style="color:#94a3b8;">{info['mean_ti_flow_ms']:.1f} ± {info['std_ti_flow_ms']:.1f} ms</span></div>
              <div class="stat-detail" style="margin-top:4px; padding-top:4px; border-top:1px dashed #334155;">
                Offset Delay &Delta;t: <b style="color:{ep_col}; font-size:13px;">{sign}{delta:.1f} ms</b>
              </div>
            </div>
            """

    out_html = out_dir / "intact_window_raw_emg_flow_interactive.html"
    logger.info(f"Assembling interactive HTML dashboard at {out_html}...")

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>e097 Intact Window (420.0s) &mdash; Diaphragm EMG & Airflow Dynamics</title>
  <script src="https://cdn.plot.ly/plotly-2.29.1.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-dark: #090d16;
      --card-bg: #0f172a;
      --card-sub: #131d31;
      --border: #1e293b;
      --border-accent: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent-flow: #38bdf8;
      --accent-emg-env: #f59e0b;
      --accent-emg-raw: #10b981;
      --accent-peak: #fbbf24;
      --accent-ioff: #c084fc;
      --accent-rose: #f43f5e;
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}
    body {{
      background: var(--bg-dark);
      color: var(--text);
      padding: 24px;
      line-height: 1.5;
    }}
    .mono {{
      font-family: 'JetBrains Mono', monospace;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 20px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
      flex-wrap: wrap;
      gap: 16px;
    }}
    .header-left h1 {{
      font-size: 24px;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: #f8fafc;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .badge-animal {{
      background: #1e293b;
      color: var(--accent-flow);
      font-size: 13px;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 6px;
      border: 1px solid #334155;
    }}
    .header-left p {{
      color: var(--text-muted);
      font-size: 14px;
      margin-top: 6px;
      max-width: 950px;
    }}
    .header-actions {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .btn {{
      background: #1e293b;
      color: #e2e8f0;
      border: 1px solid #334155;
      padding: 8px 14px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }}
    .btn:hover {{
      background: #334155;
      color: #fff;
      border-color: #475569;
    }}
    .btn-primary {{
      background: #2563eb;
      color: #fff;
      border-color: #3b82f6;
    }}
    .btn-primary:hover {{
      background: #1d4ed8;
      border-color: #2563eb;
    }}

    /* Stat Cards Row */
    .stat-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }}
    .stat-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 16px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }}
    .stat-title {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 4px;
    }}
    .stat-value {{
      font-size: 22px;
      font-weight: 800;
      color: #f8fafc;
      letter-spacing: -0.01em;
    }}
    .stat-detail {{
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 3px;
    }}

    /* Controls Panel */
    .controls-panel {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 18px;
      margin-bottom: 20px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .controls-row {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .controls-label {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      min-width: 90px;
    }}
    .stage-btn {{
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      background: #131d31;
      border: 1px solid #1e293b;
      color: #cbd5e1;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }}
    .stage-btn:hover {{
      background: #1e293b;
      color: #fff;
      border-color: #475569;
    }}
    .stage-btn.active {{
      border-color: currentColor;
      box-shadow: 0 0 10px -2px currentColor;
    }}
    .stage-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      display: inline-block;
    }}

    /* Toggle Switches */
    .toggle-group {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .toggle-label {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: #cbd5e1;
      cursor: pointer;
      user-select: none;
    }}
    .toggle-label input {{
      cursor: pointer;
      accent-color: #38bdf8;
    }}

    /* Main Plot Container */
    .plot-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
      position: relative;
    }}
    #plot-div {{
      width: 100%;
      height: 820px;
    }}
    .plot-status-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--border);
      font-size: 12px;
      color: var(--text-muted);
      flex-wrap: wrap;
      gap: 10px;
    }}

    /* Scientific Callout Box */
    .callout-box {{
      background: #0b1329;
      border: 1px solid #1d4ed8;
      border-left: 4px solid #38bdf8;
      border-radius: 8px;
      padding: 14px 18px;
      margin-top: 20px;
    }}
    .callout-title {{
      font-size: 13px;
      font-weight: 700;
      color: #38bdf8;
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .callout-body {{
      font-size: 13px;
      color: #cbd5e1;
      line-height: 1.6;
    }}
    .callout-body b {{
      color: #f8fafc;
    }}

    /* Loading Overlay */
    #loading-overlay {{
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(9, 13, 22, 0.92);
      backdrop-filter: blur(4px);
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      z-index: 100;
      border-radius: 10px;
    }}
    .spinner {{
      width: 48px;
      height: 48px;
      border: 4px solid rgba(56, 189, 248, 0.2);
      border-top-color: #38bdf8;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin-bottom: 16px;
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
  </style>
</head>
<body>

  <!-- Header -->
  <div class="header">
    <div class="header-left">
      <h1>
        <span>Intact Window (420.0s) &mdash; Diaphragm EMG & Airflow Dynamics</span>
        <span class="badge-animal">Animal: e097</span>
      </h1>
      <p>
        Integrated diaphragm burst offset detection (<span style="color:#c084fc; font-weight:700;">I<sub>off</sub></span>)
        and airflow slope-inversion peak inspiration (<span style="color:#fbbf24; font-weight:700;">Peak Insp</span>) across the continuous 420.0s recording (t = 1740.0s &ndash; 2160.0s).
        Zoom into individual breath cycles to inspect flow baseline drift, negative deflection troughs, and neural drive termination.
      </p>
    </div>
    <div class="header-actions">
      <a href="intact_window_6state_population_dynamics_dashboard.html" class="btn">
        &larr; 6-State Dashboard
      </a>
      <a href="investigation_post_gasping_offswitch_shift.png" target="_blank" class="btn btn-primary">
        View Diagnostic PNG
      </a>
    </div>
  </div>

  <!-- Summary Statistics Cards -->
  <div class="stat-row">
    <div class="stat-card" style="border-top: 3px solid #38bdf8;">
      <div class="stat-title">Full Intact Window</div>
      <div class="stat-value">420.0 <span style="font-size:14px; font-weight:normal; color:#94a3b8;">sec</span></div>
      <div class="stat-detail">Absolute: <b>1740.0s &ndash; 2160.0s</b></div>
      <div class="stat-detail">Analog Sampling: <b>10,000 Hz</b></div>
    </div>
    <div class="stat-card" style="border-top: 3px solid #c084fc;">
      <div class="stat-title">Integrated Diaphragm I<sub>off</sub></div>
      <div class="stat-value">{len(valid_onsets)} <span style="font-size:14px; font-weight:normal; color:#94a3b8;">breaths</span></div>
      <div class="stat-detail">Diaphragm T<sub>I</sub> (I<sub>on</sub>&rarr;I<sub>off</sub>): <b style="color:#f8fafc;">{np.mean(ti_arr):.1f} ± {np.std(ti_arr):.1f} ms</b></div>
      <div class="stat-detail">Time to Peak Insp: <b>{np.mean(t_peak_rel):.1f} ± {np.std(t_peak_rel):.1f} ms</b></div>
    </div>
    {epoch_cards_html}
  </div>

  <!-- Interactive Controls Panel -->
  <div class="controls-panel">
    <!-- Presets -->
    <div class="controls-row">
      <div class="controls-label">Jump to Stage:</div>
      <button class="stage-btn" onclick="zoomTo(0, 420)" id="btn-full" style="color: #38bdf8; border-color: #38bdf8; background:#1e293b;">
        🔍 Full 420s Overview
      </button>
      <button class="stage-btn" onclick="zoomTo(0, 60)" style="color: #10b981;">
        <span class="stage-dot" style="background:#10b981;"></span> 1. Baseline Eupnea (0-60s)
      </button>
      <button class="stage-btn" onclick="zoomTo(60, 96)" style="color: #eab308;">
        <span class="stage-dot" style="background:#eab308;"></span> 2. Anoxic Gas Switch (60-96s)
      </button>
      <button class="stage-btn" onclick="zoomTo(96, 120)" style="color: #ef4444;">
        <span class="stage-dot" style="background:#ef4444;"></span> 3. Apnea Nadir (96-120s)
      </button>
      <button class="stage-btn" onclick="zoomTo(120, 210)" style="color: #38bdf8;">
        <span class="stage-dot" style="background:#38bdf8;"></span> 4. Early Gasping (120-210s)
      </button>
      <button class="stage-btn" onclick="zoomTo(210, 330)" style="color: #6366f1;">
        <span class="stage-dot" style="background:#6366f1;"></span> 5. Late Gasping (210-330s)
      </button>
      <button class="stage-btn" onclick="zoomTo(330, 396)" style="color: #a855f7;">
        <span class="stage-dot" style="background:#a855f7;"></span> 6. Active Flow Shift (330-396s)
      </button>
      <button class="stage-btn" onclick="zoomTo(396, 420)" style="color: #ec4899;">
        <span class="stage-dot" style="background:#ec4899;"></span> 7. Recovery (396-420s)
      </button>
    </div>

    <!-- Micro-zoom breath detail presets -->
    <div class="controls-row">
      <div class="controls-label">Micro-Zoom:</div>
      <button class="stage-btn" onclick="zoomTo(48, 52)" style="color: #34d399;">
        🔬 Baseline Breaths Detail (48-52s)
      </button>
      <button class="stage-btn" onclick="zoomTo(75, 80)" style="color: #fde047;">
        🔬 Anoxia 1 Gas Switch Shape (75-80s)
      </button>
      <button class="stage-btn" onclick="zoomTo(98, 106)" style="color: #f87171;">
        🔬 Apnea Nadir Landmarked Breaths (98-106s)
      </button>
      <button class="stage-btn" onclick="zoomTo(150, 154)" style="color: #60a5fa;">
        🔬 Ballistic Gasp Burst (150-154s)
      </button>
      <button class="stage-btn" onclick="zoomTo(350, 354)" style="color: #c084fc;">
        🔬 Flow Reshaping / Offset Decoupling (350-354s)
      </button>
    </div>

    <!-- Toggles & Y-Scaling -->
    <div class="controls-row" style="margin-top: 4px; padding-top: 8px; border-top: 1px dashed #1e293b;">
      <div class="controls-label">Layers:</div>
      <div class="toggle-group">
        <label class="toggle-label">
          <input type="checkbox" id="toggle-peak" checked onchange="toggleMarkers('flow-peak')">
          <span style="color:#fbbf24; font-weight:600;">Peak Inspiration (Flow Slope Inversion, gold triangles)</span>
        </label>
        <label class="toggle-label">
          <input type="checkbox" id="toggle-ioff-flow" checked onchange="toggleMarkers('flow-ioff')">
          <span style="color:#c084fc; font-weight:600;">Diaphragm I<sub>off</sub> on Flow (purple ticks)</span>
        </label>
        <label class="toggle-label">
          <input type="checkbox" id="toggle-onsets" checked onchange="toggleMarkers('emg-onsets')">
          <span>Inspiration-Onset (I<sub>on</sub>, green circles)</span>
        </label>
        <label class="toggle-label">
          <input type="checkbox" id="toggle-offsets" checked onchange="toggleMarkers('emg-offsets')">
          <span>Inspiration-Off (I<sub>off</sub> on EMG, purple crosses)</span>
        </label>
        <label class="toggle-label">
          <input type="checkbox" id="toggle-zcs" onchange="toggleMarkers('flow-zcs')">
          <span style="color:#94a3b8;">Flow Zero-Crossing (t<sub>Flow_ZC</sub>, red diamonds)</span>
        </label>
        <label class="toggle-label">
          <input type="checkbox" id="toggle-stages" checked onchange="toggleStageShapes()">
          <span>Stage Shading</span>
        </label>
      </div>
    </div>

    <!-- Y-Scale Controls -->
    <div class="controls-row" style="padding-top: 4px;">
      <div class="controls-label">Y-Scaling:</div>
      <button class="stage-btn" onclick="autoFitCurrentY()" style="color: #38bdf8; border-color: #0284c7; background: #0c4a6e;">
        ↕️ Auto-Fit Y to Visible Window
      </button>
      <button class="stage-btn" onclick="resetYGlobal()" style="color: #94a3b8;">
        ↕️ Reset Y Global (-6.5V to +4.5V)
      </button>
      <label class="toggle-label" style="margin-left: 10px;">
        <input type="checkbox" id="toggle-autoy" checked>
        <span>Auto-fit Y when jumping to stage / micro-zoom</span>
      </label>
    </div>
  </div>

  <!-- Plot Container -->
  <div class="plot-card">
    <div id="loading-overlay">
      <div class="spinner"></div>
      <div style="font-weight: 600; font-size: 16px; color: #f8fafc;">Decompressing WebGL Signal Buffers...</div>
      <div style="font-size: 13px; color: #94a3b8; margin-top: 6px;">Initializing synchronized 840k-point EMG and 420k-point airflow traces</div>
    </div>
    <div id="plot-div"></div>
    <div class="plot-status-bar">
      <div>
        Visible Window: <span id="window-range" class="mono" style="color: #38bdf8; font-weight:600;">0.00s &ndash; 420.00s</span>
        (Duration: <span id="window-dur" class="mono" style="font-weight:600; color:#f8fafc;">420.00s</span> |
        Absolute: <span id="window-abs" class="mono" style="color:#94a3b8;">1740.00s &ndash; 2160.00s</span>)
      </div>
      <div>
        <span style="color:#10b981; font-weight:600;">Pan & Zoom:</span> Click & drag horizontally to box-zoom &bull; Double-click to reset &bull; Hover for synchronized millisecond readout
      </div>
    </div>
  </div>

  <!-- Scientific Callout -->
  <div class="callout-box">
    <div class="callout-title">
      <span>💡 Diaphragm I<sub>off</sub> & Flow Slope-Inversion Peak Landmark Definitions</span>
    </div>
    <div class="callout-body">
      1. <b>Inspiration-Off (I<sub>off</sub>) from Integrated Diaphragm:</b> By defining inspiration-off using the integrated diaphragm burst offset (<b>{np.mean(ti_arr):.1f} ± {np.std(ti_arr):.1f} ms</b> duration), breath cycle segmentation is completely invariant to pneumotachograph baseline drift, gas density shifts during anoxic gas wash-in, and mechanical airway resistance changes.
      <br>
      2. <b>Peak Inspiration (Slope Inversion):</b> Identified on the airflow trace at the local minimum of the negative deflection where the slope transitions from negative ($dFlow/dt &lt; 0$, accelerating inward airflow) to positive ($dFlow/dt &gt; 0$, deceleration toward expiration). Occurs at <b>{np.mean(t_peak_rel):.1f} ± {np.std(t_peak_rel):.1f} ms</b> after breath onset.
      <br>
      3. <b>Anoxic Challenge Observations:</b> Notice that during <b>Anoxic Challenge 1 (75&ndash;82s)</b> and <b>Apnea Nadir (98&ndash;116s)</b>, the airflow baseline elevates and changes shape, but the integrated diaphragm motor burst cleanly marks the exact start and termination of active neural inspiration.
    </div>
  </div>

  <!-- Scripts -->
  <script>
    // Embedded Data Payloads (Base64 Float32Array)
    const B64_T_RAW = "{b64_t_raw}";
    const B64_RAW_EMG = "{b64_raw_emg}";
    const B64_T_FLOW = "{b64_t_flow}";
    const B64_FLOW = "{b64_flow}";
    const B64_EMG_ENV = "{b64_emg_env}";

    const B64_PEAK_INSP_FLOW = "{b64_peak_insp_flow}";
    const B64_PEAK_INSP_FLOW_VALS = "{b64_peak_insp_flow_vals}";
    const B64_FLOW_AT_IOFF_VALS = "{b64_flow_at_ioff_vals}";

    const B64_ONSETS = "{b64_onsets}";
    const B64_ONSET_ENV_VALS = "{b64_onset_env_vals}";
    const B64_OFFSETS = "{b64_offsets}";
    const B64_OFFSET_ENV_VALS = "{b64_offset_env_vals}";

    const B64_FLOW_ZCS = "{b64_flow_zcs}";
    const B64_ZC_FLOW_VALS = "{b64_zc_flow_vals}";

    const STAGES = {json.dumps(stages_data)};
    const INTACT_T_START = {INTACT_T_START};

    // Fast Base64 to Float32Array
    function decodeB64Float32(base64) {{
      const binaryString = atob(base64);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {{
        bytes[i] = binaryString.charCodeAt(i);
      }}
      return new Float32Array(bytes.buffer);
    }}

    console.time("Decoding buffers");
    const tRaw = decodeB64Float32(B64_T_RAW);
    const rawEmg = decodeB64Float32(B64_RAW_EMG);
    const tFlow = decodeB64Float32(B64_T_FLOW);
    const flow = decodeB64Float32(B64_FLOW);
    const emgEnv = decodeB64Float32(B64_EMG_ENV);

    const peakInspFlow = decodeB64Float32(B64_PEAK_INSP_FLOW);
    const peakInspFlowVals = decodeB64Float32(B64_PEAK_INSP_FLOW_VALS);
    const flowAtIoffVals = decodeB64Float32(B64_FLOW_AT_IOFF_VALS);

    const onsets = decodeB64Float32(B64_ONSETS);
    const onsetEnvVals = decodeB64Float32(B64_ONSET_ENV_VALS);
    const offsets = decodeB64Float32(B64_OFFSETS);
    const offsetEnvVals = decodeB64Float32(B64_OFFSET_ENV_VALS);

    const flowZcs = decodeB64Float32(B64_FLOW_ZCS);
    const zcFlowVals = decodeB64Float32(B64_ZC_FLOW_VALS);
    console.timeEnd("Decoding buffers");

    // Build Stage Background Shapes
    const stageShapes = STAGES.map(s => ({{
      type: 'rect',
      xref: 'x',
      yref: 'paper',
      x0: s.rel_t0,
      x1: s.rel_t1,
      y0: 0,
      y1: 1,
      fillcolor: s.color,
      opacity: 0.08,
      line: {{ width: 1, color: s.color, dash: 'dot' }},
      layer: 'below'
    }}));

    // Build Stage Annotations along top
    const stageAnnotations = STAGES.map(s => ({{
      x: (s.rel_t0 + s.rel_t1) / 2,
      y: 1.0,
      xref: 'x',
      yref: 'paper',
      text: `<b>${{s.name}}</b>`,
      showarrow: false,
      font: {{ size: 10, color: s.color }},
      yanchor: 'bottom',
      bgcolor: 'rgba(15, 23, 42, 0.75)',
      bordercolor: s.color,
      borderwidth: 1,
      borderpad: 2
    }}));

    // Zero line for airflow
    const zeroLine = {{
      type: 'line',
      xref: 'x',
      yref: 'y',
      x0: 0,
      x1: 420,
      y0: 0,
      y1: 0,
      line: {{ color: '#64748b', width: 1, dash: 'dash' }}
    }};

    // Traces
    // 1. Flow Trace (Top)
    const traceFlow = {{
      x: tFlow,
      y: flow,
      type: 'scattergl',
      mode: 'lines',
      name: 'Airflow (V)',
      line: {{ color: '#38bdf8', width: 1.2 }},
      xaxis: 'x',
      yaxis: 'y'
    }};

    // Peak Inspiration (Slope Inversion: dFlow/dt = 0, minimum of negative deflection)
    const tracePeakInsp = {{
      x: peakInspFlow,
      y: peakInspFlowVals,
      type: 'scattergl',
      mode: 'markers',
      name: 'Peak Inspiration (Slope Inversion)',
      marker: {{ color: '#fbbf24', size: 6, symbol: 'triangle-down' }},
      xaxis: 'x',
      yaxis: 'y'
    }};

    // Diaphragm I_off projected onto Flow
    const traceFlowIoff = {{
      x: offsets,
      y: flowAtIoffVals,
      type: 'scattergl',
      mode: 'markers',
      name: 'Inspiration-Off (Diaphragm I_off on Flow)',
      marker: {{ color: '#c084fc', size: 6, symbol: 'line-ns-open' }},
      xaxis: 'x',
      yaxis: 'y'
    }};

    // Legacy Flow Zero Crossings
    const traceZcs = {{
      x: flowZcs,
      y: zcFlowVals,
      type: 'scattergl',
      mode: 'markers',
      name: 'Flow Zero-Crossing (Legacy t_Flow_ZC)',
      visible: 'legendonly',
      marker: {{ color: '#f43f5e', size: 5, symbol: 'diamond' }},
      xaxis: 'x',
      yaxis: 'y'
    }};

    // 2. Integrated EMG Envelope (Middle)
    const traceEmgEnv = {{
      x: tFlow,
      y: emgEnv,
      type: 'scattergl',
      mode: 'lines',
      name: 'Int. Diaphragm EMG (Envelope)',
      line: {{ color: '#f59e0b', width: 1.5 }},
      xaxis: 'x',
      yaxis: 'y2'
    }};

    // EMG Onsets (I_on)
    const traceOnsets = {{
      x: onsets,
      y: onsetEnvVals,
      type: 'scattergl',
      mode: 'markers',
      name: 'Inspiration-Onset (I_on)',
      marker: {{ color: '#10b981', size: 5, symbol: 'circle' }},
      xaxis: 'x',
      yaxis: 'y2'
    }};

    // EMG Offsets (I_off)
    const traceOffsets = {{
      x: offsets,
      y: offsetEnvVals,
      type: 'scattergl',
      mode: 'markers',
      name: 'Inspiration-Off (I_off from Integrated EMG)',
      marker: {{ color: '#c084fc', size: 6, symbol: 'x' }},
      xaxis: 'x',
      yaxis: 'y2'
    }};

    // 3. Raw Diaphragm EMG (Bottom)
    const traceRawEmg = {{
      x: tRaw,
      y: rawEmg,
      type: 'scattergl',
      mode: 'lines',
      name: 'Raw Diaphragm EMG (V)',
      line: {{ color: '#10b981', width: 0.9 }},
      xaxis: 'x',
      yaxis: 'y3'
    }};

    const traces = [
      traceFlow,
      tracePeakInsp,
      traceFlowIoff,
      traceZcs,
      traceEmgEnv,
      traceOnsets,
      traceOffsets,
      traceRawEmg
    ];

    const layout = {{
      paper_bgcolor: '#0f172a',
      plot_bgcolor: '#090d16',
      font: {{ color: '#f8fafc', family: 'Inter, sans-serif' }},
      margin: {{ l: 70, r: 30, t: 36, b: 60 }},
      showlegend: true,
      legend: {{
        orientation: 'h',
        x: 0.5,
        y: 1.08,
        xanchor: 'center',
        yanchor: 'bottom',
        font: {{ size: 11, color: '#cbd5e1' }},
        bgcolor: 'rgba(15, 23, 42, 0.85)',
        bordercolor: '#334155',
        borderwidth: 1
      }},
      xaxis: {{
        domain: [0, 1],
        range: [0, 420],
        title: {{
          text: 'Time (s) from Intact Window Start [0.0s &ndash; 420.0s | Absolute: 1740s &ndash; 2160s]',
          font: {{ size: 12, color: '#cbd5e1' }},
          standoff: 10
        }},
        gridcolor: '#1e293b',
        zerolinecolor: '#334155',
        showspikes: true,
        spikemode: 'across',
        spikethickness: 1,
        spikecolor: '#94a3b8',
        spikedash: 'dot',
        rangeslider: {{ visible: true, thickness: 0.05, bgcolor: '#0b1120', bordercolor: '#334155' }}
      }},
      yaxis: {{
        domain: [0.69, 1.0],
        title: {{ text: 'Airflow (V)', font: {{ size: 11, color: '#38bdf8' }} }},
        gridcolor: '#1e293b',
        zerolinecolor: '#475569',
        range: [-6.5, 4.5],
        fixedrange: false
      }},
      yaxis2: {{
        domain: [0.36, 0.65],
        title: {{ text: 'Int. EMG (a.u.)', font: {{ size: 11, color: '#f59e0b' }} }},
        gridcolor: '#1e293b',
        zerolinecolor: '#475569',
        anchor: 'x',
        range: [0.0, 1.15],
        fixedrange: false
      }},
      yaxis3: {{
        domain: [0.0, 0.32],
        title: {{ text: 'Raw EMG (V)', font: {{ size: 11, color: '#10b981' }} }},
        gridcolor: '#1e293b',
        zerolinecolor: '#475569',
        anchor: 'x',
        range: [-10.5, 8.0],
        fixedrange: false
      }},
      shapes: [zeroLine, ...stageShapes],
      annotations: stageAnnotations
    }};

    const config = {{
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      displaylogo: false,
      scrollZoom: true
    }};

    let currentXRange = [0, 420];

    Plotly.newPlot('plot-div', traces, layout, config).then(() => {{
      document.getElementById('loading-overlay').style.display = 'none';

      const plotDiv = document.getElementById('plot-div');
      plotDiv.on('plotly_relayout', function(eventData) {{
        let x0 = currentXRange[0], x1 = currentXRange[1];
        if (eventData['xaxis.range[0]'] !== undefined) {{
          x0 = eventData['xaxis.range[0]'];
          x1 = eventData['xaxis.range[1]'];
        }} else if (eventData['xaxis.range'] !== undefined) {{
          x0 = eventData['xaxis.range'][0];
          x1 = eventData['xaxis.range'][1];
        }}
        currentXRange = [x0, x1];
        const dur = Math.max(0, x1 - x0);
        document.getElementById('window-range').innerText = `${{x0.toFixed(2)}}s &ndash; ${{x1.toFixed(2)}}s`;
        document.getElementById('window-dur').innerText = `${{dur.toFixed(2)}}s`;
        document.getElementById('window-abs').innerText = `${{(x0 + INTACT_T_START).toFixed(2)}}s &ndash; ${{(x1 + INTACT_T_START).toFixed(2)}}s`;
      }});
    }});

    function zoomTo(start, end) {{
      const autoY = document.getElementById('toggle-autoy') ? document.getElementById('toggle-autoy').checked : true;
      currentXRange = [start, end];
      const dur = end - start;
      document.getElementById('window-range').innerText = `${{start.toFixed(2)}}s &ndash; ${{end.toFixed(2)}}s`;
      document.getElementById('window-dur').innerText = `${{dur.toFixed(2)}}s`;
      document.getElementById('window-abs').innerText = `${{(start + INTACT_T_START).toFixed(2)}}s &ndash; ${{(end + INTACT_T_START).toFixed(2)}}s`;

      if (start <= 0 && end >= 420) {{
        // Reset to full global range
        resetYGlobal();
        return;
      }}

      if (autoY) {{
        fitYToRange(start, end);
      }} else {{
        Plotly.relayout('plot-div', {{
          'xaxis.range': [start, end]
        }});
      }}
    }}

    function fitYToRange(start, end) {{
      // Slice flow and emgEnv
      const idx0 = Math.max(0, Math.floor(start * 1000));
      const idx1 = Math.min(flow.length - 1, Math.ceil(end * 1000));
      let fMin = Infinity, fMax = -Infinity;
      let eMin = Infinity, eMax = -Infinity;
      for (let i = idx0; i <= idx1; i++) {{
        const fv = flow[i];
        if (fv < fMin) fMin = fv;
        if (fv > fMax) fMax = fv;
        const ev = emgEnv[i];
        if (ev < eMin) eMin = ev;
        if (ev > eMax) eMax = ev;
      }}
      if (fMin === Infinity) {{ fMin = -6.5; fMax = 4.5; }}
      if (eMin === Infinity) {{ eMin = 0; eMax = 1.15; }}

      // Slice raw EMG
      const rIdx0 = Math.max(0, Math.floor(start * 2000));
      const rIdx1 = Math.min(rawEmg.length - 1, Math.ceil(end * 2000));
      let rMin = Infinity, rMax = -Infinity;
      for (let i = rIdx0; i <= rIdx1; i++) {{
        const rv = rawEmg[i];
        if (rv < rMin) rMin = rv;
        if (rv > rMax) rMax = rv;
      }}
      if (rMin === Infinity) {{ rMin = -10.5; rMax = 8.0; }}

      const fPad = Math.max(0.2, (fMax - fMin) * 0.1);
      const ePad = Math.max(0.04, (eMax - eMin) * 0.1);
      const rPad = Math.max(0.5, (rMax - rMin) * 0.1);

      Plotly.relayout('plot-div', {{
        'xaxis.range': [start, end],
        'yaxis.range': [fMin - fPad, fMax + fPad],
        'yaxis2.range': [Math.max(0, eMin - ePad), eMax + ePad],
        'yaxis3.range': [rMin - rPad, rMax + rPad]
      }});
    }}

    function autoFitCurrentY() {{
      fitYToRange(currentXRange[0], currentXRange[1]);
    }}

    function resetYGlobal() {{
      Plotly.relayout('plot-div', {{
        'xaxis.range': [currentXRange[0], currentXRange[1]],
        'yaxis.range': [-6.5, 4.5],
        'yaxis2.range': [0.0, 1.15],
        'yaxis3.range': [-10.5, 8.0]
      }});
    }}

    function toggleMarkers(markerType) {{
      const plotDiv = document.getElementById('plot-div');
      let updateIndices = [];
      let isVisible = true;
      if (markerType === 'flow-peak') {{
        updateIndices = [1];
        isVisible = document.getElementById('toggle-peak').checked;
      }} else if (markerType === 'flow-ioff') {{
        updateIndices = [2];
        isVisible = document.getElementById('toggle-ioff-flow').checked;
      }} else if (markerType === 'flow-zcs') {{
        updateIndices = [3];
        isVisible = document.getElementById('toggle-zcs').checked;
      }} else if (markerType === 'emg-onsets') {{
        updateIndices = [5];
        isVisible = document.getElementById('toggle-onsets').checked;
      }} else if (markerType === 'emg-offsets') {{
        updateIndices = [6];
        isVisible = document.getElementById('toggle-offsets').checked;
      }}
      Plotly.restyle(plotDiv, {{ visible: isVisible }}, updateIndices);
    }}

    let stagesVisible = true;
    function toggleStageShapes() {{
      const plotDiv = document.getElementById('plot-div');
      stagesVisible = document.getElementById('toggle-stages').checked;
      const currentShapes = stagesVisible ? [zeroLine, ...stageShapes] : [zeroLine];
      const currentAnnotations = stagesVisible ? stageAnnotations : [];
      Plotly.relayout(plotDiv, {{
        shapes: currentShapes,
        annotations: currentAnnotations
      }});
    }}
  </script>
</body>
</html>
"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_template)
    
    logger.info(f"Successfully generated interactive zoomable plot at {out_html} (HTML size: {out_html.stat().st_size / (1024*1024):.2f} MB, runtime: {time.time() - t0_all:.1f}s)")

    # Copy to artifacts directory
    artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\f68da1c7-291f-411b-8163-80610ca6d1a2")
    if artifact_dir.exists():
        shutil.copy2(out_html, artifact_dir / "intact_window_raw_emg_flow_interactive.html")
        logger.info(f"Copied interactive dashboard to artifact directory: {artifact_dir}")

if __name__ == '__main__':
    main()
