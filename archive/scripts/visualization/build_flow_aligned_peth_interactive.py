#!/usr/bin/env python3
"""
Generate Interactive HTML of PETHs and Rasters for All 96 Single Units with Flow Meter-Determined Expiration Onset.
Features:
1. Alignment to Inspiration Onset (I_on) [Sorted by T_I flow duration]
2. Alignment to Flow Expiration Onset (E_on flow) [Sorted by T_I flow duration]
3. Dual Side-by-Side Comparison Mode
4. Full 96-Neuron Gallery with filter by phase classification (Inspiratory, Expiratory, Tonic)
5. Synchronized PETH with Mean FR + SD Shadow, Diaphragm EMG Envelope, and Flow Meter Trace

Output:
e097_prelaser_respiratory_neurodynamics/06_exploratory_flow_expiration/all_96_neurons_flow_aligned_peth_interactive.html
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal

PROJECT_ROOT = Path(r"C:\Users\lpascu\analysis_src")
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

OUT_DIR = PROJECT_ROOT / "e097_prelaser_respiratory_neurodynamics" / "06_exploratory_flow_expiration"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("BUILDING INTERACTIVE PETH & RASTER TOOL FOR ALL 96 NEURONS (FLOW-ALIGNED)")
    print("=" * 80)

    # 1. Load Data
    cons_dir = PROJECT_ROOT / "output_e097_prelaser_60s" / "lupin" / "consensus_master"
    alf_dir = cons_dir / "alf"
    spk_times_all = np.load(alf_dir / "spike_times.npy")
    spk_clusters_all = np.load(alf_dir / "spike_clusters.npy")
    df_info = pd.read_csv(cons_dir / "cluster_info.tsv", sep="\t")

    qc_col = "passed_qc" if "passed_qc" in df_info.columns else "passed_allen_qc"
    df_sua = df_info[df_info[qc_col] == True].copy().sort_values("depth_um", ascending=True).reset_index(drop=True)

    # Load Phase classifications
    phase_csv = PROJECT_ROOT / "e097_prelaser_respiratory_neurodynamics" / "unit_respiratory_phase_classification.csv"
    df_phase = pd.read_csv(phase_csv)
    df_sua = df_sua.merge(df_phase[["cluster_id", "vector_strength", "unit_class", "mean_phase"]], on="cluster_id", how="left")

    # Load Breath Metrics (Channel 5 Flow & Diaphragm EMG)
    metrics_csv = OUT_DIR / "breath_by_breath_flow_vs_emg_metrics.csv"
    df_metrics = pd.read_csv(metrics_csv)

    # Filter eupneic breaths (skip breath 99 sigh for raster uniformity, or include with flag)
    # We will sort by T_I_flow_slope_ms
    eupnea_mask = ~df_metrics["is_sigh"]
    df_eupnea = df_metrics[eupnea_mask].copy().sort_values("T_I_flow_slope_ms", ascending=True).reset_index(drop=True)
    n_trials = len(df_eupnea)
    print(f"Loaded {len(df_sua)} clean single units across {n_trials} sorted eupneic breaths.")

    # Load NIDQ raw signals for average EMG and Flow envelopes
    raw_dir = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2")
    nidq_bin = list(raw_dir.glob("*.nidq.bin"))[0]
    data = read_nidq_bin(nidq_bin, nidq_bin.with_suffix(".meta"), emg_chan=0, ecg_chan=1)
    fs = data.fs

    start_t = 796.827
    duration_s = 60.0
    mask_nidq = (data.time_vector >= start_t) & (data.time_vector <= start_t + duration_s)
    t_nidq = data.time_vector[mask_nidq] - start_t
    emg_raw = data.all_channels[0][mask_nidq]
    flow_raw = data.all_channels[5][mask_nidq]

    # Process EMG envelope
    emg_cfg = EMGConfig(fs=fs, highpass_cutoff=300.0, lowpass_cutoff=3000.0, smoothing_window_ms=50.0)
    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    emg_env = emg_proc.envelope

    # Filter flow
    b_low, a_low = signal.butter(2, 50.0 / (fs / 2.0), btype='low')
    flow_lp = signal.filtfilt(b_low, a_low, flow_raw)

    # 2. Define Alignment Windows
    # Onset window: -150 ms to +300 ms (t=0 is I_on)
    # Offset window: -250 ms to +250 ms (t=0 is E_on flow)
    dt_bin = 0.005  # 5 ms bins
    bins_onset = np.arange(-0.150, 0.305, dt_bin)
    centers_onset = np.round((bins_onset[:-1] + bins_onset[1:]) / 2.0 * 1000.0, 1)

    bins_offset = np.arange(-0.250, 0.255, dt_bin)
    centers_offset = np.round((bins_offset[:-1] + bins_offset[1:]) / 2.0 * 1000.0, 1)

    kernel = signal.windows.gaussian(11, std=2)
    kernel /= np.sum(kernel)

    # 3. Extract Trial Landmarks (Sorted by T_I_flow)
    t_on_arr = df_eupnea["t_on_emg"].values
    t_off_emg_arr = df_eupnea["t_off_emg"].values
    t_flow_slope_arr = df_eupnea["t_flow_slope"].values
    ti_flow_arr = df_eupnea["T_I_flow_slope_ms"].values
    ti_emg_arr = df_eupnea["T_I_emg_ms"].values
    delta_t_arr = df_eupnea["delta_t_slope_ms"].values

    # Landmarks relative to I_on:
    # E_on_flow curve = ti_flow_arr (ms)
    # I_off_emg curve = ti_emg_arr (ms)
    landmarks_onset = {
        "t_i_flow": np.round(ti_flow_arr, 1).tolist(),
        "t_i_emg": np.round(ti_emg_arr, 1).tolist()
    }

    # Landmarks relative to E_on_flow:
    # I_on curve = -ti_flow_arr (ms)
    # I_off_emg curve = -delta_t_arr (ms) (since delta_t = t_flow - t_emg, t_emg - t_flow = -delta_t)
    landmarks_offset = {
        "t_i_on_rel": np.round(-ti_flow_arr, 1).tolist(),
        "t_off_emg_rel": np.round(-delta_t_arr, 1).tolist()
    }

    # 4. Compute Average Diaphragm EMG and Flow Meter Traces for Both Alignments
    print("Computing average EMG and Flow profiles across trials...")
    emg_trials_onset = []
    flow_trials_onset = []
    emg_trials_offset = []
    flow_trials_offset = []

    for i in range(n_trials):
        t0_on = t_on_arr[i]
        t0_off = t_flow_slope_arr[i]

        # Onset window
        t_eval_on = t0_on + centers_onset / 1000.0
        emg_trials_onset.append(np.interp(t_eval_on, t_nidq, emg_env))
        flow_trials_onset.append(np.interp(t_eval_on, t_nidq, flow_lp))

        # Offset window
        t_eval_off = t0_off + centers_offset / 1000.0
        emg_trials_offset.append(np.interp(t_eval_off, t_nidq, emg_env))
        flow_trials_offset.append(np.interp(t_eval_off, t_nidq, flow_lp))

    avg_signals = {
        "time_onset_ms": centers_onset.tolist(),
        "time_offset_ms": centers_offset.tolist(),
        "emg_mean_onset": np.round(np.mean(emg_trials_onset, axis=0), 4).tolist(),
        "flow_mean_onset": np.round(np.mean(flow_trials_onset, axis=0), 4).tolist(),
        "emg_mean_offset": np.round(np.mean(emg_trials_offset, axis=0), 4).tolist(),
        "flow_mean_offset": np.round(np.mean(flow_trials_offset, axis=0), 4).tolist(),
    }

    # 5. Process Each Single Unit (All 96 Neurons)
    print("Processing all 96 neurons for both alignments...")
    units_data = []

    for idx, row in df_sua.iterrows():
        uid = int(row["cluster_id"])
        depth = float(row["depth_um"]) if not np.isnan(row["depth_um"]) else 0.0
        unit_class = str(row["unit_class"]) if pd.notna(row["unit_class"]) else "Unknown"
        r_val = float(row["vector_strength"]) if pd.notna(row["vector_strength"]) else 0.0
        mean_phase = float(row["mean_phase"]) if pd.notna(row["mean_phase"]) else 0.0

        spks = spk_times_all[spk_clusters_all == uid]
        overall_fr = float(len(spks) / duration_s)

        # Onset-aligned trials: spikes and rates
        rates_onset_matrix = np.zeros((n_trials, len(centers_onset)))
        spk_trials_onset = []

        # Offset-aligned trials: spikes and rates
        rates_offset_matrix = np.zeros((n_trials, len(centers_offset)))
        spk_trials_offset = []

        for tr_idx in range(n_trials):
            t_on = t_on_arr[tr_idx]
            t_off = t_flow_slope_arr[tr_idx]

            # 1. Onset alignment
            spk_rel_on = (spks - t_on) * 1000.0  # ms
            spk_in_on = spk_rel_on[(spk_rel_on >= -150.0) & (spk_rel_on <= 300.0)]
            spk_trials_onset.append(np.round(spk_in_on, 1).tolist())

            # Histogram rate
            counts_on, _ = np.histogram(spk_in_on / 1000.0, bins=bins_onset)
            rates_onset_matrix[tr_idx, :] = np.convolve(counts_on / dt_bin, kernel, mode="same")

            # 2. Offset alignment
            spk_rel_off = (spks - t_off) * 1000.0  # ms
            spk_in_off = spk_rel_off[(spk_rel_off >= -250.0) & (spk_rel_off <= 250.0)]
            spk_trials_offset.append(np.round(spk_in_off, 1).tolist())

            # Histogram rate
            counts_off, _ = np.histogram(spk_in_off / 1000.0, bins=bins_offset)
            rates_offset_matrix[tr_idx, :] = np.convolve(counts_off / dt_bin, kernel, mode="same")

        peth_mean_onset = np.round(np.mean(rates_onset_matrix, axis=0), 2).tolist()
        peth_std_onset = np.round(np.std(rates_onset_matrix, axis=0), 2).tolist()

        peth_mean_offset = np.round(np.mean(rates_offset_matrix, axis=0), 2).tolist()
        peth_std_offset = np.round(np.std(rates_offset_matrix, axis=0), 2).tolist()

        peak_on_idx = int(np.argmax(peth_mean_onset))
        peak_off_idx = int(np.argmax(peth_mean_offset))

        units_data.append({
            "cluster_id": uid,
            "depth_um": depth,
            "overall_fr": round(overall_fr, 2),
            "unit_class": unit_class,
            "vector_strength": round(r_val, 3),
            "mean_phase": round(mean_phase, 3),
            "peak_time_on_ms": float(centers_onset[peak_on_idx]),
            "peak_time_off_ms": float(centers_offset[peak_off_idx]),
            "peth_onset": {
                "mean": peth_mean_onset,
                "std": peth_std_onset
            },
            "peth_offset": {
                "mean": peth_mean_offset,
                "std": peth_std_offset
            },
            "spikes_onset": spk_trials_onset,
            "spikes_offset": spk_trials_offset
        })

    print(f"Processed all {len(units_data)} units. Building HTML...")

    # 6. Build High-Performance Standalone HTML Tool
    payload = {
        "n_trials": n_trials,
        "landmarks_onset": landmarks_onset,
        "landmarks_offset": landmarks_offset,
        "avg_signals": avg_signals,
        "units": units_data
    }
    json_data = json.dumps(payload)

    html_file = OUT_DIR / "all_96_neurons_flow_aligned_peth_interactive.html"
    generate_html_page(html_file, json_data)
    print(f"[SUCCESS] Generated interactive HTML: {html_file}")


def generate_html_page(out_path, json_str):
    html_code = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>All 96 Neurons Flow-Aligned PETH & Raster Explorer — e097</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card: #111827;
      --card-alt: #1a2234;
      --border: rgba(255, 255, 255, 0.08);
      --border-accent: rgba(56, 189, 248, 0.3);
      --text: #f1f5f9;
      --muted: #94a3b8;
      --cyan: #38bdf8;
      --purple: #a855f7;
      --green: #22c55e;
      --red: #ef4444;
      --yellow: #facc15;
      --blue: #3b82f6;
      --insp-fill: rgba(34, 197, 94, 0.12);
      --braking-fill: rgba(250, 204, 21, 0.18);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; overflow-x: hidden; }
    
    header {
      background: rgba(17, 24, 39, 0.85);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
      padding: 14px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .header-left { display: flex; align-items: center; gap: 16px; }
    h1 { font-size: 18px; font-weight: 700; color: #fff; letter-spacing: -0.01em; }
    .badge-sub { font-size: 12px; color: var(--muted); }

    .nav-tabs { display: flex; gap: 6px; background: #060910; padding: 4px; border-radius: 8px; border: 1px solid var(--border); }
    .tab-btn {
      background: transparent; color: var(--muted); border: none; padding: 6px 14px; border-radius: 6px;
      font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.15s ease;
    }
    .tab-btn.active { background: #1e293b; color: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.3); }
    .tab-btn:hover:not(.active) { color: #fff; }

    .app-container { max-width: 1750px; margin: 0 auto; padding: 20px; display: flex; flex-direction: column; gap: 20px; }

    /* Control Toolbar */
    .toolbar {
      background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 20px;
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;
    }
    .unit-selector-group { display: flex; align-items: center; gap: 10px; }
    select, input[type="text"] {
      background: #060910; color: #fff; border: 1px solid var(--border); padding: 8px 14px;
      border-radius: 8px; font-size: 13px; font-family: inherit; outline: none;
    }
    select:focus, input[type="text"]:focus { border-color: var(--cyan); }
    .btn {
      background: #1e293b; color: #fff; border: 1px solid var(--border); padding: 8px 14px;
      border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; transition: background 0.15s;
    }
    .btn:hover { background: #334155; }
    .btn-accent { background: #0284c7; border-color: #0284c7; }
    .btn-accent:hover { background: #0369a1; }

    .align-switch { display: flex; gap: 8px; align-items: center; }
    .switch-label { font-size: 12px; color: var(--muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }

    .meta-cards-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
    .meta-card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px; }
    .meta-title { font-size: 11px; text-transform: uppercase; color: var(--muted); font-weight: 600; letter-spacing: 0.05em; margin-bottom: 4px; }
    .meta-value { font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 700; color: #fff; }

    /* Canvas Panels Layout */
    .view-container { display: grid; grid-template-columns: 1fr; gap: 20px; }
    .view-dual { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }

    .plot-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px; display: flex; flex-direction: column; gap: 12px; position: relative; }
    .plot-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 10px; }
    .plot-title { font-size: 15px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }
    .tag { font-size: 11px; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .tag-green { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
    .tag-purple { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }

    canvas { width: 100%; display: block; border-radius: 8px; background: #060910; cursor: crosshair; }

    /* Gallery Grid View */
    .gallery-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; display: none; }
    .gallery-card {
      background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 12px;
      cursor: pointer; transition: transform 0.15s, border-color 0.15s; display: flex; flex-direction: column; gap: 8px;
    }
    .gallery-card:hover { transform: translateY(-2px); border-color: var(--cyan); }
    .gallery-card.active { border-color: var(--cyan); box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.3); }
    .gal-header { display: flex; justify-content: space-between; font-size: 13px; font-weight: 700; }
    .gal-canvas { width: 100%; height: 110px; background: #060910; border-radius: 6px; }

    /* Legend */
    .legend-bar {
      display: flex; gap: 16px; flex-wrap: wrap; background: #060910; padding: 10px 16px;
      border-radius: 8px; border: 1px solid var(--border); font-size: 12px; align-items: center;
    }
    .leg-item { display: flex; align-items: center; gap: 6px; }
    .leg-line { width: 16px; height: 3px; border-radius: 2px; }
    .leg-box { width: 14px; height: 14px; border-radius: 3px; }
  </style>
</head>
<body>

  <header>
    <div class="header-left">
      <h1>All 96 Neurons Flow-Aligned PETH & Spike Raster Explorer</h1>
      <span class="badge-sub">Adult Mouse e097 (60s Pre-Laser) • 198 Eupneic Cycles Sorted by $T_I^{flow}$</span>
    </div>
    <div class="nav-tabs">
      <button class="tab-btn active" id="tabDeep">Neuron Deep-Dive</button>
      <button class="tab-btn" id="tabGallery">Population Gallery (96 Neurons)</button>
    </div>
  </header>

  <div class="app-container">

    <!-- TOOLBAR (Neuron Picker, Filters, Alignment Mode) -->
    <div class="toolbar">
      <div class="unit-selector-group">
        <button class="btn" id="btnPrev">◀ Prev</button>
        <select id="unitSelect"></select>
        <button class="btn" id="btnNext">Next ▶</button>
        <input type="text" id="searchUnit" placeholder="Jump to Unit #..." style="width: 130px;">
      </div>

      <div class="align-switch">
        <span class="switch-label">Display Mode:</span>
        <button class="btn btn-accent" id="btnModeDual">Dual Comparison (Both Alignments)</button>
        <button class="btn" id="btnModeOnset">Inspiration Onset (I_on)</button>
        <button class="btn" id="btnModeOffset">Flow Expiration Onset (E_on)</button>
      </div>

      <div class="align-switch">
        <span class="switch-label">Filter:</span>
        <button class="btn" id="fltAll" style="background:#334155;">All (96)</button>
        <button class="btn" id="fltInsp">Inspiratory (12)</button>
        <button class="btn" id="fltExp">Expiratory (73)</button>
        <button class="btn" id="fltTonic">Tonic (11)</button>
      </div>
    </div>

    <!-- METADATA CARDS FOR SELECTED NEURON -->
    <div class="meta-cards-row" id="metaRow">
      <div class="meta-card">
        <div class="meta-title">Selected Unit</div>
        <div class="meta-value" id="metaUnit">Unit #--</div>
      </div>
      <div class="meta-card">
        <div class="meta-title">Anatomical Depth</div>
        <div class="meta-value" id="metaDepth">-- µm</div>
      </div>
      <div class="meta-card">
        <div class="meta-title">Phase Archetype</div>
        <div class="meta-value" id="metaClass">--</div>
      </div>
      <div class="meta-card">
        <div class="meta-title">Overall Firing Rate</div>
        <div class="meta-value" id="metaFR">-- Hz</div>
      </div>
      <div class="meta-card">
        <div class="meta-title">Rayleigh Strength (R)</div>
        <div class="meta-value" id="metaR" style="color: var(--yellow);">--</div>
      </div>
      <div class="meta-card">
        <div class="meta-title">Peak Latency (Onset)</div>
        <div class="meta-value" id="metaPeakOn" style="color: var(--green);">-- ms</div>
      </div>
      <div class="meta-card">
        <div class="meta-title">Peak Latency (Offset)</div>
        <div class="meta-value" id="metaPeakOff" style="color: var(--purple);">-- ms</div>
      </div>
    </div>

    <!-- LEGEND BAR -->
    <div class="legend-bar">
      <span style="font-weight: 700; color: #fff;">Legend:</span>
      <div class="leg-item"><div class="leg-line" style="background: #22c55e;"></div> Inspiration Onset ($I_{on}^{EMG}$)</div>
      <div class="leg-item"><div class="leg-line" style="background: #a855f7;"></div> Flow Expiration Onset ($E_{on}^{flow, slope}$, Trough)</div>
      <div class="leg-item"><div class="leg-line" style="background: #ef4444; border-top: 1px dashed #ef4444;"></div> Diaphragm EMG Offset ($I_{off}^{EMG}$)</div>
      <div class="leg-item"><div class="leg-box" style="background: rgba(34, 197, 94, 0.2); border: 1px solid #22c55e;"></div> Inspiratory Phase ($T_I^{flow}$)</div>
      <div class="leg-item"><div class="leg-box" style="background: rgba(250, 204, 21, 0.25); border: 1px solid #ca8a04;"></div> Post-I Diaphragm Braking Phase (~27 ms)</div>
      <div class="leg-item"><div class="leg-line" style="background: #38bdf8;"></div> Mean Firing Rate (Hz) ± 1 SD</div>
      <div class="leg-item"><div class="leg-line" style="background: #059669;"></div> Flow Meter Average</div>
      <div class="leg-item"><div class="leg-line" style="background: #ea580c;"></div> Diaphragm EMG Envelope Average</div>
    </div>

    <!-- DEEP-DIVE INTERACTIVE CANVAS CONTAINERS -->
    <div id="deepDiveView" class="view-dual">
      <!-- PANEL 1: INSPIRATION ONSET ALIGNED -->
      <div class="plot-card" id="cardOnset">
        <div class="plot-header">
          <div class="plot-title">
            <span>Aligned to Inspiration Onset ($I_{on}^{EMG}$, $t=0$)</span>
            <span class="tag tag-green">Sorted by $T_I^{flow}$ (Ascending)</span>
          </div>
          <span style="font-size: 12px; color: var(--muted); font-family: monospace;" id="hudOnset">t: -- ms | FR: -- Hz</span>
        </div>
        <!-- Raster Canvas -->
        <canvas id="canvasRasterOnset" width="820" height="340"></canvas>
        <!-- PETH Canvas -->
        <canvas id="canvasPethOnset" width="820" height="230"></canvas>
      </div>

      <!-- PANEL 2: FLOW EXPIRATION ONSET ALIGNED -->
      <div class="plot-card" id="cardOffset">
        <div class="plot-header">
          <div class="plot-title">
            <span>Aligned to Flow Expiration Onset ($E_{on}^{flow}$, $t=0$)</span>
            <span class="tag tag-purple">Sorted by $T_I^{flow}$ (Ascending)</span>
          </div>
          <span style="font-size: 12px; color: var(--muted); font-family: monospace;" id="hudOffset">t: -- ms | FR: -- Hz</span>
        </div>
        <!-- Raster Canvas -->
        <canvas id="canvasRasterOffset" width="820" height="340"></canvas>
        <!-- PETH Canvas -->
        <canvas id="canvasPethOffset" width="820" height="230"></canvas>
      </div>
    </div>

    <!-- GALLERY VIEW OF ALL 96 NEURONS -->
    <div id="galleryView" class="gallery-grid"></div>

  </div>

  <script>
    const RAW_DATA = __JSON_DATA__;
    let currentUnitIndex = 0;
    let displayMode = 'dual'; // 'dual', 'onset', 'offset'
    let currentFilter = 'all'; // 'all', 'insp', 'exp', 'tonic'
    let filteredUnits = [...RAW_DATA.units];

    // Find exemplary Unit 255 by default if present
    const defaultIdx = RAW_DATA.units.findIndex(u => u.cluster_id === 255);
    if (defaultIdx >= 0) currentUnitIndex = defaultIdx;

    // Elements
    const unitSelect = document.getElementById('unitSelect');
    const deepDiveView = document.getElementById('deepDiveView');
    const galleryView = document.getElementById('galleryView');
    const tabDeep = document.getElementById('tabDeep');
    const tabGallery = document.getElementById('tabGallery');

    // Populate Select Dropdown
    function populateUnitSelect() {
      unitSelect.innerHTML = '';
      RAW_DATA.units.forEach((u, i) => {
        const opt = document.createElement('option');
        opt.value = i;
        opt.innerText = `Unit #${u.cluster_id} (${u.depth_um} µm, ${u.unit_class}, ${u.overall_fr} Hz)`;
        unitSelect.appendChild(opt);
      });
      unitSelect.value = currentUnitIndex;
    }

    function updateMetadata(u) {
      document.getElementById('metaUnit').innerText = `Unit #${u.cluster_id}`;
      document.getElementById('metaDepth').innerText = `${u.depth_um.toFixed(0)} µm`;
      document.getElementById('metaClass').innerText = u.unit_class;
      document.getElementById('metaFR').innerText = `${u.overall_fr.toFixed(2)} Hz`;
      document.getElementById('metaR').innerText = u.vector_strength.toFixed(3);
      document.getElementById('metaPeakOn').innerText = `${u.peak_time_on_ms > 0 ? '+' : ''}${u.peak_time_on_ms.toFixed(1)} ms`;
      document.getElementById('metaPeakOff').innerText = `${u.peak_time_off_ms > 0 ? '+' : ''}${u.peak_time_off_ms.toFixed(1)} ms`;
    }

    // -----------------------------------------------------------------
    // CANVAS RENDERING ENGINES
    // -----------------------------------------------------------------
    function drawRaster(canvasId, mode, u, mouseX = -1) {
      const cv = document.getElementById(canvasId);
      const ctx = cv.getContext('2d');
      const w = cv.width;
      const h = cv.height;
      ctx.clearRect(0, 0, w, h);

      const padL = 60, padR = 25, padT = 15, padB = 30;
      const plotW = w - padL - padR;
      const plotH = h - padT - padB;

      const tMin = mode === 'onset' ? -150.0 : -250.0;
      const tMax = mode === 'onset' ? 300.0 : 250.0;
      const nTrials = RAW_DATA.n_trials;

      function toX(t) { return padL + (t - tMin) / (tMax - tMin) * plotW; }
      function toY(tr) { return padT + plotH - (tr / nTrials) * plotH; }

      // 1. Shaded Inspiration Bands & Post-I Diaphragm Braking Bands
      for (let tr = 0; tr < nTrials; tr++) {
        const yTop = toY(tr + 1);
        const yH = Math.max(1.5, plotH / nTrials);

        if (mode === 'onset') {
          // Inspiration: [0, t_i_flow]
          const x0 = toX(0);
          const xFlow = toX(RAW_DATA.landmarks_onset.t_i_flow[tr]);
          const xEmg = toX(RAW_DATA.landmarks_onset.t_i_emg[tr]);

          // Inspiratory shaded band (green)
          ctx.fillStyle = 'rgba(34, 197, 94, 0.12)';
          ctx.fillRect(x0, yTop, xFlow - x0, yH);

          // Post-I Braking band (yellow): [t_i_flow, t_i_emg]
          ctx.fillStyle = 'rgba(250, 204, 21, 0.22)';
          ctx.fillRect(xFlow, yTop, xEmg - xFlow, yH);
        } else {
          // Expiration aligned:
          // Inspiration: [-t_i_flow, 0]
          const xOn = toX(RAW_DATA.landmarks_offset.t_i_on_rel[tr]);
          const x0 = toX(0);
          const xEmg = toX(RAW_DATA.landmarks_offset.t_off_emg_rel[tr]);

          // Inspiratory shaded band (green)
          ctx.fillStyle = 'rgba(34, 197, 94, 0.12)';
          ctx.fillRect(xOn, yTop, x0 - xOn, yH);

          // Post-I Braking band (yellow): [0, t_off_emg_rel]
          ctx.fillStyle = 'rgba(250, 204, 21, 0.22)';
          ctx.fillRect(x0, yTop, xEmg - x0, yH);
        }
      }

      // 2. Draw Landmark Lines
      // Zero Line (t = 0)
      const xZero = toX(0);
      ctx.strokeStyle = mode === 'onset' ? '#22c55e' : '#a855f7';
      ctx.lineWidth = 2.0;
      ctx.beginPath(); ctx.moveTo(xZero, padT); ctx.lineTo(xZero, padT + plotH); ctx.stroke();

      // Flow Expiration Onset / T_I Flow curve
      ctx.strokeStyle = '#a855f7';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let tr = 0; tr < nTrials; tr++) {
        const xPt = mode === 'onset' ? toX(RAW_DATA.landmarks_onset.t_i_flow[tr]) : toX(RAW_DATA.landmarks_offset.t_i_on_rel[tr]);
        const yPt = toY(tr + 0.5);
        if (tr === 0) ctx.moveTo(xPt, yPt); else ctx.lineTo(xPt, yPt);
      }
      ctx.stroke();

      // Diaphragm EMG Offset curve (Red Dotted)
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      for (let tr = 0; tr < nTrials; tr++) {
        const xPt = mode === 'onset' ? toX(RAW_DATA.landmarks_onset.t_i_emg[tr]) : toX(RAW_DATA.landmarks_offset.t_off_emg_rel[tr]);
        const yPt = toY(tr + 0.5);
        if (tr === 0) ctx.moveTo(xPt, yPt); else ctx.lineTo(xPt, yPt);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // 3. Draw Spike Ticks
      const spkList = mode === 'onset' ? u.spikes_onset : u.spikes_offset;
      ctx.fillStyle = '#f8fafc';
      const tickH = Math.max(2.0, plotH / nTrials * 0.95);
      for (let tr = 0; tr < nTrials; tr++) {
        const yTick = toY(tr + 1);
        const trSpks = spkList[tr];
        for (let s = 0; s < trSpks.length; s++) {
          const xSpk = toX(trSpks[s]);
          ctx.fillRect(xSpk, yTick, 1.6, tickH);
        }
      }

      // 4. Axis & Labels
      ctx.strokeStyle = 'rgba(255,255,255,0.15)';
      ctx.lineWidth = 1;
      ctx.strokeRect(padL, padT, plotW, plotH);

      // Y-axis labels
      ctx.fillStyle = '#94a3b8';
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`Trial ${nTrials} (Long T_I)`, padL - 8, padT + 12);
      ctx.fillText('Trial 1 (Short T_I)', padL - 8, padT + plotH);
      ctx.fillText('Breaths', padL - 8, padT + plotH / 2);

      // Crosshair
      if (mouseX >= padL && mouseX <= padL + plotW) {
        ctx.strokeStyle = 'rgba(255,255,255,0.35)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(mouseX, padT); ctx.lineTo(mouseX, padT + plotH); ctx.stroke();
      }
    }

    function drawPETH(canvasId, mode, u, mouseX = -1) {
      const cv = document.getElementById(canvasId);
      const ctx = cv.getContext('2d');
      const w = cv.width;
      const h = cv.height;
      ctx.clearRect(0, 0, w, h);

      const padL = 60, padR = 25, padT = 15, padB = 30;
      const plotW = w - padL - padR;
      const plotH = h - padT - padB;

      const tMin = mode === 'onset' ? -150.0 : -250.0;
      const tMax = mode === 'onset' ? 300.0 : 250.0;
      const timeArr = mode === 'onset' ? RAW_DATA.avg_signals.time_onset_ms : RAW_DATA.avg_signals.time_offset_ms;
      const peth = mode === 'onset' ? u.peth_onset : u.peth_offset;

      const maxRate = Math.max(10, Math.max(...peth.mean) * 1.35);

      function toX(t) { return padL + (t - tMin) / (tMax - tMin) * plotW; }
      function toYRate(r) { return padT + plotH - (r / maxRate) * plotH; }

      // 1. Grid lines
      ctx.strokeStyle = 'rgba(255,255,255,0.06)';
      ctx.lineWidth = 1;
      for (let g = 0; g <= 4; g++) {
        const val = (maxRate / 4) * g;
        const yG = toYRate(val);
        ctx.beginPath(); ctx.moveTo(padL, yG); ctx.lineTo(padL + plotW, yG); ctx.stroke();
        ctx.fillStyle = '#64748b';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.textAlign = 'right';
        ctx.fillText(val.toFixed(0), padL - 8, yG + 3);
      }

      // 2. Average Signals Overlay (Secondary Scales)
      // Flow Meter (Green, Normalized)
      const flowAvg = mode === 'onset' ? RAW_DATA.avg_signals.flow_mean_onset : RAW_DATA.avg_signals.flow_mean_offset;
      ctx.strokeStyle = 'rgba(5, 150, 105, 0.45)';
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let i = 0; i < timeArr.length; i++) {
        const x = toX(timeArr[i]);
        // Normalize flow (-4 to +3 V) into bottom 35%
        const normF = (flowAvg[i] - (-3.0)) / (3.0 - (-3.0));
        const y = padT + plotH - normF * (plotH * 0.45);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Diaphragm EMG Envelope (Orange, Normalized)
      const emgAvg = mode === 'onset' ? RAW_DATA.avg_signals.emg_mean_onset : RAW_DATA.avg_signals.emg_mean_offset;
      ctx.strokeStyle = 'rgba(234, 88, 12, 0.45)';
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let i = 0; i < timeArr.length; i++) {
        const x = toX(timeArr[i]);
        // Normalize emg (0 to 0.4 V) into bottom 40%
        const normE = emgAvg[i] / 0.35;
        const y = padT + plotH - normE * (plotH * 0.45);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // 3. PETH ± 1 SD Shaded Shadow
      ctx.fillStyle = 'rgba(56, 189, 248, 0.18)';
      ctx.beginPath();
      for (let i = 0; i < timeArr.length; i++) {
        const x = toX(timeArr[i]);
        const yUpper = toYRate(peth.mean[i] + peth.std[i]);
        if (i === 0) ctx.moveTo(x, yUpper); else ctx.lineTo(x, yUpper);
      }
      for (let i = timeArr.length - 1; i >= 0; i--) {
        const x = toX(timeArr[i]);
        const yLower = toYRate(Math.max(0, peth.mean[i] - peth.std[i]));
        ctx.lineTo(x, yLower);
      }
      ctx.closePath();
      ctx.fill();

      // 4. Mean Firing Rate Curve
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      for (let i = 0; i < timeArr.length; i++) {
        const x = toX(timeArr[i]);
        const y = toYRate(peth.mean[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // 5. Landmark Vertical Lines
      const xZero = toX(0);
      ctx.strokeStyle = mode === 'onset' ? '#22c55e' : '#a855f7';
      ctx.lineWidth = 2.0;
      ctx.beginPath(); ctx.moveTo(xZero, padT); ctx.lineTo(xZero, padT + plotH); ctx.stroke();

      // Mean T_I line
      const meanTIFlow = 68.0;
      const xMeanFlow = mode === 'onset' ? toX(meanTIFlow) : toX(-meanTIFlow);
      ctx.strokeStyle = '#a855f7';
      ctx.lineWidth = 1.6;
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(xMeanFlow, padT); ctx.lineTo(xMeanFlow, padT + plotH); ctx.stroke();
      ctx.setLineDash([]);

      // Mean Diaphragm Offset line
      const meanTIEmg = 94.9;
      const xMeanEmg = mode === 'onset' ? toX(meanTIEmg) : toX(27.3);
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 1.4;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(xMeanEmg, padT); ctx.lineTo(xMeanEmg, padT + plotH); ctx.stroke();
      ctx.setLineDash([]);

      // 6. X-Axis Ticks & Labels
      ctx.strokeStyle = 'rgba(255,255,255,0.15)';
      ctx.lineWidth = 1;
      ctx.strokeRect(padL, padT, plotW, plotH);

      ctx.fillStyle = '#94a3b8';
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      const stepT = mode === 'onset' ? 50 : 50;
      for (let t = tMin; t <= tMax; t += stepT) {
        const xTick = toX(t);
        ctx.beginPath(); ctx.moveTo(xTick, padT + plotH); ctx.lineTo(xTick, padT + plotH + 5); ctx.stroke();
        ctx.fillText(`${t > 0 ? '+' : ''}${t}`, xTick, padT + plotH + 18);
      }

      ctx.fillText(mode === 'onset' ? 'Time from Inspiration Onset I_on (ms)' : 'Time from Flow Expiration Onset E_on (ms)', padL + plotW / 2, padT + plotH + 30);
      ctx.save();
      ctx.translate(15, padT + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Firing Rate (Hz)', 0, 0);
      ctx.restore();

      // Crosshair & Readout
      if (mouseX >= padL && mouseX <= padL + plotW) {
        ctx.strokeStyle = 'rgba(255,255,255,0.35)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(mouseX, padT); ctx.lineTo(mouseX, padT + plotH); ctx.stroke();

        const curT = tMin + (mouseX - padL) / plotW * (tMax - tMin);
        // Find nearest bin
        let nearestIdx = 0;
        let minD = 9999;
        for (let i = 0; i < timeArr.length; i++) {
          const d = Math.abs(timeArr[i] - curT);
          if (d < minD) { minD = d; nearestIdx = i; }
        }
        const hud = document.getElementById(mode === 'onset' ? 'hudOnset' : 'hudOffset');
        hud.innerText = `t: ${timeArr[nearestIdx] > 0 ? '+' : ''}${timeArr[nearestIdx]} ms | FR: ${peth.mean[nearestIdx].toFixed(1)} ± ${peth.std[nearestIdx].toFixed(1)} Hz`;
      }
    }

    function renderCurrentUnit() {
      const u = RAW_DATA.units[currentUnitIndex];
      updateMetadata(u);
      unitSelect.value = currentUnitIndex;

      drawRaster('canvasRasterOnset', 'onset', u);
      drawPETH('canvasPethOnset', 'onset', u);

      drawRaster('canvasRasterOffset', 'offset', u);
      drawPETH('canvasPethOffset', 'offset', u);
    }

    // Interactive Hover Listeners
    function attachCrosshair(canvasRasterId, canvasPethId, mode) {
      const rCv = document.getElementById(canvasRasterId);
      const pCv = document.getElementById(canvasPethId);

      function onMove(e) {
        const rect = e.target.getBoundingClientRect();
        const scaleX = e.target.width / rect.width;
        const mX = (e.clientX - rect.left) * scaleX;
        const u = RAW_DATA.units[currentUnitIndex];
        drawRaster(canvasRasterId, mode, u, mX);
        drawPETH(canvasPethId, mode, u, mX);
      }

      function onLeave() {
        const u = RAW_DATA.units[currentUnitIndex];
        drawRaster(canvasRasterId, mode, u, -1);
        drawPETH(canvasPethId, mode, u, -1);
      }

      rCv.addEventListener('mousemove', onMove);
      rCv.addEventListener('mouseleave', onLeave);
      pCv.addEventListener('mousemove', onMove);
      pCv.addEventListener('mouseleave', onLeave);
    }

    attachCrosshair('canvasRasterOnset', 'canvasPethOnset', 'onset');
    attachCrosshair('canvasRasterOffset', 'canvasPethOffset', 'offset');

    // Navigation Events
    document.getElementById('btnPrev').onclick = () => {
      currentUnitIndex = (currentUnitIndex - 1 + RAW_DATA.units.length) % RAW_DATA.units.length;
      renderCurrentUnit();
    };
    document.getElementById('btnNext').onclick = () => {
      currentUnitIndex = (currentUnitIndex + 1) % RAW_DATA.units.length;
      renderCurrentUnit();
    };
    unitSelect.onchange = (e) => {
      currentUnitIndex = parseInt(e.target.value);
      renderCurrentUnit();
    };
    document.getElementById('searchUnit').onchange = (e) => {
      const val = parseInt(e.target.value.trim());
      const idx = RAW_DATA.units.findIndex(u => u.cluster_id === val);
      if (idx >= 0) { currentUnitIndex = idx; renderCurrentUnit(); }
      else { alert(`Unit #${val} not found in SUA population.`); }
    };

    // Keyboard Shortcuts (Left / Right Arrow)
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.key === 'ArrowLeft') { document.getElementById('btnPrev').click(); }
      else if (e.key === 'ArrowRight') { document.getElementById('btnNext').click(); }
    });

    // Display Mode Toggles
    const cardOnset = document.getElementById('cardOnset');
    const cardOffset = document.getElementById('cardOffset');
    const btnDual = document.getElementById('btnModeDual');
    const btnOn = document.getElementById('btnModeOnset');
    const btnOff = document.getElementById('btnModeOffset');

    function setDisplayMode(mode) {
      displayMode = mode;
      [btnDual, btnOn, btnOff].forEach(b => b.classList.remove('btn-accent'));
      if (mode === 'dual') {
        btnDual.classList.add('btn-accent');
        deepDiveView.className = 'view-dual';
        cardOnset.style.display = 'flex';
        cardOffset.style.display = 'flex';
      } else if (mode === 'onset') {
        btnOn.classList.add('btn-accent');
        deepDiveView.className = 'view-container';
        cardOnset.style.display = 'flex';
        cardOffset.style.display = 'none';
      } else {
        btnOff.classList.add('btn-accent');
        deepDiveView.className = 'view-container';
        cardOnset.style.display = 'none';
        cardOffset.style.display = 'flex';
      }
      renderCurrentUnit();
    }

    btnDual.onclick = () => setDisplayMode('dual');
    btnOn.onclick = () => setDisplayMode('onset');
    btnOff.onclick = () => setDisplayMode('offset');

    // Tab Navigation (Deep Dive vs Gallery)
    tabDeep.onclick = () => {
      tabDeep.classList.add('active'); tabGallery.classList.remove('active');
      deepDiveView.style.display = displayMode === 'dual' ? 'grid' : 'grid';
      galleryView.style.display = 'none';
    };
    tabGallery.onclick = () => {
      tabGallery.classList.add('active'); tabDeep.classList.remove('active');
      deepDiveView.style.display = 'none';
      galleryView.style.display = 'grid';
      buildGallery();
    };

    // Build Gallery View of All 96 Neurons
    function buildGallery() {
      galleryView.innerHTML = '';
      RAW_DATA.units.forEach((u, i) => {
        if (currentFilter !== 'all' && u.unit_class.toLowerCase() !== currentFilter) return;

        const card = document.createElement('div');
        card.className = `gallery-card ${i === currentUnitIndex ? 'active' : ''}`;
        card.innerHTML = `
          <div class="gal-header">
            <span>Unit #${u.cluster_id}</span>
            <span style="color: var(--muted);">${u.depth_um.toFixed(0)} µm</span>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--muted);">
            <span>${u.unit_class}</span>
            <span>FR: ${u.overall_fr} Hz</span>
          </div>
          <canvas class="gal-canvas" id="galCv_${u.cluster_id}" width="260" height="110"></canvas>
        `;

        card.onclick = () => {
          currentUnitIndex = i;
          renderCurrentUnit();
          tabDeep.click();
        };

        galleryView.appendChild(card);

        // Render mini-PETH on canvas
        setTimeout(() => {
          const cv = document.getElementById(`galCv_${u.cluster_id}`);
          if (!cv) return;
          const ctx = cv.getContext('2d');
          const w = cv.width, h = cv.height;
          const peth = u.peth_offset; // Default to offset in gallery
          const maxR = Math.max(5, Math.max(...peth.mean));

          // Draw Zero line
          const x0 = (250 / 500) * w;
          ctx.strokeStyle = '#a855f7';
          ctx.lineWidth = 1.2;
          ctx.beginPath(); ctx.moveTo(x0, 0); ctx.lineTo(x0, h); ctx.stroke();

          // Curve
          ctx.strokeStyle = '#38bdf8';
          ctx.lineWidth = 1.8;
          ctx.beginPath();
          for (let p = 0; p < peth.mean.length; p++) {
            const x = (p / peth.mean.length) * w;
            const y = h - (peth.mean[p] / maxR) * (h - 10) - 5;
            if (p === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }, 10);
      });
    }

    // Filter Buttons
    document.getElementById('fltAll').onclick = () => { currentFilter = 'all'; buildGallery(); };
    document.getElementById('fltInsp').onclick = () => { currentFilter = 'inspiratory'; buildGallery(); };
    document.getElementById('fltExp').onclick = () => { currentFilter = 'expiratory'; buildGallery(); };
    document.getElementById('fltTonic').onclick = () => { currentFilter = 'tonic'; buildGallery(); };

    // Initial load
    populateUnitSelect();
    renderCurrentUnit();
  </script>
</body>
</html>
"""
    final_html = html_code.replace("__JSON_DATA__", json_str)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_html)


if __name__ == "__main__":
    main()
