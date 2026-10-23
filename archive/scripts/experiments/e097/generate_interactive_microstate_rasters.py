"""Generates an Interactive, Zoomable Population Micro-State Raster Hub
for e097 win1_baseline and win2_anox1.

Features:
- Subwindow toggle: win1_baseline (187 Breaths) vs win2_anox1 (90 Breaths).
- Alignment toggle: Inspiration Offset (I_off) vs Inspiration Onset (I_on).
- Sorter toggle: Kilosort 4 (GPU) vs Lupin (SVD-OMP).
- Sort order toggle: Chronological vs Sorted by Inspiratory Duration (T_I).
- Fully interactive Plotly heatmap with box-zoom, pan, trial-level inspection,
  synchronized state occupancy probability PETH, and analog airflow / EMG waveforms.
- Standalone zero-dependency HTML dashboard.
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from hmmlearn import hmm

REPO_ROOT = Path(r"c:\Users\lpascu\resp_analysis")
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.processing.analog_nidq.proc_emg import process_emg
from src.pipeline.processing.analog_nidq.proc_flow import process_flow_meter
from src.pipeline.config import EMGConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("INTERACTIVE_HMM_RASTER")


def extract_subwindow_breaths(start_t: float, dur: float) -> Dict[str, Any]:
    """Extracts NIDQ analog signals and breath segmentation for a subwindow."""
    nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
    fs = 10000.0
    n_ch = 9
    s0 = int(start_t * fs)
    s1 = int((start_t + dur) * fs)

    logger.info(f"Loading analog data from {nidq_file} (t={start_t}s to {start_t+dur}s)...")
    raw = np.memmap(nidq_file, dtype="int16", mode="r")
    slice_data = raw[s0 * n_ch : s1 * n_ch].reshape(-1, n_ch)
    emg_v = slice_data[:, 0].astype(np.float32) * (10.0 / 32768.0)
    flow_v = slice_data[:, 5].astype(np.float32) * (10.0 / 32768.0)
    t_vec = np.arange(len(emg_v)) / fs

    emg_cfg = EMGConfig(fs=fs, highpass_cutoff=300.0, lowpass_cutoff=3000.0, smoothing_window_ms=50.0, peak_threshold_sd=2.0)
    emg_proc = process_emg(emg_v, t_vec, emg_cfg)
    valid_pairs = [(on, off) for on, off in zip(emg_proc.breath_onsets, emg_proc.breath_offsets) if on > 0.05 and on < t_vec[-1] - 0.05 and off > on]
    valid_onsets = np.array([p[0] for p in valid_pairs])
    valid_offsets = np.array([p[1] for p in valid_pairs])

    flow_res = process_flow_meter(flow_v, t_vec, valid_onsets, valid_offsets, fs=fs)
    flow_zcs = flow_res.flow_zero_crossings
    n_breaths = len(valid_onsets)

    cycle_ends = np.append(valid_onsets[1:], dur)
    ti_arr = (flow_zcs - valid_onsets) * 1000.0  # ms
    te_arr = (cycle_ends - flow_zcs) * 1000.0   # ms
    tot_arr = (cycle_ends - valid_onsets) * 1000.0 # ms

    # Gasps detection for anoxia window (valid_onsets >= 27.0s)
    gasp_mask = np.zeros(n_breaths, dtype=bool)
    if start_t >= 1790.0:
        for i in range(n_breaths):
            if valid_onsets[i] >= 27.0:
                gasp_mask[i] = True

    logger.info(f"Loaded {n_breaths} breaths. Mean T_I = {np.mean(ti_arr):.1f} ± {np.std(ti_arr):.1f} ms.")

    return {
        "fs": fs,
        "dur": dur,
        "start_t": start_t,
        "t_vec": t_vec,
        "emg_v": emg_v,
        "emg_env": emg_proc.envelope,
        "flow_v": flow_v,
        "valid_onsets": valid_onsets,
        "valid_offsets": valid_offsets,
        "flow_zcs": flow_zcs,
        "cycle_ends": cycle_ends,
        "ti_arr": ti_arr,
        "te_arr": te_arr,
        "tot_arr": tot_arr,
        "gasp_mask": gasp_mask,
        "n_breaths": n_breaths
    }


def fit_and_extract_rasters(subwin_key: str, sorter_key: str, sorter_label: str, breath_data: Dict[str, Any], dt: float = 0.010, n_states: int = 5):
    """Fits 5-state Poisson HMM and extracts both I_off and I_on aligned microstate matrices."""
    base_dir = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / subwin_key / sorter_key
    spk_times = np.load(base_dir / "alf" / "spike_times.npy")
    spk_clusters = np.load(base_dir / "alf" / "spike_clusters.npy")
    qc_csv = base_dir / f"{sorter_key}_qc_summary.csv"
    if not qc_csv.exists():
        qc_csv = base_dir / "cluster_info.tsv"
    sep = "\t" if qc_csv.suffix == ".tsv" else ","
    qc_df = pd.read_csv(qc_csv, sep=sep)
    if "passed_allen_qc" in qc_df.columns:
        df_qc = qc_df[qc_df["passed_allen_qc"] == True].copy()
    else:
        df_qc = qc_df.copy()

    df_qc = df_qc.sort_values(by="depth_um", ascending=False).reset_index(drop=True)
    if len(df_qc) > 120:
        df_qc = df_qc.head(120)

    cids = df_qc["cluster_id"].values
    depths = df_qc["depth_um"].values
    n_units = len(cids)
    dur = breath_data["dur"]

    time_bins = np.arange(0, dur + dt, dt)
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2.0
    n_time_bins = len(bin_centers)

    counts = np.zeros((n_time_bins, n_units), dtype=np.int32)
    for u_idx, cid in enumerate(cids):
        c_spk = spk_times[spk_clusters == cid]
        counts[:, u_idx] = np.histogram(c_spk, bins=time_bins)[0][:n_time_bins]

    logger.info(f"[{subwin_key} - {sorter_label}] Fitting {n_states}-state Poisson HMM on {counts.shape}...")
    model = hmm.PoissonHMM(n_components=n_states, n_iter=100, random_state=42, tol=1e-3)
    model.fit(counts)
    raw_states = model.predict(counts)

    valid_onsets = breath_data["valid_onsets"]
    flow_zcs = breath_data["flow_zcs"]
    valid_offsets = breath_data["valid_offsets"]
    n_breaths = breath_data["n_breaths"]

    # Compute phase labels for canonical mapping
    phase_labels = np.zeros(n_time_bins, dtype=int)
    for b_idx in range(n_breaths):
        t_on = valid_onsets[b_idx]
        t_zc = flow_zcs[b_idx]
        t_off = valid_offsets[b_idx]
        t_cend = breath_data["cycle_ends"][b_idx]

        mask_insp = (bin_centers >= t_on) & (bin_centers < t_zc)
        phase_labels[mask_insp] = 1
        mask_post = (bin_centers >= t_zc) & (bin_centers < min(t_off + 0.08, t_cend))
        phase_labels[mask_post] = 2

    # Map raw states to canonical 5 states
    state_stats = []
    for s in range(n_states):
        idx_s = np.where(raw_states == s)[0]
        if len(idx_s) == 0:
            pct_insp, pct_post, mean_rate = 0.0, 0.0, 0.0
        else:
            pct_insp = np.mean(phase_labels[idx_s] == 1) * 100.0
            pct_post = np.mean(phase_labels[idx_s] == 2) * 100.0
            mean_rate = np.mean(counts[idx_s]) / dt
        state_stats.append({
            "raw_id": s,
            "pct_insp": pct_insp,
            "pct_post": pct_post,
            "mean_rate": mean_rate
        })

    # Sort to establish canonical ordering
    sorted_by_insp = sorted(state_stats, key=lambda x: (x["pct_insp"], x["mean_rate"]), reverse=True)
    c_s0 = sorted_by_insp[0]["raw_id"]  # State 1: Insp Burst
    c_s1 = sorted_by_insp[1]["raw_id"]  # State 2: Peak / Off-switch

    rem = [x for x in state_stats if x["raw_id"] not in [c_s0, c_s1]]
    rem_by_rate = sorted(rem, key=lambda x: x["mean_rate"], reverse=True)
    c_s2 = rem_by_rate[0]["raw_id"]  # State 3: Eupneic Exp Drive
    c_s3 = rem_by_rate[-1]["raw_id"] # State 4: Late-E Baseline
    c_s4 = [x["raw_id"] for x in rem if x["raw_id"] not in [c_s2, c_s3]][0] # State 5: Quiescence / Gasp

    canonical_map = {c_s0: 0, c_s1: 1, c_s2: 2, c_s3: 3, c_s4: 4}
    canonical_states = np.array([canonical_map[s] for s in raw_states])

    state_names = [
        "State 1: Inspiratory Burst (Insp)",
        "State 2: Peak Transition / Off-Switch",
        "State 3: Eupneic Expiratory Drive",
        "State 4: Late-Expiratory Baseline (E2)",
        "State 5: Inter-Breath Quiescence / Gasp"
    ]
    state_short_names = ["1: Insp Burst", "2: Peak Off-Switch", "3: Exp Drive", "4: Late-E", "5: Quiescence"]
    state_colors = ["#10b981", "#f97316", "#38bdf8", "#64748b", "#a855f7"]

    # -------------------------------------------------------------
    # 1. ALIGNMENT BY INSPIRATION OFFSET (I_off, Flow Zero-Crossing)
    # -------------------------------------------------------------
    # Window: -250 ms to +350 ms relative to I_off
    ioff_pre_ms = 250.0
    ioff_post_ms = 350.0
    ioff_axis_ms = np.arange(-ioff_pre_ms, ioff_post_ms, dt * 1000.0)
    n_ioff_bins = len(ioff_axis_ms)

    raster_ioff = []
    flow_ioff_trials = []
    emg_ioff_trials = []
    trial_records_ioff = []

    fs_nidq = breath_data["fs"]
    flow_v = breath_data["flow_v"]
    emg_env = breath_data["emg_env"]

    for b_idx in range(n_breaths):
        t_zc = flow_zcs[b_idx]
        t_on = valid_onsets[b_idx]
        ti = breath_data["ti_arr"][b_idx]
        tot = breath_data["tot_arr"][b_idx]
        is_gasp = bool(breath_data["gasp_mask"][b_idx])

        # HMM microstate slice
        t_start = t_zc - (ioff_pre_ms / 1000.0)
        t_end = t_zc + (ioff_post_ms / 1000.0)
        mask_tr = (bin_centers >= t_start) & (bin_centers < t_end)
        s_slice = canonical_states[mask_tr]

        if len(s_slice) < n_ioff_bins:
            pad_val = s_slice[-1] if len(s_slice) > 0 else 3
            s_fixed = np.pad(s_slice, (0, n_ioff_bins - len(s_slice)), constant_values=pad_val)
        else:
            s_fixed = s_slice[:n_ioff_bins]
        raster_ioff.append(s_fixed)

        # Analog waveform slices matching HMM bins
        bin_eval_times = t_zc + (ioff_axis_ms / 1000.0)
        idx_eval = np.clip((bin_eval_times * fs_nidq).astype(int), 0, len(flow_v) - 1)
        flow_ioff_trials.append(flow_v[idx_eval])
        emg_ioff_trials.append(emg_env[idx_eval])

        trial_records_ioff.append({
            "trial_idx": b_idx,
            "trial_num": b_idx + 1,
            "is_gasp": is_gasp,
            "regime": "Gasping" if is_gasp else ("Apnea/Trans" if (subwin_key=="win2_anox1" and 24.0 <= t_on < 27.0) else "Eupnea"),
            "t_on_s": float(t_on),
            "t_zc_s": float(t_zc),
            "ti_ms": float(ti),
            "tot_ms": float(tot),
            "ion_relative_ms": float(-ti)  # When aligned to I_off, I_on is at -T_I
        })

    raster_ioff = np.array(raster_ioff)
    flow_ioff_mean = np.mean(flow_ioff_trials, axis=0).tolist()
    emg_ioff_mean = np.mean(emg_ioff_trials, axis=0).tolist()

    # Compute State Probability PETH (%)
    prob_ioff = []
    for k in range(n_states):
        prob_k = np.mean(raster_ioff == k, axis=0) * 100.0
        prob_ioff.append(prob_k.tolist())

    # -------------------------------------------------------------
    # 2. ALIGNMENT BY INSPIRATION ONSET (I_on, Diaphragm Onset)
    # -------------------------------------------------------------
    # Window: -100 ms to +500 ms relative to I_on
    ion_pre_ms = 100.0
    ion_post_ms = 500.0
    ion_axis_ms = np.arange(-ion_pre_ms, ion_post_ms, dt * 1000.0)
    n_ion_bins = len(ion_axis_ms)

    raster_ion = []
    flow_ion_trials = []
    emg_ion_trials = []
    trial_records_ion = []

    for b_idx in range(n_breaths):
        t_on = valid_onsets[b_idx]
        t_zc = flow_zcs[b_idx]
        ti = breath_data["ti_arr"][b_idx]
        tot = breath_data["tot_arr"][b_idx]
        is_gasp = bool(breath_data["gasp_mask"][b_idx])

        t_start = t_on - (ion_pre_ms / 1000.0)
        t_end = t_on + (ion_post_ms / 1000.0)
        mask_tr = (bin_centers >= t_start) & (bin_centers < t_end)
        s_slice = canonical_states[mask_tr]

        if len(s_slice) < n_ion_bins:
            pad_val = s_slice[-1] if len(s_slice) > 0 else 3
            s_fixed = np.pad(s_slice, (0, n_ion_bins - len(s_slice)), constant_values=pad_val)
        else:
            s_fixed = s_slice[:n_ion_bins]
        raster_ion.append(s_fixed)

        bin_eval_times = t_on + (ion_axis_ms / 1000.0)
        idx_eval = np.clip((bin_eval_times * fs_nidq).astype(int), 0, len(flow_v) - 1)
        flow_ion_trials.append(flow_v[idx_eval])
        emg_ion_trials.append(emg_env[idx_eval])

        trial_records_ion.append({
            "trial_idx": b_idx,
            "trial_num": b_idx + 1,
            "is_gasp": is_gasp,
            "regime": "Gasping" if is_gasp else ("Apnea/Trans" if (subwin_key=="win2_anox1" and 24.0 <= t_on < 27.0) else "Eupnea"),
            "t_on_s": float(t_on),
            "t_zc_s": float(t_zc),
            "ti_ms": float(ti),
            "tot_ms": float(tot),
            "ioff_relative_ms": float(ti)  # When aligned to I_on, I_off is at +T_I
        })

    raster_ion = np.array(raster_ion)
    flow_ion_mean = np.mean(flow_ion_trials, axis=0).tolist()
    emg_ion_mean = np.mean(emg_ion_trials, axis=0).tolist()

    prob_ion = []
    for k in range(n_states):
        prob_k = np.mean(raster_ion == k, axis=0) * 100.0
        prob_ion.append(prob_k.tolist())

    return {
        "subwin_key": subwin_key,
        "sorter_key": sorter_key,
        "sorter_label": sorter_label,
        "n_units": n_units,
        "n_breaths": n_breaths,
        "state_names": state_names,
        "state_short_names": state_short_names,
        "state_colors": state_colors,
        "ioff": {
            "axis_ms": ioff_axis_ms.tolist(),
            "raster": [r.tolist() for r in raster_ioff],
            "prob": prob_ioff,
            "trials": trial_records_ioff,
            "flow_mean": flow_ioff_mean,
            "emg_mean": emg_ioff_mean
        },
        "ion": {
            "axis_ms": ion_axis_ms.tolist(),
            "raster": [r.tolist() for r in raster_ion],
            "prob": prob_ion,
            "trials": trial_records_ion,
            "flow_mean": flow_ion_mean,
            "emg_mean": emg_ion_mean
        }
    }


def build_interactive_dashboard_html(all_data: Dict[str, Any], output_path: Path, default_subwin: str = "win1_baseline"):
    """Builds a rich, interactive, standalone HTML dashboard with Plotly."""
    json_str = json.dumps(all_data)

    if default_subwin == "win1_baseline":
        back_url = "win1_baseline_population_dynamics_dashboard.html"
        back_label = "&larr; Back to win1_baseline Dynamics Hub"
        sister_url = "../../win2_anox1/population_dynamics/win2_anox1_interactive_microstate_raster.html"
        sister_label = "⚡ win2_anox1 Interactive Raster &rarr;"
        benchmark_url = "../win1_baseline_sorter_comparison.html"
    else:
        back_url = "win2_anox1_population_dynamics_dashboard.html"
        back_label = "&larr; Back to win2_anox1 Dynamics Hub"
        sister_url = "../../win1_baseline/population_dynamics/win1_baseline_interactive_microstate_raster.html"
        sister_label = "🔬 win1_baseline Interactive Raster &rarr;"
        benchmark_url = "../win2_anox1_sorter_comparison.html"

    btn_win1_cls = "toggle-btn active" if default_subwin == "win1_baseline" else "toggle-btn"
    btn_win2_cls = "toggle-btn active anox" if default_subwin == "win2_anox1" else "toggle-btn"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>e097 Interactive Micro-State Raster &amp; Population Dynamics Hub ({default_subwin})</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: #07090e;
      color: #f1f5f9;
      line-height: 1.5;
      padding: 24px 32px;
    }}
    .container {{ max-width: 1480px; margin: 0 auto; }}
    header {{
      background: linear-gradient(180deg, #0f172a 0%, #07090e 100%);
      border-bottom: 1px solid #1e293b;
      padding-bottom: 20px;
      margin-bottom: 24px;
    }}
    .nav-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
    }}
    .nav-links {{ display: flex; gap: 10px; }}
    .back-btn {{
      background: #1e293b;
      color: #38bdf8;
      text-decoration: none;
      padding: 7px 14px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      border: 1px solid #334155;
      transition: all 0.2s;
    }}
    .back-btn:hover {{ background: #38bdf8; color: #07090e; }}
    h1 {{
      font-size: 26px;
      font-weight: 800;
      color: #f8fafc;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      color: #94a3b8;
      font-size: 14px;
      margin-top: 4px;
      max-width: 1250px;
    }}
    /* CONTROL PANEL */
    .controls-panel {{
      background: #0f1523;
      border: 1px solid #1e293b;
      border-radius: 12px;
      padding: 18px 24px;
      margin-bottom: 24px;
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 20px;
    }}
    .control-group {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .control-label {{
      font-size: 11.5px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #94a3b8;
    }}
    .btn-group {{
      display: flex;
      gap: 6px;
      background: #090d16;
      padding: 4px;
      border-radius: 8px;
      border: 1px solid #1e293b;
    }}
    .toggle-btn {{
      flex: 1;
      padding: 8px 10px;
      background: transparent;
      border: none;
      color: #94a3b8;
      font-size: 12px;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
      text-align: center;
    }}
    .toggle-btn:hover {{ color: #f8fafc; background: rgba(255,255,255,0.05); }}
    .toggle-btn.active {{
      background: #0284c7;
      color: #ffffff;
      box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }}
    .toggle-btn.active.anox {{ background: #dc2626; }}
    .toggle-btn.active.gold {{ background: #d97706; }}
    .toggle-btn.active.purple {{ background: #7c3aed; }}
    /* SUMMARY STATS */
    .stats-row {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }}
    .stat-card {{
      background: #0f1523;
      border: 1px solid #1e293b;
      border-radius: 10px;
      padding: 14px 18px;
    }}
    .stat-val {{
      font-size: 24px;
      font-weight: 800;
      color: #38bdf8;
      font-family: 'JetBrains Mono', monospace;
    }}
    .stat-lbl {{
      font-size: 11px;
      color: #94a3b8;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    /* STATE LEGEND BAR */
    .legend-bar {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 12px;
      margin-bottom: 20px;
    }}
    .legend-pill {{
      background: #0f1523;
      border: 1px solid #1e293b;
      border-radius: 8px;
      padding: 10px 14px;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .legend-color {{
      width: 14px;
      height: 14px;
      border-radius: 4px;
      flex-shrink: 0;
    }}
    .legend-text strong {{
      font-size: 12px;
      display: block;
      color: #f8fafc;
    }}
    .legend-text span {{
      font-size: 11px;
      color: #94a3b8;
    }}
    /* PLOT CONTAINERS */
    .plot-card {{
      background: #0f1523;
      border: 1px solid #1e293b;
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 24px;
    }}
    .plot-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }}
    .plot-title {{
      font-size: 16px;
      font-weight: 700;
      color: #f8fafc;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .zoom-helpers {{
      display: flex;
      gap: 8px;
    }}
    .zoom-btn {{
      background: #1e293b;
      color: #cbd5e1;
      border: 1px solid #334155;
      padding: 5px 10px;
      border-radius: 6px;
      font-size: 11.5px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .zoom-btn:hover {{
      background: #38bdf8;
      color: #07090e;
      border-color: #38bdf8;
    }}
    #rasterPlot {{ width: 100%; height: 560px; }}
    #probPlot {{ width: 100%; height: 260px; }}
    #analogPlot {{ width: 100%; height: 210px; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="nav-bar">
        <a href="{back_url}" class="back-btn">{back_label}</a>
        <div class="nav-links">
          <a href="{sister_url}" class="back-btn" style="border-color:#38bdf8;color:#38bdf8;">{sister_label}</a>
          <a href="{benchmark_url}" class="back-btn">Benchmark Hub</a>
        </div>
      </div>
      <h1>e097 &mdash; Interactive Population Micro-State Raster &amp; PETH Inspector</h1>
      <p class="subtitle">
        Zoom, pan, and inspect individual breath micro-states across 10 ms raw spiking Poisson HMM states. Toggle alignment between <strong>Inspiration Offset ($I_{{off}}$, flow zero-crossing)</strong> and <strong>Inspiration Onset ($I_{{on}}$, diaphragm EMG)</strong>.
      </p>
    </header>

    <!-- INTERACTIVE CONTROLS -->
    <div class="controls-panel">
      <!-- 1. SUBWINDOW -->
      <div class="control-group">
        <div class="control-label">1. Subwindow &amp; Regime</div>
        <div class="btn-group">
          <button class="{btn_win1_cls}" id="btn-win1" onclick="setSubwindow('win1_baseline')">win1_baseline (187 B)</button>
          <button class="{btn_win2_cls}" id="btn-win2" onclick="setSubwindow('win2_anox1')">win2_anox1 (90 B)</button>
        </div>
      </div>

      <!-- 2. ALIGNMENT (DEFAULT I_OFF AS REQUESTED) -->
      <div class="control-group">
        <div class="control-label">2. Breath Alignment Event</div>
        <div class="btn-group">
          <button class="toggle-btn active gold" id="btn-ioff" onclick="setAlignment('ioff')">⭐ Inspiration Offset (I_off)</button>
          <button class="toggle-btn" id="btn-ion" onclick="setAlignment('ion')">Inspiration Onset (I_on)</button>
        </div>
      </div>

      <!-- 3. SORTER -->
      <div class="control-group">
        <div class="control-label">3. Spike Sorter</div>
        <div class="btn-group">
          <button class="toggle-btn active" id="btn-ks4" onclick="setSorter('kilosort4')">Kilosort 4 (GPU)</button>
          <button class="toggle-btn" id="btn-lupin" onclick="setSorter('lupin')">Lupin (SVD-OMP)</button>
        </div>
      </div>

      <!-- 4. SORT ORDER -->
      <div class="control-group">
        <div class="control-label">4. Trial Sorting Order</div>
        <div class="btn-group">
          <button class="toggle-btn active purple" id="btn-sort-chrono" onclick="setSortOrder('chrono')">Chronological (1 &rarr; N)</button>
          <button class="toggle-btn" id="btn-sort-ti" onclick="setSortOrder('ti')">Sorted by T_I Duration</button>
        </div>
      </div>
    </div>

    <!-- STATS ROW -->
    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-lbl">Active Breaths</div>
        <div class="stat-val" id="stat-nbreaths">187 Breaths</div>
      </div>
      <div class="stat-card">
        <div class="stat-lbl">Mean Insp Time (T_I)</div>
        <div class="stat-val" id="stat-ti" style="color:#10b981;">120.7 ms</div>
      </div>
      <div class="stat-card">
        <div class="stat-lbl">Alignment Reference</div>
        <div class="stat-val" id="stat-align" style="color:#fbbf24;">I_off (t = 0 ms)</div>
      </div>
      <div class="stat-card">
        <div class="stat-lbl">Sorter / Units</div>
        <div class="stat-val" id="stat-units" style="color:#38bdf8;">KS4 / 120 Units</div>
      </div>
    </div>

    <!-- HMM STATE LEGEND -->
    <div class="legend-bar">
      <div class="legend-pill" style="border-left: 3px solid #10b981;">
        <div class="legend-color" style="background: #10b981;"></div>
        <div class="legend-text">
          <strong>State 1: Insp Burst</strong>
          <span>High-rate sync drive (&gt;99% in Insp)</span>
        </div>
      </div>
      <div class="legend-pill" style="border-left: 3px solid #f97316;">
        <div class="legend-color" style="background: #f97316;"></div>
        <div class="legend-text">
          <strong>State 2: Peak / Off-Switch</strong>
          <span>Flow zero-crossing quench ($I_{{off}}$)</span>
        </div>
      </div>
      <div class="legend-pill" style="border-left: 3px solid #38bdf8;">
        <div class="legend-color" style="background: #38bdf8;"></div>
        <div class="legend-text">
          <strong>State 3: Exp Drive</strong>
          <span>Post-inspiratory expiratory phase</span>
        </div>
      </div>
      <div class="legend-pill" style="border-left: 3px solid #64748b;">
        <div class="legend-color" style="background: #64748b;"></div>
        <div class="legend-text">
          <strong>State 4: Late-E Baseline</strong>
          <span>Inter-breath baseline period</span>
        </div>
      </div>
      <div class="legend-pill" style="border-left: 3px solid #a855f7;">
        <div class="legend-color" style="background: #a855f7;"></div>
        <div class="legend-text">
          <strong>State 5: Quiescence / Gasp</strong>
          <span>Inter-breath pause / Anoxic gasp</span>
        </div>
      </div>
    </div>

    <!-- MAIN PLOT CARD 1: TRIAL-BY-TRIAL RASTER -->
    <div class="plot-card">
      <div class="plot-header">
        <div class="plot-title">
          <span>🧬 1. Interactive Micro-State Trial-by-Trial Raster</span>
          <span style="font-size:12px;color:#94a3b8;font-weight:400;" id="raster-subtitle">(187 Breaths, Raw Poisson 10ms Spikes)</span>
        </div>
        <div class="zoom-helpers">
          <button class="zoom-btn" onclick="zoomTime(-150, 50)">🔍 Zoom Off-Switch Boundary (-150 to +50ms)</button>
          <button class="zoom-btn" onclick="zoomTime(-200, 150)">🔍 Zoom Full Burst (-200 to +150ms)</button>
          <button class="zoom-btn" onclick="zoomTrials(1, 30)">Breaths 1–30</button>
          <button class="zoom-btn" onclick="zoomTrials(31, 60)">Breaths 31–60</button>
          <button class="zoom-btn" onclick="resetZoom()">↺ Reset Zoom</button>
        </div>
      </div>
      <div id="rasterPlot"></div>
    </div>

    <!-- MAIN PLOT CARD 2: POPULATION STATE OCCUPANCY PROBABILITY PETH -->
    <div class="plot-card">
      <div class="plot-header">
        <div class="plot-title">
          <span>📊 2. Population State Occupancy Probability PETH (%)</span>
          <span style="font-size:12px;color:#94a3b8;font-weight:400;">Synchronized across all breath trials</span>
        </div>
      </div>
      <div id="probPlot"></div>
    </div>

    <!-- MAIN PLOT CARD 3: AVERAGE AIRFLOW & DIAPHRAGM EMG WAVEFORMS -->
    <div class="plot-card">
      <div class="plot-header">
        <div class="plot-title">
          <span>🫁 3. Average Physiological Waveforms (Airflow &amp; Diaphragm EMG Envelope)</span>
          <span style="font-size:12px;color:#94a3b8;font-weight:400;">Physiological alignment validation</span>
        </div>
      </div>
      <div id="analogPlot"></div>
    </div>

  </div>

  <script>
    // RAW DATA PAYLOAD
    const DATA = {json_str};

    // ACTIVE STATE
    let currentSubwin = '{default_subwin}';
    let currentAlign = 'ioff'; // Default to Inspiration Offset as requested!
    let currentSorter = 'kilosort4';
    let currentSortOrder = 'chrono';

    const STATE_COLORS = ["#10b981", "#f97316", "#38bdf8", "#64748b", "#a855f7"];
    const STATE_NAMES = [
      "State 1: Insp Burst",
      "State 2: Peak Off-Switch",
      "State 3: Exp Drive",
      "State 4: Late-E Baseline",
      "State 5: Quiescence / Gasp"
    ];

    // Discrete colorscale for Plotly Heatmap (0 to 4 mapped to distinct bands)
    const customColorscale = [
      [0.0, '#10b981'], [0.19, '#10b981'],
      [0.2, '#f97316'], [0.39, '#f97316'],
      [0.4, '#38bdf8'], [0.59, '#38bdf8'],
      [0.6, '#64748b'], [0.79, '#64748b'],
      [0.8, '#a855f7'], [1.00, '#a855f7']
    ];

    function setSubwindow(sub) {{
      currentSubwin = sub;
      document.getElementById('btn-win1').classList.toggle('active', sub === 'win1_baseline');
      document.getElementById('btn-win2').classList.toggle('active', sub === 'win2_anox1');
      if (sub === 'win2_anox1') {{
        document.getElementById('btn-win2').classList.add('anox');
      }} else {{
        document.getElementById('btn-win2').classList.remove('anox');
      }}
      renderAll();
    }}

    function setAlignment(align) {{
      currentAlign = align;
      document.getElementById('btn-ioff').classList.toggle('active', align === 'ioff');
      document.getElementById('btn-ion').classList.toggle('active', align === 'ion');
      renderAll();
    }}

    function setSorter(sorter) {{
      currentSorter = sorter;
      document.getElementById('btn-ks4').classList.toggle('active', sorter === 'kilosort4');
      document.getElementById('btn-lupin').classList.toggle('active', sorter === 'lupin');
      renderAll();
    }}

    function setSortOrder(order) {{
      currentSortOrder = order;
      document.getElementById('btn-sort-chrono').classList.toggle('active', order === 'chrono');
      document.getElementById('btn-sort-ti').classList.toggle('active', order === 'ti');
      renderAll();
    }}

    function renderAll() {{
      const key = `${{currentSubwin}}_${{currentSorter}}`;
      const bundle = DATA[key];
      if (!bundle) return;

      const alignData = bundle[currentAlign];
      const timeAxis = alignData.axis_ms;
      let rawRaster = alignData.raster;
      let trials = alignData.trials;
      const nBreaths = bundle.n_breaths;

      // Update stat cards
      document.getElementById('stat-nbreaths').textContent = `${{nBreaths}} Breaths`;
      const meanTi = (trials.reduce((acc, t) => acc + t.ti_ms, 0) / nBreaths).toFixed(1);
      document.getElementById('stat-ti').textContent = `${{meanTi}} ms`;
      document.getElementById('stat-align').textContent = currentAlign === 'ioff' ? '⭐ I_off (t = 0 ms)' : 'I_on (t = 0 ms)';
      document.getElementById('stat-units').textContent = `${{currentSorter === 'kilosort4' ? 'KS4' : 'Lupin'}} / ${{bundle.n_units}} Units`;

      document.getElementById('raster-subtitle').textContent =
        `(${{currentSubwin}} | ${{nBreaths}} Breaths | ${{currentSorter === 'kilosort4' ? 'Kilosort 4' : 'Lupin'}} | Aligned to ${{currentAlign === 'ioff' ? 'Inspiration Offset (I_off)' : 'Inspiration Onset (I_on)'}})`;

      // Handle sort order
      let rowIndices = Array.from({{ length: nBreaths }}, (_, i) => i);
      if (currentSortOrder === 'ti') {{
        rowIndices.sort((a, b) => trials[a].ti_ms - trials[b].ti_ms);
      }}

      // Build sorted heatmap matrix and hover text
      const sortedZ = [];
      const hoverText = [];
      const markerCurveX = [];
      const markerCurveY = [];

      for (let r = 0; r < nBreaths; r++) {{
        const origIdx = rowIndices[r];
        const tr = trials[origIdx];
        sortedZ.push(rawRaster[origIdx]);

        const rowHover = [];
        for (let c = 0; c < timeAxis.length; c++) {{
          const stateVal = rawRaster[origIdx][c];
          const stName = STATE_NAMES[stateVal];
          const tMs = timeAxis[c];
          rowHover.push(
            `<b>Trial #${{tr.trial_num}}</b> (${{tr.regime}})<br>` +
            `Time: <b>${{tMs >= 0 ? '+' : ''}}${{tMs.toFixed(1)}} ms</b> rel to ${{currentAlign.toUpperCase()}}<br>` +
            `State: <b>${{stName}}</b><br>` +
            `Diaphragm T_I: <b>${{tr.ti_ms.toFixed(1)}} ms</b> | T_tot: <b>${{tr.tot_ms.toFixed(1)}} ms</b>`
          );
        }}
        hoverText.push(rowHover);

        // Marker position
        if (currentAlign === 'ioff') {{
          markerCurveX.push(tr.ion_relative_ms); // -T_I
        }} else {{
          markerCurveX.push(tr.ioff_relative_ms); // +T_I
        }}
        markerCurveY.push(r + 1);
      }}

      const yTrials = Array.from({{ length: nBreaths }}, (_, i) => i + 1);

      // ---------------------------------------------
      // 1. RASTER PLOT
      // ---------------------------------------------
      const traceHeatmap = {{
        x: timeAxis,
        y: yTrials,
        z: sortedZ,
        type: 'heatmap',
        colorscale: customColorscale,
        zmin: 0,
        zmax: 4,
        showscale: false,
        hoverinfo: 'text',
        text: hoverText
      }};

      const markerLabel = currentAlign === 'ioff' ? 'Diaphragm Onset (I_on = -T_I)' : 'Flow Zero-Crossing (I_off = +T_I)';
      const markerColor = currentAlign === 'ioff' ? '#34d399' : '#fbbf24';

      const traceMarker = {{
        x: markerCurveX,
        y: markerCurveY,
        mode: 'lines+markers',
        line: {{ color: markerColor, width: 2 }},
        marker: {{ size: 4, color: markerColor }},
        name: markerLabel,
        hoverinfo: 'skip'
      }};

      const rasterLayout = {{
        margin: {{ l: 65, r: 30, t: 30, bottom: 45 }},
        paper_bgcolor: '#0f1523',
        plot_bgcolor: '#090d16',
        xaxis: {{
          title: {{ text: `Time from ${{currentAlign === 'ioff' ? 'Inspiration Offset (I_off)' : 'Inspiration Onset (I_on)'}} (ms)`, font: {{ color: '#f8fafc', size: 12, weight: 700 }} }},
          tickcolor: '#334155',
          tickfont: {{ color: '#94a3b8', size: 11 }},
          gridcolor: '#1e293b',
          zeroline: true,
          zerolinecolor: '#ffffff',
          zerolinewidth: 2
        }},
        yaxis: {{
          title: {{ text: `Breath Trial Index (${{currentSortOrder === 'chrono' ? 'Chronological 1 to ' + nBreaths : 'Sorted by T_I Duration'}})`, font: {{ color: '#f8fafc', size: 12, weight: 700 }} }},
          tickcolor: '#334155',
          tickfont: {{ color: '#94a3b8', size: 11 }},
          gridcolor: '#1e293b',
          autorange: 'reversed'
        }},
        shapes: [
          {{
            type: 'line',
            x0: 0, x1: 0,
            y0: 0.5, y1: nBreaths + 0.5,
            line: {{ color: '#ffffff', width: 2, dash: 'dash' }}
          }}
        ],
        showlegend: true,
        legend: {{
          x: 0.01, y: 0.02,
          bgcolor: 'rgba(15, 23, 42, 0.9)',
          bordercolor: '#334155',
          borderwidth: 1,
          font: {{ color: '#f8fafc', size: 11 }}
        }}
      }};

      Plotly.react('rasterPlot', [traceHeatmap, traceMarker], rasterLayout, {{ responsive: true }});

      // ---------------------------------------------
      // 2. OCCUPANCY PROBABILITY PETH PLOT
      // ---------------------------------------------
      const probTraces = [];
      for (let k = 0; k < 5; k++) {{
        probTraces.push({{
          x: timeAxis,
          y: alignData.prob[k],
          name: STATE_NAMES[k],
          mode: 'lines',
          line: {{ color: STATE_COLORS[k], width: 2.2 }},
          hoverinfo: 'x+y+name'
        }});
      }}

      const probLayout = {{
        margin: {{ l: 65, r: 30, t: 15, bottom: 40 }},
        paper_bgcolor: '#0f1523',
        plot_bgcolor: '#090d16',
        xaxis: {{
          title: {{ text: `Time from ${{currentAlign === 'ioff' ? 'I_off' : 'I_on'}} (ms)`, font: {{ color: '#f8fafc', size: 11 }} }},
          tickcolor: '#334155',
          tickfont: {{ color: '#94a3b8', size: 10 }},
          gridcolor: '#1e293b',
          zeroline: true,
          zerolinecolor: '#ffffff',
          zerolinewidth: 1.5
        }},
        yaxis: {{
          title: {{ text: 'Occupancy (%)', font: {{ color: '#f8fafc', size: 11, weight: 700 }} }},
          tickcolor: '#334155',
          tickfont: {{ color: '#94a3b8', size: 10 }},
          gridcolor: '#1e293b',
          range: [0, 102]
        }},
        shapes: [
          {{
            type: 'line',
            x0: 0, x1: 0,
            y0: 0, y1: 100,
            line: {{ color: '#ffffff', width: 1.5, dash: 'dash' }}
          }}
        ],
        showlegend: true,
        legend: {{
          orientation: 'h',
          x: 0.05, y: 1.18,
          font: {{ color: '#cbd5e1', size: 10.5 }}
        }}
      }};

      Plotly.react('probPlot', probTraces, probLayout, {{ responsive: true }});

      // ---------------------------------------------
      // 3. ANALOG WAVEFORMS PLOT (AIRFLOW & EMG)
      // ---------------------------------------------
      const traceFlow = {{
        x: timeAxis,
        y: alignData.flow_mean,
        name: 'Mean Airflow (V)',
        mode: 'lines',
        line: {{ color: '#10b981', width: 2.0 }}
      }};

      const traceEMG = {{
        x: timeAxis,
        y: alignData.emg_mean,
        name: 'Mean Diaphragm EMG Env (V)',
        yaxis: 'y2',
        mode: 'lines',
        line: {{ color: '#ec4899', width: 2.0 }}
      }};

      const analogLayout = {{
        margin: {{ l: 65, r: 65, t: 15, bottom: 40 }},
        paper_bgcolor: '#0f1523',
        plot_bgcolor: '#090d16',
        xaxis: {{
          title: {{ text: `Time from ${{currentAlign === 'ioff' ? 'I_off' : 'I_on'}} (ms)`, font: {{ color: '#f8fafc', size: 11 }} }},
          tickcolor: '#334155',
          tickfont: {{ color: '#94a3b8', size: 10 }},
          gridcolor: '#1e293b',
          zeroline: true,
          zerolinecolor: '#ffffff',
          zerolinewidth: 1.5
        }},
        yaxis: {{
          title: {{ text: 'Airflow (V)', font: {{ color: '#10b981', size: 11, weight: 700 }} }},
          tickfont: {{ color: '#10b981', size: 10 }},
          gridcolor: '#1e293b'
        }},
        yaxis2: {{
          title: {{ text: 'EMG Env (V)', font: {{ color: '#ec4899', size: 11, weight: 700 }} }},
          tickfont: {{ color: '#ec4899', size: 10 }},
          overlaying: 'y',
          side: 'right'
        }},
        shapes: [
          {{
            type: 'line',
            x0: 0, x1: 0,
            y0: -0.5, y1: 0.5,
            line: {{ color: '#ffffff', width: 1.5, dash: 'dash' }}
          }}
        ],
        showlegend: true,
        legend: {{
          orientation: 'h',
          x: 0.3, y: 1.2,
          font: {{ color: '#cbd5e1', size: 10.5 }}
        }}
      }};

      Plotly.react('analogPlot', [traceFlow, traceEMG], analogLayout, {{ responsive: true }});
    }}

    // Zoom helper functions
    function zoomTime(minT, maxT) {{
      Plotly.relayout('rasterPlot', {{ 'xaxis.range': [minT, maxT] }});
      Plotly.relayout('probPlot', {{ 'xaxis.range': [minT, maxT] }});
      Plotly.relayout('analogPlot', {{ 'xaxis.range': [minT, maxT] }});
    }}

    function zoomTrials(minTrial, maxTrial) {{
      Plotly.relayout('rasterPlot', {{ 'yaxis.range': [maxTrial + 0.5, minTrial - 0.5] }});
    }}

    function resetZoom() {{
      Plotly.relayout('rasterPlot', {{ 'xaxis.autorange': true, 'yaxis.autorange': 'reversed' }});
      Plotly.relayout('probPlot', {{ 'xaxis.autorange': true, 'yaxis.autorange': true }});
      Plotly.relayout('analogPlot', {{ 'xaxis.autorange': true, 'yaxis.autorange': true }});
    }}

    // Initialize on page load
    window.addEventListener('DOMContentLoaded', () => {{
      renderAll();
    }});
  </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Interactive Dashboard successfully written to {output_path} ({len(html_content)} bytes).")


def main():
    t0 = time.time()
    logger.info("=== Starting Interactive Micro-State Rasters Generation ===")

    # 1. win1_baseline (60s, 1740s - 1800s)
    logger.info("--- Processing win1_baseline ---")
    win1_breath = extract_subwindow_breaths(1740.0, 60.0)
    win1_ks4 = fit_and_extract_rasters("win1_baseline", "kilosort4", "Kilosort 4 (GPU)", win1_breath)
    win1_lup = fit_and_extract_rasters("win1_baseline", "lupin", "Lupin (SVD-OMP)", win1_breath)

    # 2. win2_anox1 (36s, 1800s - 1836s)
    logger.info("--- Processing win2_anox1 ---")
    win2_breath = extract_subwindow_breaths(1800.0, 36.0)
    win2_ks4 = fit_and_extract_rasters("win2_anox1", "kilosort4", "Kilosort 4 (GPU)", win2_breath)
    win2_lup = fit_and_extract_rasters("win2_anox1", "lupin", "Lupin (SVD-OMP)", win2_breath)

    all_data = {
        "win1_baseline_kilosort4": win1_ks4,
        "win1_baseline_lupin": win1_lup,
        "win2_anox1_kilosort4": win2_ks4,
        "win2_anox1_lupin": win2_lup
    }

    # 3. Write Interactive HTMLs
    out_win1 = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / "win1_baseline" / "population_dynamics" / "win1_baseline_interactive_microstate_raster.html"
    build_interactive_dashboard_html(all_data, out_win1, default_subwin="win1_baseline")

    out_win2 = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / "win2_anox1" / "population_dynamics" / "win2_anox1_interactive_microstate_raster.html"
    build_interactive_dashboard_html(all_data, out_win2, default_subwin="win2_anox1")

    logger.info(f"All processing complete in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
