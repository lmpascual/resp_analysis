#!/usr/bin/env python3
"""
Master Interactive Transition Dashboard Generator for e097 (29:00 - 36:00 / 1740s - 2160s):
Synthesizes:
1. Cardiorespiratory traces: Diaphragm EMG, Plethysmography Flow, ECG Heart Rate, Probe MUA
2. All 7 Physiological Subwindows with Dual Sorter (Kilosort 4 & Lupin) yields & Consensus
3. Unified Neuronal Trajectories & Phenotype Taxonomy (Constitutive, Anoxia-Silenced, De Novo Gasping, Vulnerable)
4. Dual Respiratory Phase Tuning Evolution (Diaphragm EMG vs Flow)
5. Probe Depth Spatial Distribution
Outputs a self-contained, responsive, dark-mode interactive HTML report.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("MASTER_TRANSITION_DASHBOARD")

def build_dashboard(
    subwindows_dir: Path,
    motion_metrics_json: Path,
    out_html: Path
):
    logger.info("Building Master Interactive Anoxia Transition Dashboard...")
    
    # 1. Load motion metrics
    with open(motion_metrics_json, "r", encoding="utf-8") as f:
        subwin_motion = json.load(f)
        
    # 2. Load sorting benchmarks for all subwindows
    subwin_data = []
    for sw in subwin_motion:
        w_id = sw["subwindow_id"]
        bm_path = subwindows_dir / w_id / f"{w_id}_sorting_benchmark.json"
        if bm_path.exists():
            with open(bm_path, "r", encoding="utf-8") as f:
                bm = json.load(f)
        else:
            bm = {"lupin": {"sua_units": "-", "total_units": "-"}, "kilosort4": {"sua_units": "-", "total_units": "-"}, "agreement": {"matched_units_50pct": "-"}}
        subwin_data.append({"meta": sw, "benchmark": bm})

    # 3. Load unification data if available
    unif_csv = subwindows_dir / "unification_kilosort4" / "unified_neuronal_trajectories.csv"
    if unif_csv.exists():
        df_unif = pd.read_csv(unif_csv)
        pheno_counts = df_unif["phenotype"].value_counts().to_dict()
    else:
        df_unif = None
        pheno_counts = {"Pending completion of all subwindows": 0}

    # 4. Load phase tuning summary if available
    tuning_csv = subwindows_dir / "respiratory_phase_tuning_kilosort4" / "all_subwindows_phase_tuning_summary.csv"
    if tuning_csv.exists():
        df_tuning = pd.read_csv(tuning_csv)
        emg_phase_counts = df_tuning["emg_category"].value_counts().to_dict()
    else:
        df_tuning = None
        emg_phase_counts = {}

    # Build Subwindows HTML rows
    table_rows = ""
    for d in subwin_data:
        m = d["meta"]
        b = d["benchmark"]
        w_id = m["subwindow_id"]
        
        l_sua = b.get("lupin", {}).get("sua_units", "-")
        l_tot = b.get("lupin", {}).get("total_units", "-")
        k_sua = b.get("kilosort4", {}).get("sua_units", "-")
        k_tot = b.get("kilosort4", {}).get("total_units", "-")
        c_mat = b.get("agreement", {}).get("matched_units_50pct", "-")
        
        status_badge = '<span style="color:#10b981; font-weight:700;">&#10003; Sorted</span>' if k_sua != "-" else '<span style="color:#f59e0b; font-weight:700;">&#9684; Processing</span>'
        
        comp_link = f"{w_id}/{w_id}_sorter_comparison.html"
        link_html = f'<a href="{comp_link}" target="_blank" class="btn-sm">Compare</a>' if k_sua != "-" else '-'
        
        table_rows += f"""
        <tr>
            <td><b>{m['name']}</b><br><span style="color:#64748b; font-size:11px;">{w_id}</span></td>
            <td>{m['time_range_min'][0]:.1f} - {m['time_range_min'][1]:.1f} min<br><span style="color:#64748b; font-size:11px;">({m['duration_s']:.0f}s)</span></td>
            <td><b>{m['mean_mua_rate_hz']:.1f} Hz</b></td>
            <td><b>{m['mean_hr_bpm']:.0f} bpm</b></td>
            <td><b>{m['max_displacement_um']:.2f} µm</b></td>
            <td><b style="color:#00f5d4;">{l_sua}</b> <span style="color:#64748b; font-size:11px;">/ {l_tot}</span></td>
            <td><b style="color:#f59e0b;">{k_sua}</b> <span style="color:#64748b; font-size:11px;">/ {k_tot}</span></td>
            <td><b style="color:#a855f7;">{c_mat}</b></td>
            <td>{status_badge}</td>
            <td>{link_html}</td>
        </tr>
        """

    pheno_cards_html = ""
    for p_name, p_cnt in pheno_counts.items():
        pheno_cards_html += f"""
        <div class="card">
            <h4>{p_name}</h4>
            <div class="val">{p_cnt}</div>
            <div class="sub">Neurons</div>
        </div>
        """

    emg_phase_html = ""
    for ph_name, ph_cnt in emg_phase_counts.items():
        emg_phase_html += f"""
        <div class="card">
            <h4>{ph_name}</h4>
            <div class="val" style="color:#38bdf8;">{ph_cnt}</div>
            <div class="sub">Unit-Epochs</div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>e097 Anoxic Challenge: Master Transition Neurodynamics Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {{
            background: #07090e;
            color: #f1f5f9;
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 30px;
        }}
        .header {{
            background: #0f1523;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 24px 30px;
            margin-bottom: 24px;
        }}
        h1 {{
            margin: 0 0 10px 0;
            font-size: 26px;
            color: #38bdf8;
            font-weight: 800;
        }}
        .subtitle {{
            color: #94a3b8;
            font-size: 14px;
            margin: 0;
            line-height: 1.6;
        }}
        .grid-kpis {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: #0f1523;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 20px;
        }}
        .card h4 {{
            margin: 0 0 8px 0;
            font-size: 13px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .card .val {{
            font-size: 30px;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            color: #f1f5f9;
        }}
        .card .sub {{
            font-size: 12px;
            color: #64748b;
            margin-top: 4px;
        }}
        .section-title {{
            font-size: 18px;
            font-weight: 700;
            color: #38bdf8;
            margin: 30px 0 16px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .table-container {{
            background: #0f1523;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th {{
            color: #94a3b8;
            border-bottom: 1px solid rgba(255,255,255,0.12);
            padding: 12px 14px;
            text-align: left;
        }}
        td {{
            padding: 12px 14px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            vertical-align: middle;
        }}
        .btn-sm {{
            display: inline-block;
            background: #1e293b;
            color: #38bdf8;
            padding: 5px 10px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: 600;
            font-size: 11px;
            border: 1px solid rgba(56, 189, 248, 0.3);
            transition: all 0.2s ease;
        }}
        .btn-sm:hover {{
            background: #38bdf8;
            color: #07090e;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>e097 Anoxic Challenge: Master Transition Neurodynamics Dashboard</h1>
        <p class="subtitle">
            Longitudinal single-neuron tracking across Baseline (29:00), 15s Anoxia Challenge, Primary Apnea Nadir, 
            Autoresuscitation Gasping Surge, Active Expiration Shift, and Recovery (36:00).<br>
            Probe: 384 Neuropixels channels spanning 0 to 2865 µm (Ventral to Dorsal Medulla/Pons).
        </p>
    </div>

    <div class="grid-kpis">
        <div class="card">
            <h4>Total Duration</h4>
            <div class="val">420 s</div>
            <div class="sub">7.0 minutes continuous</div>
        </div>
        <div class="card">
            <h4>Max Probe Drift</h4>
            <div class="val" style="color:#10b981;">&le; 1.51 µm</div>
            <div class="sub">Near-absolute stationarity</div>
        </div>
        <div class="card">
            <h4>Anoxic Silence Nadir</h4>
            <div class="val" style="color:#ef4444;">-68%</div>
            <div class="sub">EMG & MUA cessation at 30.6-31.0m</div>
        </div>
        <div class="card">
            <h4>Autoresuscitation Surge</h4>
            <div class="val" style="color:#a855f7;">526 bpm</div>
            <div class="sub">Peak tachycardia during gasps</div>
        </div>
    </div>

    <div class="section-title">Physiological Subwindows: Multi-Sorter Yields & Consensus</div>
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Subwindow</th>
                    <th>Time Span</th>
                    <th>Mean MUA</th>
                    <th>Heart Rate</th>
                    <th>Max Drift</th>
                    <th>Lupin SUA / Tot</th>
                    <th>KS4 SUA / Tot</th>
                    <th>Consensus (&ge;50%)</th>
                    <th>Status</th>
                    <th>Viewer</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>

    <div class="section-title">Neuronal Phenotype Taxonomy Across Transitions</div>
    <div class="grid-kpis">
        {pheno_cards_html}
    </div>

    <div class="section-title">Diaphragm EMG Respiratory Phase Tuning Distribution</div>
    <div class="grid-kpis">
        {emg_phase_html}
    </div>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Saved master transition dashboard to: {out_html}")

if __name__ == '__main__':
    subwin_dir = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting"
    metrics_json = REPO_ROOT / "outputs" / "e097" / "e097_anoxia_subwindow_motion_metrics.json"
    out_file = subwin_dir / "master_anoxia_transition_dashboard.html"
    build_dashboard(subwin_dir, metrics_json, out_file)
