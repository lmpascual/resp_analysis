#!/usr/bin/env python3
"""
Generate Dedicated 2-Channel Spike Trace Validation Suite for E097 Intact Window (420.0s).
Window: 1740.0s - 2160.0s (420.0s Intact Eupnea, Anoxia Challenge, Apnea, & Recovery)
Dual Sorters: Kilosort 4 (1,102 clusters) and Lupin (273 clusters)

Visualizes:
1. The top 2 peak channels (highest PTP amplitude somatic contacts) for each unit.
2. Raw 300 - 6000 Hz zero-phase bandpass-filtered AP voltage signal at native 30 kHz (1.5s window).
3. Exact spike times marked on top of BOTH voltage signals (trough markers, refractory ISI violation flags, and vertical spike guide lines).
4. Interactive gain scaling controls (1x, 1.5x, 2x, 3x, Bigger, Smaller, Reset).
5. Fast keyboard (<-, ->, J, K) and filter cycling across both sorters.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import scipy.signal as signal
import spikeinterface.extractors as se
import plotly.graph_objects as go

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe UTF-8 encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from src.pipeline.diagnostics.cluster_validator import find_unit_channel_footprint

# Experiment & Path Definitions
EXPERIMENT_ID = "e097_g2"
WINDOW_ID = "intact_window"
WINDOW_NAME = "Intact Window (420.0s Eupnea & Anoxia)"
START_T_S = 1740.0
DURATION_S = 420.0
FS_AP = 29999.821974390747
N_CHANS_AP = 385

ROOT_DIR = Path(r"c:\Users\lpascu\resp_analysis")
INTACT_DIR = ROOT_DIR / "outputs" / "e097" / WINDOW_ID
KS4_DIR = INTACT_DIR / "kilosort4"
LUPIN_DIR = INTACT_DIR / "lupin"

RAW_AP_BIN = Path(r"C:\SCRATCH_neuropixels\catgt_e097_intact_420s\catgt_e097_g2\e097_g2_tcat.imec0.ap.bin")
CATGT_FOLDER = RAW_AP_BIN.parent

TOP_VAL_DIR = INTACT_DIR / "two_channel_validation"
KS4_VAL_DIR = KS4_DIR / "two_channel_validation"
LUPIN_VAL_DIR = LUPIN_DIR / "two_channel_validation"


def load_probe_geometry() -> Dict[int, float]:
    """Loads physical Neuropixels channel depths and creates depth lookup."""
    try:
        rec = se.read_spikeglx(str(CATGT_FOLDER), stream_id="imec0.ap")
        locs = rec.get_channel_locations()
        channel_depths = {ch: float(locs[ch, 1]) for ch in range(len(locs))}
        if 384 not in channel_depths:
            channel_depths[384] = 0.0
        print(f"Loaded Neuropixels probe geometry: {len(locs)} channels (Span: {locs[:,1].min():.0f}um to {locs[:,1].max():.0f}um)")
    except Exception as e:
        print(f"Fallback probe geometry ({e}): using standard spacing.")
        channel_depths = {ch: float((ch // 2) * 15.0) for ch in range(385)}

    return channel_depths


def load_consensus_map() -> Tuple[Dict[int, Tuple[int, float]], Dict[int, Tuple[int, float]]]:
    """Loads pre-computed cross-sorter consensus match table."""
    matches_file = INTACT_DIR / "cross_sorter_matches.json"
    if matches_file.exists():
        with open(matches_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        lup_to_ks4 = {int(k): (int(v[0]), float(v[1])) for k, v in data.get("lup_to_ks4", {}).items()}
        ks4_to_lup = {int(k): (int(v[0]), float(v[1])) for k, v in data.get("ks4_to_lup", {}).items()}
        print(f"Loaded consensus matches: {len(lup_to_ks4)} Lupin and {len(ks4_to_lup)} KS4 pairings.")
        return lup_to_ks4, ks4_to_lup
    return {}, {}


def build_two_channel_unit_html(
    uid: int,
    sorter_name: str,
    top_2: List[int],
    channel_depths: Dict[int, float],
    channel_ptps: Dict[int, float],
    filt_chunk: np.ndarray,
    t_full: np.ndarray,
    t_start_s: float,
    t_end_s: float,
    spks_in_win: np.ndarray,
    troughs_t_ch1: np.ndarray,
    troughs_v_ch1: np.ndarray,
    troughs_t_ch2: np.ndarray,
    troughs_v_ch2: np.ndarray,
    viol_flags: np.ndarray,
    fr: float,
    total_spikes: int,
    qc_label: str,
    qc_badge_html: str,
    n_viol_session: int,
    q_isi: float,
    dest_path: Path
):
    """Renders a single-unit 2-channel interactive trace inspection HTML report."""
    ch1, ch2 = top_2[0], top_2[1]
    d1 = channel_depths.get(ch1, 0.0)
    d2 = channel_depths.get(ch2, 0.0)
    ptp1 = channel_ptps.get(ch1, float(np.ptp(filt_chunk[:, 0])))
    ptp2 = channel_ptps.get(ch2, float(np.ptp(filt_chunk[:, 1])))

    max_ptp = max(float(np.ptp(filt_chunk[:, 0])), float(np.ptp(filt_chunk[:, 1])), 30.0)
    lane_spacing = float(max_ptp / 0.50)  # Generous headroom for 2 channels

    y_base1 = 0.0
    y_base2 = lane_spacing

    fig = go.Figure()

    t_full_list = t_full.tolist()

    # Trace 0: Channel 1 filtered AP trace (Primary Peak)
    fig.add_trace(go.Scatter(
        x=t_full_list,
        y=(filt_chunk[:, 0] + y_base1).tolist(),
        mode="lines",
        line=dict(color="#2563eb", width=2.0),
        name=f"Ch {ch1} (Primary Peak - {d1:.0f}um)",
        hoverinfo="x+y"
    ))

    # Trace 1: Channel 2 filtered AP trace (Secondary Peak)
    fig.add_trace(go.Scatter(
        x=t_full_list,
        y=(filt_chunk[:, 1] + y_base2).tolist(),
        mode="lines",
        line=dict(color="#0d9488", width=1.8),
        name=f"Ch {ch2} (Secondary Peak - {d2:.0f}um)",
        hoverinfo="x+y"
    ))

    # Trace 2: Detected spikes on Channel 1 (troughs)
    norm_mask = ~viol_flags
    has_spks = len(spks_in_win) > 0
    t1_norm = troughs_t_ch1[norm_mask].tolist() if has_spks else []
    v1_norm = (troughs_v_ch1[norm_mask] + y_base1).tolist() if has_spks else []

    fig.add_trace(go.Scatter(
        x=t1_norm,
        y=v1_norm,
        mode="markers",
        marker=dict(symbol="circle", size=8, color="#ef4444", line=dict(color="#ffffff", width=0.8)),
        name=f"Spikes on Ch {ch1}",
        hoverinfo="x+y"
    ))

    # Trace 3: Detected spikes on Channel 2 (troughs)
    t2_norm = troughs_t_ch2[norm_mask].tolist() if has_spks else []
    v2_norm = (troughs_v_ch2[norm_mask] + y_base2).tolist() if has_spks else []

    fig.add_trace(go.Scatter(
        x=t2_norm,
        y=v2_norm,
        mode="markers",
        marker=dict(symbol="circle", size=8, color="#f43f5e", line=dict(color="#ffffff", width=0.8)),
        name=f"Spikes on Ch {ch2}",
        hoverinfo="x+y"
    ))

    # Trace 4 & 5: ISI Refractory Violations (< 1.0 ms)
    has_viol = bool(np.any(viol_flags))
    t1_viol = troughs_t_ch1[viol_flags].tolist() if has_viol else []
    v1_viol = (troughs_v_ch1[viol_flags] + y_base1).tolist() if has_viol else []
    t2_viol = troughs_t_ch2[viol_flags].tolist() if has_viol else []
    v2_viol = (troughs_v_ch2[viol_flags] + y_base2).tolist() if has_viol else []

    fig.add_trace(go.Scatter(
        x=t1_viol,
        y=v1_viol,
        mode="markers",
        marker=dict(symbol="triangle-down", size=13, color="#f59e0b", line=dict(color="#b45309", width=1.5)),
        name=f"ISI Violations (<1.0ms) [Ch {ch1}]",
        showlegend=bool(has_viol),
        hoverinfo="x+y"
    ))

    fig.add_trace(go.Scatter(
        x=t2_viol,
        y=v2_viol,
        mode="markers",
        marker=dict(symbol="triangle-down", size=13, color="#f59e0b", line=dict(color="#b45309", width=1.5)),
        name=f"ISI Violations (<1.0ms) [Ch {ch2}]",
        showlegend=False,
        hoverinfo="x+y"
    ))

    # Vertical dotted spike time guide lines connecting both channels
    shapes = []
    for st_val, is_viol in zip(spks_in_win, viol_flags):
        col = "rgba(245, 158, 11, 0.50)" if is_viol else "rgba(239, 68, 68, 0.40)"
        shapes.append(dict(
            type="line",
            x0=st_val, x1=st_val,
            y0=y_base1 - lane_spacing * 0.40,
            y1=y_base2 + lane_spacing * 0.40,
            line=dict(color=col, width=1.2, dash="dot")
        ))

    fig.update_layout(
        shapes=shapes,
        height=660,
        margin=dict(l=110, r=40, t=50, b=50),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            font=dict(size=12)
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=[y_base1, y_base2],
            ticktext=[
                f"<b>Ch {ch1}</b> ({d1:.0f}um)<br>[PRIMARY PEAK]",
                f"<b>Ch {ch2}</b> ({d2:.0f}um)<br>[SECONDARY PEAK]"
            ],
            title_text="Neuropixels Channels",
            range=[-lane_spacing * 0.50, lane_spacing * 1.50],
            zeroline=False,
            showgrid=True,
            gridcolor="#f1f5f9"
        ),
        xaxis=dict(
            title="Recording Time (seconds)",
            range=[t_start_s, t_end_s],
            showgrid=True,
            gridcolor="#f1f5f9"
        ),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif")
    )

    fig_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unit {uid:04d} (Ch {ch1} & Ch {ch2}) — 2-Channel Trace Inspector | {sorter_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f8fafc;
            color: #0f172a;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1440px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
            padding: 22px 28px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 14px;
            margin-bottom: 16px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 1.5rem;
            color: #1e293b;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .header-meta {{
            font-size: 0.88rem;
            color: #64748b;
            margin-top: 4px;
        }}
        .metrics-card {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 12px;
            background: #f1f5f9;
            padding: 14px 18px;
            border-radius: 8px;
            margin-bottom: 16px;
        }}
        .metric-item {{
            display: flex;
            flex-direction: column;
        }}
        .metric-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 700;
            color: #64748b;
            letter-spacing: 0.05em;
        }}
        .metric-value {{
            font-size: 1.25rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 2px;
        }}
        .gain-toolbar {{
            display: flex;
            align-items: center;
            gap: 8px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 10px 16px;
            margin-bottom: 14px;
            flex-wrap: wrap;
        }}
        .gain-label {{
            font-size: 0.85rem;
            font-weight: 700;
            color: #475569;
            margin-right: 6px;
        }}
        .gain-btn {{
            background: #ffffff;
            color: #334155;
            border: 1px solid #cbd5e1;
            padding: 5px 12px;
            border-radius: 5px;
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .gain-btn:hover {{
            background: #f1f5f9;
            border-color: #94a3b8;
        }}
        .gain-btn.active {{
            background: #2563eb;
            color: #ffffff;
            border-color: #1d4ed8;
        }}
        .trace-card {{
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
            background: #ffffff;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1><span>⚡</span> Unit {uid:04d} — 2-Channel Spike Trace Inspector</h1>
                <div class="header-meta">
                    {sorter_name} | Primary Peak: <b>Ch {ch1}</b> ({d1:.0f}&mu;m, PTP {ptp1:.1f}&mu;V) &bull; Secondary Peak: <b>Ch {ch2}</b> ({d2:.0f}&mu;m, PTP {ptp2:.1f}&mu;V)
                </div>
            </div>
            <div>
                {qc_badge_html}
            </div>
        </div>

        <div class="metrics-card">
            <div class="metric-item">
                <span class="metric-label">Firing Rate</span>
                <span class="metric-value">{fr:.2f} Hz</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Total Spikes</span>
                <span class="metric-value">{total_spikes:,}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Spikes in Snippet</span>
                <span class="metric-value">{len(spks_in_win)} spikes ({t_start_s:.2f}s - {t_end_s:.2f}s)</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Refractory Violations</span>
                <span class="metric-value" style="color: {'#10b981' if n_viol_session == 0 else '#d97706'};">{n_viol_session} viol (Q = {q_isi:.3f})</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Primary PTP Amplitude</span>
                <span class="metric-value" style="color: #2563eb;">{ptp1:.1f} &mu;V (Ch {ch1})</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Secondary PTP Amplitude</span>
                <span class="metric-value" style="color: #0d9488;">{ptp2:.1f} &mu;V (Ch {ch2})</span>
            </div>
        </div>

        <div class="gain-toolbar">
            <span class="gain-label">🔍 Waveform Vertical Scale:</span>
            <button class="gain-btn active" id="btn-gain-10" onclick="setTraceGain(1.0)">1x (Default)</button>
            <button class="gain-btn" id="btn-gain-15" onclick="setTraceGain(1.5)">1.5x</button>
            <button class="gain-btn" id="btn-gain-20" onclick="setTraceGain(2.0)">2x</button>
            <button class="gain-btn" id="btn-gain-30" onclick="setTraceGain(3.0)">3x</button>
            <button class="gain-btn" onclick="adjustTraceGain(1.25)">➕ Bigger</button>
            <button class="gain-btn" onclick="adjustTraceGain(0.80)">➖ Smaller</button>
            <button class="gain-btn" onclick="setTraceGain(1.0)">↺ Reset</button>
            <span style="margin-left:auto;font-size:0.80rem;color:#64748b;">
                Red Circles: Detected Spikes &bull; Amber Triangles: Violations (&lt;1ms) &bull; Dotted Lines: Spike Coincidence
            </span>
        </div>

        <div class="trace-card" id="plotCard">
            {fig_html}
        </div>
    </div>

    <script>
    var currentGain = 1.0;
    var base1 = {y_base1};
    var base2 = {y_base2};
    var hasSpikes = {str(has_spks).lower()};
    var hasViol = {str(has_viol).lower()};

    function getPlotDiv() {{
        var card = document.getElementById('plotCard');
        if (!card) return null;
        var gd = card.querySelector('.plotly-graph-div');
        return gd;
    }}

    function setTraceGain(newGain) {{
        var gd = getPlotDiv();
        if (!gd || !gd.data) return;
        var ratio = newGain / currentGain;

        var allNewY = [];
        var allIndices = [];

        // Trace 0: Channel 1
        if (gd.data[0]) {{
            var y0 = gd.data[0].y;
            var newY0 = new Array(y0.length);
            for (var j = 0; j < y0.length; j++) newY0[j] = base1 + (y0[j] - base1) * ratio;
            allNewY.push(newY0);
            allIndices.push(0);
        }}

        // Trace 1: Channel 2
        if (gd.data[1]) {{
            var y1 = gd.data[1].y;
            var newY1 = new Array(y1.length);
            for (var j = 0; j < y1.length; j++) newY1[j] = base2 + (y1[j] - base2) * ratio;
            allNewY.push(newY1);
            allIndices.push(1);
        }}

        // Trace 2: Normal spikes Ch 1
        if (hasSpikes && gd.data[2] && gd.data[2].y.length > 0) {{
            var y2 = gd.data[2].y;
            var newY2 = new Array(y2.length);
            for (var k = 0; k < y2.length; k++) newY2[k] = base1 + (y2[k] - base1) * ratio;
            allNewY.push(newY2);
            allIndices.push(2);
        }}

        // Trace 3: Normal spikes Ch 2
        if (hasSpikes && gd.data[3] && gd.data[3].y.length > 0) {{
            var y3 = gd.data[3].y;
            var newY3 = new Array(y3.length);
            for (var k = 0; k < y3.length; k++) newY3[k] = base2 + (y3[k] - base2) * ratio;
            allNewY.push(newY3);
            allIndices.push(3);
        }}

        // Trace 4: Violations Ch 1
        if (hasViol && gd.data[4] && gd.data[4].y.length > 0) {{
            var y4 = gd.data[4].y;
            var newY4 = new Array(y4.length);
            for (var m = 0; m < y4.length; m++) newY4[m] = base1 + (y4[m] - base1) * ratio;
            allNewY.push(newY4);
            allIndices.push(4);
        }}

        // Trace 5: Violations Ch 2
        if (hasViol && gd.data[5] && gd.data[5].y.length > 0) {{
            var y5 = gd.data[5].y;
            var newY5 = new Array(y5.length);
            for (var m = 0; m < y5.length; m++) newY5[m] = base2 + (y5[m] - base2) * ratio;
            allNewY.push(newY5);
            allIndices.push(5);
        }}

        Plotly.restyle(gd, {{'y': allNewY}}, allIndices);

        currentGain = newGain;
        document.querySelectorAll('.gain-btn').forEach(function(b) {{ b.classList.remove('active'); }});
        var btnId = 'btn-gain-' + Math.round(newGain * 10);
        var activeBtn = document.getElementById(btnId);
        if (activeBtn) activeBtn.classList.add('active');
    }}

    function adjustTraceGain(factor) {{
        var target = Math.max(0.5, Math.min(10.0, currentGain * factor));
        setTraceGain(Math.round(target * 10) / 10);
    }}

    // Propagate keyboard cycling events to parent browser shell
    window.addEventListener('keydown', function(e) {{
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.key === 'ArrowRight' || e.key === 'k' || e.key === 'K') {{
            window.parent.postMessage({{action: 'nextUnit'}}, '*');
        }} else if (e.key === 'ArrowLeft' || e.key === 'j' || e.key === 'J') {{
            window.parent.postMessage({{action: 'prevUnit'}}, '*');
        }}
    }});
    </script>
</body>
</html>
"""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def build_two_channel_browser_html(
    units_manifest: List[Dict],
    dest_path: Path,
    sorter_name: str,
    other_sorter_name: str,
    other_suite_relative_url: str,
    full_5panel_url: str
):
    """Generates the multi-cluster browser shell for the 2-channel suite."""
    manifest_json = json.dumps(units_manifest)
    first_file = units_manifest[0]["file_name"] if len(units_manifest) > 0 else ""

    n_all = len(units_manifest)
    n_consensus = sum(1 for u in units_manifest if u.get("is_consensus", False))
    n_strict = sum(1 for u in units_manifest if u.get("passed_allen_strict", False))
    n_resp = sum(1 for u in units_manifest if u.get("passed_respiratory_continuous", False))
    n_mua = sum(1 for u in units_manifest if u.get("is_mua", False))
    n_fail = sum(1 for u in units_manifest if not u.get("passed_allen_strict", False) and not u.get("passed_respiratory_continuous", False))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E097 Intact Window (420s) {sorter_name} 2-Channel Trace Validation Browser</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }}
        .navbar {{
            background: #1e293b;
            border-bottom: 1px solid #334155;
            padding: 10px 20px;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            z-index: 100;
        }}
        .nav-left {{
            display: flex;
            align-items: center;
            gap: 14px;
            flex-wrap: wrap;
        }}
        .nav-title {{
            font-size: 1.15rem;
            font-weight: 700;
            color: #38bdf8;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .selector-box {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        select {{
            background: #0f172a;
            color: #f8fafc;
            border: 1px solid #475569;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.88rem;
            font-weight: 600;
            cursor: pointer;
            outline: none;
            max-width: 540px;
        }}
        select:focus {{
            border-color: #38bdf8;
        }}
        .btn {{
            background: #334155;
            color: #f8fafc;
            border: 1px solid #475569;
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 0.80rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .btn:hover {{
            background: #475569;
        }}
        .btn-active {{
            background: #0284c7 !important;
            border-color: #38bdf8 !important;
            color: #ffffff !important;
        }}
        .btn-consensus {{
            border-color: #a855f7 !important;
            color: #c084fc !important;
        }}
        .btn-consensus.btn-active {{
            background: #7e22ce !important;
            color: #ffffff !important;
        }}
        .nav-links {{
            display: flex;
            gap: 10px;
            align-items: center;
            margin-left: auto;
        }}
        .link-badge {{
            background: #0369a1;
            color: #ffffff;
            text-decoration: none;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 5px 12px;
            border-radius: 6px;
            transition: background 0.15s;
            border: 1px solid #0284c7;
            white-space: nowrap;
        }}
        .link-badge:hover {{
            background: #0284c7;
        }}
        .link-switch {{
            background: #4c1d95;
            border-color: #7c3aed;
        }}
        .link-switch:hover {{
            background: #6d28d9;
        }}
        .link-fullsuite {{
            background: #0f766e;
            border-color: #14b8a6;
        }}
        .link-fullsuite:hover {{
            background: #0d9488;
        }}
        .iframe-container {{
            flex: 1;
            width: 100%;
            height: 100%;
            border: none;
            background: #f8fafc;
        }}
        iframe {{
            width: 100%;
            height: 100%;
            border: none;
        }}
        .shortcuts-hint {{
            font-size: 0.78rem;
            color: #94a3b8;
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        kbd {{
            background: #0f172a;
            border: 1px solid #475569;
            border-radius: 3px;
            padding: 2px 5px;
            font-size: 0.72rem;
            color: #cbd5e1;
        }}
    </style>
</head>
<body>
    <div class="navbar">
        <div class="nav-left">
            <span class="nav-title">
                <span>⚡</span> {sorter_name} 2-Channel Trace Browser
            </span>
            <div class="selector-box">
                <label for="unitSelect" style="font-size:0.85rem;color:#94a3b8;">Cluster:</label>
                <select id="unitSelect" onchange="loadSelectedUnit()"></select>
                <button class="btn" onclick="prevUnit()" title="Previous Unit (Left Arrow or J)">&#9664; Prev</button>
                <button class="btn" onclick="nextUnit()" title="Next Unit (Right Arrow or K)">Next &#9654;</button>
            </div>
            <div class="selector-box">
                <button class="btn btn-active" id="filterAll" onclick="applyFilter('all')">All ({n_all})</button>
                <button class="btn btn-consensus" id="filterConsensus" onclick="applyFilter('consensus')">Consensus ({n_consensus})</button>
                <button class="btn" id="filterStrict" onclick="applyFilter('strict')">Strict SUA ({n_strict})</button>
                <button class="btn" id="filterResp" onclick="applyFilter('resp')">Continuous SUA ({n_resp})</button>
                <button class="btn" id="filterMua" onclick="applyFilter('mua')">MUA / Candidate ({n_mua})</button>
                <button class="btn" id="filterFail" onclick="applyFilter('fail')">Failed QC ({n_fail})</button>
            </div>
        </div>
        <div class="nav-links">
            <div class="shortcuts-hint">
                <kbd>&larr;</kbd> / <kbd>&rarr;</kbd> or <kbd>J</kbd> / <kbd>K</kbd> to cycle
            </div>
            <a href="{other_suite_relative_url}" class="link-badge link-switch">🔄 Switch to {other_sorter_name} 2-Ch Suite &rarr;</a>
            <a href="{full_5panel_url}" target="_blank" class="link-badge link-fullsuite">🔬 Full 5-Panel Suite &rarr;</a>
            <a href="../../intact_window_raw_emg_flow_interactive.html" target="_blank" class="link-badge">🫁 EMG & Flow</a>
            <a href="../../sorting_validation/index.html" class="link-badge" style="background:#334155;border-color:#64748b;">🏠 Hub</a>
        </div>
    </div>
    <div class="iframe-container">
        <iframe id="unitFrame" src="{first_file}"></iframe>
    </div>

    <script>
    const manifest = {manifest_json};
    let currentFilter = 'all';
    let filteredUnits = [...manifest];
    let currentIndex = 0;

    const selectEl = document.getElementById('unitSelect');
    const frameEl = document.getElementById('unitFrame');

    function populateDropdown() {{
        selectEl.innerHTML = '';
        filteredUnits.forEach((u, idx) => {{
            const opt = document.createElement('option');
            opt.value = idx;
            opt.textContent = `Unit ${{String(u.cluster_id).padStart(4, '0')}} | Ch ${{u.ch1}} (${{Math.round(u.depth1)}}um) & Ch ${{u.ch2}} (${{Math.round(u.depth2)}}um) | ${{u.firing_rate_hz.toFixed(1)}}Hz | ${{u.qc_label}}`;
            selectEl.appendChild(opt);
        }});
        if (filteredUnits.length > 0) {{
            currentIndex = 0;
            selectEl.selectedIndex = 0;
            frameEl.src = filteredUnits[0].file_name;
        }} else {{
            frameEl.src = 'about:blank';
        }}
    }}

    function loadSelectedUnit() {{
        const idx = parseInt(selectEl.value, 10);
        if (idx >= 0 && idx < filteredUnits.length) {{
            currentIndex = idx;
            frameEl.src = filteredUnits[currentIndex].file_name;
        }}
    }}

    function nextUnit() {{
        if (filteredUnits.length === 0) return;
        currentIndex = (currentIndex + 1) % filteredUnits.length;
        selectEl.selectedIndex = currentIndex;
        frameEl.src = filteredUnits[currentIndex].file_name;
    }}

    function prevUnit() {{
        if (filteredUnits.length === 0) return;
        currentIndex = (currentIndex - 1 + filteredUnits.length) % filteredUnits.length;
        selectEl.selectedIndex = currentIndex;
        frameEl.src = filteredUnits[currentIndex].file_name;
    }}

    function applyFilter(filterType) {{
        currentFilter = filterType;
        document.querySelectorAll('.btn').forEach(b => {{
            if (['filterAll', 'filterConsensus', 'filterStrict', 'filterResp', 'filterMua', 'filterFail'].includes(b.id)) {{
                b.classList.remove('btn-active');
            }}
        }});

        if (filterType === 'all') {{
            document.getElementById('filterAll').classList.add('btn-active');
            filteredUnits = [...manifest];
        }} else if (filterType === 'consensus') {{
            document.getElementById('filterConsensus').classList.add('btn-active');
            filteredUnits = manifest.filter(u => u.is_consensus);
        }} else if (filterType === 'strict') {{
            document.getElementById('filterStrict').classList.add('btn-active');
            filteredUnits = manifest.filter(u => u.passed_allen_strict);
        }} else if (filterType === 'resp') {{
            document.getElementById('filterResp').classList.add('btn-active');
            filteredUnits = manifest.filter(u => u.passed_respiratory_continuous);
        }} else if (filterType === 'mua') {{
            document.getElementById('filterMua').classList.add('btn-active');
            filteredUnits = manifest.filter(u => u.is_mua);
        }} else if (filterType === 'fail') {{
            document.getElementById('filterFail').classList.add('btn-active');
            filteredUnits = manifest.filter(u => !u.passed_allen_strict && !u.passed_respiratory_continuous);
        }}

        populateDropdown();
    }}

    // Keyboard navigation
    window.addEventListener('keydown', (e) => {{
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.key === 'ArrowRight' || e.key === 'k' || e.key === 'K') {{
            nextUnit();
        }} else if (e.key === 'ArrowLeft' || e.key === 'j' || e.key === 'J') {{
            prevUnit();
        }}
    }});

    // Listen for iframe navigation events
    window.addEventListener('message', (event) => {{
        if (event.data && event.data.action === 'nextUnit') {{
            nextUnit();
        }} else if (event.data && event.data.action === 'prevUnit') {{
            prevUnit();
        }}
    }});

    // Initialize
    populateDropdown();
    </script>
</body>
</html>
"""
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Built interactive 2-channel validation browser: {dest_path}")


def process_sorter_two_channel_suite(
    sorter_name: str,
    sorter_dir: Path,
    val_out_dir: Path,
    mm_ap: np.memmap,
    channel_depths: Dict[int, float],
    consensus_map: Dict[int, Tuple[int, float]],
    other_sorter_name: str,
    other_suite_url: str,
    full_5panel_url: str
) -> List[Dict]:
    """Processes 2-channel spike trace reports for all units of a sorter."""
    print("=" * 80)
    print(f" PROCESSING 2-CHANNEL TRACE SUITE FOR: {sorter_name.upper()}")
    print(f" Source: {sorter_dir}")
    print(f" Dest: {val_out_dir}")
    print("=" * 80)

    val_out_dir.mkdir(parents=True, exist_ok=True)
    units_dir = val_out_dir / "units"
    units_dir.mkdir(parents=True, exist_ok=True)

    spk_times = np.load(sorter_dir / "alf" / "spike_times.npy")
    spk_clusts = np.load(sorter_dir / "alf" / "spike_clusters.npy")
    df_clusters = pd.read_csv(sorter_dir / "cluster_info.tsv", sep="\t")
    print(f"Loaded {len(df_clusters)} clusters and {len(spk_times):,} spikes.")

    sort_idx = np.argsort(spk_clusts)
    sorted_clusts = spk_clusts[sort_idx]
    sorted_times = spk_times[sort_idx]
    unique_c, split_idx = np.unique(sorted_clusts, return_index=True)
    splits = np.split(sorted_times, split_idx[1:])
    cluster_spike_dict = {int(c): np.sort(st) for c, st in zip(unique_c, splits)}

    all_depths = np.array(list(channel_depths.values()))
    all_chans = np.array(list(channel_depths.keys()))

    def sort_priority(row):
        uid = int(row["cluster_id"])
        is_cons = uid in consensus_map and consensus_map[uid][1] >= 0.50
        passed_allen = bool(row.get("passed_allen_qc", False))
        fr = float(row.get("firing_rate_hz", 0.0))
        if is_cons:
            return (0, -consensus_map[uid][1], -fr)
        if passed_allen:
            return (1, 0, -fr)
        return (2, 0, -fr)

    df_clusters["sort_key"] = [sort_priority(r) for _, r in df_clusters.iterrows()]
    df_sorted = df_clusters.sort_values(by="sort_key").reset_index(drop=True)

    # Butterworth bandpass filter coefficients (300 - 6000 Hz)
    nyq = 0.5 * FS_AP
    b_filt, a_filt = signal.butter(4, [300.0 / nyq, 6000.0 / nyq], btype="band")

    manifest = []
    t_start_all = time.time()
    n_total = len(df_sorted)
    n_samples_total = mm_ap.shape[0]

    for i, (_, row) in enumerate(df_sorted.iterrows()):
        t_u0 = time.time()
        uid = int(row["cluster_id"])
        fr = float(row.get("firing_rate_hz", 0.0))
        passed_allen = bool(row.get("passed_allen_qc", False))
        depth_um = float(row.get("depth_um", 0.0))

        u_spk_times = cluster_spike_dict.get(uid, np.array([], dtype=np.float64))
        if len(u_spk_times) < 5:
            continue

        # Candidate channels around depth
        d_diff = np.abs(all_depths - depth_um)
        est_ch = all_chans[np.argmin(d_diff)]
        cands = np.clip(np.arange(est_ch - 25, est_ch + 26), 0, 383)

        # Footprint channels
        peak_chan, footprint_chans, channel_ptps, _ = find_unit_channel_footprint(
            mm_ap=mm_ap,
            spike_times_s=u_spk_times,
            fs=FS_AP,
            candidate_channels=cands,
            n_footprint_channels=8,
            max_sample_spikes=100,
        )

        sorted_by_ptp = sorted(channel_ptps.items(), key=lambda x: x[1], reverse=True)
        top_2 = [ch for ch, ptp in sorted_by_ptp[:2]]
        if len(top_2) < 2:
            top_2 = [peak_chan, min(383, peak_chan + 1)]

        # Find 1.5s window with highest burst spike density
        window_duration_s = 1.5
        t_min, t_max = float(np.min(u_spk_times)), float(np.max(u_spk_times))
        t_start_s = t_min
        if (t_max - t_min) > window_duration_s:
            eval_times = np.arange(t_min, t_max - window_duration_s, 0.5)
            if len(eval_times) > 0:
                counts = [np.sum((u_spk_times >= t) & (u_spk_times < t + window_duration_s)) for t in eval_times[:40]]
                t_start_s = float(eval_times[int(np.argmax(counts))])

        t_end_s = t_start_s + window_duration_s
        s0 = max(0, int(t_start_s * FS_AP))
        s1 = min(n_samples_total, int(t_end_s * FS_AP))

        # Extract 2-channel raw chunk & filter
        raw_chunk = mm_ap[s0:s1, top_2].astype(np.float32)
        filt_chunk = signal.filtfilt(b_filt, a_filt, raw_chunk, axis=0) * 0.195  # in uV
        t_full = np.arange(s0, s1) / FS_AP

        # Spikes in window
        spks_in_win = np.sort(u_spk_times[(u_spk_times >= t_start_s) & (u_spk_times < t_end_s)])

        # ISI violation check (< 1.0 ms)
        ref_s = 0.001
        viol_flags = np.zeros(len(spks_in_win), dtype=bool)
        if len(spks_in_win) > 1:
            diffs = np.diff(spks_in_win)
            for k in range(len(spks_in_win)):
                prev_dt = diffs[k - 1] if k > 0 else 999.0
                next_dt = diffs[k] if k < len(diffs) else 999.0
                if prev_dt < ref_s or next_dt < ref_s:
                    viol_flags[k] = True

        # Trough alignment on BOTH channel 1 and channel 2
        search_half = int(0.0006 * FS_AP)
        troughs_t_ch1, troughs_v_ch1 = [], []
        troughs_t_ch2, troughs_v_ch2 = [], []

        for st_val in spks_in_win:
            center = int((st_val - t_start_s) * FS_AP)
            w0 = max(0, center - search_half)
            w1 = min(len(filt_chunk), center + search_half)

            # Ch 1 trough
            if w1 > w0:
                m1 = w0 + int(np.argmin(filt_chunk[w0:w1, 0]))
                troughs_t_ch1.append(t_full[m1])
                troughs_v_ch1.append(filt_chunk[m1, 0])
            else:
                troughs_t_ch1.append(st_val)
                troughs_v_ch1.append(0.0)

            # Ch 2 trough
            if w1 > w0:
                m2 = w0 + int(np.argmin(filt_chunk[w0:w1, 1]))
                troughs_t_ch2.append(t_full[m2])
                troughs_v_ch2.append(filt_chunk[m2, 1])
            else:
                troughs_t_ch2.append(st_val)
                troughs_v_ch2.append(0.0)

        troughs_t_ch1 = np.array(troughs_t_ch1, dtype=np.float64)
        troughs_v_ch1 = np.array(troughs_v_ch1, dtype=np.float32)
        troughs_t_ch2 = np.array(troughs_t_ch2, dtype=np.float64)
        troughs_v_ch2 = np.array(troughs_v_ch2, dtype=np.float32)

        # Full session ISI metrics
        diffs_all = np.diff(np.sort(u_spk_times))
        n_viol_sess = int(np.sum(diffs_all < ref_s))
        dur = max(float(np.ptp(u_spk_times)), 1e-3)
        q_isi = float((n_viol_sess * dur) / (2.0 * ref_s * (len(u_spk_times) ** 2)))
        contam_pct = float(n_viol_sess / len(u_spk_times) * 100.0)

        # Cross-sorter consensus
        is_consensus = False
        cons_partner = -1
        cons_score = 0.0
        if uid in consensus_map:
            cons_partner, cons_score = consensus_map[uid]
            if cons_score >= 0.50:
                is_consensus = True

        # Badges
        is_strict = passed_allen and q_isi <= 0.50
        is_resp = passed_allen and fr >= 2.0 and contam_pct <= 5.0
        is_mua = not is_strict and not is_resp and contam_pct > 10.0

        if is_consensus:
            qc_badge = f'<span style="background-color:#581c87;color:#c084fc;border:1px solid #7e22ce;padding:4px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;">CONSENSUS ({other_sorter_name} #{cons_partner} • {cons_score*100:.0f}% match)</span>'
            qc_label = f"Consensus with {other_sorter_name} #{cons_partner} ({cons_score*100:.0f}%)"
        elif is_strict:
            qc_badge = '<span style="background-color:#10b981;color:#ffffff;padding:4px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;">PASS (Strict Allen SUA)</span>'
            qc_label = "Strict Allen SUA [PASS]"
        elif is_resp:
            qc_badge = '<span style="background-color:#0284c7;color:#ffffff;padding:4px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;">PASS (Continuous Resp SUA)</span>'
            qc_label = "Continuous Resp SUA [PASS]"
        elif is_mua:
            qc_badge = f'<span style="background-color:#78350f;color:#fcd34d;border:1px solid #d97706;padding:4px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;">MUA / Contaminated ({contam_pct:.1f}% viol)</span>'
            qc_label = f"MUA ({contam_pct:.1f}% viol)"
        else:
            qc_badge = '<span style="background-color:#7f1d1d;color:#f87171;border:1px solid #dc2626;padding:4px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;">FAIL (QC Rejected)</span>'
            qc_label = "QC Rejected [FAIL]"

        unit_file_name = f"unit_{uid:04d}_ch{top_2[0]}_ch{top_2[1]}.html"
        unit_file_path = units_dir / unit_file_name

        build_two_channel_unit_html(
            uid=uid,
            sorter_name=sorter_name,
            top_2=top_2,
            channel_depths=channel_depths,
            channel_ptps=channel_ptps,
            filt_chunk=filt_chunk,
            t_full=t_full,
            t_start_s=t_start_s,
            t_end_s=t_end_s,
            spks_in_win=spks_in_win,
            troughs_t_ch1=troughs_t_ch1,
            troughs_v_ch1=troughs_v_ch1,
            troughs_t_ch2=troughs_t_ch2,
            troughs_v_ch2=troughs_v_ch2,
            viol_flags=viol_flags,
            fr=fr,
            total_spikes=len(u_spk_times),
            qc_label=qc_label,
            qc_badge_html=qc_badge,
            n_viol_session=n_viol_sess,
            q_isi=q_isi,
            dest_path=unit_file_path
        )

        manifest.append({
            "cluster_id": uid,
            "ch1": top_2[0],
            "ch2": top_2[1],
            "depth1": channel_depths.get(top_2[0], depth_um),
            "depth2": channel_depths.get(top_2[1], depth_um),
            "firing_rate_hz": fr,
            "total_spikes": len(u_spk_times),
            "passed_allen_strict": is_strict,
            "passed_respiratory_continuous": is_resp,
            "is_consensus": is_consensus,
            "is_mua": is_mua,
            "consensus_partner": cons_partner,
            "consensus_score": round(cons_score, 3),
            "qc_label": qc_label,
            "file_name": f"units/{unit_file_name}",
        })

        t_elapsed = time.time() - t_u0
        if (i + 1) % 50 == 0 or (i + 1) == n_total or is_consensus:
            print(f"[{i+1:04d}/{n_total:04d}] Unit {uid:04d} | Ch {top_2[0]} & Ch {top_2[1]} | {fr:.1f}Hz | {qc_label} ({t_elapsed:.2f}s)")

    browser_path = val_out_dir / "cluster_validation_browser.html"
    build_two_channel_browser_html(
        units_manifest=manifest,
        dest_path=browser_path,
        sorter_name=sorter_name,
        other_sorter_name=other_sorter_name,
        other_suite_relative_url=other_suite_url,
        full_5panel_url=full_5panel_url
    )

    t_total = time.time() - t_start_all
    print(f"[{sorter_name}] Completed {len(manifest)} 2-channel units in {t_total:.1f}s ({t_total/max(1, len(manifest)):.2f}s/unit).")
    return manifest


def build_top_level_two_channel_hub(
    ks4_manifest: List[Dict],
    lupin_manifest: List[Dict],
    dest_path: Path
):
    """Builds top-level hub landing page for the 2-channel validation suites."""
    n_ks4_tot = len(ks4_manifest)
    n_ks4_sua = sum(1 for u in ks4_manifest if u.get("passed_allen_strict", False) or u.get("passed_respiratory_continuous", False))
    n_ks4_cons = sum(1 for u in ks4_manifest if u.get("is_consensus", False))

    n_lup_tot = len(lupin_manifest)
    n_lup_sua = sum(1 for u in lupin_manifest if u.get("passed_allen_strict", False) or u.get("passed_respiratory_continuous", False))
    n_lup_cons = sum(1 for u in lupin_manifest if u.get("is_consensus", False))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>E097 Intact Window 2-Channel Spike Trace Validation Hub</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {{
            background: #07090e;
            color: #f1f5f9;
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
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
        .grid-cards {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 24px;
            margin-bottom: 24px;
        }}
        .card {{
            background: #0f1523;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .card:hover {{
            transform: translateY(-2px);
            border-color: rgba(56, 189, 248, 0.4);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}
        .card-title {{
            font-size: 20px;
            font-weight: 700;
            margin: 0;
        }}
        .badge {{
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 700;
        }}
        .badge-ks4 {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }}
        .badge-lupin {{ background: rgba(0, 245, 212, 0.2); color: #00f5d4; border: 1px solid #00f5d4; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin: 16px 0;
            background: rgba(0,0,0,0.25);
            padding: 14px;
            border-radius: 10px;
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-label {{
            font-size: 11px;
            color: #94a3b8;
            text-transform: uppercase;
            margin-bottom: 4px;
        }}
        .stat-val {{
            font-size: 22px;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
        }}
        .btn-launch {{
            display: block;
            text-align: center;
            background: #0284c7;
            color: #ffffff;
            font-weight: 700;
            font-size: 14px;
            padding: 12px 20px;
            border-radius: 8px;
            text-decoration: none;
            margin-top: 16px;
            transition: background 0.15s ease;
        }}
        .btn-launch:hover {{
            background: #0369a1;
        }}
        .btn-lupin {{
            background: #0f766e;
        }}
        .btn-lupin:hover {{
            background: #115e59;
        }}
        .consensus-banner {{
            background: linear-gradient(135deg, rgba(88, 28, 135, 0.4), rgba(126, 34, 206, 0.2));
            border: 1px solid #7e22ce;
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .consensus-text h3 {{
            margin: 0 0 6px 0;
            font-size: 17px;
            color: #d8b4fe;
        }}
        .consensus-text p {{
            margin: 0;
            font-size: 13px;
            color: #c084fc;
        }}
        .consensus-num {{
            font-size: 32px;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            color: #f3e8ff;
        }}
        .links-row {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }}
        .link-pill {{
            background: #1e293b;
            color: #38bdf8;
            padding: 8px 16px;
            border-radius: 6px;
            text-decoration: none;
            font-size: 13px;
            font-weight: 600;
            border: 1px solid rgba(56, 189, 248, 0.25);
            transition: background 0.15s;
        }}
        .link-pill:hover {{
            background: #38bdf8;
            color: #07090e;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ E097 Intact Window (420.0s) 2-Channel Spike Trace Validation Hub</h1>
            <p class="subtitle">
                Focused Raw AP Voltage Inspection: Primary Peak & Secondary Peak Contacts @ Native 30 kHz (300&ndash;6000 Hz)<br>
                Spike Times & Trough Markers Snapped to Both Channels &bull; Refractory Violation Highlighting &bull; Real-Time Gain Scaling
            </p>
        </div>

        <div class="consensus-banner">
            <div class="consensus-text">
                <h3>🤝 Cross-Sorter Mutual Consensus Agreement</h3>
                <p>SpikeInterface bipartite spike-train cross-matching (&Delta;t = 0.4 ms, agreement &ge; 50%). Direct multi-electrode verification.</p>
            </div>
            <div class="consensus-num">{n_lup_cons} Units ({n_lup_cons/max(1, n_lup_tot)*100:.1f}%)</div>
        </div>

        <div class="grid-cards">
            <!-- Kilosort 4 Card -->
            <div class="card">
                <div>
                    <div class="card-header">
                        <h2 class="card-title" style="color: #fbbf24;">Kilosort 4 2-Channel Suite</h2>
                        <span class="badge badge-ks4">GPU Accelerated</span>
                    </div>
                    <p style="color: #94a3b8; font-size: 13px; margin: 0 0 14px 0;">
                        Streamlined 2-channel action potential trace inspector ({n_ks4_tot} units). Primary soma peak and secondary footprint contact with spike alignment.
                    </p>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-label">Total Units</div>
                            <div class="stat-val" style="color: #f1f5f9;">{n_ks4_tot}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Clean SUA</div>
                            <div class="stat-val" style="color: #10b981;">{n_ks4_sua}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Consensus</div>
                            <div class="stat-val" style="color: #c084fc;">{n_ks4_cons}</div>
                        </div>
                    </div>
                </div>
                <div>
                    <a href="../kilosort4/two_channel_validation/cluster_validation_browser.html" class="btn-launch">
                        🚀 Launch KS4 2-Channel Trace Browser ({n_ks4_tot} Units) &rarr;
                    </a>
                </div>
            </div>

            <!-- Lupin Card -->
            <div class="card">
                <div>
                    <div class="card-header">
                        <h2 class="card-title" style="color: #00f5d4;">Lupin (SVD-OMP) 2-Channel Suite</h2>
                        <span class="badge badge-lupin">Sparse Pursuit</span>
                    </div>
                    <p style="color: #94a3b8; font-size: 13px; margin: 0 0 14px 0;">
                        Sparse pursuit 2-channel action potential trace inspector ({n_lup_tot} units). High SNR waveforms with synchronized dual-contact spike alignment.
                    </p>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-label">Total Units</div>
                            <div class="stat-val" style="color: #f1f5f9;">{n_lup_tot}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Clean SUA</div>
                            <div class="stat-val" style="color: #10b981;">{n_lup_sua}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Consensus</div>
                            <div class="stat-val" style="color: #c084fc;">{n_lup_cons}</div>
                        </div>
                    </div>
                </div>
                <div>
                    <a href="../lupin/two_channel_validation/cluster_validation_browser.html" class="btn-launch btn-lupin">
                        🚀 Launch Lupin 2-Channel Trace Browser ({n_lup_tot} Units) &rarr;
                    </a>
                </div>
            </div>
        </div>

        <div style="background: #0f1523; border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 20px 24px;">
            <h3 style="margin: 0 0 14px 0; font-size: 15px; color: #94a3b8; text-transform: uppercase;">Direct Analysis Shortcuts</h3>
            <div class="links-row">
                <a href="../sorting_validation/index.html" class="link-pill">🔬 Full 5-Panel Diagnostic Hub</a>
                <a href="../intact_window_raw_emg_flow_interactive.html" class="link-pill">🫁 Interactive EMG & Airflow Dynamics</a>
                <a href="../intact_window_population_dynamics_dashboard.html" class="link-pill">📊 Population Dynamics & Dropout</a>
                <a href="../kilosort4/two_channel_validation/cluster_validation_browser.html" class="link-pill">⚡ KS4 2-Channel Browser</a>
                <a href="../lupin/two_channel_validation/cluster_validation_browser.html" class="link-pill">⚡ Lupin 2-Channel Browser</a>
            </div>
        </div>
    </div>
</body>
</html>
"""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    # Also write alias cluster_validation_browser.html
    alt_dest = dest_path.parent / "cluster_validation_browser.html"
    with open(alt_dest, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Built top-level 2-channel validation hub: {dest_path}")


def main():
    print("=" * 80)
    print(" E097 INTACT WINDOW (420.0s) 2-CHANNEL SPIKE TRACE VALIDATION SUITE GENERATOR")
    print(f" Time Range: {START_T_S}s - {START_T_S + DURATION_S}s ({DURATION_S}s)")
    print("=" * 80)

    channel_depths = load_probe_geometry()
    lup_to_ks4, ks4_to_lup = load_consensus_map()

    ap_bytes = RAW_AP_BIN.stat().st_size
    n_samples = ap_bytes // (2 * N_CHANS_AP)
    print(f"Opening AP Binary: {RAW_AP_BIN.name} ({n_samples:,} samples, {n_samples / FS_AP:.2f}s)")
    mm_ap = np.memmap(RAW_AP_BIN, dtype=np.int16, mode="r", shape=(n_samples, N_CHANS_AP))

    # 1. Process Lupin 2-Channel Suite (273 units)
    lupin_manifest = process_sorter_two_channel_suite(
        sorter_name="Lupin (SVD-OMP)",
        sorter_dir=LUPIN_DIR,
        val_out_dir=LUPIN_VAL_DIR,
        mm_ap=mm_ap,
        channel_depths=channel_depths,
        consensus_map=lup_to_ks4,
        other_sorter_name="Kilosort 4",
        other_suite_url="../../kilosort4/two_channel_validation/cluster_validation_browser.html",
        full_5panel_url="../sorting_validation/cluster_validation_browser.html"
    )

    # 2. Process Kilosort 4 2-Channel Suite (1,102 units)
    ks4_manifest = process_sorter_two_channel_suite(
        sorter_name="Kilosort 4",
        sorter_dir=KS4_DIR,
        val_out_dir=KS4_VAL_DIR,
        mm_ap=mm_ap,
        channel_depths=channel_depths,
        consensus_map=ks4_to_lup,
        other_sorter_name="Lupin",
        other_suite_url="../../lupin/two_channel_validation/cluster_validation_browser.html",
        full_5panel_url="../sorting_validation/cluster_validation_browser.html"
    )

    # 3. Build Top-Level 2-Channel Hub
    hub_path = TOP_VAL_DIR / "index.html"
    build_top_level_two_channel_hub(
        ks4_manifest=ks4_manifest,
        lupin_manifest=lupin_manifest,
        dest_path=hub_path
    )

    print("=" * 80)
    print(" ALL 2-CHANNEL VALIDATION SUITES SUCCESSFULLY GENERATED!")
    print(f" Top-Level Hub: file:///{hub_path.resolve().as_posix()}")
    print(f" Kilosort 4 Suite: file:///{(KS4_VAL_DIR / 'cluster_validation_browser.html').resolve().as_posix()}")
    print(f" Lupin Suite: file:///{(LUPIN_VAL_DIR / 'cluster_validation_browser.html').resolve().as_posix()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
