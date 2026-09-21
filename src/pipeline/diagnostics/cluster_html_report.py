"""
Interactive HTML Dashboard Generator for Spike Sorting Cluster Validation.

Builds a modern, responsive, Phy-like browser validation suite featuring:
1. Multi-channel filtered voltage traces (6-8 channels) with highlighted spike markers.
2. ISI distribution histogram with refractory cutoff threshold (1.5 ms).
3. Stationarity Suite: Spike amplitude drift vs time and time-resolved ISI PDF ridge lines.
4. Respiratory Phase Vector: Circular polar rose plot, mean vector (r, theta), Rayleigh test.
5. Literature Quality Controls: Autocorrelogram (ACG), template waveform kinetics, and amplitude cutoff fit.
6. Unified multi-unit browser with dropdown selector, filters, and keyboard navigation ([J/K], [Left/Right]).
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.pipeline.diagnostics.cluster_validator import UnitValidationBundle


def build_unit_validation_figure(bundle: UnitValidationBundle) -> go.Figure:
    """
    Constructs a 5-panel comprehensive validation figure for a single unit.
    """
    fig = make_subplots(
        rows=4, cols=2,
        column_widths=[0.55, 0.45],
        row_heights=[0.35, 0.22, 0.22, 0.21],
        specs=[
            [{"colspan": 2}, None],                       # Row 1: Multi-channel Voltage Traces
            [{"type": "xy"}, {"type": "polar"}],          # Row 2: ISI Distribution & Respiratory Phase Rose
            [{"type": "xy"}, {"type": "xy"}],             # Row 3: Amplitude Stationarity & Time-Resolved ISI PDFs
            [{"type": "xy"}, {"type": "xy"}]              # Row 4: Autocorrelogram & Template Waveforms
        ],
        subplot_titles=(
            f"<b>1. Multi-Channel Filtered Footprint Traces (300-6000 Hz)</b> & Detected Spikes (Red Markers)",
            f"<b>2. ISI Distribution</b> (Refractory Threshold: 1.5 ms)",
            f"<b>4. Respiratory Phase Rose Plot & Vector</b> ({bundle.respiratory_phase.landmark_provenance})",
            f"<b>3A. Amplitude Stationarity Over Session</b> (Drift: {bundle.stationarity.drift_slope_uv_per_min} uV/min)",
            f"<b>3B. Time-Resolved ISI PDF Lines</b> (5 Temporal Epochs)",
            f"<b>5A. Autocorrelogram (ACG)</b> (Dip Ratio: {bundle.acg.refractory_dip_ratio})",
            f"<b>5B. Footprint Template Waveforms</b> ({bundle.waveform.putative_cell_type})"
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )

    # -------------------------------------------------------------
    # Row 1: Multi-Channel Filtered Voltage Traces
    # -------------------------------------------------------------
    if bundle.trace_snippet is not None:
        snip = bundle.trace_snippet
        t_ax = snip.t_axis_s
        n_ch = len(snip.channels)
        offsets = np.arange(n_ch) * 120.0  # 120 uV spacing between channels

        for c_idx, ch in enumerate(snip.channels):
            y_base = offsets[c_idx]
            y_tr = snip.traces_uv[c_idx] + y_base
            d_um = snip.channel_depths_um[c_idx] if c_idx < len(snip.channel_depths_um) else 0.0
            is_peak = (ch == bundle.peak_channel)
            line_col = "#2563eb" if is_peak else "#64748b"
            line_w = 1.5 if is_peak else 1.0

            fig.add_trace(
                go.Scatter(
                    x=t_ax,
                    y=y_tr,
                    mode="lines",
                    name=f"Ch {ch} ({d_um:.0f}um){' [PEAK]' if is_peak else ''}",
                    line=dict(color=line_col, width=line_w),
                    hoverinfo="x+y"
                ),
                row=1, col=1
            )

            # Highlight spikes on this channel
            if len(snip.spike_sample_indices_in_window) > 0:
                valid_s_idxs = snip.spike_sample_indices_in_window[
                    (snip.spike_sample_indices_in_window >= 0) & 
                    (snip.spike_sample_indices_in_window < len(y_tr))
                ]
                if len(valid_s_idxs) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=t_ax[valid_s_idxs],
                            y=y_tr[valid_s_idxs],
                            mode="markers",
                            marker=dict(symbol="circle", size=6, color="#ef4444", line=dict(color="#ffffff", width=0.5)),
                            name=f"Spikes Ch {ch}",
                            showlegend=False,
                            hoverinfo="x+y"
                        ),
                        row=1, col=1
                    )

        # Update y-axis with channel ticks
        fig.update_yaxes(
            tickmode="array",
            tickvals=offsets,
            ticktext=[f"Ch {ch}<br>({snip.channel_depths_um[i]:.0f}um)" for i, ch in enumerate(snip.channels)],
            title_text="Electrode Channels",
            row=1, col=1
        )
        fig.update_xaxes(title_text="Recording Time (s)", row=1, col=1)

    # -------------------------------------------------------------
    # Row 2, Col 1: ISI Distribution
    # -------------------------------------------------------------
    isi_res = bundle.isi
    bin_mids = 0.5 * (isi_res.hist_bins_ms[:-1] + isi_res.hist_bins_ms[1:])
    bar_cols = ["#ef4444" if b < isi_res.refractory_ms else "#0ea5e9" for b in bin_mids]

    fig.add_trace(
        go.Bar(
            x=bin_mids,
            y=isi_res.hist_counts,
            marker_color=bar_cols,
            name="ISI (ms)",
            hoverinfo="x+y"
        ),
        row=2, col=1
    )
    # Refractory threshold line
    fig.add_vline(
        x=isi_res.refractory_ms,
        line=dict(color="#dc2626", width=2, dash="dash"),
        annotation_text=f"Refractory Threshold: {isi_res.refractory_ms}ms",
        annotation_position="top right",
        row=2, col=1
    )
    fig.update_xaxes(title_text="Inter-Spike Interval (ms)", range=[0, 60], row=2, col=1)
    fig.update_yaxes(title_text="Spike Pairs Count", row=2, col=1)

    # -------------------------------------------------------------
    # Row 2, Col 2: Respiratory Phase Rose Plot & Vector
    # -------------------------------------------------------------
    rp = bundle.respiratory_phase
    mids_rose = 0.5 * (rp.rose_bins_deg[:-1] + rp.rose_bins_deg[1:])

    fig.add_trace(
        go.Barpolar(
            r=rp.rose_density,
            theta=mids_rose,
            width=10.0,
            marker=dict(
                color=["#fb7185" if (0 <= t < 180) else "#38bdf8" for t in mids_rose],
                line=dict(color="#ffffff", width=0.5)
            ),
            name="Phase Density",
            hoverinfo="r+theta"
        ),
        row=2, col=2
    )

    # Mean resultant vector arrow
    r_len = rp.mean_vector_r * float(np.max(rp.rose_density)) if np.max(rp.rose_density) > 0 else 0.5
    fig.add_trace(
        go.Scatterpolar(
            r=[0, r_len],
            theta=[0, rp.preferred_phase_deg],
            mode="lines+markers",
            line=dict(color="#10b981", width=3.5),
            marker=dict(symbol="arrow-up", size=10, color="#10b981"),
            name=f"Mean Vector (R={rp.mean_vector_r}, {rp.preferred_phase_deg}deg)"
        ),
        row=2, col=2
    )
    fig.update_polars(
        angularaxis=dict(
            direction="clockwise",
            period=360,
            tickmode="array",
            tickvals=[0, 90, 180, 270],
            ticktext=["0 deg<br>(Insp Onset)", "90 deg<br>(Late Insp)", "180 deg<br>(Exp Onset)", "270 deg<br>(Late Exp)"]
        ),
        radialaxis=dict(visible=True, showticklabels=False),
        row=2, col=2
    )

    # -------------------------------------------------------------
    # Row 3, Col 1: Amplitude Stationarity
    # -------------------------------------------------------------
    st = bundle.stationarity
    if len(st.spike_times_s) > 0:
        # Scatter of spikes (subsample if massive)
        n_pts = len(st.spike_times_s)
        samp_idx = np.random.choice(n_pts, min(3000, n_pts), replace=False)
        fig.add_trace(
            go.Scatter(
                x=st.spike_times_s[samp_idx],
                y=st.spike_amplitudes_uv[samp_idx],
                mode="markers",
                marker=dict(size=3, color="#94a3b8", opacity=0.4),
                name="Spike Amps (uV)",
                showlegend=False
            ),
            row=3, col=1
        )
        # Rolling median trendline
        fig.add_trace(
            go.Scatter(
                x=st.time_bin_centers_s,
                y=st.median_amplitudes_uv,
                mode="lines",
                line=dict(color="#4f46e5", width=2.5),
                name="Median Amp (uV)"
            ),
            row=3, col=1
        )
        # IQR ribbon
        fig.add_trace(
            go.Scatter(
                x=list(st.time_bin_centers_s) + list(st.time_bin_centers_s[::-1]),
                y=list(st.iqr_upper_uv) + list(st.iqr_lower_uv[::-1]),
                fill="toself",
                fillcolor="rgba(79, 70, 229, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="IQR (25-75%)",
                showlegend=False
            ),
            row=3, col=1
        )
    fig.update_xaxes(title_text="Session Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Amplitude (uV)", row=3, col=1)

    # -------------------------------------------------------------
    # Row 3, Col 2: Time-Resolved ISI PDF Lines
    # -------------------------------------------------------------
    tr_isi = bundle.time_resolved_isi
    palette = ["#2563eb", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"]
    for ep_idx, (lbl, pdf) in enumerate(zip(tr_isi.epoch_labels, tr_isi.epoch_pdfs)):
        c_code = palette[ep_idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=tr_isi.pdf_x_ms,
                y=pdf,
                mode="lines",
                name=lbl,
                line=dict(color=c_code, width=2.0)
            ),
            row=3, col=2
        )
    fig.update_xaxes(title_text="Inter-Spike Interval (ms)", range=[0, 60], row=3, col=2)
    fig.update_yaxes(title_text="Probability Density", row=3, col=2)

    # -------------------------------------------------------------
    # Row 4, Col 1: Autocorrelogram (ACG)
    # -------------------------------------------------------------
    acg = bundle.acg
    fig.add_trace(
        go.Bar(
            x=acg.lags_ms,
            y=acg.counts,
            marker_color="#8b5cf6",
            name="ACG Counts",
            hoverinfo="x+y"
        ),
        row=4, col=1
    )
    fig.update_xaxes(title_text="Lag (ms)", range=[-50, 50], row=4, col=1)
    fig.update_yaxes(title_text="Coincidence Counts", row=4, col=1)

    # -------------------------------------------------------------
    # Row 4, Col 2: Footprint Template Waveforms
    # -------------------------------------------------------------
    wf = bundle.waveform
    for ch, tmpl in wf.template_waveforms.items():
        is_peak = (ch == bundle.peak_channel)
        fig.add_trace(
            go.Scatter(
                x=wf.time_axis_ms,
                y=tmpl,
                mode="lines",
                name=f"Template Ch {ch}{' (Peak)' if is_peak else ''}",
                line=dict(color="#dc2626" if is_peak else "#94a3b8", width=2.5 if is_peak else 1.0)
            ),
            row=4, col=2
        )
    fig.update_xaxes(title_text="Time Relative to Spike (ms)", range=[-1.0, 1.5], row=4, col=2)
    fig.update_yaxes(title_text="Voltage (uV)", row=4, col=2)

    # Global Layout
    fig.update_layout(
        template="plotly_white",
        height=1450,
        margin=dict(l=60, r=40, t=70, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif")
    )
    return fig


def generate_single_unit_html(bundle: UnitValidationBundle, output_path: Path) -> Path:
    """
    Renders an individual self-contained HTML validator file for a single unit.
    """
    fig = build_unit_validation_figure(bundle)
    fig_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    rp = bundle.respiratory_phase
    isi = bundle.isi
    wf = bundle.waveform

    qc_badge = (
        '<span style="background-color:#10b981;color:#ffffff;padding:4px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;">PASS (Allen QC)</span>'
        if bundle.passed_allen_qc else
        '<span style="background-color:#ef4444;color:#ffffff;padding:4px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;">FAIL (QC Flagged)</span>'
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unit {bundle.cluster_id:04d} Validation Suite | Phy-Like Inspector</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f8fafc;
            color: #0f172a;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
            padding: 24px 32px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 16px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 1.6rem;
            color: #1e293b;
        }}
        .metrics-card {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            background: #f1f5f9;
            padding: 16px 20px;
            border-radius: 8px;
            margin-bottom: 24px;
        }}
        .metric-item {{
            display: flex;
            flex-direction: column;
        }}
        .metric-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            font-weight: 600;
        }}
        .metric-val {{
            font-size: 1.15rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 4px;
        }}
        .provenance-banner {{
            background: #e0f2fe;
            border-left: 4px solid #0284c7;
            padding: 10px 16px;
            border-radius: 4px;
            margin-bottom: 20px;
            font-size: 0.88rem;
            color: #0369a1;
            font-weight: 500;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>Unit {bundle.cluster_id:04d} &mdash; Single-Cluster Electrophysiological Validation</h1>
                <div style="margin-top:6px;color:#64748b;font-size:0.9rem;">
                    Neuropixels 2.0 | Peak Channel {bundle.peak_channel} | Depth {bundle.depth_um:.0f} &mu;m | Shank {bundle.shank}
                </div>
            </div>
            <div>{qc_badge}</div>
        </div>

        <div class="provenance-banner">
            <strong>Respiratory Phase Vector Provenance:</strong> {rp.landmark_provenance} &mdash; Cycle: 0&deg; Insp Onset &rarr; 180&deg; Exp Onset
        </div>

        <div class="metrics-card">
            <div class="metric-item">
                <span class="metric-label">Firing Rate</span>
                <span class="metric-val">{bundle.firing_rate_hz:.2f} Hz</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Total Spikes</span>
                <span class="metric-val">{bundle.total_spikes:,}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Estimated SNR</span>
                <span class="metric-val">{wf.snr:.2f}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">ISI Violations (Q)</span>
                <span class="metric-val" style="color:{'#10b981' if isi.violation_rate_q <= 0.5 else '#ef4444'};">
                    {isi.violation_rate_q:.3f} ({isi.contamination_pct:.1f}%)
                </span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Phase Modulation (R)</span>
                <span class="metric-val">{rp.mean_vector_r:.3f} ({rp.functional_class})</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Preferred Phase</span>
                <span class="metric-val">{rp.preferred_phase_deg:.1f}&deg; ({rp.subtype_class})</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Action Potential Kinetics</span>
                <span class="metric-val">{wf.trough_to_peak_us:.0f} &mu;s ({wf.putative_cell_type.split()[0]})</span>
            </div>
        </div>

        {fig_html}
    </div>
</body>
</html>
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    return output_path


def generate_multi_cluster_browser_html(
    bundles: List[UnitValidationBundle],
    output_path: Path
) -> Path:
    """
    Renders a unified multi-cluster browser HTML with an interactive dropdown selector,
    instant keyboard navigation, and fast rendering.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Render plots for each unit
    unit_payloads = []
    for b in bundles:
        fig = build_unit_validation_figure(b)
        fig_div = fig.to_html(include_plotlyjs=False, full_html=False)
        rp = b.respiratory_phase
        isi = b.isi
        wf = b.waveform

        unit_payloads.append({
            "cluster_id": b.cluster_id,
            "firing_rate_hz": b.firing_rate_hz,
            "total_spikes": b.total_spikes,
            "depth_um": b.depth_um,
            "shank": b.shank,
            "peak_channel": b.peak_channel,
            "passed_allen_qc": b.passed_allen_qc,
            "landmark_provenance": rp.landmark_provenance,
            "snr": wf.snr,
            "isi_q": isi.violation_rate_q,
            "isi_pct": isi.contamination_pct,
            "phase_r": rp.mean_vector_r,
            "phase_deg": rp.preferred_phase_deg,
            "functional_class": rp.functional_class,
            "subtype_class": rp.subtype_class,
            "trough_to_peak_us": wf.trough_to_peak_us,
            "cell_type": wf.putative_cell_type.split()[0],
            "fig_html": fig_div
        })

    # Convert payloads to json
    units_json = json.dumps(unit_payloads)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phy-Like Multi-Cluster Validation Browser</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 20px;
        }}
        .browser-card {{
            max-width: 1440px;
            margin: 0 auto;
            background: #1e293b;
            border-radius: 12px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
            padding: 20px 28px;
            border: 1px solid #334155;
        }}
        .top-bar {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #334155;
            padding-bottom: 16px;
            margin-bottom: 16px;
            gap: 12px;
        }}
        .selector-controls {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        select {{
            background: #0f172a;
            color: #f8fafc;
            border: 1px solid #475569;
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 0.95rem;
            outline: none;
            cursor: pointer;
        }}
        button {{
            background: #3b82f6;
            color: #ffffff;
            border: none;
            padding: 8px 14px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s ease;
        }}
        button:hover {{
            background: #2563eb;
        }}
        .filter-btn {{
            background: #334155;
            font-size: 0.85rem;
            padding: 6px 12px;
        }}
        .filter-btn.active {{
            background: #0284c7;
        }}
        .metrics-card {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 12px;
            background: #0f172a;
            padding: 14px 18px;
            border-radius: 8px;
            border: 1px solid #334155;
            margin-bottom: 16px;
        }}
        .metric-item {{
            display: flex;
            flex-direction: column;
        }}
        .metric-label {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            font-weight: 600;
        }}
        .metric-val {{
            font-size: 1.1rem;
            font-weight: 700;
            color: #f8fafc;
            margin-top: 4px;
        }}
        .provenance-banner {{
            background: #0c4a6e;
            border-left: 4px solid #38bdf8;
            padding: 8px 14px;
            border-radius: 4px;
            margin-bottom: 16px;
            font-size: 0.85rem;
            color: #bae6fd;
        }}
        #plot-container {{
            background: #ffffff;
            border-radius: 8px;
            padding: 10px;
        }}
    </style>
</head>
<body>
    <div class="browser-card">
        <div class="top-bar">
            <div>
                <h2 style="margin:0;font-size:1.4rem;">Phy-Like Cluster Validation Browser</h2>
                <div style="color:#94a3b8;font-size:0.85rem;margin-top:4px;">
                    Use dropdown or keys <strong>[J/K]</strong> or <strong>[&larr;/&rarr;]</strong> to switch clusters.
                </div>
            </div>

            <div class="selector-controls">
                <button onclick="prevUnit()">&larr; Prev</button>
                <select id="unitSelect" onchange="onSelectUnit(this.value)"></select>
                <button onclick="nextUnit()">Next &rarr;</button>
            </div>

            <div style="display:flex;gap:6px;">
                <button class="filter-btn active" id="filter-all" onclick="applyFilter('all')">All Units</button>
                <button class="filter-btn" id="filter-qc" onclick="applyFilter('qc')">Passed QC</button>
                <button class="filter-btn" id="filter-insp" onclick="applyFilter('insp')">Inspiratory</button>
                <button class="filter-btn" id="filter-exp" onclick="applyFilter('exp')">Expiratory</button>
                <button class="filter-btn" id="filter-tonic" onclick="applyFilter('tonic')">Tonic</button>
            </div>
        </div>

        <div class="provenance-banner" id="provBanner">
            <strong>Respiratory Landmark Provenance:</strong> <span id="provText">Loading...</span>
        </div>

        <div class="metrics-card">
            <div class="metric-item">
                <span class="metric-label">Unit / Location</span>
                <span class="metric-val" id="mUnitLoc">--</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Allen QC Status</span>
                <span class="metric-val" id="mQC">--</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Firing Rate / Spikes</span>
                <span class="metric-val" id="mFR">--</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Estimated SNR</span>
                <span class="metric-val" id="mSNR">--</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">ISI Violations (Q)</span>
                <span class="metric-val" id="mISI">--</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Phase Vector (R, &phi;)</span>
                <span class="metric-val" id="mPhase">--</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">AP Trough-to-Peak</span>
                <span class="metric-val" id="mKinetics">--</span>
            </div>
        </div>

        <div id="plot-container"></div>
    </div>

    <script>
        const unitsData = {units_json};
        let currentFilter = 'all';
        let filteredIndices = [];
        let currentIndex = 0;

        function populateSelector() {{
            const sel = document.getElementById('unitSelect');
            sel.innerHTML = '';
            filteredIndices = [];

            unitsData.forEach((u, idx) => {{
                let matches = true;
                if (currentFilter === 'qc' && !u.passed_allen_qc) matches = false;
                if (currentFilter === 'insp' && u.functional_class !== 'Inspiratory') matches = false;
                if (currentFilter === 'exp' && u.functional_class !== 'Expiratory') matches = false;
                if (currentFilter === 'tonic' && u.functional_class !== 'Tonic') matches = false;

                if (matches) {{
                    filteredIndices.push(idx);
                    const opt = document.createElement('option');
                    opt.value = idx;
                    opt.textContent = `Unit ${{String(u.cluster_id).padStart(4, '0')}} | Ch ${{u.peak_channel}} (${{u.depth_um.toFixed(0)}}um) | ${{u.firing_rate_hz.toFixed(1)}}Hz | ${{u.subtype_class}}`;
                    sel.appendChild(opt);
                }}
            }});

            if (filteredIndices.length > 0) {{
                renderUnit(filteredIndices[0]);
            }} else {{
                document.getElementById('plot-container').innerHTML = '<div style="padding:40px;text-align:center;color:#64748b;">No units match the selected filter.</div>';
            }}
        }}

        function renderUnit(index) {{
            currentIndex = index;
            const u = unitsData[index];
            document.getElementById('unitSelect').value = index;

            document.getElementById('provText').textContent = `${{u.landmark_provenance}} (0 deg Insp Onset -> 180 deg Exp Onset)`;
            document.getElementById('mUnitLoc').textContent = `Unit ${{u.cluster_id}} (Ch ${{u.peak_channel}}, ${{u.depth_um.toFixed(0)}}um)`;
            document.getElementById('mQC').innerHTML = u.passed_allen_qc ? 
                '<span style="color:#10b981;">PASS (Good)</span>' : 
                '<span style="color:#ef4444;">FAIL (Flagged)</span>';
            document.getElementById('mFR').textContent = `${{u.firing_rate_hz.toFixed(2)}} Hz (${{u.total_spikes.toLocaleString()}} spk)`;
            document.getElementById('mSNR').textContent = `${{u.snr.toFixed(2)}}`;
            document.getElementById('mISI').innerHTML = `<span style="color:${{u.isi_q <= 0.5 ? '#10b981' : '#ef4444'}}">${{u.isi_q.toFixed(3)}} (${{u.isi_pct.toFixed(1)}}%)</span>`;
            document.getElementById('mPhase').textContent = `R=${{u.phase_r.toFixed(3)}} | ${{u.phase_deg.toFixed(1)}} deg (${{u.subtype_class}})`;
            document.getElementById('mKinetics').textContent = `${{u.trough_to_peak_us.toFixed(0)}} us (${{u.cell_type}})`;

            // Inject figure HTML
            const container = document.getElementById('plot-container');
            container.innerHTML = u.fig_html;

            // Execute Plotly scripts inside injected HTML
            const scripts = container.getElementsByTagName('script');
            for (let i = 0; i < scripts.length; i++) {{
                eval(scripts[i].innerText);
            }}
        }}

        function onSelectUnit(idx) {{
            renderUnit(parseInt(idx));
        }}

        function prevUnit() {{
            const currentPos = filteredIndices.indexOf(currentIndex);
            if (currentPos > 0) {{
                renderUnit(filteredIndices[currentPos - 1]);
            }}
        }}

        function nextUnit() {{
            const currentPos = filteredIndices.indexOf(currentIndex);
            if (currentPos >= 0 && currentPos < filteredIndices.length - 1) {{
                renderUnit(filteredIndices[currentPos + 1]);
            }}
        }}

        function applyFilter(f) {{
            currentFilter = f;
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('filter-' + f).classList.add('active');
            populateSelector();
        }}

        // Keyboard navigation
        window.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft' || e.key === 'j' || e.key === 'J') {{
                prevUnit();
            }} else if (e.key === 'ArrowRight' || e.key === 'k' || e.key === 'K') {{
                nextUnit();
            }}
        }});

        window.onload = populateSelector;
    </script>
</body>
</html>
"""
    output_path.write_text(html_content, encoding="utf-8")
    return output_path
