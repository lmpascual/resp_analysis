"""
Interactive HTML Validation Report Generator for Single Clusters.

Builds a modern, responsive, browser validation suite featuring:
- Panel 1: Multi-Channel AP Footprint (1.5s Burst Snippet @ 30 kHz Native) with interactive waveform scaling buttons
- Panel 2: ISI Distribution with 1.0 ms Refractory Cutoff
- Panel 3A: Amplitude Stationarity Scatter & Rolling Median Trendline
- Panel 3B: Time-Resolved Unsmoothed ISI CDFs (5 Epochs, Linear Scale)
- Panel 4: Respiratory Phase Rose Plot & Mazurek Vector Strength
- Panel 5A: Autocorrelogram (ACG)
- Panel 5B: Footprint Template Waveforms with generous headroom (decreased scale)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.pipeline.diagnostics.cluster_validator import UnitValidationBundle


def build_unit_validation_figure(bundle: UnitValidationBundle) -> go.Figure:
    """
    Constructs a 5-panel comprehensive validation figure for a single unit.
    Expanded height (1850px) and bottom margin to prevent any legend or subplot collision.
    """
    fig = make_subplots(
        rows=4, cols=2,
        column_widths=[0.55, 0.45],
        row_heights=[0.35, 0.21, 0.22, 0.22],
        specs=[
            [{"colspan": 2}, None],                       # Row 1: Multi-channel Voltage Traces
            [{"type": "xy"}, {"type": "polar"}],          # Row 2: ISI Distribution & Respiratory Phase Rose
            [{"type": "xy"}, {"type": "xy"}],             # Row 3: Amplitude Stationarity & Time-Resolved ISI CDFs
            [{"type": "xy"}, {"type": "xy"}]              # Row 4: Autocorrelogram & Template Waveforms
        ],
        subplot_titles=(
            f"<b>1. Multi-Channel Action Potential Footprint (1.5s Burst Snippet @ 30 kHz Native)</b> • Ch {bundle.peak_channel} [PEAK] | Red: Spikes • Amber: ISI Violations (<{bundle.isi.refractory_ms:.1f}ms)",
            f"<b>2. ISI Distribution</b> (Refractory Threshold: {bundle.isi.refractory_ms:.1f} ms)",
            f"<b>4. Respiratory Phase Rose Plot & Vector</b> ({bundle.respiratory_phase.landmark_provenance})",
            f"<b>3A. Amplitude Stationarity Over Session</b> (Drift: {bundle.stationarity.drift_slope_uv_per_min} uV/min)",
            f"<b>3B. Time-Resolved Unsmoothed ISI CDFs</b> (5 Epochs, Linear Scale)",
            f"<b>5A. Autocorrelogram (ACG)</b> (Dip Ratio: {bundle.acg.refractory_dip_ratio})",
            f"<b>5B. Footprint Template Waveforms</b> ({bundle.waveform.putative_cell_type})"
        ),
        vertical_spacing=0.09,
        horizontal_spacing=0.08
    )

    # -------------------------------------------------------------
    # Row 1: Multi-Channel Filtered Voltage Traces (Dynamic Auto-Scaled Lanes)
    # -------------------------------------------------------------
    if bundle.trace_snippet is not None:
        snip = bundle.trace_snippet
        t_ax = snip.t_axis_s
        n_ch = len(snip.channels)

        chan_depth_pairs = [(ch, snip.channel_depths_um[i], i) for i, ch in enumerate(snip.channels)]
        chan_depth_pairs.sort(key=lambda x: x[1])  # Ascending depth

        # Dynamic channel min/max examination (<0.15ms execution time)
        # Sizes lanes so waveforms comfortably occupy ~70% of lane height by default
        ch_ptps = [float(np.ptp(snip.traces_uv[i])) for i in range(n_ch)]
        max_ptp = max(ch_ptps) if len(ch_ptps) > 0 else 60.0
        max_ptp = max(max_ptp, 30.0)
        lane_spacing = float(max_ptp / 0.70)
        offsets = np.arange(n_ch) * lane_spacing

        peak_y_base = 0.0
        for plot_idx, (ch, d_um, orig_idx) in enumerate(chan_depth_pairs):
            y_base = offsets[plot_idx]
            y_tr = snip.traces_uv[orig_idx] + y_base
            is_peak = (ch == bundle.peak_channel)
            if is_peak:
                peak_y_base = y_base

            line_col = "#2563eb" if is_peak else "#64748b"
            line_w = 2.0 if is_peak else 1.0

            fig.add_trace(
                go.Scatter(
                    x=t_ax,
                    y=y_tr,
                    mode="lines",
                    name=f"Peak Trace (Ch {ch})" if is_peak else f"Ch {ch}",
                    showlegend=is_peak,
                    line=dict(color=line_col, width=line_w),
                    hoverinfo="x+y"
                ),
                row=1, col=1
            )

        # Highlight detected spikes ONLY on the PEAK CHANNEL at aligned negative troughs
        if len(snip.aligned_trough_times) > 0:
            spk_y = snip.aligned_trough_voltages + peak_y_base
            fig.add_trace(
                go.Scatter(
                    x=snip.aligned_trough_times,
                    y=spk_y,
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=7,
                        color="#ef4444",
                        line=dict(color="#ffffff", width=0.5)
                    ),
                    name=f"Detected Spikes (Ch {bundle.peak_channel})",
                    showlegend=True,
                    hoverinfo="x+y"
                ),
                row=1, col=1
            )

        # Highlight ISI Refractory Violations (< refractory_ms) with prominent warning triangles
        if hasattr(snip, "violation_trough_times") and len(snip.violation_trough_times) > 0:
            viol_y = snip.violation_trough_voltages + peak_y_base
            fig.add_trace(
                go.Scatter(
                    x=snip.violation_trough_times,
                    y=viol_y,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color="#f59e0b",
                        line=dict(color="#b45309", width=1.8)
                    ),
                    name=f"ISI Violations (<{bundle.isi.refractory_ms:.1f}ms)",
                    showlegend=True,
                    hoverinfo="x+y+name"
                ),
                row=1, col=1
            )

        # Update y-axis with channel depth ticks and tight viewport range
        fig.update_yaxes(
            tickmode="array",
            tickvals=offsets,
            ticktext=[
                f"<b>Ch {ch}</b> ({d_um:.0f}um){'<br>[PEAK]' if ch == bundle.peak_channel else ''}"
                for ch, d_um, _ in chan_depth_pairs
            ],
            title_text="Electrode Contacts",
            range=[-lane_spacing * 0.5, (n_ch - 0.5) * lane_spacing],
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
    fig.add_vline(
        x=isi_res.refractory_ms,
        line=dict(color="#dc2626", width=2, dash="dash"),
        annotation_text=f"Refractory Threshold: {isi_res.refractory_ms:.1f}ms",
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
            ticktext=["0 deg<br>(Insp Onset)", "90 deg", "180 deg<br>(Exp Onset)", "270 deg"]
        ),
        radialaxis=dict(showticklabels=False, ticks=""),
        row=2, col=2
    )

    # -------------------------------------------------------------
    # Row 3, Col 1: Amplitude Stationarity
    # -------------------------------------------------------------
    st = bundle.stationarity
    if len(st.spike_times_s) > 0:
        fig.add_trace(
            go.Scattergl(
                x=st.spike_times_s,
                y=st.spike_amplitudes_uv,
                mode="markers",
                marker=dict(size=3, color="#94a3b8", opacity=0.4),
                name="Individual Spikes (uV)",
                hoverinfo="x+y"
            ),
            row=3, col=1
        )
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
    # Row 3, Col 2: Time-Resolved Unsmoothed ISI CDFs (Linear Scale)
    # -------------------------------------------------------------
    tr_isi = bundle.time_resolved_isi
    palette = ["#2563eb", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"]
    has_cdf = hasattr(tr_isi, "epoch_cdfs") and len(tr_isi.epoch_cdfs) > 0 and len(tr_isi.epoch_isis_ms) > 0

    if has_cdf:
        for ep_idx, (lbl, isis_ep, cdf_ep) in enumerate(zip(tr_isi.epoch_labels, tr_isi.epoch_isis_ms, tr_isi.epoch_cdfs)):
            if len(isis_ep) == 0:
                continue
            c_code = palette[ep_idx % len(palette)]
            ks_val = tr_isi.ks_drift_p_values[ep_idx] if hasattr(tr_isi, "ks_drift_p_values") and ep_idx < len(tr_isi.ks_drift_p_values) else 1.0
            ks_text = f" (KS p={ks_val:.2f})" if ep_idx > 0 else " (Ref)"
            fig.add_trace(
                go.Scatter(
                    x=isis_ep,
                    y=cdf_ep,
                    mode="lines",
                    line_shape="hv",  # Exact unsmoothed empirical step CDF
                    name=f"{lbl}{ks_text}",
                    line=dict(color=c_code, width=2.0),
                    hovertemplate=f"<b>{lbl}</b><br>ISI: %{{x:.2f}} ms<br>Cumulative: %{{y:.3f}}<extra></extra>"
                ),
                row=3, col=2
            )

        fig.add_trace(
            go.Scatter(
                x=[bundle.isi.refractory_ms, bundle.isi.refractory_ms], y=[0.0, 1.0],
                mode="lines",
                line=dict(color="#ef4444", dash="dash", width=1.5),
                name=f"{bundle.isi.refractory_ms:.1f}ms Refractory Threshold",
                hoverinfo="name"
            ),
            row=3, col=2
        )
        fig.update_xaxes(
            title_text="Inter-Spike Interval (ms)",
            range=[0, 60],
            row=3, col=2
        )
        fig.update_yaxes(
            title_text="Cumulative Probability (ECDF)",
            range=[0.0, 1.05],
            row=3, col=2
        )
    else:
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
    # Row 4, Col 2: Footprint Template Waveforms (Decreased Scale / Zoomed Out)
    # -------------------------------------------------------------
    wf = bundle.waveform
    first_footprint = True
    for ch, tmpl in wf.template_waveforms.items():
        is_peak = (ch == bundle.peak_channel)
        if is_peak:
            leg_name = f"Template Ch {ch} (Peak Soma)"
            show_leg = True
        else:
            leg_name = "Adjacent Footprint Templates"
            show_leg = first_footprint
            if first_footprint:
                first_footprint = False

        fig.add_trace(
            go.Scatter(
                x=wf.time_axis_ms,
                y=tmpl,
                mode="lines",
                name=leg_name,
                showlegend=show_leg,
                line=dict(color="#dc2626" if is_peak else "#94a3b8", width=3.0 if is_peak else 1.2),
                hoverinfo="x+y+name"
            ),
            row=4, col=2
        )

    # Compute generous y-axis bounds to decrease scale (zoom out)
    all_tmpls = list(wf.template_waveforms.values())
    if len(all_tmpls) > 0:
        min_v = float(min(np.min(t) for t in all_tmpls))
        max_v = float(max(np.max(t) for t in all_tmpls))
        ptp_v = max_v - min_v
        pad = max(25.0, ptp_v * 0.60)
        y_min_wf = min_v - pad
        y_max_wf = max_v + pad
    else:
        y_min_wf, y_max_wf = -100.0, 50.0

    fig.update_xaxes(title_text="Time Relative to Spike (ms)", range=[-1.0, 1.5], row=4, col=2)
    fig.update_yaxes(title_text="Voltage (uV)", range=[y_min_wf, y_max_wf], row=4, col=2)

    # Global Layout: Expanded height and bottom margin to guarantee zero overlap
    fig.update_layout(
        template="plotly_white",
        height=1950,
        margin=dict(l=75, r=45, t=75, b=140),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.06,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif")
    )
    return fig


def generate_single_unit_html(bundle: UnitValidationBundle, output_path: Path) -> Path:
    """
    Renders an individual self-contained HTML validator file for a single unit.
    Includes interactive waveform vertical scale zoom buttons and full session violation tools.
    """
    fig = build_unit_validation_figure(bundle)
    fig_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    rp = bundle.respiratory_phase
    isi = bundle.isi
    wf = bundle.waveform
    snip = bundle.trace_snippet
    snip_t0 = snip.t_start_s if snip else 0.0
    snip_t1 = snip.t_end_s if snip else 1.5
    custom_badge = getattr(bundle, "custom_badge_html", None)
    if custom_badge:
        qc_badge = custom_badge
    else:
        qc_badge = (
            '<span style="background-color:#10b981;color:#ffffff;padding:4px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;">PASS (Allen QC)</span>'
            if bundle.passed_allen_qc else
            f'<span style="background-color:#ef4444;color:#ffffff;padding:4px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;" title="{getattr(bundle, "qc_summary", "QC Flagged")}">FAIL ({getattr(bundle, "qc_summary", "QC Flagged")})</span>'
        )

    n_ch = len(snip.channels) if snip else 0
    if snip:
        ch_ptps = [float(np.ptp(snip.traces_uv[i])) for i in range(n_ch)]
        max_ptp = max(ch_ptps) if len(ch_ptps) > 0 else 60.0
        max_ptp = max(max_ptp, 30.0)
        lane_spacing = float(max_ptp / 0.70)
    else:
        lane_spacing = 100.0

    trace_offsets = [float(i * lane_spacing) for i in range(n_ch)]
    chan_depth_pairs = [(ch, snip.channel_depths_um[i], i) for i, ch in enumerate(snip.channels)] if snip else []
    chan_depth_pairs.sort(key=lambda x: x[1])
    peak_y_base = 0.0
    for plot_idx, (ch, _, _) in enumerate(chan_depth_pairs):
        if ch == bundle.peak_channel:
            peak_y_base = float(plot_idx * lane_spacing)
            break

    has_spk = bool(snip and len(snip.aligned_trough_times) > 0)
    has_viol = bool(snip and hasattr(snip, "violation_trough_times") and len(snip.violation_trough_times) > 0)

    cli_dir = getattr(bundle, "custom_cli_dir", "outputs/LP21/LP21_fent_EXPT_g1_60s_120s_lupin_run")
    if isi.n_violations == 0:
        viol_summary_html = '<span style="color:#10b981;font-weight:600;">0 Violations (Clean single-unit isolation, Q = 0.000)</span>'
        cli_window_cmd = f"python scripts/visualization/generate_cluster_trace_window.py --output-dir {cli_dir} --unit {bundle.cluster_id} --t-start {max(0.0, snip_t0):.1f} --duration 3.0 --open"
    else:
        viol_summary_html = f'<span style="color:#d97706;font-weight:600;">{isi.n_violations} Violations (&lt; {isi.refractory_ms:.1f} ms, Q = {isi.violation_rate_q:.3f})</span>'
        cli_window_cmd = f"python scripts/visualization/generate_cluster_trace_window.py --output-dir {cli_dir} --unit {bundle.cluster_id} --violation 1 --open"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unit {bundle.cluster_id:04d} Validation Suite | Sorting Validation Inspector</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f8fafc;
            color: #0f172a;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1440px;
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
        .gain-btn {{
            background: #ffffff;
            border: 1px solid #cbd5e1;
            color: #334155;
            padding: 5px 12px;
            border-radius: 6px;
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
            box-shadow: 0 1px 2px rgba(37, 99, 235, 0.2);
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

        <div style="background:#f8fafc;border:1px solid #cbd5e1;border-left:4px solid #3b82f6;border-radius:8px;padding:14px 18px;margin-bottom:16px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <span style="font-weight:700;font-size:0.95rem;color:#1e293b;">
                    🔬 Panel 1 Trace Inspector &amp; Interactive Waveform Zoom
                </span>
                <span style="font-size:0.82rem;background:#e2e8f0;padding:4px 10px;border-radius:6px;color:#334155;font-weight:600;">
                    Snippet: {snip_t0:.2f}s &ndash; {snip_t1:.2f}s (1.5s @ 30,000 pts/sec)
                </span>
            </div>
            <div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:12px;background:#ffffff;border:1px solid #e2e8f0;padding:8px 12px;border-radius:6px;">
                <span style="font-size:0.84rem;font-weight:700;color:#1e293b;margin-right:4px;">🔍 Waveform Vertical Scale:</span>
                <button onclick="setTraceGain(1.0)" class="gain-btn active" id="btn-gain-10">1x (Default)</button>
                <button onclick="setTraceGain(1.5)" class="gain-btn" id="btn-gain-15">1.5x</button>
                <button onclick="setTraceGain(2.0)" class="gain-btn" id="btn-gain-20">2x</button>
                <button onclick="setTraceGain(3.0)" class="gain-btn" id="btn-gain-30">3x</button>
                <button onclick="adjustTraceGain(1.25)" class="gain-btn" title="Increase trace height across all channels">➕ Bigger</button>
                <button onclick="adjustTraceGain(0.80)" class="gain-btn" title="Decrease trace height across all channels">➖ Smaller</button>
                <button onclick="setTraceGain(1.0)" class="gain-btn" title="Reset to default scale">↺ Reset</button>
            </div>
            <div style="font-size:0.85rem;color:#475569;margin-bottom:10px;line-height:1.5;">
                Panel 1 displays raw bandpass-filtered (300&ndash;6000 Hz) traces with exact action potential trough snapping and ISI refractory violation markers (&lt; {isi.refractory_ms:.1f} ms).<br>
                <strong>Full Session Refractory Violations:</strong> {viol_summary_html}
            </div>
            <div style="font-size:0.82rem;color:#64748b;margin-bottom:6px;font-weight:600;">
                ⚡ Generate Custom Window or Jump to Violations (Calculates single cluster on-demand in &lt;1.5s):
            </div>
            <div style="background:#0f172a;color:#e2e8f0;border-radius:6px;padding:9px 14px;font-family:Consolas, Monaco, monospace;font-size:0.82rem;display:flex;justify-content:space-between;align-items:center;">
                <span id="cliCmd">{cli_window_cmd}</span>
                <button onclick="navigator.clipboard.writeText(document.getElementById('cliCmd').innerText); this.innerText='Copied!'; setTimeout(() => this.innerText='Copy Command', 2000);" style="background:#2563eb;border:none;color:#ffffff;padding:5px 12px;border-radius:4px;cursor:pointer;font-size:0.75rem;font-weight:600;margin-left:14px;white-space:nowrap;">Copy Command</button>
            </div>
        </div>

        {fig_html}
    </div>

    <script>
    var currentGain = 1.0;
    var traceOffsets = {json.dumps(trace_offsets)};
    var peakBase = {peak_y_base};
    var nChannels = {n_ch};
    var hasSpikes = {json.dumps(has_spk)};
    var hasViol = {json.dumps(has_viol)};

    function setTraceGain(newGain) {{
        var gd = document.querySelector('.plotly-graph-div');
        if (!gd || !gd.data) return;
        var ratio = newGain / currentGain;

        var allNewY = [];
        var allIndices = [];

        for (var i = 0; i < nChannels; i++) {{
            if (!gd.data[i]) continue;
            var base = traceOffsets[i];
            var yData = gd.data[i].y;
            var newY = new Float32Array(yData.length);
            for (var j = 0; j < yData.length; j++) {{
                newY[j] = base + (yData[j] - base) * ratio;
            }}
            allNewY.push(newY);
            allIndices.push(i);
        }}

        var curIdx = nChannels;
        if (hasSpikes && gd.data[curIdx]) {{
            var spkY = gd.data[curIdx].y;
            var newSpkY = new Float64Array(spkY.length);
            for (var k = 0; k < spkY.length; k++) {{
                newSpkY[k] = peakBase + (spkY[k] - peakBase) * ratio;
            }}
            allNewY.push(newSpkY);
            allIndices.push(curIdx);
            curIdx++;
        }}

        if (hasViol && gd.data[curIdx]) {{
            var violY = gd.data[curIdx].y;
            var newViolY = new Float64Array(violY.length);
            for (var m = 0; m < violY.length; m++) {{
                newViolY[m] = peakBase + (violY[m] - peakBase) * ratio;
            }}
            allNewY.push(newViolY);
            allIndices.push(curIdx);
        }}

        // Single batch restyle call for instant WebGL update (<15ms)
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
    </script>
</body>
</html>
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    return output_path


def generate_multi_cluster_browser_html(
    bundles: List[UnitValidationBundle],
    output_path: Path,
    units_rel_dir: str = "units"
) -> Path:
    """
    Renders an ultra-lightweight (<50 KB) multi-cluster browser shell that dynamically
    loads individual unit reports inside an iframe. Zero memory bloat, instant opening,
    no risk of OOM in IDE or browser.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    units_manifest = []
    for b in bundles:
        rp = b.respiratory_phase
        isi = b.isi
        wf = b.waveform
        unit_file_name = f"unit_{b.cluster_id:04d}_ch{b.peak_channel}.html"
        units_manifest.append({
            "cluster_id": b.cluster_id,
            "firing_rate_hz": b.firing_rate_hz,
            "total_spikes": b.total_spikes,
            "depth_um": b.depth_um,
            "shank": b.shank,
            "peak_channel": b.peak_channel,
            "passed_allen_qc": b.passed_allen_qc,
            "mean_vector_r": rp.mean_vector_r,
            "preferred_phase_deg": rp.preferred_phase_deg,
            "functional_class": rp.functional_class,
            "subtype_class": rp.subtype_class,
            "violation_rate_q": isi.violation_rate_q,
            "contamination_pct": isi.contamination_pct,
            "snr": wf.snr,
            "trough_to_peak_us": wf.trough_to_peak_us,
            "putative_cell_type": wf.putative_cell_type,
            "file_name": f"{units_rel_dir}/{unit_file_name}"
        })

    manifest_json = json.dumps(units_manifest)
    first_unit_file = units_manifest[0]["file_name"] if len(units_manifest) > 0 else ""

    browser_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spike Sorting Validation Browser</title>
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
            gap: 16px;
        }}
        .nav-title {{
            font-size: 1.15rem;
            font-weight: 700;
            color: #38bdf8;
            letter-spacing: -0.02em;
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
        }}
        select:focus {{
            border-color: #38bdf8;
        }}
        .btn {{
            background: #334155;
            color: #f8fafc;
            border: 1px solid #475569;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .btn:hover {{
            background: #475569;
        }}
        .btn-active {{
            background: #0284c7;
            border-color: #38bdf8;
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
        }}
        kbd {{
            background: #0f172a;
            border: 1px solid #475569;
            border-radius: 3px;
            padding: 2px 5px;
            font-size: 0.75rem;
            color: #cbd5e1;
        }}
    </style>
</head>
<body>
    <div class="navbar">
        <div class="nav-left">
            <span class="nav-title">Spike Sorting Validation Browser</span>
            <div class="selector-box">
                <label for="unitSelect" style="font-size:0.85rem;color:#94a3b8;">Cluster:</label>
                <select id="unitSelect" onchange="loadSelectedUnit()"></select>
                <button class="btn" onclick="prevUnit()" title="Previous Unit (Left Arrow or J)">&#9664; Prev</button>
                <button class="btn" onclick="nextUnit()" title="Next Unit (Right Arrow or K)">Next &#9654;</button>
            </div>
            <div class="selector-box">
                <button class="btn btn-active" id="filterAll" onclick="applyFilter('all')">All</button>
                <button class="btn" id="filterPass" onclick="applyFilter('pass')">Passed QC</button>
                <button class="btn" id="filterInsp" onclick="applyFilter('insp')">Inspiratory</button>
                <button class="btn" id="filterExp" onclick="applyFilter('exp')">Expiratory</button>
                <button class="btn" id="filterTonic" onclick="applyFilter('tonic')">Tonic</button>
            </div>
        </div>
        <div class="shortcuts-hint">
            Keyboard: <kbd>&larr;</kbd> / <kbd>&rarr;</kbd> or <kbd>J</kbd> / <kbd>K</kbd> to cycle units
        </div>
    </div>
    <div class="iframe-container">
        <iframe id="unitFrame" src="{first_unit_file}"></iframe>
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
            opt.textContent = `Unit ${{String(u.cluster_id).padStart(4, '0')}} | Ch ${{u.peak_channel}} (${{Math.round(u.depth_um)}}um) | ${{u.firing_rate_hz.toFixed(1)}}Hz | ${{u.subtype_class}}`;
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
            if (['filterAll', 'filterPass', 'filterInsp', 'filterExp', 'filterTonic'].includes(b.id)) {{
                b.classList.remove('btn-active');
            }}
        }});
        if (filterType === 'all') {{
            document.getElementById('filterAll').classList.add('btn-active');
            filteredUnits = [...manifest];
        }} else if (filterType === 'pass') {{
            document.getElementById('filterPass').classList.add('btn-active');
            filteredUnits = manifest.filter(u => u.passed_allen_qc);
        }} else if (filterType === 'insp') {{
            document.getElementById('filterInsp').classList.add('btn-active');
            filteredUnits = manifest.filter(u => u.functional_class === 'Inspiratory');
        }} else if (filterType === 'exp') {{
            document.getElementById('filterExp').classList.add('btn-active');
            filteredUnits = manifest.filter(u => u.functional_class === 'Expiratory');
        }} else if (filterType === 'tonic') {{
            document.getElementById('filterTonic').classList.add('btn-active');
            filteredUnits = manifest.filter(u => u.functional_class === 'Tonic');
        }}
        populateDropdown();
    }}

    window.addEventListener('keydown', (e) => {{
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.key === 'ArrowRight' || e.key === 'k' || e.key === 'K') {{
            nextUnit();
        }} else if (e.key === 'ArrowLeft' || e.key === 'j' || e.key === 'J') {{
            prevUnit();
        }}
    }});

    populateDropdown();
    </script>
</body>
</html>
"""
    output_path.write_text(browser_html, encoding="utf-8")
    return output_path
