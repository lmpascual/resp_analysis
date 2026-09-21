#!/usr/bin/env python3
"""
Interactive Breath Detection Diagnostic Dashboard (cibrrig Methodology):
Plots synchronized across time:
1. Raw Diaphragm EMG (Channel 0)
2. Processed/Integrated Diaphragm EMG Envelope with Onsets (Green) & Offsets (Red)
3. Flow Meter Signal (Channel 5)
4. Vertical marker lines & shaded inspiratory durations across all 3 traces.
"""
import sys
import os
import argparse
import logging
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BREATH_DIAGNOSTICS")


def main():
    parser = argparse.ArgumentParser(description="Diagnostic dashboard for breath onset/offset detection on EMG and Flow signals")
    parser.add_argument("--raw-dir", type=str, default=r"Z:\ramirezlab\pascual\rec_data\LP21_fent_EXPT_g1")
    parser.add_argument("--out-dir", type=str, default="output_publication_figures_LP21_60s")
    parser.add_argument("--start-time", type=float, default=60.0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=20.0, help="Duration in seconds to display interactively")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir.strip('"'))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nidq_bins = list(raw_dir.glob("*.nidq.bin"))
    if not nidq_bins:
        raise FileNotFoundError(f"No .nidq.bin found in {raw_dir}")
    nidq_bin = nidq_bins[0]
    nidq_meta = nidq_bin.with_suffix(".meta")

    # Read NIDQ channels: Ch 0 = EMG, Ch 1 = ECG, Ch 5 = Flow
    nidq_data = read_nidq_bin(nidq_bin, nidq_meta, emg_chan=0, ecg_chan=1)
    t_nidq = nidq_data.time_vector
    fs = nidq_data.fs

    t0 = args.start_time
    t1 = t0 + args.duration
    mask = (t_nidq >= t0) & (t_nidq <= t1)

    t_sub = t_nidq[mask]
    emg_raw = nidq_data.all_channels[0][mask]
    flow_raw = nidq_data.all_channels[5][mask] if 5 in nidq_data.all_channels else np.zeros_like(t_sub)

    # Run EMG processor with cibrrig peak_widths algorithm
    emg_cfg = EMGConfig(fs=fs, highpass_cutoff=300.0, lowpass_cutoff=3000.0, smoothing_window_ms=20.0)
    emg_proc = process_emg(emg_raw, t_sub, emg_cfg)
    detected_onsets = emg_proc.breath_onsets
    detected_offsets = emg_proc.breath_offsets

    # Downsample for smooth web browser interaction (~2000 Hz)
    step = max(1, int(fs / 2000.0))
    t_plot = t_sub[::step]
    emg_raw_plot = emg_raw[::step]
    emg_env_plot = emg_proc.envelope[::step]
    flow_plot = flow_raw[::step]

    # Create 3-Panel Synchronized Interactive Dashboard
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            "<b>1. Raw Diaphragm EMG (Channel 0)</b>",
            "<b>2. Integrated Diaphragm EMG Envelope — Onsets (Green) & Offsets (Red)</b>",
            "<b>3. Flow Meter Signal (Channel 5)</b>"
        ]
    )

    # 1. Raw Diaphragm EMG
    fig.add_trace(
        go.Scattergl(x=t_plot, y=emg_raw_plot, mode="lines", line=dict(color="#475569", width=1), name="Raw EMG"),
        row=1, col=1
    )

    # 2. EMG Envelope
    fig.add_trace(
        go.Scattergl(x=t_plot, y=emg_env_plot, mode="lines", line=dict(color="#0284c7", width=2), name="EMG Envelope"),
        row=2, col=1
    )

    # Mark Onsets (Green Triangles) on Envelope
    onset_amplitudes = np.interp(detected_onsets, t_sub, emg_proc.envelope) if len(detected_onsets) > 0 else []
    fig.add_trace(
        go.Scatter(
            x=detected_onsets,
            y=onset_amplitudes,
            mode="markers",
            marker=dict(symbol="triangle-up", size=10, color="#16a34a"),
            name="Breath Onset (I-on)"
        ),
        row=2, col=1
    )

    # Mark Offsets (Red Triangles) on Envelope
    offset_amplitudes = np.interp(detected_offsets, t_sub, emg_proc.envelope) if len(detected_offsets) > 0 else []
    fig.add_trace(
        go.Scatter(
            x=detected_offsets,
            y=offset_amplitudes,
            mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="#dc2626"),
            name="Breath Offset (I-off)"
        ),
        row=2, col=1
    )

    # 3. Flow Meter
    fig.add_trace(
        go.Scattergl(x=t_plot, y=flow_plot, mode="lines", line=dict(color="#059669", width=1.8), name="Flow Signal"),
        row=3, col=1
    )

    # Add vertical lines across all 3 subplots: Green for Onsets, Red for Offsets in batch
    shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    for on in detected_onsets:
        for r in range(1, 4):
            y_axis = f"y{r}" if r > 1 else "y"
            shapes.append(dict(
                type="line",
                x0=on, x1=on,
                y0=0, y1=1,
                xref="x", yref=f"{y_axis} domain",
                line=dict(width=1.2, dash="dash", color="rgba(22, 163, 74, 0.65)")
            ))
    for off in detected_offsets:
        for r in range(1, 4):
            y_axis = f"y{r}" if r > 1 else "y"
            shapes.append(dict(
                type="line",
                x0=off, x1=off,
                y0=0, y1=1,
                xref="x", yref=f"{y_axis} domain",
                line=dict(width=1.2, dash="dot", color="rgba(220, 38, 38, 0.65)")
            ))
    fig.layout.shapes = shapes

    fig.update_layout(
        title=f"<b>Breath Detection Diagnostic Dashboard (cibrrig Method) — {len(detected_onsets)} Breaths Detected ({t0:.1f}s to {t1:.1f}s)</b><br><sup>Green Dashed Lines = True Breath Onset (I-on) | Red Dotted Lines = Breath Offset (I-off)</sup>",
        template="plotly_white",
        height=880,
        hovermode="x unified",
        margin=dict(l=60, r=40, b=40, t=90)
    )

    fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
    fig.update_yaxes(title_text="Envelope Amplitude (V)", row=2, col=1)
    fig.update_yaxes(title_text="Flow Meter (V)", row=3, col=1)

    html_path = out_dir / "breath_detection_diagnostic_dashboard.html"
    fig.write_html(str(html_path))
    logger.info(f"Saved Interactive Breath Detection Dashboard: {html_path.resolve()}")

    print("\n" + "=" * 85)
    print(" [SUCCESS] BREATH DETECTION DIAGNOSTIC DASHBOARD UPDATED")
    print("=" * 85)
    print(f" • Interactive Dashboard: {html_path.resolve()}")
    print("=" * 85)


if __name__ == "__main__":
    main()
