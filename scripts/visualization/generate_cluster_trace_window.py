#!/usr/bin/env python3
"""
On-Demand Single-Cluster High-Resolution Trace Inspector (Native 30 kHz, Zero Downsampling).

Enables electrophysiologists to inspect arbitrary time windows, fuller windows, or specific
ISI refractory violation events at 100% native 30,000 samples/sec without downsampling.
Calculates only one cluster at a time on-demand, executing in < 1.5 seconds without
computational overhead on the batch pipeline.

Modes:
  1. Custom Time Window:
     python scripts/visualization/generate_cluster_trace_window.py \
         --output-dir outputs/LP21/LP21_fent_EXPT_g1_60s_120s_lupin_run \
         --unit 65 --t-start 10.0 --duration 3.0 --open

  2. Center on Specific ISI Refractory Violation (< 1.0 ms):
     python scripts/visualization/generate_cluster_trace_window.py \
         --output-dir outputs/LP21/LP21_fent_EXPT_g1_60s_120s_lupin_run \
         --unit 64 --violation 1 --duration 2.0 --open

  3. Inspect All ISI Violations (Multi-Violation Dashboard):
     python scripts/visualization/generate_cluster_trace_window.py \
         --output-dir outputs/LP21/LP21_fent_EXPT_g1_60s_120s_lupin_run \
         --unit 64 --all-violations --open

  4. Interactive Multi-Chunk Browser (Page Through 2s Windows across Session):
     python scripts/visualization/generate_cluster_trace_window.py \
         --output-dir outputs/LP21/LP21_fent_EXPT_g1_60s_120s_lupin_run \
         --unit 65 --chunk-browser --open
"""

import argparse
import os
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.diagnostics.cluster_validator import find_unit_channel_footprint


def extract_native_trace_window(
    mm_ap: np.memmap,
    channels: List[int],
    channel_depths: Dict[int, float],
    spike_times_s: np.ndarray,
    peak_channel: int,
    t_start_s: float,
    duration_s: float,
    fs: float = 30000.0,
    refractory_ms: float = 1.0
) -> Dict[str, Any]:
    """
    Extracts raw voltage traces across specified channels, applies 300-6000 Hz zero-phase
    Butterworth filter, and performs trough snapping at 100% native 30 kHz resolution (zero decimation).
    """
    n_total_samples = mm_ap.shape[0]
    t_end_s = t_start_s + duration_s
    s0 = max(0, int(t_start_s * fs))
    s1 = min(n_total_samples, int(t_end_s * fs))

    raw_chunk = mm_ap[s0:s1, channels].astype(np.float32)

    # 4th-order zero-phase Butterworth bandpass filter (300 - 6000 Hz)
    nyq = 0.5 * fs
    b, a = signal.butter(4, [300.0 / nyq, 6000.0 / nyq], btype="band")
    filt_chunk = signal.filtfilt(b, a, raw_chunk, axis=0) * 0.195  # in uV

    t_full = np.arange(s0, s1) / fs

    peak_idx = channels.index(peak_channel) if peak_channel in channels else 0
    filt_peak = filt_chunk[:, peak_idx]

    spks_in_win = np.sort(spike_times_s[(spike_times_s >= t_start_s) & (spike_times_s < t_end_s)])

    # Identify refractory violation spikes (< refractory_ms)
    ref_s = refractory_ms / 1000.0
    viol_flags = np.zeros(len(spks_in_win), dtype=bool)
    if len(spks_in_win) > 1:
        diffs = np.diff(spks_in_win)
        for i in range(len(spks_in_win)):
            prev_dt = diffs[i - 1] if i > 0 else 999.0
            next_dt = diffs[i] if i < len(diffs) else 999.0
            if prev_dt < ref_s or next_dt < ref_s:
                viol_flags[i] = True

    # Align each spike to exact negative trough within +/- 0.6 ms (+/- 18 samples)
    search_half_samples = int(0.0006 * fs)
    aligned_troughs_t = []
    aligned_troughs_v = []
    viol_troughs_t = []
    viol_troughs_v = []

    for i, st in enumerate(spks_in_win):
        s_center = int((st - t_start_s) * fs)
        w_start = max(0, s_center - search_half_samples)
        w_end = min(len(filt_peak), s_center + search_half_samples)
        if w_end > w_start:
            local_min_idx = w_start + int(np.argmin(filt_peak[w_start:w_end]))
            t_tr = t_full[local_min_idx]
            v_tr = filt_peak[local_min_idx]
        else:
            t_tr = st
            v_tr = 0.0

        aligned_troughs_t.append(t_tr)
        aligned_troughs_v.append(v_tr)
        if viol_flags[i]:
            viol_troughs_t.append(t_tr)
            viol_troughs_v.append(v_tr)

    return {
        "t_full": t_full,
        "filt_traces": filt_chunk.T,  # (n_channels, n_samples)
        "channels": channels,
        "peak_channel": peak_channel,
        "channel_depths": [channel_depths.get(ch, (ch // 2) * 15.0) for ch in channels],
        "spikes_t": np.array(aligned_troughs_t, dtype=np.float64),
        "spikes_v": np.array(aligned_troughs_v, dtype=np.float32),
        "viol_t": np.array(viol_troughs_t, dtype=np.float64),
        "viol_v": np.array(viol_troughs_v, dtype=np.float32),
        "t_start_s": t_start_s,
        "t_end_s": t_end_s,
        "n_samples": len(t_full)
    }


def find_session_refractory_violations(
    spike_times_s: np.ndarray,
    refractory_ms: float = 1.0
) -> List[Dict[str, float]]:
    """
    Scans entire spike train and returns all refractory violation events (< refractory_ms).
    """
    sorted_spks = np.sort(spike_times_s)
    if len(sorted_spks) < 2:
        return []

    diffs_s = np.diff(sorted_spks)
    ref_s = refractory_ms / 1000.0

    violations = []
    for i, dt in enumerate(diffs_s):
        if dt < ref_s:
            violations.append({
                "index": len(violations) + 1,
                "spike_1_t": float(sorted_spks[i]),
                "spike_2_t": float(sorted_spks[i + 1]),
                "isi_ms": float(dt * 1000.0),
                "midpoint_t": float(0.5 * (sorted_spks[i] + sorted_spks[i + 1]))
            })
    return violations


def build_window_figure(
    window_data: Dict[str, Any],
    title_text: str,
    time_offset_s: float = 0.0,
    trace_spacing_uv: float = 180.0
) -> go.Figure:
    """
    Builds a high-performance Plotly figure for an un-downsampled window.
    """
    channels = window_data["channels"]
    peak_channel = window_data["peak_channel"]
    depths = window_data["channel_depths"]
    traces = window_data["filt_traces"]
    t_axis = window_data["t_full"] - time_offset_s

    chan_depth_pairs = [(ch, depths[i], i) for i, ch in enumerate(channels)]
    chan_depth_pairs.sort(key=lambda x: x[1])

    offsets = np.arange(len(channels)) * trace_spacing_uv
    fig = go.Figure()

    peak_y_base = 0.0
    for plot_idx, (ch, d_um, orig_idx) in enumerate(chan_depth_pairs):
        y_base = offsets[plot_idx]
        y_tr = traces[orig_idx] + y_base
        is_peak = (ch == peak_chan)
        if is_peak:
            peak_y_base = y_base

        fig.add_trace(
            go.Scattergl(
                x=t_axis,
                y=y_tr,
                mode="lines",
                line=dict(color="#2563eb" if is_peak else "#64748b", width=1.8 if is_peak else 0.9),
                name=f"Ch {ch} ({d_um:.0f}um){' [PEAK]' if is_peak else ''}",
                hoverinfo="x+y+name"
            )
        )

    # Overlay normal spikes
    if len(window_data["spikes_t"]) > 0:
        spk_t = window_data["spikes_t"] - time_offset_s
        spk_y = window_data["spikes_v"] + peak_y_base
        fig.add_trace(
            go.Scatter(
                x=spk_t,
                y=spk_y,
                mode="markers",
                marker=dict(symbol="circle", size=7, color="#ef4444", line=dict(color="#ffffff", width=0.8)),
                name=f"Spikes (Ch {peak_channel})",
                hoverinfo="x+y"
            )
        )

    # Overlay refractory violations (< refractory_ms)
    if len(window_data["viol_t"]) > 0:
        viol_t = window_data["viol_t"] - time_offset_s
        viol_y = window_data["viol_v"] + peak_y_base
        fig.add_trace(
            go.Scatter(
                x=viol_t,
                y=viol_y,
                mode="markers",
                marker=dict(symbol="triangle-down", size=13, color="#f59e0b", line=dict(color="#b45309", width=2.0)),
                name=f"ISI Violations (&lt;1.0ms, N={len(window_data['viol_t'])})",
                hoverinfo="x+y+name"
            )
        )

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=15, color="#1e293b")),
        template="plotly_white",
        height=650,
        margin=dict(l=80, r=40, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif")
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=offsets,
        ticktext=[
            f"<b>Ch {ch}</b> ({d_um:.0f}um){'<br>[PEAK]' if ch == peak_chan else ''}"
            for ch, d_um, _ in chan_depth_pairs
        ],
        title_text="Electrode Contacts"
    )
    fig.update_xaxes(title_text="Recording Time (s)")
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="On-Demand Single-Cluster High-Resolution Trace Inspector (Native 30 kHz, Zero Downsampling)"
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Spike sorting output directory.")
    parser.add_argument("--unit", type=int, required=True, help="Target cluster/unit ID.")
    parser.add_argument("--ap-bin", type=Path, default=None, help="Path to raw AP binary file.")
    parser.add_argument("--t-start", type=float, default=None, help="Start time in seconds (relative to recording snippet).")
    parser.add_argument("--duration", type=float, default=2.0, help="Window duration in seconds (default: 2.0s).")
    parser.add_argument("--violation", type=int, default=None, help="1-indexed violation number to center window on.")
    parser.add_argument("--all-violations", action="store_true", help="Generate multi-panel report of all ISI violations.")
    parser.add_argument("--chunk-browser", action="store_true", help="Generate interactive multi-window chunk browser.")
    parser.add_argument("--refractory-ms", type=float, default=1.0, help="Refractory threshold in ms (default: 1.0).")
    parser.add_argument("--n-channels", type=int, default=8, help="Number of footprint channels to display (default: 8).")
    parser.add_argument("--dest-file", type=Path, default=None, help="Custom output HTML path.")
    parser.add_argument("--open", action="store_true", help="Automatically open generated HTML in browser.")

    args = parser.parse_args()
    out_dir = args.output_dir
    uid = args.unit

    # 1. Locate spike times & clusters
    alf_dir = out_dir / "00_spike_sorting_data" / "alf"
    if not alf_dir.exists():
        alf_dir = out_dir / "alf"

    spk_times = np.load(alf_dir / "spike_times.npy")
    spk_clusts = np.load(alf_dir / "spike_clusters.npy")
    u_spk_times = np.sort(spk_times[spk_clusts == uid])

    if len(u_spk_times) == 0:
        print(f"Error: No spikes found for Unit {uid}!")
        sys.exit(1)

    # 2. Get session time offset
    time_offset_s = 0.0
    cfg_snapshot = out_dir / "00_spike_sorting_data" / "pipeline_config_snapshot.json"
    if cfg_snapshot.exists():
        import json
        with open(cfg_snapshot, "r") as f:
            cfg_dict = json.load(f)
            time_offset_s = float(cfg_dict.get("start_time_s", 0.0))

    u_abs_times = time_offset_s + u_spk_times

    # 3. Locate AP binary
    ap_bin = args.ap_bin
    if ap_bin is None:
        ap_cand = Path("Z:/ramirezlab/pascual/rec_data/LP21_fent_EXPT_g1/LP21_fent_EXPT_g1_imec0/LP21_fent_EXPT_g1_t0.imec0.ap.bin")
        if ap_cand.exists():
            ap_bin = ap_cand
        else:
            print("Error: Could not locate AP binary file. Please pass --ap-bin.")
            sys.exit(1)

    fs_ap = 30000.0
    n_chans_ap = 385
    ap_file_bytes = ap_bin.stat().st_size
    n_ap_samples = ap_file_bytes // (2 * n_chans_ap)
    mm_ap = np.memmap(ap_bin, dtype=np.int16, mode="r", shape=(n_ap_samples, n_chans_ap))

    # Channel geometry
    meta_path = ap_bin.with_suffix(".meta")
    channel_depths = {}
    if meta_path.exists():
        from scripts.visualization.generate_phy_like_cluster_validator import parse_geom_depths_from_meta
        channel_depths = parse_geom_depths_from_meta(meta_path)

    # 4. Footprint discovery
    peak_chan, footprint_chans, channel_ptps, _ = find_unit_channel_footprint(
        mm_ap=mm_ap,
        spike_times_s=u_abs_times,
        fs=fs_ap,
        candidate_channels=None,
        n_footprint_channels=args.n_channels,
        max_sample_spikes=80
    )

    # 5. Find refractory violations (< refractory_ms)
    violations = find_session_refractory_violations(u_spk_times, refractory_ms=args.refractory_ms)
    print(f"Unit {uid:04d} (Peak Ch {peak_chan}): {len(u_spk_times)} total spikes | {len(violations)} refractory violations (<{args.refractory_ms:.1f}ms)")

    # Determine destination directory
    custom_dir = out_dir / "sorting_validation" / "custom_traces"
    custom_dir.mkdir(parents=True, exist_ok=True)

    # Handle Mode 2: Center on Specific Violation
    if args.violation is not None:
        if len(violations) == 0:
            print(f"Unit {uid:04d} has ZERO refractory violations (<{args.refractory_ms:.1f}ms)! No violation to inspect.")
            sys.exit(0)
        v_idx = max(1, min(len(violations), args.violation))
        viol_info = violations[v_idx - 1]
        v_center = viol_info["midpoint_t"]
        t_start = max(float(np.min(u_spk_times)), v_center - 0.5 * args.duration)
        abs_start = time_offset_s + t_start

        win_data = extract_native_trace_window(
            mm_ap=mm_ap,
            channels=footprint_chans,
            channel_depths=channel_depths,
            spike_times_s=u_abs_times,
            peak_channel=peak_chan,
            t_start_s=abs_start,
            duration_s=args.duration,
            fs=fs_ap,
            refractory_ms=args.refractory_ms
        )

        title = f"<b>Unit {uid:04d} &bull; Refractory Violation #{v_idx} of {len(violations)}</b> (ISI = {viol_info['isi_ms']:.2f} ms @ t = {viol_info['spike_1_t']:.3f}s) &bull; Native 30 kHz (Zero Downsampling)"
        fig = build_window_figure(win_data, title, time_offset_s=time_offset_s)

        dest_file = args.dest_file or (custom_dir / f"unit_{uid:04d}_violation_{v_idx:02d}_t{t_start:.1f}s.html")
        fig.write_html(dest_file, include_plotlyjs="cdn")
        print(f"Generated Violation Inspector: file:///{dest_file.resolve().as_posix()}")

        if args.open:
            webbrowser.open(dest_file.resolve().as_uri())
        return

    # Handle Mode 3: All Violations Dashboard
    if args.all_violations:
        if len(violations) == 0:
            print(f"Unit {uid:04d} has ZERO refractory violations (<{args.refractory_ms:.1f}ms)! Nothing to display.")
            sys.exit(0)

        from plotly.subplots import make_subplots
        n_v = min(len(violations), 10)  # Top 10 violations
        fig = make_subplots(
            rows=n_v, cols=1,
            subplot_titles=[
                f"<b>Violation #{v['index']}</b> (ISI = {v['isi_ms']:.2f} ms, t = {v['spike_1_t']:.3f}s)"
                for v in violations[:n_v]
            ],
            vertical_spacing=max(0.04, 0.6 / n_v)
        )

        trace_spacing_uv = 160.0
        offsets = np.arange(len(footprint_chans)) * trace_spacing_uv
        chan_depth_pairs = [(ch, channel_depths.get(ch, (ch // 2) * 15.0), i) for i, ch in enumerate(footprint_chans)]
        chan_depth_pairs.sort(key=lambda x: x[1])

        for r_idx, v in enumerate(violations[:n_v]):
            v_center = v["midpoint_t"]
            dur = 1.0  # 1.0s window per violation
            t_s = max(float(np.min(u_spk_times)), v_center - 0.5 * dur)
            abs_s = time_offset_s + t_s

            w_data = extract_native_trace_window(
                mm_ap=mm_ap,
                channels=footprint_chans,
                channel_depths=channel_depths,
                spike_times_s=u_abs_times,
                peak_channel=peak_chan,
                t_start_s=abs_s,
                duration_s=dur,
                fs=fs_ap,
                refractory_ms=args.refractory_ms
            )
            t_ax = w_data["t_full"] - time_offset_s
            p_base = 0.0
            for plot_idx, (ch, d_um, orig_idx) in enumerate(chan_depth_pairs):
                y_base = offsets[plot_idx]
                y_tr = w_data["filt_traces"][orig_idx] + y_base
                is_p = (ch == peak_chan)
                if is_p:
                    p_base = y_base
                fig.add_trace(
                    go.Scattergl(
                        x=t_ax, y=y_tr, mode="lines",
                        line=dict(color="#2563eb" if is_p else "#94a3b8", width=1.5 if is_p else 0.8),
                        showlegend=(r_idx == 0)
                    ),
                    row=r_idx + 1, col=1
                )

            # Spikes
            if len(w_data["spikes_t"]) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=w_data["spikes_t"] - time_offset_s,
                        y=w_data["spikes_v"] + p_base,
                        mode="markers",
                        marker=dict(symbol="circle", size=6, color="#ef4444"),
                        showlegend=(r_idx == 0),
                        name="Spikes"
                    ),
                    row=r_idx + 1, col=1
                )
            # Violations
            if len(w_data["viol_t"]) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=w_data["viol_t"] - time_offset_s,
                        y=w_data["viol_v"] + p_base,
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=12, color="#f59e0b", line=dict(color="#b45309", width=2)),
                        showlegend=(r_idx == 0),
                        name="ISI Violations"
                    ),
                    row=r_idx + 1, col=1
                )

        fig.update_layout(
            title=f"<b>Unit {uid:04d} &bull; All Refractory Period Violations ({len(violations)} Total)</b> &bull; Native 30 kHz (Zero Downsampling)",
            template="plotly_white",
            height=320 * n_v,
            margin=dict(l=70, r=40, t=70, b=50)
        )
        dest_file = args.dest_file or (custom_dir / f"unit_{uid:04d}_all_violations.html")
        fig.write_html(dest_file, include_plotlyjs="cdn")
        print(f"Generated All Violations Report: file:///{dest_file.resolve().as_posix()}")

        if args.open:
            webbrowser.open(dest_file.resolve().as_uri())
        return

    # Handle Mode 4: Chunk Browser (Page Through Windows)
    if args.chunk_browser:
        t_min = float(np.min(u_spk_times))
        t_max = float(np.max(u_spk_times))
        chunk_len = 2.0
        n_chunks = int(np.ceil((t_max - t_min) / chunk_len))
        n_chunks = min(n_chunks, 12)  # up to 12 chunks (24 seconds)

        print(f"Building interactive multi-chunk browser with {n_chunks} un-downsampled 2.0s windows...")
        fig = go.Figure()
        buttons = []

        total_traces_per_chunk = len(footprint_chans) + 2  # traces + normal spikes + viol spikes
        for c_idx in range(n_chunks):
            t_s = t_min + c_idx * chunk_len
            abs_s = time_offset_s + t_s
            w_data = extract_native_trace_window(
                mm_ap=mm_ap,
                channels=footprint_chans,
                channel_depths=channel_depths,
                spike_times_s=u_abs_times,
                peak_channel=peak_chan,
                t_start_s=abs_s,
                duration_s=chunk_len,
                fs=fs_ap,
                refractory_ms=args.refractory_ms
            )

            t_ax = w_data["t_full"] - time_offset_s
            chan_depth_pairs = [(ch, channel_depths.get(ch, (ch // 2) * 15.0), i) for i, ch in enumerate(footprint_chans)]
            chan_depth_pairs.sort(key=lambda x: x[1])
            offsets = np.arange(len(footprint_chans)) * 180.0

            p_base = 0.0
            is_visible = (c_idx == 0)
            for plot_idx, (ch, d_um, orig_idx) in enumerate(chan_depth_pairs):
                y_base = offsets[plot_idx]
                y_tr = w_data["filt_traces"][orig_idx] + y_base
                is_p = (ch == peak_chan)
                if is_p:
                    p_base = y_base
                fig.add_trace(
                    go.Scattergl(
                        x=t_ax, y=y_tr, mode="lines",
                        line=dict(color="#2563eb" if is_p else "#64748b", width=1.8 if is_p else 0.9),
                        name=f"Ch {ch}",
                        visible=is_visible,
                        showlegend=(plot_idx == 0 and c_idx == 0)
                    )
                )

            # Spikes
            spk_t = w_data["spikes_t"] - time_offset_s if len(w_data["spikes_t"]) > 0 else []
            spk_y = w_data["spikes_v"] + p_base if len(w_data["spikes_v"]) > 0 else []
            fig.add_trace(
                go.Scatter(
                    x=spk_t, y=spk_y, mode="markers",
                    marker=dict(symbol="circle", size=7, color="#ef4444"),
                    name="Spikes",
                    visible=is_visible
                )
            )

            # Violations
            viol_t = w_data["viol_t"] - time_offset_s if len(w_data["viol_t"]) > 0 else []
            viol_y = w_data["viol_v"] + p_base if len(w_data["viol_v"]) > 0 else []
            fig.add_trace(
                go.Scatter(
                    x=viol_t, y=viol_y, mode="markers",
                    marker=dict(symbol="triangle-down", size=13, color="#f59e0b", line=dict(color="#b45309", width=2)),
                    name="Violations",
                    visible=is_visible
                )
            )

        # Build dropdown buttons
        n_traces_total = len(fig.data)
        traces_per_chunk = n_traces_total // n_chunks
        for c_idx in range(n_chunks):
            t_s = t_min + c_idx * chunk_len
            t_e = t_s + chunk_len
            vis = [False] * n_traces_total
            for t_idx in range(c_idx * traces_per_chunk, (c_idx + 1) * traces_per_chunk):
                vis[t_idx] = True

            buttons.append(dict(
                label=f"Window {c_idx + 1} ({t_s:.1f}s - {t_e:.1f}s)",
                method="update",
                args=[
                    {"visible": vis},
                    {"title": f"<b>Unit {uid:04d} Multi-Window Browser</b> &bull; Window {c_idx + 1}: {t_s:.1f}s &ndash; {t_e:.1f}s (Native 30 kHz, Zero Downsampling)"}
                ]
            ))

        fig.update_layout(
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                x=0.0, y=1.12,
                xanchor="left", yanchor="top",
                direction="down",
                showactive=True
            )],
            title=f"<b>Unit {uid:04d} Multi-Window Browser</b> &bull; Window 1: {t_min:.1f}s &ndash; {t_min + chunk_len:.1f}s (Native 30 kHz, Zero Downsampling)",
            template="plotly_white",
            height=680,
            margin=dict(l=80, r=40, t=90, b=50),
            font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif")
        )
        chan_depth_pairs = [(ch, channel_depths.get(ch, (ch // 2) * 15.0), i) for i, ch in enumerate(footprint_chans)]
        chan_depth_pairs.sort(key=lambda x: x[1])
        offsets = np.arange(len(footprint_chans)) * 180.0
        fig.update_yaxes(
            tickmode="array",
            tickvals=offsets,
            ticktext=[
                f"<b>Ch {ch}</b> ({d_um:.0f}um){'<br>[PEAK]' if ch == peak_chan else ''}"
                for ch, d_um, _ in chan_depth_pairs
            ],
            title_text="Electrode Contacts"
        )
        fig.update_xaxes(title_text="Recording Time (s)")

        dest_file = args.dest_file or (custom_dir / f"unit_{uid:04d}_chunk_browser.html")
        fig.write_html(dest_file, include_plotlyjs="cdn")
        print(f"Generated Multi-Window Chunk Browser: file:///{dest_file.resolve().as_posix()}")

        if args.open:
            webbrowser.open(dest_file.resolve().as_uri())
        return

    # Mode 1: Custom Time Window
    if args.t_start is None:
        # Default to highest firing rate burst window if not provided
        t_min = float(np.min(u_spk_times))
        t_max = float(np.max(u_spk_times))
        eval_times = np.arange(t_min, max(t_min + 0.5, t_max - args.duration), 0.25)
        if len(eval_times) > 0:
            counts = [np.sum((u_spk_times >= t) & (u_spk_times < t + args.duration)) for t in eval_times]
            t_start = float(eval_times[int(np.argmax(counts))])
        else:
            t_start = t_min
    else:
        t_start = args.t_start

    abs_start = time_offset_s + t_start
    win_data = extract_native_trace_window(
        mm_ap=mm_ap,
        channels=footprint_chans,
        channel_depths=channel_depths,
        spike_times_s=u_abs_times,
        peak_channel=peak_chan,
        t_start_s=abs_start,
        duration_s=args.duration,
        fs=fs_ap,
        refractory_ms=args.refractory_ms
    )

    t_end = t_start + args.duration
    title = f"<b>Unit {uid:04d} Voltage Trace Window ({t_start:.2f}s &ndash; {t_end:.2f}s, {args.duration:.1f}s Window)</b> &bull; Native 30 kHz (Zero Downsampling) &bull; Peak Ch {peak_chan}"
    fig = build_window_figure(win_data, title, time_offset_s=time_offset_s)

    dest_file = args.dest_file or (custom_dir / f"unit_{uid:04d}_window_{t_start:.1f}s_{args.duration:.1f}s.html")
    fig.write_html(dest_file, include_plotlyjs="cdn")
    print(f"Generated Custom Trace Window: file:///{dest_file.resolve().as_posix()}")

    if args.open:
        webbrowser.open(dest_file.resolve().as_uri())


if __name__ == "__main__":
    main()
