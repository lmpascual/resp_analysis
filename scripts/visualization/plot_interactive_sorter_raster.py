#!/usr/bin/env python3
"""
Interactive Population Raster & Diagnostic Dashboard for QC-Passed Sorter Units
Supports MountainSort 5, Lupin, Kilosort 4, SpyKING-CIRCUS 2, Tridesclous 2.
Generates:
1. Plotly Interactive WebGL Raster (Depth vs Time & PSTH)
2. Standalone Web Dashboard with Real-Time Filters (Depth, Rate, SNR) and Single-Unit Inspector
3. High-Resolution PNG Raster Preview
"""
import sys
import json
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
orig_np_solve = np.linalg.solve
def safe_solve(a, b):
    try:
        t_a = torch.as_tensor(a, dtype=torch.float64)
        t_b = torch.as_tensor(b, dtype=torch.float64)
        return torch.linalg.solve(t_a, t_b).numpy()
    except Exception:
        return orig_np_solve(a, b)
np.linalg.solve = safe_solve

orig_np_norm = np.linalg.norm
def safe_norm(x, ord=None, axis=None, keepdims=False):
    try:
        t = torch.as_tensor(x)
        if axis is None and not keepdims:
            return float(torch.linalg.norm(t, ord=ord))
        return torch.linalg.norm(t, ord=ord, dim=axis, keepdim=keepdims).numpy()
    except Exception:
        return orig_np_norm(x, ord=ord, axis=axis, keepdims=keepdims)
np.linalg.norm = safe_norm

orig_np_lstsq = np.linalg.lstsq
def safe_lstsq(a, b, rcond=None):
    try:
        t_a = torch.as_tensor(a, dtype=torch.float64)
        t_b = torch.as_tensor(b, dtype=torch.float64)
        res = torch.linalg.lstsq(t_a, t_b)
        x = res.solution.numpy()
        residuals = res.residuals.numpy() if res.residuals.numel() > 0 else np.array([])
        rank = int(res.rank)
        s = res.singular_values.numpy()
        return x, residuals, rank, s
    except Exception:
        return orig_np_lstsq(a, b, rcond=rcond)

np.linalg.lstsq = safe_lstsq
try:
    import numpy.linalg as nla
    nla.lstsq = safe_lstsq
    import numpy.lib._polynomial_impl as npi
    npi.lstsq = safe_lstsq
except Exception:
    pass

try:
    import matplotlib.transforms as mtransforms
    def safe_transform_dot(a, b):
        t_a = torch.as_tensor(a, dtype=torch.float64)
        t_b = torch.as_tensor(b, dtype=torch.float64)
        return (t_a @ t_b).numpy()
    mtransforms.np.dot = safe_transform_dot
    mtransforms.inv = lambda m: torch.linalg.inv(torch.as_tensor(m, dtype=torch.float64)).numpy()
except Exception:
    pass

try:
    import matplotlib.bezier as mb
    orig_bezier_call = mb.BezierSegment.__call__
    def safe_bezier_call(self, t):
        try:
            t = np.asarray(t)
            basis = (np.power.outer(1 - t, self._orders[::-1]) * np.power.outer(t, self._orders))
            t_basis = torch.as_tensor(basis, dtype=torch.float64)
            t_px = torch.as_tensor(self._px, dtype=torch.float64)
            return (t_basis @ t_px).numpy()
        except Exception:
            return orig_bezier_call(self, t)
    mb.BezierSegment.__call__ = safe_bezier_call
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("INTERACTIVE_SORTER_RASTER")

def build_interactive_sorter_raster(sorter_dir: Path, sorter_name: str = None, bin_size_ms: float = 20.0):
    sorter_dir = Path(sorter_dir)
    if not sorter_dir.exists():
        p_str = str(sorter_dir).replace('"', '').replace("'", "")
        # Try with literal single quotes around output folder
        for cand in [
            Path(p_str),
            Path(r"C:\Users\lpascu\analysis_src") / p_str,
            Path(r"C:\Users\lpascu\analysis_src") / "'output_LP21_60s_120s'" / "sliding_window_pipeline" / "lupin" / "consensus_master",
            Path(r"C:\Users\lpascu\analysis_src\output_LP21_60s_120s\sliding_window_pipeline\lupin\consensus_master")
        ]:
            if cand.exists():
                sorter_dir = cand
                break

    if sorter_name is None:
        sorter_name = sorter_dir.name
        
    sorter_name_clean = sorter_name.replace(" ", "_").lower()
    
    # Locate QC summary or cluster_info.tsv
    qc_csv = None
    for candidate in [
        sorter_dir / f"{sorter_name_clean}_qc_summary.csv",
        sorter_dir / "cluster_info.tsv",
        sorter_dir / "qc_summary.csv"
    ]:
        if candidate.exists():
            qc_csv = candidate
            break
            
    if qc_csv is None:
        # Check any csv in folder
        csvs = list(sorter_dir.glob("*qc*.csv"))
        if csvs:
            qc_csv = csvs[0]
        else:
            raise FileNotFoundError(f"Could not locate QC summary or cluster_info.tsv in {sorter_dir}")

    # Load metrics
    sep = "\t" if qc_csv.suffix == ".tsv" else ","
    df_metrics = pd.read_csv(qc_csv, sep=sep)
    
    # Filter for QC passed single units
    qc_col = None
    if "passed_qc" in df_metrics.columns:
        qc_col = "passed_qc"
    elif "passed_allen_qc" in df_metrics.columns:
        qc_col = "passed_allen_qc"

    if qc_col is not None:
        df_qc = df_metrics[df_metrics[qc_col] == True].copy()
        logger.info(f"Filtering for QC-passed single units using column '{qc_col}': {len(df_qc)} / {len(df_metrics)} units kept.")
    else:
        df_qc = df_metrics.copy()

    if len(df_qc) == 0:
        logger.warning(f"No units passed QC filter in {sorter_name}. Using units with firing_rate >= 0.1 Hz.")
        df_qc = df_metrics[df_metrics["firing_rate_hz"] >= 0.1].copy()

    # Sort by probe depth (ascending)
    if "depth_um" in df_qc.columns:
        df_qc = df_qc.sort_values(by="depth_um", ascending=True).reset_index(drop=True)
    qc_unit_ids = set(df_qc["cluster_id"].values)

    # Load spikes
    alf_dir = sorter_dir / "alf"
    if not (alf_dir / "spike_times.npy").exists():
        raise FileNotFoundError(f"Missing spike_times.npy in {alf_dir}")

    times = np.load(alf_dir / "spike_times.npy")
    clusters = np.load(alf_dir / "spike_clusters.npy")

    # Filter spikes belonging to QC units
    mask = np.isin(clusters, list(qc_unit_ids))
    qc_times = times[mask]
    qc_clusters = clusters[mask]

    unit_map = {row["cluster_id"]: row for _, row in df_qc.iterrows()}
    unit_depths = np.array([unit_map[c]["depth_um"] if "depth_um" in unit_map[c] else 0.0 for c in qc_clusters])
    unit_rates = np.array([unit_map[c]["firing_rate_hz"] if "firing_rate_hz" in unit_map[c] else 1.0 for c in qc_clusters])

    # Determine recording duration
    duration_s = float(np.max(qc_times)) if len(qc_times) > 0 else 60.0

    # Locate NIDAQ flow rate, EMG, and ECG data if available
    flow_times, flow_volts = None, None
    laser_times, laser_volts = None, None
    emg_times, emg_volts = None, None
    ecg_times, ecg_volts = None, None
    emg_chan = 0   # XA0 diaphragm EMG channel
    ecg_chan = 2 if "e097" in str(sorter_dir).lower() else 1   # XA2 for e097, XA1 for LP21
    flow_chan = 5  # XA5 zero-centered flow rate channel
    laser_chan = 6 if "e097" in str(sorter_dir).lower() else None  # XA6 opto photostimulation for e097
    
    # Check possible locations for NIDAQ binary
    nidq_candidates = [
        Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin"),
        sorter_dir.parent / "raw_ephys_data" / "LP21_fent_EXPT_g1_t0.nidq.bin",
        Path(r"Z:\ramirezlab\pascual\rec_data\LP21_fent_EXPT_g1\LP21_fent_EXPT_g1_t0.nidq.bin"),
        Path(r"Z:\ramirezlab\pascual\rec_data\LP21_fent_EXPT_g1\LP21_fent_EXPT_g1_imec0\LP21_fent_EXPT_g1_t0.nidq.bin"),
    ]
    
    # Also check if pipeline config snapshot exists in sorter_dir or parent
    for cfg_cand in [sorter_dir / "pipeline_config_snapshot.json", sorter_dir.parent / "pipeline_config_snapshot.json", Path("pipeline_config_e097_prelaser_60s.json")]:
        if cfg_cand.exists():
            try:
                import json
                with open(cfg_cand, "r") as f_cfg:
                    cfg_data = json.load(f_cfg)
                    raw_dir_cfg = Path(cfg_data.get("raw_data_dir", ""))
                    if raw_dir_cfg.exists():
                        for f_bin in list(raw_dir_cfg.glob("*.nidq.bin")) + list(raw_dir_cfg.parent.glob("*.nidq.bin")):
                            nidq_candidates.insert(0, f_bin)
            except Exception:
                pass
    
    nidq_file = None
    for cand in nidq_candidates:
        if cand.exists():
            nidq_file = cand
            break
            
    if nidq_file is not None:
        try:
            logger.info(f"Loading NIDAQ Flow Rate (Ch {flow_chan}) & EMG (Ch {emg_chan}) from: {nidq_file}")
            n_ch = 9
            fs_nidq = 10000.0
            
            # Synchronize start time offset if sliced run
            if "e097" in str(sorter_dir).lower() or "e097" in str(nidq_file).lower():
                start_t_s = 796.827
            elif "60s_120s" in str(sorter_dir) or "60s_120s" in str(sorter_dir.parent):
                start_t_s = 60.0
            else:
                start_t_s = 0.0
            start_samp = int(start_t_s * fs_nidq)
            n_samp_total = int(duration_s * fs_nidq)
            
            raw_nidq = np.memmap(nidq_file, dtype="int16", mode="r")
            avail_samps = len(raw_nidq) // n_ch
            end_samp = min(avail_samps, start_samp + n_samp_total)
            
            nidq_slice_raw = raw_nidq[start_samp * n_ch : end_samp * n_ch].reshape(-1, n_ch)
            
            from scipy import signal as sp_signal
            from scipy.ndimage import median_filter

            # 1. Flow Rate (Downsample to 500 Hz)
            v_flow_raw = nidq_slice_raw[:, flow_chan].astype(np.float32) * (10.0 / 32768.0)
            ds_flow = 20  # 10 kHz -> 500 Hz
            flow_volts = v_flow_raw[::ds_flow]
            flow_times = (np.arange(len(flow_volts)) * ds_flow) / fs_nidq

            # 2. Raw Diaphragm EMG (Bandpass 300-5000 Hz & Downsample to 1000 Hz)
            v_emg_raw = nidq_slice_raw[:, emg_chan].astype(np.float32) * (10.0 / 32768.0)
            b_emg, a_emg = sp_signal.butter(2, [300.0, min(4900.0, fs_nidq * 0.48)], btype='bandpass', fs=fs_nidq)
            v_emg_filt = sp_signal.filtfilt(b_emg, a_emg, v_emg_raw)
            ds_emg = 10  # 10 kHz -> 1000 Hz
            emg_volts = v_emg_filt[::ds_emg]
            emg_times = (np.arange(len(emg_volts)) * ds_emg) / fs_nidq
            
            # 3. Integrated Diaphragm EMG Envelope (50 ms Median Filter + 10 ms Savitzky-Golay)
            rectified = np.abs(v_emg_filt)
            med_win = int(0.050 * fs_nidq)
            if med_win % 2 == 0:
                med_win += 1
            env_med = median_filter(rectified, size=med_win)
            savgol_win = int(0.010 * fs_nidq)
            if savgol_win % 2 == 0:
                savgol_win += 1
            emg_env_full = sp_signal.savgol_filter(env_med, window_length=savgol_win, polyorder=1)
            int_emg_volts = emg_env_full[::ds_emg]
            int_emg_times = emg_times
            
            # 4. Cardiac ECG (Bandpass 5-100 Hz, 60 Hz notch & Downsample to 1000 Hz)
            v_ecg_raw = nidq_slice_raw[:, ecg_chan].astype(np.float32) * (10.0 / 32768.0)
            b_notch, a_notch = sp_signal.iirnotch(60.0, 30.0, fs=fs_nidq)
            v_ecg_notched = sp_signal.filtfilt(b_notch, a_notch, v_ecg_raw)
            b_ecg, a_ecg = sp_signal.butter(2, [5.0, 100.0], btype='bandpass', fs=fs_nidq)
            v_ecg_filt = sp_signal.filtfilt(b_ecg, a_ecg, v_ecg_notched)
            ds_ecg = 10  # 10 kHz -> 1000 Hz
            ecg_volts = v_ecg_filt[::ds_ecg]
            ecg_times = (np.arange(len(ecg_volts)) * ds_ecg) / fs_nidq
            
            # 5. Laser Photostimulation Channel (Opto Ch 6)
            if laser_chan is not None and laser_chan < n_ch:
                v_laser_raw = nidq_slice_raw[:, laser_chan].astype(np.float32) * (10.0 / 32768.0)
                ds_laser = 5  # 10 kHz -> 2000 Hz (preserves 5 ms pulses with 10 samples per pulse)
                laser_volts = v_laser_raw[::ds_laser]
                laser_times = (np.arange(len(laser_volts)) * ds_laser) / fs_nidq
                logger.info(f"Extracted {len(laser_volts):,} Laser Stim points from Channel {laser_chan} (peak={laser_volts.max():.2f}V).")

            logger.info(f"Extracted {len(flow_volts):,} Flow, {len(emg_volts):,} Raw EMG, {len(int_emg_volts):,} Int. EMG, & {len(ecg_volts):,} Cardiac ECG points across slice {start_t_s:.1f}s to {start_t_s + duration_s:.1f}s.")
        except Exception as e:
            logger.warning(f"Could not load NIDAQ signals: {e}")

    has_flow = flow_volts is not None and len(flow_volts) > 0
    has_laser = laser_volts is not None and len(laser_volts) > 0 and np.max(laser_volts) > 0.3
    has_emg = emg_volts is not None and len(emg_volts) > 0
    has_int_emg = 'int_emg_volts' in locals() and int_emg_volts is not None and len(int_emg_volts) > 0
    has_ecg = ecg_volts is not None and len(ecg_volts) > 0

    # Output paths
    out_html_path = sorter_dir / f"{sorter_name_clean}_qc_interactive_raster.html"
    dashboard_html_path = sorter_dir / f"{sorter_name_clean}_qc_interactive_dashboard.html"
    preview_png_path = sorter_dir / f"{sorter_name_clean}_qc_raster_preview.png"

    # Synchronized Subplots:
    # 6-Panel (or 5-Panel): Flow Rate -> Raw Diaphragm EMG -> Integrated Diaphragm EMG -> Cardiac ECG -> Population PSTH -> Spike Raster
    n_psth_bins = max(int(np.ceil((duration_s * 1000.0) / bin_size_ms)), 1)
    psth_bins = np.linspace(0, duration_s, n_psth_bins + 1)
    psth_counts, _ = np.histogram(qc_times, bins=psth_bins)
    psth_rate_hz = psth_counts / (bin_size_ms / 1000.0)
    psth_centers = (psth_bins[:-1] + psth_bins[1:]) / 2.0

    if has_flow and has_laser and has_emg and has_int_emg and has_ecg:
        fig = make_subplots(
            rows=7, cols=1,
            row_heights=[0.07, 0.07, 0.07, 0.07, 0.07, 0.09, 0.56],
            shared_xaxes=True,
            vertical_spacing=0.020,
            subplot_titles=(
                "Respiratory Flow Rate (Plethysmography Airflow)",
                "Laser Photostimulation (Ch 6 Opto — 20 Hz, 5 ms Pulses)",
                "Raw Diaphragm EMG (High-Pass Filtered Motor Drive)",
                "Integrated Diaphragm EMG Envelope (Smooth Inspiratory Burst)",
                "Cardiac Electrocardiogram (Filtered ECG - QRS Complexes & Heart Rate)",
                f"Population Instantaneous Firing Rate ({bin_size_ms:.0f} ms bins)",
                f"Depth-Resolved Single-Unit Spike Raster ({len(df_qc)} QC-Passed Units)"
            )
        )

        # 1. Row 1: Flow Rate Trace
        fig.add_trace(go.Scattergl(x=flow_times, y=flow_volts, mode="lines", line=dict(color="#059669", width=1.5), name="Airflow (V)", hoverinfo="x+y"), row=1, col=1)
        # 2. Row 2: Laser Photostimulation Trace
        fig.add_trace(go.Scattergl(x=laser_times, y=laser_volts, mode="lines", line=dict(color="#f59e0b", width=1.5), name="Laser Opto (V)", hoverinfo="x+y"), row=2, col=1)
        # 3. Row 3: Raw Diaphragm EMG Trace
        fig.add_trace(go.Scattergl(x=emg_times, y=emg_volts, mode="lines", name="Diaphragm EMG (V)", line=dict(color="#a29bfe", width=1.2)), row=3, col=1)
        # 4. Row 4: Integrated Diaphragm EMG Envelope Trace
        fig.add_trace(go.Scattergl(x=int_emg_times, y=int_emg_volts, mode="lines", name="Integrated Diaphragm (V)", line=dict(color="#0284c7", width=1.8)), row=4, col=1)
        # 5. Row 5: Cardiac ECG Trace
        fig.add_trace(go.Scattergl(x=ecg_times, y=ecg_volts, mode="lines", name="Cardiac ECG (V)", line=dict(color="#f43f5e", width=1.3)), row=5, col=1)
        # 6. Row 6: Population PSTH
        fig.add_trace(go.Scatter(x=psth_centers, y=psth_rate_hz, mode="lines", name="Population Rate", line=dict(color="#00cec9", width=2.0), fill="tozeroy", fillcolor="rgba(0, 206, 201, 0.18)"), row=6, col=1)

        # 7. Row 7: Spike Raster
        hover_text = [f"Cluster ID: #{c}<br>Depth: {d:.1f} µm<br>Unit FR: {fr:.2f} Hz" for c, d, fr in zip(qc_clusters, unit_depths, unit_rates)]
        fig.add_trace(go.Scattergl(x=qc_times, y=unit_depths, mode="markers", name="QC Spikes", marker=dict(size=3.5, color=unit_rates, colorscale="Viridis", showscale=True, colorbar=dict(title="FR (Hz)", thickness=14, len=0.50, y=0.25, x=1.02), opacity=0.85), text=hover_text), row=7, col=1)

        fig.update_yaxes(title_text="Flow (V)", row=1, col=1, gridcolor="#2d3436")
        fig.update_yaxes(title_text="Laser (V)", row=2, col=1, gridcolor="#2d3436")
        fig.update_yaxes(title_text="EMG (V)", row=3, col=1, gridcolor="#2d3436")
        fig.update_yaxes(title_text="Int. EMG (V)", row=4, col=1, gridcolor="#2d3436")
        fig.update_yaxes(title_text="ECG (V)", row=5, col=1, gridcolor="#2d3436")
        fig.update_yaxes(title_text="Rate (Hz)", row=6, col=1, gridcolor="#2d3436")
        fig.update_yaxes(title_text="Probe Depth (µm)", row=7, col=1, gridcolor="#2d3436")
        fig.update_xaxes(title_text="Recording Time (seconds)", row=7, col=1, gridcolor="#2d3436")

        # Highlight the 3 laser train periods across all rows
        for (l_s, l_e) in [(60.0, 61.955), (119.664, 121.619), (164.288, 166.243)]:
            fig.add_vrect(
                x0=l_s, x1=l_e,
                fillcolor="#f59e0b", opacity=0.12,
                layer="below", line_width=0,
                annotation_text="Laser Stim", annotation_position="top left",
                annotation_font=dict(color="#f59e0b", size=9)
            )

    elif has_flow and has_emg and has_int_emg and has_ecg:
        fig = make_subplots(
            rows=6, cols=1,
            row_heights=[0.08, 0.08, 0.08, 0.08, 0.10, 0.58],
            shared_xaxes=True,
            vertical_spacing=0.022,
            subplot_titles=(
                "Respiratory Flow Rate (Plethysmography Airflow)",
                "Raw Diaphragm EMG (High-Pass Filtered Motor Drive)",
                "Integrated Diaphragm EMG Envelope (Smooth Inspiratory Burst)",
                "Cardiac Electrocardiogram (Filtered ECG - QRS Complexes & Heart Rate)",
                f"Population Instantaneous Firing Rate ({bin_size_ms:.0f} ms bins)",
                f"Depth-Resolved Single-Unit Spike Raster ({len(df_qc)} QC-Passed Units)"
            )
        )

        # 1. Row 1: Flow Rate Trace
        fig.add_trace(
            go.Scattergl(
                x=flow_times,
                y=flow_volts,
                mode="lines",
                line=dict(color="#059669", width=1.5),
                name="Airflow (V)",
                hoverinfo="x+y"
            ),
            row=1, col=1
        )

        # 2. Row 2: Raw Diaphragm EMG Trace
        fig.add_trace(
            go.Scattergl(
                x=emg_times,
                y=emg_volts,
                mode="lines",
                name="Diaphragm EMG (V)",
                line=dict(color="#a29bfe", width=1.2),
                hovertemplate="Time: %{x:.3f} s<br>EMG: %{y:.2f} V<extra></extra>"
            ),
            row=2, col=1
        )

        # 3. Row 3: Integrated Diaphragm EMG Envelope Trace
        fig.add_trace(
            go.Scattergl(
                x=int_emg_times,
                y=int_emg_volts,
                mode="lines",
                name="Integrated Diaphragm (V)",
                line=dict(color="#0284c7", width=1.8),
                hovertemplate="Time: %{x:.3f} s<br>Envelope: %{y:.2f} V<extra></extra>"
            ),
            row=3, col=1
        )

        # 4. Row 4: Cardiac ECG Trace
        fig.add_trace(
            go.Scattergl(
                x=ecg_times,
                y=ecg_volts,
                mode="lines",
                name="Cardiac ECG (V)",
                line=dict(color="#f43f5e", width=1.3),
                hovertemplate="Time: %{x:.3f} s<br>ECG: %{y:.2f} V<extra></extra>"
            ),
            row=4, col=1
        )

        # 5. Row 5: Population PSTH
        fig.add_trace(
            go.Scatter(
                x=psth_centers,
                y=psth_rate_hz,
                mode="lines",
                name="Population Rate",
                line=dict(color="#00cec9", width=2.0),
                fill="tozeroy",
                fillcolor="rgba(0, 206, 201, 0.18)",
                hovertemplate="Time: %{x:.2f} s<br>Rate: %{y:.1f} Hz<extra></extra>"
            ),
            row=5, col=1
        )

        # 6. Row 6: Spike Raster
        hover_text = [
            f"Cluster ID: #{c}<br>Depth: {d:.1f} µm<br>Unit FR: {fr:.2f} Hz"
            for c, d, fr in zip(qc_clusters, unit_depths, unit_rates)
        ]

        fig.add_trace(
            go.Scattergl(
                x=qc_times,
                y=unit_depths,
                mode="markers",
                name="QC Spikes",
                marker=dict(
                    size=3.5,
                    color=unit_rates,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(
                        title="FR (Hz)",
                        thickness=14,
                        len=0.50,
                        y=0.25,
                        x=1.02
                    ),
                    opacity=0.85
                ),
                text=hover_text,
                hovertemplate="Time: %{x:.3f} s<br>Depth: %{y:.1f} µm<br>%{text}<extra></extra>"
            ),
            row=6, col=1
        )

        fig.update_yaxes(title_text="Flow (V)", row=1, col=1, gridcolor="#2d3436")
        fig.update_yaxes(title_text="EMG (V)", row=2, col=1, gridcolor="#2d3436")
        fig.update_yaxes(title_text="Int. EMG (V)", row=3, col=1, gridcolor="#2d3436")
        fig.update_yaxes(title_text="ECG (V)", row=4, col=1, gridcolor="#2d3436")
        fig.update_yaxes(title_text="Rate (Hz)", row=5, col=1, gridcolor="#2d3436")
        fig.update_yaxes(title_text="Probe Depth (µm)", row=6, col=1, gridcolor="#2d3436")
        fig.update_xaxes(title_text="Recording Time (seconds)", row=6, col=1, gridcolor="#2d3436")

    elif has_flow and has_emg and has_int_emg:
        fig = make_subplots(
            rows=5, cols=1,
            row_heights=[0.11, 0.11, 0.11, 0.13, 0.54],
            shared_xaxes=True,
            vertical_spacing=0.025,
            subplot_titles=(
                "Respiratory Flow Rate (Plethysmography Airflow)",
                "Raw Diaphragm EMG (High-Pass Filtered Motor Drive)",
                "Integrated Diaphragm EMG Envelope (Smooth Inspiratory Burst)",
                f"Population Instantaneous Firing Rate ({bin_size_ms:.0f} ms bins)",
                f"Depth-Resolved Single-Unit Spike Raster ({len(df_qc)} QC-Passed Units)"
            )
        )

        fig.add_trace(go.Scattergl(x=flow_times, y=flow_volts, mode="lines", line=dict(color="#059669", width=1.5), name="Airflow (V)"), row=1, col=1)
        fig.add_trace(go.Scattergl(x=emg_times, y=emg_volts, mode="lines", name="Diaphragm EMG (V)", line=dict(color="#a29bfe", width=1.2)), row=2, col=1)
        fig.add_trace(go.Scattergl(x=int_emg_times, y=int_emg_volts, mode="lines", name="Integrated Diaphragm (V)", line=dict(color="#0284c7", width=1.8)), row=3, col=1)
        fig.add_trace(go.Scatter(x=psth_centers, y=psth_rate_hz, mode="lines", name="Population Rate", line=dict(color="#00cec9", width=2.0), fill="tozeroy", fillcolor="rgba(0, 206, 201, 0.18)"), row=4, col=1)

        hover_text = [f"Cluster ID: #{c}<br>Depth: {d:.1f} µm<br>Unit FR: {fr:.2f} Hz" for c, d, fr in zip(qc_clusters, unit_depths, unit_rates)]
        fig.add_trace(go.Scattergl(x=qc_times, y=unit_depths, mode="markers", name="QC Spikes", marker=dict(size=3.5, color=unit_rates, colorscale="Viridis", showscale=True, colorbar=dict(title="FR (Hz)", thickness=14, len=0.50, y=0.25, x=1.02), opacity=0.85), text=hover_text), row=5, col=1)

        fig.update_yaxes(title_text="Flow (V)", row=1, col=1, gridcolor="#2d3436")
        fig.update_yaxes(title_text="EMG (V)", row=2, col=1, gridcolor="#2d3436")
        fig.update_yaxes(title_text="Int. EMG (V)", row=3, col=1, gridcolor="#2d3436")
        fig.update_yaxes(title_text="Rate (Hz)", row=4, col=1, gridcolor="#2d3436")
        fig.update_yaxes(title_text="Probe Depth (µm)", row=5, col=1, gridcolor="#2d3436")
        fig.update_xaxes(title_text="Recording Time (seconds)", row=5, col=1, gridcolor="#2d3436")

    else:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.22, 0.78],
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                f"Population Instantaneous Firing Rate ({bin_size_ms:.0f} ms bins)",
                f"Depth-Resolved Single-Unit Spike Raster ({len(df_qc)} QC-Passed Units)"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=psth_centers,
                y=psth_rate_hz,
                mode="lines",
                name="Population Rate",
                line=dict(color="#38bdf8", width=2.0),
                fill="tozeroy",
                fillcolor="rgba(56, 189, 248, 0.15)",
                hovertemplate="Time: %{x:.2f} s<br>Rate: %{y:.1f} Hz<extra></extra>"
            ),
            row=1, col=1
        )

        hover_text = [
            f"Cluster ID: #{c}<br>Depth: {d:.1f} µm<br>Unit FR: {fr:.2f} Hz"
            for c, d, fr in zip(qc_clusters, unit_depths, unit_rates)
        ]

        fig.add_trace(
            go.Scattergl(
                x=qc_times,
                y=unit_depths,
                mode="markers",
                name="QC Spikes",
                marker=dict(
                    size=3.5,
                    color=unit_rates,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(
                        title="FR (Hz)",
                        thickness=14,
                        len=0.75,
                        y=0.38,
                        x=1.02
                    ),
                    opacity=0.85
                ),
                text=hover_text,
                hovertemplate="Time: %{x:.3f} s<br>Depth: %{y:.1f} µm<br>%{text}<extra></extra>"
            ),
            row=2, col=1
        )

        fig.update_yaxes(title_text="Rate (Hz)", row=1, col=1, gridcolor="#2d3436")
        fig.update_yaxes(title_text="Probe Depth (µm)", row=2, col=1, gridcolor="#2d3436")
        fig.update_xaxes(title_text="Recording Time (seconds)", row=2, col=1, gridcolor="#2d3436")

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>{sorter_name.upper()} — QC Single Units Raster & Cardiorespiratory (Flow + EMG + ECG) Alignment</b><br><sup>{len(df_qc)} Passing Single Units | Synchronized with Respiratory Airflow, Diaphragm EMG, and Cardiac ECG</sup>",
            font=dict(size=18, color="#f1f2f6"),
            x=0.03,
            y=0.97
        ),
        hovermode="closest",
        margin=dict(l=60, r=90, t=90, b=50),
        height=1200,
        legend=dict(orientation="h", y=1.03, x=0.60),
        paper_bgcolor="#1e272e",
        plot_bgcolor="#141a1f"
    )

    fig.write_html(str(out_html_path), include_plotlyjs="cdn", full_html=True)
    logger.info(f"Saved synchronized Plotly raster with Flow, EMG & ECG to: {out_html_path}")

    # Generate Static PNG Preview
    if has_flow and has_laser and has_emg and has_int_emg and has_ecg:
        fig_static, axes_s = plt.subplots(7, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1.2, 3.5]}, sharex=True)
        axes_s[0].plot(flow_times, flow_volts, color='#059669', lw=1.2)
        axes_s[0].set_ylabel('Airflow (V)', fontsize=9, fontweight='bold')
        axes_s[0].set_title(f'{sorter_name.upper()} Airflow, Laser Opto (Ch 6), EMG, ECG, Population PSTH & Spike Raster ({len(df_qc)} Passing Units)', fontsize=13, fontweight='bold')
        axes_s[0].grid(True, linestyle=':', alpha=0.6)

        axes_s[1].plot(laser_times, laser_volts, color='#f59e0b', lw=1.2)
        axes_s[1].set_ylabel('Laser Opto (V)', fontsize=9, fontweight='bold')
        axes_s[1].grid(True, linestyle=':', alpha=0.6)

        axes_s[2].plot(emg_times, emg_volts, color='#475569', lw=1.0)
        axes_s[2].set_ylabel('Raw EMG (V)', fontsize=9, fontweight='bold')
        axes_s[2].grid(True, linestyle=':', alpha=0.6)

        axes_s[3].plot(int_emg_times, int_emg_volts, color='#0284c7', lw=1.5)
        axes_s[3].set_ylabel('Int. EMG (V)', fontsize=9, fontweight='bold')
        axes_s[3].grid(True, linestyle=':', alpha=0.6)

        axes_s[4].plot(ecg_times, ecg_volts, color='#f43f5e', lw=1.1)
        axes_s[4].set_ylabel('ECG (V)', fontsize=9, fontweight='bold')
        axes_s[4].grid(True, linestyle=':', alpha=0.6)

        axes_s[5].plot(psth_centers, psth_rate_hz, color='#00cec9', lw=1.5)
        axes_s[5].set_ylabel('Pop FR (Hz)', fontsize=9, fontweight='bold')
        axes_s[5].grid(True, linestyle=':', alpha=0.6)

        axes_s[6].scatter(qc_times, unit_depths, s=3, c=unit_depths, cmap='viridis', alpha=0.7, rasterized=True)
        axes_s[6].set_ylabel('Probe Depth (µm)', fontsize=10, fontweight='bold')
        axes_s[6].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        axes_s[6].grid(True, linestyle=':', alpha=0.6)

        for ax in axes_s:
            ax.axvspan(60.0, 61.955, color='#f59e0b', alpha=0.15, zorder=0)
            ax.axvspan(119.664, 121.619, color='#f59e0b', alpha=0.15, zorder=0)
            ax.axvspan(164.288, 166.243, color='#f59e0b', alpha=0.15, zorder=0)

    elif has_flow and has_emg and has_int_emg and has_ecg:
        fig_static, axes_s = plt.subplots(6, 1, figsize=(14, 15), gridspec_kw={'height_ratios': [1, 1, 1, 1, 1.2, 3.5]}, sharex=True)
        axes_s[0].plot(flow_times, flow_volts, color='#059669', lw=1.2)
        axes_s[0].set_ylabel('Airflow (V)', fontsize=9, fontweight='bold')
        axes_s[0].set_title(f'{sorter_name.upper()} Airflow, Raw EMG, Integrated Diaphragm EMG, Cardiac ECG, Population PSTH & Spike Raster ({len(df_qc)} Passing Units)', fontsize=13, fontweight='bold')
        axes_s[0].grid(True, linestyle=':', alpha=0.6)

        axes_s[1].plot(emg_times, emg_volts, color='#475569', lw=1.0)
        axes_s[1].set_ylabel('Raw EMG (V)', fontsize=9, fontweight='bold')
        axes_s[1].grid(True, linestyle=':', alpha=0.6)

        axes_s[2].plot(int_emg_times, int_emg_volts, color='#0284c7', lw=1.5)
        axes_s[2].set_ylabel('Int. EMG (V)', fontsize=9, fontweight='bold')
        axes_s[2].grid(True, linestyle=':', alpha=0.6)

        axes_s[3].plot(ecg_times, ecg_volts, color='#f43f5e', lw=1.1)
        axes_s[3].set_ylabel('ECG (V)', fontsize=9, fontweight='bold')
        axes_s[3].grid(True, linestyle=':', alpha=0.6)

        axes_s[4].plot(psth_centers, psth_rate_hz, color='#00cec9', lw=1.5)
        axes_s[4].set_ylabel('Pop FR (Hz)', fontsize=9, fontweight='bold')
        axes_s[4].grid(True, linestyle=':', alpha=0.6)

        axes_s[5].scatter(qc_times, unit_depths, s=3, c=unit_depths, cmap='viridis', alpha=0.7, rasterized=True)
        axes_s[5].set_ylabel('Probe Depth (µm)', fontsize=10, fontweight='bold')
        axes_s[5].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        axes_s[5].grid(True, linestyle=':', alpha=0.6)
    elif has_flow and has_emg and has_int_emg:
        fig_static, axes_s = plt.subplots(5, 1, figsize=(14, 13), gridspec_kw={'height_ratios': [1, 1, 1, 1, 3.5]}, sharex=True)
        axes_s[0].plot(flow_times, flow_volts, color='#059669', lw=1.2)
        axes_s[0].set_ylabel('Airflow (V)', fontsize=9, fontweight='bold')
        axes_s[0].set_title(f'{sorter_name.upper()} Airflow, Raw EMG, Integrated Diaphragm EMG, Population PSTH & Spike Raster ({len(df_qc)} Passing Units)', fontsize=13, fontweight='bold')
        axes_s[0].grid(True, linestyle=':', alpha=0.6)

        axes_s[1].plot(emg_times, emg_volts, color='#475569', lw=1.0)
        axes_s[1].set_ylabel('Raw EMG (V)', fontsize=9, fontweight='bold')
        axes_s[1].grid(True, linestyle=':', alpha=0.6)

        axes_s[2].plot(int_emg_times, int_emg_volts, color='#0284c7', lw=1.5)
        axes_s[2].set_ylabel('Int. EMG (V)', fontsize=9, fontweight='bold')
        axes_s[2].grid(True, linestyle=':', alpha=0.6)

        axes_s[3].plot(psth_centers, psth_rate_hz, color='#00cec9', lw=1.5)
        axes_s[3].set_ylabel('Pop FR (Hz)', fontsize=9, fontweight='bold')
        axes_s[3].grid(True, linestyle=':', alpha=0.6)

        axes_s[4].scatter(qc_times, unit_depths, s=3, c=unit_depths, cmap='viridis', alpha=0.7, rasterized=True)
        axes_s[4].set_ylabel('Probe Depth (µm)', fontsize=10, fontweight='bold')
        axes_s[4].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        axes_s[4].grid(True, linestyle=':', alpha=0.6)
    elif has_flow:
        fig_static, axes_s = plt.subplots(3, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [1, 1, 3.5]}, sharex=True)
        axes_s[0].plot(flow_times, flow_volts, color='#e17055', lw=1.2)
        axes_s[0].set_ylabel('Flow (V)', fontsize=10, fontweight='bold')
        axes_s[0].set_title(f'{sorter_name.upper()} Respiratory Flow Rate, Population PSTH & Spike Raster ({len(df_qc)} Units)', fontsize=13, fontweight='bold')
        axes_s[0].grid(True, linestyle=':', alpha=0.6)

        axes_s[1].plot(psth_centers, psth_rate_hz, color='#00cec9', lw=1.5)
        axes_s[1].set_ylabel('Pop FR (Hz)', fontsize=10, fontweight='bold')
        axes_s[1].grid(True, linestyle=':', alpha=0.6)

        axes_s[2].scatter(qc_times, unit_depths, s=3, c=unit_depths, cmap='viridis', alpha=0.7, rasterized=True)
        axes_s[2].set_ylabel('Probe Depth (µm)', fontsize=10, fontweight='bold')
        axes_s[2].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        axes_s[2].grid(True, linestyle=':', alpha=0.6)
    else:
        fig_static, axes_s = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 4]}, sharex=True)
        axes_s[0].plot(psth_centers, psth_rate_hz, color='#00cec9', lw=1.5)
        axes_s[0].set_ylabel('Pop FR (Hz)', fontsize=11)
        axes_s[0].set_title(f'{sorter_name.upper()} Single-Unit QC Passed Population Raster ({len(df_qc)} Units)', fontsize=13, fontweight='bold')
        axes_s[0].grid(True, linestyle=':', alpha=0.6)

        axes_s[1].scatter(qc_times, unit_depths, s=3, c=unit_depths, cmap='viridis', alpha=0.7, rasterized=True)
        axes_s[1].set_ylabel('Probe Depth (µm)', fontsize=11)
        axes_s[1].set_xlabel('Time (s)', fontsize=11)
        axes_s[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(preview_png_path, dpi=300)
    plt.close(fig_static)
    logger.info(f"Saved static preview to: {preview_png_path}")

    return out_html_path, dashboard_html_path, preview_png_path

def generate_full_dashboard_generic(
    df_qc: pd.DataFrame,
    qc_times: np.ndarray,
    qc_clusters: np.ndarray,
    psth_time: np.ndarray,
    psth_rate: np.ndarray,
    out_path: Path,
    sorter_name: str,
    duration_s: float
):
    """
    Creates an interactive HTML5 / CSS3 / JavaScript dashboard with instant unit filtering,
    probe schematic, and unit metric table for any sorter.
    """
    units_data = df_qc.to_dict(orient="records")
    
    # Subsample spikes for embedded fast viewer (limit 80k spikes max)
    if len(qc_times) > 80000:
        sub_idx = np.random.choice(len(qc_times), 80000, replace=False)
        sub_times = qc_times[sub_idx].tolist()
        sub_clusters = qc_clusters[sub_idx].tolist()
    else:
        sub_times = qc_times.tolist()
        sub_clusters = qc_clusters.tolist()

    max_depth = float(np.max(df_qc["depth_um"])) + 50.0 if "depth_um" in df_qc.columns else 3840.0

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{sorter_name} QC Units Interactive Raster Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        body {{
            background: #0f141c;
            color: #e2e8f0;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }}
        header {{
            background: #182232;
            padding: 12px 24px;
            border-bottom: 1px solid #2d3d54;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header-title h1 {{
            font-size: 1.25rem;
            font-weight: 700;
            color: #38bdf8;
        }}
        .header-title p {{
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 2px;
        }}
        .stats-badge-group {{
            display: flex;
            gap: 12px;
        }}
        .stat-badge {{
            background: #202e42;
            padding: 6px 14px;
            border-radius: 8px;
            border: 1px solid #334966;
            font-size: 0.85rem;
        }}
        .stat-badge strong {{
            color: #38bdf8;
            margin-right: 4px;
        }}
        .main-container {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}
        .sidebar {{
            width: 340px;
            background: #141c28;
            border-right: 1px solid #2d3d54;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 14px;
            overflow-y: auto;
        }}
        .panel-card {{
            background: #1c2738;
            padding: 14px;
            border-radius: 10px;
            border: 1px solid #2e405b;
        }}
        .panel-card h3 {{
            font-size: 0.9rem;
            font-weight: 600;
            color: #f1f5f9;
            margin-bottom: 10px;
        }}
        .control-group {{
            margin-bottom: 12px;
        }}
        .control-group label {{
            display: flex;
            justify-content: space-between;
            font-size: 0.78rem;
            color: #94a3b8;
            margin-bottom: 4px;
        }}
        input[type="range"] {{
            width: 100%;
            accent-color: #38bdf8;
        }}
        select, input[type="text"] {{
            width: 100%;
            background: #0f141c;
            border: 1px solid #334966;
            color: #f1f5f9;
            padding: 7px 10px;
            border-radius: 6px;
            font-size: 0.82rem;
        }}
        .unit-details-card {{
            background: #0f1722;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #0284c7;
            font-size: 0.8rem;
            display: none;
            flex-direction: column;
            gap: 4px;
        }}
        .unit-details-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
            margin-top: 6px;
        }}
        .unit-details-item {{
            background: #182232;
            padding: 4px 8px;
            border-radius: 4px;
        }}
        .unit-details-item span {{
            color: #94a3b8;
            font-size: 0.72rem;
            display: block;
        }}
        .unit-details-item strong {{
            color: #38bdf8;
            font-size: 0.85rem;
        }}
        .unit-list {{
            max-height: 240px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .unit-item {{
            display: flex;
            justify-content: space-between;
            padding: 6px 10px;
            background: #0f1722;
            border-radius: 6px;
            font-size: 0.78rem;
            cursor: pointer;
            border: 1px solid transparent;
            transition: all 0.15s ease;
        }}
        .unit-item:hover, .unit-item.active {{
            background: #25354d;
            border-color: #38bdf8;
            color: #38bdf8;
        }}
        .plot-container {{
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 10px;
            background: #0b0f17;
            overflow: hidden;
            position: relative;
        }}
        #plot-div {{
            flex: 1;
            width: 100%;
            height: 100%;
            min-height: 400px;
        }}
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <h1>{sorter_name} — Single-Unit QC Population Raster</h1>
            <p>Dataset: LP21_fent_EXPT_g1 (Neuropixels 2.0 | First 60s) | Sorter: {sorter_name}</p>
        </div>
        <div class="stats-badge-group">
            <div class="stat-badge"><strong>{len(df_qc)}</strong> QC Passed Units</div>
            <div class="stat-badge"><strong>{len(qc_times):,}</strong> Total Spikes</div>
            <div class="stat-badge"><strong>{df_qc['firing_rate_hz'].mean():.2f} Hz</strong> Mean FR</div>
        </div>
    </header>

    <div class="main-container">
        <div class="sidebar">
            <div class="panel-card">
                <h3>Raster Filter Controls</h3>
                <div class="control-group">
                    <label><span>Min Firing Rate</span><span id="lbl-fr">0.0 Hz</span></label>
                    <input type="range" id="slider-fr" min="0" max="40" step="0.5" value="0">
                </div>
                <div class="control-group">
                    <label><span>Depth Range (µm)</span><span id="lbl-depth">0 - {max_depth:.0f} µm</span></label>
                    <input type="range" id="slider-depth-min" min="0" max="{max_depth:.0f}" step="50" value="0">
                    <input type="range" id="slider-depth-max" min="0" max="{max_depth:.0f}" step="50" value="{max_depth:.0f}" style="margin-top: 6px;">
                </div>
                <div class="control-group">
                    <label><span>Min Estimated SNR</span><span id="lbl-snr">3.0</span></label>
                    <input type="range" id="slider-snr" min="3.0" max="15.0" step="0.5" value="3.0">
                </div>
                <button id="btn-reset" style="width:100%; padding:8px; background:#0284c7; color:#fff; border:none; border-radius:6px; cursor:pointer; font-size:0.8rem; font-weight:600;">Show All QC Units</button>
            </div>

            <div class="unit-details-card" id="unit-details-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:700; color:#38bdf8;" id="ud-title">Cluster #</span>
                    <button onclick="resetUnitSelection()" style="background:transparent; border:none; color:#94a3b8; cursor:pointer; font-size:0.75rem;">✕ Close</button>
                </div>
                <div class="unit-details-grid">
                    <div class="unit-details-item"><span>Depth</span><strong id="ud-depth">-- µm</strong></div>
                    <div class="unit-details-item"><span>Firing Rate</span><strong id="ud-fr">-- Hz</strong></div>
                    <div class="unit-details-item"><span>SNR</span><strong id="ud-snr">--</strong></div>
                    <div class="unit-details-item"><span>ISI Violations</span><strong id="ud-isi">-- %</strong></div>
                </div>
            </div>

            <div class="panel-card" style="flex:1; display:flex; flex-direction:column;">
                <h3>QC Single Units ({len(df_qc)})</h3>
                <input type="text" id="unit-search" placeholder="Search Unit ID..." style="margin-bottom:8px;">
                <div class="unit-list" id="unit-list"></div>
            </div>
        </div>

        <div class="plot-container">
            <div id="plot-div"></div>
        </div>
    </div>

    <script>
        const units = {json.dumps(units_data)};
        const times = {json.dumps(sub_times)};
        const clusters = {json.dumps(sub_clusters)};
        const psthTime = {json.dumps(psth_time.tolist())};
        const psthRate = {json.dumps(psth_rate.tolist())};

        const unitMap = {{}};
        units.forEach(u => {{ unitMap[u.cluster_id] = u; }});

        let selectedUnitId = null;

        function populateUnitList() {{
            const container = document.getElementById("unit-list");
            container.innerHTML = "";
            units.forEach(u => {{
                const div = document.createElement("div");
                div.className = "unit-item";
                div.id = `unit-item-${{u.cluster_id}}`;
                div.innerHTML = `<span><b>#${{u.cluster_id}}</b> (${{u.depth_um.toFixed(0)}} µm)</span><span>${{u.firing_rate_hz.toFixed(1)}} Hz</span>`;
                div.onclick = () => selectUnit(u.cluster_id);
                container.appendChild(div);
            }});
        }}

        function selectUnit(clusterId) {{
            selectedUnitId = clusterId;
            document.querySelectorAll(".unit-item").forEach(el => el.classList.remove("active"));
            const target = document.getElementById(`unit-item-${{clusterId}}`);
            if (target) target.classList.add("active");

            const u = unitMap[clusterId];
            if (u) {{
                const card = document.getElementById("unit-details-card");
                card.style.display = "flex";
                document.getElementById("ud-title").textContent = `Selected Unit #${{clusterId}}`;
                document.getElementById("ud-depth").textContent = `${{u.depth_um.toFixed(1)}} µm`;
                document.getElementById("ud-fr").textContent = `${{u.firing_rate_hz.toFixed(2)}} Hz`;
                document.getElementById("ud-snr").textContent = `${{u.snr.toFixed(1)}}`;
                document.getElementById("ud-isi").textContent = `${{(u.isi_violations * 100).toFixed(2)}}%`;
            }}

            renderPlot([clusterId]);
        }}

        function resetUnitSelection() {{
            selectedUnitId = null;
            document.getElementById("unit-details-card").style.display = "none";
            document.querySelectorAll(".unit-item").forEach(el => el.classList.remove("active"));
            applyFilters();
        }}

        function renderPlot(filteredUnitIds = null) {{
            const validSet = filteredUnitIds ? new Set(filteredUnitIds) : new Set(units.map(u => u.cluster_id));
            
            const plotTimes = [];
            const plotDepths = [];
            const plotRates = [];
            const hoverText = [];

            for (let i = 0; i < times.length; i++) {{
                const cid = clusters[i];
                if (validSet.has(cid)) {{
                    const u = unitMap[cid];
                    if (u) {{
                        plotTimes.push(times[i]);
                        plotDepths.push(u.depth_um);
                        plotRates.push(u.firing_rate_hz);
                        hoverText.push(`Cluster: #${{cid}}<br>Depth: ${{u.depth_um.toFixed(1)}} µm<br>FR: ${{u.firing_rate_hz.toFixed(2)}} Hz<br>SNR: ${{u.snr.toFixed(1)}}`);
                    }}
                }}
            }}

            const tracePSTH = {{
                x: psthTime,
                y: psthRate,
                type: 'scatter',
                mode: 'lines',
                name: 'Population PSTH',
                line: {{ color: '#38bdf8', width: 2 }},
                fill: 'tozeroy',
                fillcolor: 'rgba(56, 189, 248, 0.15)',
                xaxis: 'x',
                yaxis: 'y2',
                hoverinfo: 'x+y'
            }};

            const traceRaster = {{
                x: plotTimes,
                y: plotDepths,
                type: 'scattergl',
                mode: 'markers',
                name: 'Spikes',
                marker: {{
                    size: filteredUnitIds ? 5.0 : 3.5,
                    color: filteredUnitIds ? '#38bdf8' : plotRates,
                    colorscale: 'Viridis',
                    showscale: !filteredUnitIds,
                    colorbar: {{
                        title: 'FR (Hz)',
                        thickness: 14,
                        len: 0.72,
                        y: 0.36,
                        x: 1.01
                    }},
                    opacity: 0.85
                }},
                text: hoverText,
                hoverinfo: 'text+x',
                xaxis: 'x',
                yaxis: 'y'
            }};

            const layout = {{
                paper_bgcolor: '#0b0f17',
                plot_bgcolor: '#101723',
                margin: {{ l: 65, r: 70, t: 20, b: 45 }},
                yaxis2: {{
                    title: 'Rate (Hz)',
                    domain: [0.80, 1.0],
                    gridcolor: '#1e293b',
                    color: '#94a3b8'
                }},
                yaxis: {{
                    title: 'Probe Depth (µm)',
                    domain: [0.0, 0.76],
                    gridcolor: '#1e293b',
                    color: '#94a3b8'
                }},
                xaxis: {{
                    title: 'Recording Time (s)',
                    gridcolor: '#1e293b',
                    color: '#94a3b8'
                }},
                hovermode: 'closest',
                showlegend: false
            }};

            Plotly.newPlot('plot-div', [tracePSTH, traceRaster], layout, {{ responsive: true, displayModeBar: true }});
        }}

        function applyFilters() {{
            if (selectedUnitId !== null) return;

            const minFR = parseFloat(document.getElementById("slider-fr").value);
            const minDepth = parseFloat(document.getElementById("slider-depth-min").value);
            const maxDepth = parseFloat(document.getElementById("slider-depth-max").value);
            const minSNR = parseFloat(document.getElementById("slider-snr").value);

            document.getElementById("lbl-fr").textContent = `${{minFR}} Hz`;
            document.getElementById("lbl-depth").textContent = `${{minDepth}} - ${{maxDepth}} µm`;
            document.getElementById("lbl-snr").textContent = `${{minSNR}}`;

            const filtered = units.filter(u => 
                u.firing_rate_hz >= minFR &&
                u.depth_um >= minDepth &&
                u.depth_um <= maxDepth &&
                u.snr >= minSNR
            ).map(u => u.cluster_id);

            renderPlot(filtered);
        }}

        document.getElementById("slider-fr").oninput = applyFilters;
        document.getElementById("slider-depth-min").oninput = applyFilters;
        document.getElementById("slider-depth-max").oninput = applyFilters;
        document.getElementById("slider-snr").oninput = applyFilters;
        document.getElementById("btn-reset").onclick = resetUnitSelection;

        document.getElementById("unit-search").oninput = (e) => {{
            const q = e.target.value.toLowerCase();
            document.querySelectorAll(".unit-item").forEach(el => {{
                el.style.display = el.textContent.toLowerCase().includes(q) ? "flex" : "none";
            }});
        }};

        window.addEventListener('DOMContentLoaded', () => {{
            populateUnitList();
            renderPlot();
        }});
    </script>
</body>
</html>
"""
    out_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Dashboard generated at: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate interactive raster plots for any spike sorter")
    parser.add_argument("--sorter-dir", type=str, default=r"c:\Users\lpascu\analysis_src\output_LP21_fent_EXPT_g1\mountainsort5", help="Path to sorter output directory")
    parser.add_argument("--sorter-name", type=str, default="MountainSort 5", help="Display name of the sorter")
    parser.add_argument("--bin-size", type=float, default=50.0, help="PSTH bin size in ms")
    args = parser.parse_args()

    build_interactive_sorter_raster(Path(args.sorter_dir), sorter_name=args.sorter_name, bin_size_ms=args.bin_size)

if __name__ == "__main__":
    main()
