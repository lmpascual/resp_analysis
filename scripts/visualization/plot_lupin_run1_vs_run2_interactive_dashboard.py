#!/usr/bin/env python3
"""
Interactive HTML Dashboard Comparing Lupin Run 1 (Full Probe) vs Run 2 (Rostral Slice)
Across the Full 60-Second Recording, Synchronized with Flow Rate and Diaphragm EMG.
"""
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("LUPIN_RUN1_VS_RUN2_HTML")

def main():
    run1_dir = Path(r"c:\Users\lpascu\analysis_src\output_LP21_fent_EXPT_g1\lupin")
    run2_dir = Path(r"c:\Users\lpascu\analysis_src\output_LP21_fent_EXPT_g1\rostral_tip_benchmark\lupin")
    nidq_path = Path(r"Z:\ramirezlab\pascual\rec_data\LP21_fent_EXPT_g1\LP21_fent_EXPT_g1_t0.nidq.bin")
    fs_nidq = 10000.0
    duration_s = 60.0

    # 1. Load NIDAQ signals (Flow Rate and Diaphragm EMG)
    logger.info("Loading NIDAQ Flow Rate and EMG signals...")
    n_ch = 9
    n_samp_total = int(duration_s * fs_nidq)
    raw_nidq = np.memmap(nidq_path, dtype="int16", mode="r")
    n_samp_avail = min(len(raw_nidq) // n_ch, n_samp_total)
    nidq_2d = raw_nidq[:n_samp_avail * n_ch].reshape(-1, n_ch)

    # Flow Rate (Ch 5) - Downsample to 500 Hz
    v_flow_raw = nidq_2d[:, 5].astype(np.float32) * (10.0 / 32768.0)
    ds_flow = 20  # 10 kHz -> 500 Hz
    flow_volts = v_flow_raw[::ds_flow]
    flow_times = (np.arange(len(flow_volts)) * ds_flow) / fs_nidq

    # Diaphragm EMG (Ch 0) - Bandpass 100-2000 Hz & Downsample to 1000 Hz
    v_emg_raw = nidq_2d[:, 0].astype(np.float32) * (10.0 / 32768.0)
    b_emg, a_emg = signal.butter(3, [100.0, 2000.0], btype='bandpass', fs=fs_nidq)
    v_emg_filt = signal.filtfilt(b_emg, a_emg, v_emg_raw)
    ds_emg = 10  # 10 kHz -> 1000 Hz
    emg_volts = v_emg_filt[::ds_emg]
    emg_times = (np.arange(len(emg_volts)) * ds_emg) / fs_nidq

    # 2. Load Lupin Run 1 Rostral Units (<150 um)
    logger.info("Loading Lupin Run 1 (Full 384-Channel Probe) spikes...")
    run1_t = np.load(run1_dir / "alf" / "spike_times.npy")
    run1_c = np.load(run1_dir / "alf" / "spike_clusters.npy")
    run1_qc = pd.read_csv(run1_dir / "lupin_qc_summary.csv")
    run1_sh_qc = run1_qc[run1_qc["passed_allen_qc"] & (run1_qc["depth_um"] < 150.0)]
    run1_sh_ids = set(run1_sh_qc["cluster_id"])
    run1_mask = np.isin(run1_c, list(run1_sh_ids)) & (run1_t <= duration_s)
    
    run1_spk_times = run1_t[run1_mask]
    run1_spk_clus = run1_c[run1_mask]
    u_map1_depth = {r["cluster_id"]: r["depth_um"] for _, r in run1_sh_qc.iterrows()}
    u_map1_fr = {r["cluster_id"]: r["firing_rate_hz"] for _, r in run1_sh_qc.iterrows()}
    run1_depths = np.array([u_map1_depth[c] for c in run1_spk_clus])
    run1_rates = np.array([u_map1_fr[c] for c in run1_spk_clus])

    # 3. Load Lupin Run 2 Rostral Units (24-Channel Slice)
    logger.info("Loading Lupin Run 2 (24-Channel Slice) spikes...")
    run2_t = np.load(run2_dir / "alf" / "spike_times.npy")
    run2_c = np.load(run2_dir / "alf" / "spike_clusters.npy")
    run2_qc = pd.read_csv(run2_dir / "lupin_rostral_qc_summary.csv")
    run2_sh_qc = run2_qc[run2_qc["passed_allen_qc"]]
    run2_sh_ids = set(run2_sh_qc["cluster_id"])
    run2_mask = np.isin(run2_c, list(run2_sh_ids)) & (run2_t <= duration_s)
    
    run2_spk_times = run2_t[run2_mask]
    run2_spk_clus = run2_c[run2_mask]
    u_map2_depth = {r["cluster_id"]: r["depth_um"] for _, r in run2_sh_qc.iterrows()}
    u_map2_fr = {r["cluster_id"]: r["firing_rate_hz"] for _, r in run2_sh_qc.iterrows()}
    run2_depths = np.array([u_map2_depth[c] for c in run2_spk_clus])
    run2_rates = np.array([u_map2_fr[c] for c in run2_spk_clus])

    # 4. Compute 50ms Binned PSTHs
    bin_size_ms = 50.0
    n_psth_bins = max(int(np.ceil((duration_s * 1000.0) / bin_size_ms)), 1)
    psth_bins = np.linspace(0, duration_s, n_psth_bins + 1)
    psth_centers = (psth_bins[:-1] + psth_bins[1:]) / 2.0

    counts1, _ = np.histogram(run1_spk_times, bins=psth_bins)
    rate1_hz = counts1 / (bin_size_ms / 1000.0)

    counts2, _ = np.histogram(run2_spk_times, bins=psth_bins)
    rate2_hz = counts2 / (bin_size_ms / 1000.0)

    # 5. Build Synchronized 5-Row Plotly Figure
    fig = make_subplots(
        rows=5, cols=1,
        row_heights=[0.12, 0.12, 0.28, 0.28, 0.20],
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            "Respiratory Flow Rate (Plethysmography Airflow)",
            "Diaphragm EMG (Inspiratory Motor Burst Drive)",
            f"Lupin Run 1: Full 384-Channel Probe ({len(run1_sh_ids)} Over-split Units | {len(run1_spk_times):,} Spikes - Note Collision Dropouts)",
            f"Lupin Run 2: 24-Channel Rostral Slice ({len(run2_sh_ids)} Unified Single Unit | {len(run2_spk_times):,} Spikes - Continuous Bursting)",
            f"Instantaneous Firing Rate Comparison (50 ms bins: Run 1 Combined Rate vs Run 2 Single Unit Rate)"
        )
    )

    # Row 1: Flow Rate Trace
    fig.add_trace(
        go.Scattergl(
            x=flow_times, y=flow_volts,
            mode="lines", name="Flow Rate (V)",
            line=dict(color="#ff7675", width=1.5),
            hovertemplate="Time: %{x:.3f} s<br>Flow: %{y:.2f} V<extra></extra>"
        ),
        row=1, col=1
    )

    # Row 2: Diaphragm EMG Trace
    fig.add_trace(
        go.Scattergl(
            x=emg_times, y=emg_volts,
            mode="lines", name="Diaphragm EMG (V)",
            line=dict(color="#a29bfe", width=1.2),
            hovertemplate="Time: %{x:.3f} s<br>EMG: %{y:.2f} V<extra></extra>"
        ),
        row=2, col=1
    )

    # Row 3: Run 1 Raster (Red/Coral)
    hover_run1 = [
        f"<b>[Run 1 Full Probe]</b><br>Cluster ID: #{c}<br>Depth: {d:.1f} µm<br>Unit FR: {fr:.1f} Hz"
        for c, d, fr in zip(run1_spk_clus, run1_depths, run1_rates)
    ]
    fig.add_trace(
        go.Scattergl(
            x=run1_spk_times, y=run1_depths,
            mode="markers", name="Run 1 Units (15 Units)",
            marker=dict(size=4.5, color=run1_rates, colorscale="Reds", showscale=False, opacity=0.85),
            text=hover_run1,
            hovertemplate="Time: %{x:.3f} s<br>%{text}<extra></extra>"
        ),
        row=3, col=1
    )

    # Row 4: Run 2 Raster (Green/Viridis)
    hover_run2 = [
        f"<b>[Run 2 Rostral Slice]</b><br>Cluster ID: #{c}<br>Depth: {d:.1f} µm<br>Unit FR: {fr:.1f} Hz"
        for c, d, fr in zip(run2_spk_clus, run2_depths, run2_rates)
    ]
    fig.add_trace(
        go.Scattergl(
            x=run2_spk_times, y=run2_depths,
            mode="markers", name="Run 2 Unit (1 SUA)",
            marker=dict(size=5.0, color="#2ecc71", opacity=0.90),
            text=hover_run2,
            hovertemplate="Time: %{x:.3f} s<br>%{text}<extra></extra>"
        ),
        row=4, col=1
    )

    # Row 5: Firing Rate PSTH Overlays
    fig.add_trace(
        go.Scatter(
            x=psth_centers, y=rate1_hz,
            mode="lines", name="Run 1 Rate (15 Units Combined)",
            line=dict(color="#e74c3c", width=2.0),
            hovertemplate="Time: %{x:.2f} s<br>Run 1 Rate: %{y:.1f} Hz<extra></extra>"
        ),
        row=5, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=psth_centers, y=rate2_hz,
            mode="lines", name="Run 2 Rate (1 Unified Unit)",
            line=dict(color="#2ecc71", width=2.2),
            fill="tozeroy",
            fillcolor="rgba(46, 204, 113, 0.20)",
            hovertemplate="Time: %{x:.2f} s<br>Run 2 Rate: %{y:.1f} Hz<extra></extra>"
        ),
        row=5, col=1
    )

    # Axis formatting
    fig.update_yaxes(title_text="Flow (V)", row=1, col=1, gridcolor="#2d3436")
    fig.update_yaxes(title_text="EMG (V)", row=2, col=1, gridcolor="#2d3436")
    fig.update_yaxes(title_text="Depth (µm)", row=3, col=1, range=[-10, 160], gridcolor="#2d3436")
    fig.update_yaxes(title_text="Depth (µm)", row=4, col=1, range=[-10, 160], gridcolor="#2d3436")
    fig.update_yaxes(title_text="Rate (Hz)", row=5, col=1, gridcolor="#2d3436")
    fig.update_xaxes(title_text="Recording Time (seconds)", row=5, col=1, gridcolor="#2d3436", rangeslider=dict(visible=True, thickness=0.04))

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text="<b>LUPIN FULL PROBE (RUN 1) VS ROSTRAL SLICE (RUN 2): 60-SECOND INTERACTIVE COMPARISON</b><br><sup>Synchronized Across Respiratory Airflow, Diaphragm EMG, Run 1 vs Run 2 Single-Unit Rasters, and Instantaneous Firing Rate PSTHs</sup>",
            font=dict(size=18, color="#f1f2f6"),
            x=0.03, y=0.98
        ),
        height=1100,
        margin=dict(l=60, r=90, t=90, b=50),
        legend=dict(orientation="h", y=1.04, x=0.55),
        paper_bgcolor="#1e272e",
        plot_bgcolor="#141a1f"
    )

    out_html = Path(r"c:\Users\lpascu\analysis_src\output_LP21_fent_EXPT_g1\rostral_tip_benchmark\lupin_run1_vs_run2_interactive_dashboard.html")
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)
    logger.info(f"Saved interactive comparison dashboard to: {out_html}")
    print("\n" + "=" * 85)
    print(" [SUCCESS] Interactive 60-Second Comparison Dashboard Generated!")
    print(f" Output HTML : {out_html}")
    print("=" * 85)

if __name__ == "__main__":
    main()
