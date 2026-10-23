#!/usr/bin/env python3
"""
Unified 3-Sorter Rostral Tip Comparison Dashboard (Lupin vs MountainSort 5 vs Kilosort 4)
Synchronized with Respiratory Flow Rate and Diaphragm EMG traces.
"""
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ROSTRAL_3SORTERS_RASTER")

def main():
    base_dir = Path(r"c:\Users\lpascu\analysis_src\output_LP21_fent_EXPT_g1\rostral_tip_benchmark")
    nidq_path = Path(r"Z:\ramirezlab\pascual\rec_data\LP21_fent_EXPT_g1\LP21_fent_EXPT_g1_t0.nidq.bin")
    fs_nidq = 10000.0
    duration_s = 60.0

    # 1. Load NIDAQ signals (Flow Rate and Diaphragm EMG)
    n_ch = 9
    n_samp_total = int(duration_s * fs_nidq)
    raw_nidq = np.memmap(nidq_path, dtype="int16", mode="r")
    n_samp_avail = min(len(raw_nidq) // n_ch, n_samp_total)
    nidq_2d = raw_nidq[:n_samp_avail * n_ch].reshape(-1, n_ch)

    # Flow Rate (Ch 5)
    v_flow_raw = nidq_2d[:, 5].astype(np.float32) * (10.0 / 32768.0)
    ds_flow = 20  # 10 kHz -> 500 Hz
    flow_volts = v_flow_raw[::ds_flow]
    flow_times = (np.arange(len(flow_volts)) * ds_flow) / fs_nidq

    # Diaphragm EMG (Ch 0)
    v_emg_raw = nidq_2d[:, 0].astype(np.float32) * (10.0 / 32768.0)
    b_emg, a_emg = signal.butter(3, [100.0, 2000.0], btype='bandpass', fs=fs_nidq)
    v_emg_filt = signal.filtfilt(b_emg, a_emg, v_emg_raw)
    ds_emg = 10  # 10 kHz -> 1000 Hz
    emg_volts = v_emg_filt[::ds_emg]
    emg_times = (np.arange(len(emg_volts)) * ds_emg) / fs_nidq

    # 2. Load Passing Single Units from each Sorter
    sorter_data = {}
    for s_name, color in [("lupin", "#e74c3c"), ("mountainsort5", "#2ecc71"), ("kilosort4", "#3498db")]:
        s_dir = base_dir / s_name
        alf_dir = s_dir / "alf"
        spk_t = np.load(alf_dir / "spike_times.npy")
        spk_c = np.load(alf_dir / "spike_clusters.npy")
        qc_df = pd.read_csv(s_dir / f"{s_name}_rostral_qc_summary.csv")
        
        # Filter for passing QC
        passing_df = qc_df[qc_df["passed_allen_qc"]].copy()
        passing_ids = set(passing_df["cluster_id"])
        
        mask = np.isin(spk_c, list(passing_ids)) & (spk_t <= duration_s)
        u_times = spk_t[mask]
        u_clusters = spk_c[mask]
        u_map_depth = {r["cluster_id"]: r["depth_um"] for _, r in passing_df.iterrows()}
        u_map_fr = {r["cluster_id"]: r["firing_rate_hz"] for _, r in passing_df.iterrows()}
        
        sorter_data[s_name] = {
            "times": u_times,
            "clusters": u_clusters,
            "depths": np.array([u_map_depth[c] for c in u_clusters]),
            "rates": np.array([u_map_fr[c] for c in u_clusters]),
            "df": passing_df,
            "color": color
        }

    # 3. Create Synchronized 5-Row Plotly Dashboard
    fig = make_subplots(
        rows=5, cols=1,
        row_heights=[0.14, 0.14, 0.22, 0.25, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            "Respiratory Flow Rate (Plethysmography Airflow)",
            "Diaphragm EMG (Inspiratory Motor Burst Drive)",
            f"Lupin Rostral Single Units ({len(sorter_data['lupin']['df'])} Passing SUA Unit)",
            f"MountainSort 5 Rostral Single Units ({len(sorter_data['mountainsort5']['df'])} Passing SUA Units)",
            f"Kilosort 4 Rostral Single Units ({len(sorter_data['kilosort4']['df'])} Passing SUA Units)"
        )
    )

    # Row 1: Flow
    fig.add_trace(
        go.Scattergl(x=flow_times, y=flow_volts, mode="lines", name="Flow Rate", line=dict(color="#ff7675", width=1.5)),
        row=1, col=1
    )

    # Row 2: EMG
    fig.add_trace(
        go.Scattergl(x=emg_times, y=emg_volts, mode="lines", name="Diaphragm EMG", line=dict(color="#a29bfe", width=1.1)),
        row=2, col=1
    )

    # Row 3: Lupin Raster
    lup_d = sorter_data["lupin"]
    fig.add_trace(
        go.Scattergl(
            x=lup_d["times"], y=lup_d["depths"],
            mode="markers", name="Lupin SUA",
            marker=dict(size=4.5, color=lup_d["rates"], colorscale="Viridis", showscale=False),
            text=[f"Cluster #{c} | Depth {d:.1f} µm | FR: {fr:.1f} Hz" for c, d, fr in zip(lup_d["clusters"], lup_d["depths"], lup_d["rates"])],
            hovertemplate="Time: %{x:.3f} s<br>%{text}<extra></extra>"
        ),
        row=3, col=1
    )

    # Row 4: MountainSort 5 Raster
    ms_d = sorter_data["mountainsort5"]
    fig.add_trace(
        go.Scattergl(
            x=ms_d["times"], y=ms_d["depths"],
            mode="markers", name="MountainSort 5 SUA",
            marker=dict(size=4.5, color=ms_d["rates"], colorscale="Viridis", showscale=False),
            text=[f"Cluster #{c} | Depth {d:.1f} µm | FR: {fr:.1f} Hz" for c, d, fr in zip(ms_d["clusters"], ms_d["depths"], ms_d["rates"])],
            hovertemplate="Time: %{x:.3f} s<br>%{text}<extra></extra>"
        ),
        row=4, col=1
    )

    # Row 5: Kilosort 4 Raster
    ks_d = sorter_data["kilosort4"]
    fig.add_trace(
        go.Scattergl(
            x=ks_d["times"], y=ks_d["depths"],
            mode="markers", name="Kilosort 4 SUA",
            marker=dict(size=4.5, color=ks_d["rates"], colorscale="Viridis", showscale=True, colorbar=dict(title="FR (Hz)", len=0.45, y=0.25, x=1.02)),
            text=[f"Cluster #{c} | Depth {d:.1f} µm | FR: {fr:.1f} Hz" for c, d, fr in zip(ks_d["clusters"], ks_d["depths"], ks_d["rates"])],
            hovertemplate="Time: %{x:.3f} s<br>%{text}<extra></extra>"
        ),
        row=5, col=1
    )

    fig.update_yaxes(title_text="Flow (V)", row=1, col=1, gridcolor="#2d3436")
    fig.update_yaxes(title_text="EMG (V)", row=2, col=1, gridcolor="#2d3436")
    fig.update_yaxes(title_text="Depth (µm)", row=3, col=1, range=[-10, 180], gridcolor="#2d3436")
    fig.update_yaxes(title_text="Depth (µm)", row=4, col=1, range=[-10, 180], gridcolor="#2d3436")
    fig.update_yaxes(title_text="Depth (µm)", row=5, col=1, range=[-10, 180], gridcolor="#2d3436")
    fig.update_xaxes(title_text="Recording Time (seconds)", row=5, col=1, gridcolor="#2d3436")

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text="<b>ROSTRAL PROBE TIP BENCHMARK (Channels AP0 - AP23, Depths 0 - 165 µm)</b><br><sup>Passing Single Units (Lupin vs MountainSort 5 vs Kilosort 4) Synchronized with Flow Rate and Diaphragm EMG</sup>",
            font=dict(size=18, color="#f1f2f6"),
            x=0.03, y=0.98
        ),
        height=1050,
        margin=dict(l=60, r=90, t=90, b=50),
        paper_bgcolor="#1e272e",
        plot_bgcolor="#141a1f"
    )

    out_html = base_dir / "rostral_tip_3sorters_comparison_dashboard.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)
    logger.info(f"Saved unified 3-sorter rostral dashboard to: {out_html}")

    # 4. Generate Static Publication PNG Preview
    fig_s, axes_s = plt.subplots(5, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1.2, 1.2, 1.2]})

    # Row 1: Flow
    axes_s[0].plot(flow_times, flow_volts, color='#e17055', lw=1.2)
    axes_s[0].set_ylabel('Flow (V)', fontweight='bold')
    axes_s[0].set_title('Rostral Channels (AP0 - AP23) Benchmark: Passing Single Units Synchronized with Flow & EMG', fontsize=13, fontweight='bold')
    axes_s[0].grid(True, linestyle=':', alpha=0.5)

    # Row 2: EMG
    axes_s[1].plot(emg_times, emg_volts, color='#6c5ce7', lw=1.0)
    axes_s[1].set_ylabel('EMG (V)', fontweight='bold')
    axes_s[1].grid(True, linestyle=':', alpha=0.5)

    # Row 3: Lupin
    axes_s[2].scatter(lup_d["times"], lup_d["depths"], s=8, color='#e74c3c', label=f'Lupin: 1 SUA Unit ({len(lup_d["times"]):,} spikes)')
    axes_s[2].set_ylabel('Lupin Depth (µm)', fontweight='bold')
    axes_s[2].set_ylim([-10, 180])
    axes_s[2].legend(loc='upper right', fontsize=9)
    axes_s[2].grid(True, linestyle=':', alpha=0.5)

    # Row 4: MountainSort 5
    axes_s[3].scatter(ms_d["times"], ms_d["depths"], s=8, color='#2ecc71', label=f'MountainSort 5: 2 SUA Units ({len(ms_d["times"]):,} spikes)')
    axes_s[3].set_ylabel('MS5 Depth (µm)', fontweight='bold')
    axes_s[3].set_ylim([-10, 180])
    axes_s[3].legend(loc='upper right', fontsize=9)
    axes_s[3].grid(True, linestyle=':', alpha=0.5)

    # Row 5: Kilosort 4
    axes_s[4].scatter(ks_d["times"], ks_d["depths"], s=8, color='#3498db', label=f'Kilosort 4: 4 SUA Units ({len(ks_d["times"]):,} spikes)')
    axes_s[4].set_ylabel('KS4 Depth (µm)', fontweight='bold')
    axes_s[4].set_xlabel('Recording Time (seconds)', fontsize=11, fontweight='bold')
    axes_s[4].set_ylim([-10, 180])
    axes_s[4].legend(loc='upper right', fontsize=9)
    axes_s[4].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    out_png = base_dir / "rostral_tip_3sorters_comparison_preview.png"
    plt.savefig(out_png, dpi=300)
    plt.close(fig_s)
    logger.info(f"Saved static preview to: {out_png}")

    print("\n" + "=" * 80)
    print(" ROSTRAL TIP 3-SORTER BENCHMARK DASHBOARDS GENERATED!")
    print(f" Unified Interactive Dashboard : {out_html}")
    print(f" Static Publication Preview     : {out_png}")
    print("=" * 80)

if __name__ == "__main__":
    main()
