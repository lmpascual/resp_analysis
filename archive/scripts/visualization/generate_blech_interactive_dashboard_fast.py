#!/usr/bin/env python3
"""
Fast Interactive HTML Visualizer for Single-Channel blech_clust Sorted Units.
Generates an interactive Plotly dashboard allowing manual inspection of spike detection
performance across all 17 deduplicated units on their peak channels.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import spikeinterface.extractors as se
from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig
from src.pipeline.processors.proc_blech_single_channel import filter_ap_channel

def main():
    out_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1")
    blech_alf_dir = out_dir / "blech_alf_384"
    session_dir = out_dir / "alf_session"
    raw_dir = Path(r"Z:\ramirezlab\pascual\rec_data\TP_Tacr1_1_npx2_g4")

    # Load ALF sorted units & cluster info
    info_df = pd.read_csv(blech_alf_dir / "cluster_info.tsv", sep="\t")
    spikes_times = np.load(blech_alf_dir / "spikes.times.npy")
    spikes_clusters = np.load(blech_alf_dir / "spikes.clusters.npy")

    # Load Raw AP binary on local C: drive
    ap_bin_file = session_dir / "raw_ephys_data" / "probe00" / "TP_Tacr1_1_npx2_g4_t0.imec0.ap.bin"
    n_samples_ap = ap_bin_file.stat().st_size // (2 * 385)
    fs_ap = 30000.0

    # Load NIDQ Diaphragm EMG & Laser TTL
    nidq_meta = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.meta"
    nidq_bin = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.bin"
    nidq_data = read_nidq_bin(nidq_bin, nidq_meta)

    t_nidq = nidq_data.time_vector
    fs_nidq = nidq_data.fs
    emg_raw = nidq_data.all_channels[0]
    ch6_laser = nidq_data.all_channels[6]

    emg_cfg = EMGConfig(fs=fs_nidq, highpass_cutoff=300.0, lowpass_cutoff=3000.0, notch_freq=60.0, smoothing_window_ms=20.0)
    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    breath_onsets = emg_proc.breath_onsets

    mm_ap = np.memmap(ap_bin_file, dtype=np.int16, mode='r', shape=(n_samples_ap, 385))

    # Downsample factor for smooth rendering
    ds_ap = 50 # 600 Hz display rate
    t_ap_ds = np.arange(0, n_samples_ap, ds_ap) / fs_ap

    ds_nidq = 10
    t_nidq_ds = t_nidq[::ds_nidq]
    emg_env_ds = emg_proc.envelope[::ds_nidq]
    laser_ds = (ch6_laser[::ds_nidq] > 0.8).astype(float)

    # Generate individual unit figures
    unit_html_snippets = []

    for idx, row in info_df.iterrows():
        uid = int(row['cluster_id'])
        ch = int(row['ch'])
        sh = int(row['shank'])
        depth = float(row['depth'])
        n_spk = int(row['n_spikes'])
        fr = float(row['firing_rate'])
        peak_amp = float(row['peak_amp_uv'])
        is_manual = bool(row['is_manual_pick'])

        print(f"Generating visualizer snippet for Unit {uid} (Channel {ch}, Shank {sh}, Depth {depth:.0f} um)...")

        # Extract & filter AP voltage trace
        raw_tr = mm_ap[:, ch].astype(np.float32)
        filt_tr = filter_ap_channel(raw_tr, fs=fs_ap)
        filt_ds = filt_tr[::ds_ap]

        u_spks = spikes_times[spikes_clusters == uid]
        u_samples = (u_spks * fs_ap).astype(int)
        u_samples = u_samples[(u_samples >= 0) & (u_samples < len(filt_tr))]
        u_amps = filt_tr[u_samples]

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=(
                f"Filtered AP Voltage Trace (300-3000 Hz) on Peak Channel {ch} & Detected Spikes",
                "Diaphragm EMG Envelope & Breath Onset Triggers",
                "Optogenetic Laser Stimulation TTL (Channel 6)"
            ),
            row_heights=[0.50, 0.30, 0.20]
        )

        # Track 1: AP Trace & Detected Spikes
        fig.add_trace(
            go.Scatter(
                x=t_ap_ds,
                y=filt_ds,
                mode='lines',
                name=f'AP Ch {ch} (uV)',
                line=dict(color='#1f77b4', width=1.0)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=u_spks,
                y=u_amps,
                mode='markers',
                name=f'Unit {uid} Spikes',
                marker=dict(color='crimson', size=6, symbol='x')
            ),
            row=1, col=1
        )

        # Track 2: Diaphragm EMG & Breath Onsets
        fig.add_trace(
            go.Scatter(
                x=t_nidq_ds,
                y=emg_env_ds,
                mode='lines',
                name='Diaphragm EMG (V)',
                line=dict(color='#2ca02c', width=1.2)
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=breath_onsets,
                y=np.zeros(len(breath_onsets)),
                mode='markers',
                name='Inspiration Onsets',
                marker=dict(color='blue', size=6, symbol='triangle-up')
            ),
            row=2, col=1
        )

        # Track 3: Laser TTL
        fig.add_trace(
            go.Scatter(
                x=t_nidq_ds,
                y=laser_ds,
                mode='lines',
                name='Laser TTL',
                line=dict(color='magenta', width=1.5)
            ),
            row=3, col=1
        )

        manual_tag = " [EXACT MATCH WITH MANUAL PICK]" if is_manual else ""
        fig.update_layout(
            title=f"<b>Unit {uid}</b> — Channel {ch} | Shank {sh} | Depth {depth:.0f} µm | {n_spk} spikes ({fr:.2f} Hz, Peak Amp: {peak_amp:.1f} µV){manual_tag}",
            template="plotly_white",
            height=600,
            margin=dict(l=50, r=30, t=60, b=50),
            xaxis3=dict(title="Time in Recording (seconds)", type="linear")
        )
        fig.update_xaxes(range=[60.0, 75.0]) # Default 15-second window around first laser block

        div_html = fig.to_html(full_html=False, include_plotlyjs='cdn' if idx==0 else False)
        unit_html_snippets.append(f"""
        <div id="unit-section-{uid}" class="unit-card" style="margin-bottom: 40px; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #ffffff; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
            {div_html}
        </div>
        """)

    # Combine into Master HTML Document
    master_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>TP_Tacr1_1 Single-Channel blech_clust Spike Inspection Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f8f9fa; margin: 0; padding: 20px; color: #333; }}
        .header {{ background-color: #1a252f; color: white; padding: 20px 30px; border-radius: 8px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 24px; }}
        .header p {{ margin: 0; color: #bdc3c7; font-size: 14px; }}
        .nav-container {{ position: sticky; top: 0; z-index: 1000; background: white; padding: 15px 20px; border-radius: 8px; margin-bottom: 25px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
        .nav-label {{ font-weight: bold; margin-right: 10px; color: #2c3e50; }}
        .nav-btn {{ padding: 6px 14px; background: #ecf0f1; border: 1px solid #bdc3c7; border-radius: 20px; cursor: pointer; text-decoration: none; color: #2c3e50; font-size: 13px; font-weight: 500; transition: all 0.2s; }}
        .nav-btn:hover {{ background: #3498db; color: white; border-color: #3498db; }}
        .nav-btn.manual {{ background: #e8f8f5; border-color: #2ecc71; color: #27ae60; font-weight: bold; }}
        .nav-btn.manual:hover {{ background: #2ecc71; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Subject TP_Tacr1_1: Single-Channel blech_clust Spike Inspection Dashboard</h1>
        <p>Interactive verification of detected spike timestamps overlaid on 300-3000 Hz filtered AP voltage traces, Diaphragm EMG respiration, and Optogenetic Laser pulses across all 17 deduplicated single units.</p>
    </div>

    <div class="nav-container">
        <span class="nav-label">Jump to Unit:</span>
        {" ".join([f'<a class="nav-btn {"manual" if row["is_manual_pick"] else ""}" href="#unit-section-{int(row["cluster_id"])}">Unit {int(row["cluster_id"])} (Ch {int(row["ch"])})</a>' for _, row in info_df.iterrows()])}
    </div>

    {"".join(unit_html_snippets)}
</body>
</html>
"""

    html_out = out_dir / "TP_Tacr1_1_blech_interactive_spike_inspector.html"
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(master_html)

    print(f"\n" + "=" * 85)
    print(f" [SUCCESS] Generated Master Interactive Spike Inspection Dashboard!")
    print(f" File Location: {html_out}")
    print(f" Size: {html_out.stat().st_size / (1024**2):.2f} MB")
    print("=" * 85)

if __name__ == "__main__":
    main()
