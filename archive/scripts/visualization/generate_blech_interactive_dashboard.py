#!/usr/bin/env python3
"""
Generate Interactive HTML Visualizer for Single-Channel blech_clust Sorted Units.
Plots bandpass filtered AP voltage traces, detected spike timestamps, mean waveforms,
and Diaphragm EMG respiration / Optogenetic laser TTLs for interactive inspection.
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
from src.pipeline.processors.proc_blech_single_channel import filter_ap_channel, detect_spike_snippets

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

    laser_onsets = t_nidq[np.where(np.diff((ch6_laser > 0.8).astype(int)) == 1)[0]]

    # Prepare memmap read for AP channels
    mm_ap = np.memmap(ap_bin_file, dtype=np.int16, mode='r', shape=(n_samples_ap, 385))

    # Downsample time vectors for fast interactive Plotly rendering (downsample factor 30 for AP trace)
    ds_ap = 30
    t_ap_ds = np.arange(0, n_samples_ap, ds_ap) / fs_ap

    ds_nidq = 5
    t_nidq_ds = t_nidq[::ds_nidq]
    emg_env_ds = emg_proc.envelope[::ds_nidq]
    laser_ds = (ch6_laser[::ds_nidq] > 0.8).astype(float)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "1. Filtered AP Voltage Trace (300-3000 Hz) on Peak Channel & Detected Spikes",
            "2. Detected Single-Unit Spike Timestamps",
            "3. Diaphragm EMG Envelope & Breath Onset Triggers",
            "4. Optogenetic Laser Stimulation TTL (Channel 6)"
        ),
        row_heights=[0.40, 0.15, 0.25, 0.20]
    )

    # Initial unit: Cluster 0
    init_uid = 0
    init_ch = int(info_df.loc[info_df['cluster_id'] == init_uid, 'ch'].values[0])
    init_spks = spikes_times[spikes_clusters == init_uid]

    # Extract & filter AP voltage trace for initial channel
    raw_trace_init = mm_ap[:, init_ch].astype(np.float32)
    filt_trace_init = filter_ap_channel(raw_trace_init, fs=fs_ap)
    filt_trace_ds = filt_trace_init[::ds_ap]

    # Add AP Voltage Trace
    fig.add_trace(
        go.Scatter(
            x=t_ap_ds,
            y=filt_trace_ds,
            mode='lines',
            name=f'Filtered AP Ch {init_ch} (uV)',
            line=dict(color='#1f77b4', width=1.0)
        ),
        row=1, col=1
    )

    # Add Detected Spike Peaks Overlay on AP Trace
    spk_samples = (init_spks * fs_ap).astype(int)
    spk_samples = spk_samples[(spk_samples >= 0) & (spk_samples < len(filt_trace_init))]
    spk_amps = filt_trace_init[spk_samples]

    fig.add_trace(
        go.Scatter(
            x=init_spks,
            y=spk_amps,
            mode='markers',
            name=f'Unit {init_uid} Spikes',
            marker=dict(color='crimson', size=6, symbol='x')
        ),
        row=1, col=1
    )

    # Add Spike Raster (Track 2)
    fig.add_trace(
        go.Scatter(
            x=init_spks,
            y=np.ones(len(init_spks)),
            mode='markers',
            name='Spike Timestamps',
            marker=dict(color='crimson', size=5, symbol='line-ns-open')
        ),
        row=2, col=1
    )

    # Add Diaphragm EMG Envelope (Track 3)
    fig.add_trace(
        go.Scatter(
            x=t_nidq_ds,
            y=emg_env_ds,
            mode='lines',
            name='Diaphragm EMG Envelope (V)',
            line=dict(color='#2ca02c', width=1.2)
        ),
        row=3, col=1
    )

    # Add Diaphragm Breath Onsets
    fig.add_trace(
        go.Scatter(
            x=breath_onsets,
            y=np.zeros(len(breath_onsets)),
            mode='markers',
            name='Inspiration Onsets',
            marker=dict(color='blue', size=6, symbol='triangle-up')
        ),
        row=3, col=1
    )

    # Add Laser TTL (Track 4)
    fig.add_trace(
        go.Scatter(
            x=t_nidq_ds,
            y=laser_ds,
            mode='lines',
            name='Laser TTL',
            line=dict(color='magenta', width=1.5)
        ),
        row=4, col=1
    )

    # Create Dropdown Buttons for all 17 Units
    buttons = []
    for idx, row in info_df.iterrows():
        uid = int(row['cluster_id'])
        ch = int(row['ch'])
        sh = int(row['shank'])
        depth = float(row['depth'])
        n_spk = int(row['n_spikes'])
        fr = float(row['firing_rate'])

        # Load channel data
        raw_tr = mm_ap[:, ch].astype(np.float32)
        filt_tr = filter_ap_channel(raw_tr, fs=fs_ap)
        filt_ds = filt_tr[::ds_ap]

        u_spks = spikes_times[spikes_clusters == uid]
        u_samples = (u_spks * fs_ap).astype(int)
        u_samples = u_samples[(u_samples >= 0) & (u_samples < len(filt_tr))]
        u_amps = filt_tr[u_samples]

        button = dict(
            method="update",
            label=f"Unit {uid} (Ch {ch}, Shank {sh}, Depth {depth:.0f}um)",
            args=[
                {
                    "x": [t_ap_ds, u_spks, u_spks, t_nidq_ds, breath_onsets, t_nidq_ds],
                    "y": [filt_ds, u_amps, np.ones(len(u_spks)), emg_env_ds, np.zeros(len(breath_onsets)), laser_ds],
                    "name": [
                        f'Filtered AP Ch {ch} (uV)',
                        f'Unit {uid} Spikes',
                        'Spike Timestamps',
                        'Diaphragm EMG Envelope (V)',
                        'Inspiration Onsets',
                        'Laser TTL'
                    ]
                },
                {
                    "title": f"Subject TP_Tacr1_1: Unit {uid} Inspection (Channel {ch}, Shank {sh}, Depth {depth:.0f} um | {n_spk} spikes, {fr:.2f} Hz)"
                }
            ]
        )
        buttons.append(button)

    fig.update_layout(
        title=f"Subject TP_Tacr1_1: Interactive blech_clust Single-Unit Spike Inspector (Unit 0, Ch {init_ch}, Shank 3, Depth 720 um)",
        template="plotly_white",
        height=850,
        margin=dict(l=60, r=40, t=90, b=60),
        xaxis4=dict(title="Time in Recording (seconds)", rangeslider=dict(visible=True), type="linear"),
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=0.0, y=1.12,
                xanchor="left", yanchor="top",
                direction="down",
                showactive=True
            )
        ]
    )

    # Initial 5-second zoom window around first laser pulse
    fig.update_xaxes(range=[60.0, 65.0])

    html_out = out_dir / "TP_Tacr1_1_blech_interactive_spike_inspector.html"
    fig.write_html(str(html_out))
    print(f"\n[SUCCESS] Generated Interactive HTML Spike Inspector: {html_out}")

if __name__ == "__main__":
    main()
