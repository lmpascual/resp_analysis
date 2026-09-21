#!/usr/bin/env python3
"""
Generate Lightweight Individual HTML Visualizer Files (Top 50 Kilosort4 Good Units).
Creates fast-loading 20-40 MB HTML files for each of the top 50 good units sorted by firing rate
along with an index.html navigation page.
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

MANUAL_CHANNELS = [
    4, 12, 28, 32, 96, 108, 202, 210, 222, 224, 226, 290, 316,
    48, 58, 66, 82, 90, 154, 178, 186, 190, 242, 346, 364, 374,
    279, 283, 287, 343, 349, 91, 167
]

def main():
    out_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1")
    unit_dashboards_dir = out_dir / "unit_dashboards"
    unit_dashboards_dir.mkdir(parents=True, exist_ok=True)

    session_dir = out_dir / "alf_session"
    alf_dir = session_dir / "alf" / "probe00"
    raw_dir = Path(r"Z:\ramirezlab\pascual\rec_data\TP_Tacr1_1_npx2_g4")

    # Load Kilosort ALF sorted units & cluster info
    info_df = pd.read_csv(alf_dir / "cluster_info.tsv", sep="\t")
    spikes_times = np.load(alf_dir / "spikes.times.npy")
    spikes_clusters = np.load(alf_dir / "spikes.clusters.npy")

    # Filter for 'good' units and take top 50 by firing rate
    good_df = info_df[info_df['group'] == 'good'].copy()
    good_df = good_df.sort_values(by='firing_rate', ascending=False).head(50).reset_index(drop=True)

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

    ds_ap = 20 # 1500 Hz display resolution
    t_ap_ds = np.arange(0, n_samples_ap, ds_ap) / fs_ap

    ds_nidq = 5
    t_nidq_ds = t_nidq[::ds_nidq]
    emg_env_ds = emg_proc.envelope[::ds_nidq]
    laser_ds = (ch6_laser[::ds_nidq] > 0.8).astype(float)

    generated_files = []

    print("=" * 85)
    print(f" Generating Lightweight Individual HTML Dashboards for Top {len(good_df)} Good Units")
    print(f" Output Directory: {unit_dashboards_dir}")
    print("=" * 85)

    for rank, row in good_df.iterrows():
        uid = int(row['cluster_id'])
        ch = int(row['ch'])
        sh = int(row['shank'])
        depth = float(row['depth'])
        n_spk = int(row['n_spikes'])
        fr = float(row['firing_rate'])
        isi_v = float(row['isi_violations_ratio'])
        is_manual = ch in MANUAL_CHANNELS

        file_name = f"unit_{uid:04d}_ch{ch}_shank{sh}.html"
        unit_html_path = unit_dashboards_dir / file_name

        logger_str = f"Rank {rank+1:02d} | Unit {uid} (Ch {ch}, Shank {sh}, Depth {depth:.0f} um, FR {fr:.2f} Hz)"
        print(f"Generating {file_name} for {logger_str}...")

        # Extract & filter AP voltage trace on peak channel
        raw_tr = mm_ap[:, ch].astype(np.float32)
        filt_tr = filter_ap_channel(raw_tr, fs=fs_ap)
        filt_ds = filt_tr[::ds_ap]

        u_spks = spikes_times[spikes_clusters == uid]
        u_samples = (u_spks * fs_ap).astype(int)
        u_samples = u_samples[(u_samples >= 0) & (u_samples < len(filt_tr))]
        u_amps = filt_tr[u_samples] if len(u_samples) > 0 else np.array([0.0])

        # Estimate average peak-to-trough amplitude from up to 500 sampled waveforms
        if len(u_samples) > 0:
            sample_idx = u_samples[np.random.choice(len(u_samples), min(500, len(u_samples)), replace=False)]
            snippets = []
            for s in sample_idx:
                if s - 15 >= 0 and s + 30 < len(filt_tr):
                    snippets.append(filt_tr[s-15:s+30])
            if len(snippets) > 0:
                mean_snip = np.mean(snippets, axis=0)
                peak_amp = float(np.ptp(mean_snip))
            else:
                peak_amp = float(np.ptp(u_amps[:100])) if len(u_amps) > 0 else 0.0
        else:
            peak_amp = 0.0

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=(
                f"1. Filtered AP Voltage Trace (300-3000 Hz) on Peak Channel {ch} & Kilosort4 Spikes (red x)",
                "2. Diaphragm EMG Envelope & Breath Onset Triggers (blue triangles)",
                "3. Optogenetic Laser Stimulation TTL (Channel 6)"
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
            title=f"<b>Subject TP_Tacr1_1 — Unit {uid}</b> (Rank #{rank+1} | Peak Ch {ch} | Shank {sh} | Depth {depth:.0f} µm | {n_spk} spikes, {fr:.2f} Hz, Amp: {peak_amp:.1f} µV, ISI viol: {isi_v*100:.2f}%){manual_tag}",
            template="plotly_white",
            height=750,
            margin=dict(l=60, r=40, t=80, b=60),
            xaxis3=dict(title="Time in Recording (seconds)", type="linear")
        )
        fig.update_xaxes(range=[60.0, 75.0]) # Default 15-second zoom window

        fig.write_html(str(unit_html_path))
        file_mb = unit_html_path.stat().st_size / (1024**2)
        print(f" -> Saved {file_name} ({file_mb:.1f} MB)")

        generated_files.append({
            'rank': rank + 1,
            'uid': uid,
            'ch': ch,
            'sh': sh,
            'depth': depth,
            'n_spikes': n_spk,
            'fr': fr,
            'amp': peak_amp,
            'isi_v': isi_v,
            'is_manual': is_manual,
            'file_name': file_name,
            'size_mb': file_mb
        })

    # Create Index Launcher HTML Page
    index_html_path = unit_dashboards_dir / "index.html"
    rows_html = []
    for g in generated_files:
        match_badge = '<span style="background: #2ecc71; color: white; padding: 3px 8px; border-radius: 12px; font-size: 11px; font-weight: bold;">MANUAL MATCH</span>' if g['is_manual'] else '<span style="background: #3498db; color: white; padding: 3px 8px; border-radius: 12px; font-size: 11px;">Good Unit</span>'
        rows_html.append(f"""
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0; font-weight: bold;">#{g['rank']}</td>
            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0; font-weight: bold;">Unit {g['uid']}</td>
            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">Ch {g['ch']}</td>
            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">Shank {g['sh']}</td>
            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">{g['depth']:.0f} µm</td>
            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">{g['n_spikes']:,}</td>
            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0; font-weight: bold;">{g['fr']:.2f} Hz</td>
            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">{g['amp']:.1f} µV</td>
            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">{g['isi_v']*100:.2f}%</td>
            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">{match_badge}</td>
            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">
                <a href="{g['file_name']}" target="_blank" style="display: inline-block; padding: 6px 14px; background: #3498db; color: white; text-decoration: none; border-radius: 4px; font-size: 13px; font-weight: bold;">Open Dashboard ({g['size_mb']:.1f} MB) &rarr;</a>
            </td>
        </tr>
        """)

    index_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>TP_Tacr1_1 Top 50 Kilosort4 Good Units Dashboards</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f8f9fa; margin: 0; padding: 30px; color: #333; }}
        .card {{ background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); padding: 30px; margin-bottom: 30px; border: 1px solid #e0e0e0; }}
        h1 {{ margin-top: 0; color: #2c3e50; font-size: 24px; }}
        p {{ color: #7f8c8d; font-size: 14px; line-height: 1.5; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th {{ background: #f1f2f6; text-align: left; padding: 12px; border-bottom: 2px solid #dcdde1; color: #2c3e50; font-size: 13px; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>Subject TP_Tacr1_1: Top 50 Kilosort4 Good Units Index</h1>
        <p>Click any unit below to open its individual interactive HTML dashboard (20-40 MB). Each dashboard displays the 300-3000 Hz filtered AP voltage trace on that unit's peak channel with red spike markers, Diaphragm EMG respiration, and Optogenetic Laser pulses.</p>
        
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Unit ID</th>
                    <th>Peak Channel</th>
                    <th>Shank</th>
                    <th>Depth</th>
                    <th>Spikes</th>
                    <th>Firing Rate</th>
                    <th>Peak Amp</th>
                    <th>ISI Violations</th>
                    <th>Status</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows_html)}
            </tbody>
        </table>
    </div>
</body>
</html>
"""

    with open(index_html_path, "w", encoding="utf-8") as f:
        f.write(index_content)

    print("\n" + "=" * 85)
    print(f" [SUCCESS] Generated Dashboards for Top {len(good_df)} Kilosort4 Good Units & Index Page!")
    print(f" Index Launcher: {index_html_path}")
    print("=" * 85)

if __name__ == "__main__":
    main()
