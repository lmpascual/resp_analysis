#!/usr/bin/env python3
r"""
Interactive Spiking Audio Population Raster Dashboard (Web Audio API)
Loaded EXCLUSIVELY for the Top 24 Well-Isolated Phasic Single Units.

Allows the user to click on any of the 24 units in the raster plot (or sidebar)
to synthesize and play its spiking activity as crisp audio clicks in real time!

Outputs:
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\interactive_spiking_audio_raster.html
"""

import sys
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

def main():
    out_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation")
    out_dir.mkdir(parents=True, exist_ok=True)
    session_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1\alf_session")
    alf_dir = session_dir / "alf" / "probe00"
    raw_dir = Path(r"Z:\ramirezlab\pascual\rec_data\TP_Tacr1_1_npx2_g4")
    artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\0e7be222-f160-4bc2-921c-f3fdf8b7825e")

    print("=" * 90)
    print(" Building Interactive Spiking Audio Dashboard (STRICTLY TOP 24 PHASIC UNITS)")
    print(f" ALF Directory: {alf_dir}")
    print(f" Output Directory: {out_dir}")
    print("=" * 90)

    # 1. Load Top 24 Phasic Units CSV
    top24_csv = out_dir / "top_24_phasic_single_units.csv"
    if not top24_csv.exists():
        print(f" Error: {top24_csv} not found. Running PSTH script first...")
        from scripts.plot_top_phasic_units_psth import main as run_psth
        run_psth()

    top24_df = pd.read_csv(top24_csv)
    
    # Sort units by Shank & Depth
    sorted_df = top24_df.sort_values(by=['shank', 'depth'], ascending=[True, True]).reset_index(drop=True)
    sorted_uids = list(sorted_df['cluster_id'])
    n_sorted = len(sorted_uids)

    spikes_times = np.load(alf_dir / "spike_times.npy")
    spikes_clusters = np.load(alf_dir / "spike_clusters.npy")
    info_df = pd.read_csv(alf_dir / "cluster_info.tsv", sep="\t")

    fs_ap = 30000.0
    if spikes_times.dtype in [np.int64, np.int32] and spikes_times.max() > 100000:
        spikes_times = spikes_times / fs_ap

    MAX_TIME_S = 60.0 # First 60 seconds baseline epoch
    spk_mask = (spikes_times >= 0.0) & (spikes_times <= MAX_TIME_S) & (np.isin(spikes_clusters, sorted_uids))
    spikes_times = spikes_times[spk_mask]
    spikes_clusters = spikes_clusters[spk_mask]

    # Build dictionary of spike times per unit
    unit_spiketrains_dict = {}
    for uid in sorted_uids:
        u_spks = spikes_times[spikes_clusters == uid]
        unit_spiketrains_dict[int(uid)] = np.round(u_spks, 4).tolist()

    # 2. Load EMG Signal for 0-60s
    nidq_meta = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.meta"
    nidq_bin = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.bin"
    nidq_data = read_nidq_bin(nidq_bin, nidq_meta)

    t_nidq = nidq_data.time_vector
    fs_nidq = nidq_data.fs

    nidq_mask = (t_nidq >= 0.0) & (t_nidq <= MAX_TIME_S)
    t_nidq = t_nidq[nidq_mask]
    emg_raw = nidq_data.all_channels[0][nidq_mask]

    emg_cfg = EMGConfig(fs=fs_nidq, highpass_cutoff=300.0, lowpass_cutoff=3000.0, notch_freq=60.0, smoothing_window_ms=20.0)
    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    emg_env = emg_proc.envelope
    breath_onsets = emg_proc.breath_onsets

    # Subsample EMG for HTML rendering
    sub_step = 20
    t_emg_sub = np.round(t_nidq[::sub_step], 3).tolist()
    emg_sub = np.round(emg_env[::sub_step], 2).tolist()

    # Unit Metadata JSON
    unit_meta_list = []
    for idx, row in sorted_df.iterrows():
        uid = int(row['cluster_id'])
        is_insp = (row['pref_phase'] >= 0)
        unit_meta_list.append({
            'unit_id': uid,
            'y_index': idx + 1,
            'shank': int(row['shank']),
            'depth': float(row['depth']),
            'firing_rate': float(row['firing_rate_60s']),
            'vector_strength': float(row['vector_strength']),
            'phase': 'Inspiratory' if is_insp else 'Expiratory',
            'snr': float(row['snr']),
            'isi_violations': float(row['isi_violations']),
            'label': f"Unit {uid} (S{int(row['shank'])}, {int(row['depth'])}µm) - {'Insp' if is_insp else 'Exp'}"
        })

    # 3. Construct HTML + Web Audio API Dashboard Page
    json_spikes = json.dumps(unit_spiketrains_dict)
    json_meta = json.dumps(unit_meta_list)
    json_emg_t = json.dumps(t_emg_sub)
    json_emg_v = json.dumps(emg_sub)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Subject TP_Tacr1_1: Interactive Spiking Audio Dashboard (Top 24 Phasic Units)</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 15px;
            background-color: #121212;
            color: #ffffff;
        }}
        .header-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #1e1e1e;
            padding: 12px 20px;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        h1 {{
            margin: 0;
            font-size: 20px;
            color: #00d2ff;
        }}
        .controls {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        button {{
            background: #00d2ff;
            color: #121212;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        button:hover {{
            background: #80e5ff;
            transform: scale(1.03);
        }}
        button:disabled {{
            background: #444;
            color: #888;
            cursor: not-allowed;
        }}
        select, input[type=range] {{
            background: #2a2a2a;
            color: #fff;
            border: 1px solid #444;
            padding: 6px;
            border-radius: 4px;
        }}
        .main-container {{
            display: flex;
            gap: 15px;
        }}
        #plot-div {{
            flex: 1;
            background: #1e1e1e;
            border-radius: 8px;
            padding: 10px;
            height: 750px;
        }}
        .unit-sidebar {{
            width: 320px;
            background: #1e1e1e;
            border-radius: 8px;
            padding: 15px;
            max-height: 750px;
            overflow-y: auto;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .unit-item {{
            padding: 8px 12px;
            margin-bottom: 6px;
            background: #2a2a2a;
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.2s ease;
        }}
        .unit-item:hover {{
            background: #3a3a3a;
        }}
        .unit-item.active {{
            background: #d62728;
            color: #fff;
            font-weight: bold;
        }}
        .active-info {{
            background: #262626;
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 12px;
            border-left: 4px solid #00d2ff;
        }}
    </style>
</head>
<body>

<div class="header-bar">
    <div>
        <h1>🎧 Top 24 Phasic Units Spiking Audio Raster</h1>
        <small style="color: #aaa;">Click any unit on the raster or list to play its spiking audio in real-time!</small>
    </div>
    <div class="controls">
        <button id="btn-play">▶ Play Audio</button>
        <button id="btn-pause" disabled>⏸ Pause</button>
        <label>Speed: 
            <select id="sel-speed">
                <option value="0.5">0.5x Slow</option>
                <option value="1.0" selected>1.0x Realtime</option>
                <option value="2.0">2.0x Fast</option>
            </select>
        </label>
        <label>Volume: 
            <input type="range" id="rng-vol" min="0" max="1" step="0.05" value="0.7">
        </label>
    </div>
</div>

<div class="active-info" id="info-card">
    <b>Currently Selected Unit:</b> None (Click any of the 24 units to synthesize spiking audio)
</div>

<div class="main-container">
    <div id="plot-div"></div>
    <div class="unit-sidebar">
        <h3 style="margin-top:0; font-size:15px; color:#00d2ff;">Top 24 Phasic Single Units</h3>
        <div id="unit-list"></div>
    </div>
</div>

<script>
    const SPIKES_DATA = {json_spikes};
    const UNITS_META = {json_meta};
    const EMG_T = {json_emg_t};
    const EMG_V = {json_emg_v};

    let selectedUnitId = null;
    let audioCtx = null;
    let isPlaying = false;
    let currentPlaybackTime = 0.0;
    let playbackSpeed = 1.0;
    let volumeLevel = 0.7;
    let scheduledTimeOuts = [];

    function getAudioContext() {{
        if (!audioCtx) {{
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }}
        return audioCtx;
    }}

    function playSpikeClick(audioTime) {{
        if (!audioCtx) return;
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();

        osc.type = 'sine';
        osc.frequency.setValueAtTime(850, audioTime);
        osc.frequency.exponentialRampToValueAtTime(120, audioTime + 0.004);

        gain.gain.setValueAtTime(volumeLevel, audioTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, audioTime + 0.005);

        osc.connect(gain);
        gain.connect(audioCtx.destination);

        osc.start(audioTime);
        osc.stop(audioTime + 0.006);
    }}

    const traces = [];

    // Subplot 1: EMG
    traces.push({{
        x: EMG_T,
        y: EMG_V,
        mode: 'lines',
        line: {{color: '#00d2ff', width: 1.2}},
        name: 'Diaphragm EMG',
        xaxis: 'x',
        yaxis: 'y'
    }});

    // Subplot 2: 24 Phasic Units
    UNITS_META.forEach((u, i) => {{
        const spks = SPIKES_DATA[u.unit_id] || [];
        const tickColor = (u.phase === 'Inspiratory') ? '#ff4444' : '#ffffff';
        traces.push({{
            x: spks,
            y: Array(spks.length).fill(u.y_index),
            mode: 'markers',
            marker: {{
                symbol: 'line-ns-open',
                color: tickColor,
                size: 7,
                line: {{width: 1.4}}
            }},
            hoverinfo: 'text',
            text: spks.map(t => `${{u.label}}<br>Vector Strength R: ${{u.vector_strength}}<br>Spike Time: ${{t}}s<br>FR: ${{u.firing_rate}}Hz`),
            name: `Unit ${{u.unit_id}}`,
            unit_id: u.unit_id,
            xaxis: 'x',
            yaxis: 'y2'
        }});
    }});

    const layout = {{
        grid: {{rows: 2, columns: 1, roworder: 'top to bottom', pattern: 'independent'}},
        paper_bgcolor: '#1e1e1e',
        plot_bgcolor: '#1e1e1e',
        font: {{color: '#ffffff'}},
        showlegend: false,
        height: 720,
        margin: {{l: 60, r: 20, t: 30, b: 40}},
        xaxis: {{
            domain: [0, 1],
            anchor: 'y',
            showticklabels: false,
            gridcolor: '#333'
        }},
        yaxis: {{
            domain: [0.78, 1],
            anchor: 'x',
            title: 'EMG (µV)',
            gridcolor: '#333'
        }},
        xaxis2: {{
            domain: [0, 1],
            anchor: 'y2',
            title: 'Session Time (s)',
            gridcolor: '#333',
            range: [0, 60]
        }},
        yaxis2: {{
            domain: [0, 0.72],
            anchor: 'x2',
            title: 'Top 24 Phasic Units (Sorted by Depth)',
            gridcolor: '#333'
        }}
    }};

    Plotly.newPlot('plot-div', traces, layout, {{responsive: true}}).then(gd => {{
        gd.on('plotly_click', data => {{
            if (data.points && data.points.length > 0) {{
                const pt = data.points[0];
                if (pt.data.unit_id) {{
                    selectUnit(pt.data.unit_id);
                }}
            }}
        }});
    }});

    // Sidebar List
    const listDiv = document.getElementById('unit-list');
    UNITS_META.forEach(u => {{
        const item = document.createElement('div');
        item.className = 'unit-item';
        item.id = `unit-item-${{u.unit_id}}`;
        const cTag = (u.phase === 'Inspiratory') ? 'style="color:#ff6666;"' : '';
        item.innerHTML = `<span ${{cTag}}><b>U${{u.unit_id}}</b> (${{u.depth}}µm)</span><small>R=${{u.vector_strength}}</small>`;
        item.onclick = () => selectUnit(u.unit_id);
        listDiv.appendChild(item);
    }});

    function selectUnit(uid) {{
        selectedUnitId = uid;
        const u = UNITS_META.find(x => x.unit_id === uid);
        
        document.querySelectorAll('.unit-item').forEach(el => el.classList.remove('active'));
        const activeEl = document.getElementById(`unit-item-${{uid}}`);
        if (activeEl) activeEl.classList.add('active');

        document.getElementById('info-card').innerHTML = `
            <b>Selected: Unit ${{u.unit_id}}</b> (${{u.phase}}) | Shank ${{u.shank}}, Depth ${{u.depth}}µm | Vector Strength R: ${{u.vector_strength}} | Firing Rate: ${{u.firing_rate}} Hz | SNR: ${{u.snr}}
        `;

        UNITS_META.forEach((meta, idx) => {{
            const isTarget = (meta.unit_id === uid);
            const baseColor = (meta.phase === 'Inspiratory') ? '#ff4444' : '#ffffff';
            const color = isTarget ? '#ffff00' : baseColor;
            const size = isTarget ? 10 : 7;
            Plotly.restyle('plot-div', {{'marker.color': color, 'marker.size': size}}, [idx + 1]);
        }});

        if (isPlaying) {{
            startAudioPlayback(currentPlaybackTime);
        }}
    }}

    document.getElementById('btn-play').onclick = () => {{
        if (!selectedUnitId) {{
            alert("Please click a unit first to select its spiking audio!");
            return;
        }}
        getAudioContext().resume();
        isPlaying = true;
        document.getElementById('btn-play').disabled = true;
        document.getElementById('btn-pause').disabled = false;
        startAudioPlayback(currentPlaybackTime);
    }};

    document.getElementById('btn-pause').onclick = () => {{
        isPlaying = false;
        document.getElementById('btn-play').disabled = false;
        document.getElementById('btn-pause').disabled = true;
        clearScheduledClicks();
    }};

    document.getElementById('sel-speed').onchange = (e) => {{
        playbackSpeed = parseFloat(e.target.value);
        if (isPlaying) startAudioPlayback(currentPlaybackTime);
    }};

    document.getElementById('rng-vol').oninput = (e) => {{
        volumeLevel = parseFloat(e.target.value);
    }};

    function clearScheduledClicks() {{
        scheduledTimeOuts.forEach(t => clearTimeout(t));
        scheduledTimeOuts = [];
    }}

    function startAudioPlayback(startSec) {{
        clearScheduledClicks();
        if (!selectedUnitId) return;

        const ctx = getAudioContext();
        const spks = SPIKES_DATA[selectedUnitId] || [];
        const upcomingSpikes = spks.filter(t => t >= startSec);

        const nowAudio = ctx.currentTime;

        upcomingSpikes.forEach(spkTime => {{
            const delaySec = (spkTime - startSec) / playbackSpeed;
            if (delaySec >= 0 && delaySec <= 60) {{
                const targetAudioTime = nowAudio + delaySec;
                playSpikeClick(targetAudioTime);
            }}
        }});
    }}
</script>
</body>
</html>
"""

    html_out = out_dir / "interactive_spiking_audio_raster.html"
    with open(html_out, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f" Saved Interactive Audio Dashboard (Strictly Top 24 Units): {html_out}")

    print("\n==========================================================================================")
    print(" [SUCCESS] Interactive Spiking Audio Dashboard Creation Complete!")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
