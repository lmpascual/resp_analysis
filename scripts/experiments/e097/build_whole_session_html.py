#!/usr/bin/env python3
"""
Compiles the standalone interactive HTML dashboard for e097_g2:
- Synchronized multi-track Plotly timecourse (EMG, IBI, Heart Rate, Laser)
- Interactive epoch jump buttons for all milestones in e097.docx
- 1 kHz high-resolution raw waveform zoom modals for key events
- Complete surgical protocol, laser tables, and surgeon notes
"""

import json
from pathlib import Path

DATA_JSON = Path("outputs/e097/e097_whole_session_data.json")
OUT_HTML = Path("outputs/e097/e097_session_wide_cardiorespiratory_dashboard.html")

def build_html():
    print(f"Reading {DATA_JSON}...")
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    meta = data["metadata"]
    trains = data["laser_trains"]
    milestones = data["milestones"]
    tl = data["timeline"]
    sc = data["ibi_scatter"]
    snaps = data["snapshots"]
    
    # Pre-render laser trains table rows
    trains_table_rows = ""
    for tr in trains:
        m, s = divmod(tr["t_start"], 60)
        trains_table_rows += f"""
        <tr>
          <td><span class="badge-laser">Train {tr['id']}</span></td>
          <td><b>{int(m):02d}:{s:05.2f}</b> ({tr['t_start']:.1f}s)</td>
          <td>{tr['dur']:.3f} s</td>
          <td>{tr['pulses']}</td>
          <td>{tr['freq']:.1f} Hz</td>
          <td>{tr['note']}</td>
        </tr>
        """

    # Pre-render milestones jump buttons
    milestone_buttons = ""
    for i, ms in enumerate(milestones):
        m0, s0 = divmod(ms["t_start"], 60)
        m1, s1 = divmod(ms["t_end"], 60)
        btn_class = "btn-laser" if "LASER" in ms["badge"] else ("btn-anox" if "ANOXIA" in ms["badge"] else ("btn-dr" if "DR" in ms["badge"] else ("btn-seiz" if "SEIZURE" in ms["badge"] or "STATUS" in ms["badge"] else "btn-neutral")))
        milestone_buttons += f"""
        <button class="ms-btn {btn_class}" onclick="jumpToEpoch({ms['t_start']}, {ms['t_end']})">
          <span class="ms-badge">{ms['badge']}</span>
          <span class="ms-time">{int(m0):02d}:{s0:02.0f} - {int(m1):02d}:{s1:02.0f}</span>
          <span class="ms-name">{ms['name']}</span>
        </button>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>e097_g2 Whole-Session Cardiorespiratory & Opto-Stimulation Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg-dark: #07090e;
      --bg-card: #0f1523;
      --border: rgba(255, 255, 255, 0.08);
      --border-accent: rgba(56, 189, 248, 0.3);
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --emg-color: #00f5d4;
      --ibi-color: #f59e0b;
      --ecg-color: #ff0055;
      --laser-color: #a855f7;
      --success: #10b981;
      --danger: #ef4444;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg-dark);
      color: var(--text);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      line-height: 1.5;
      padding: 24px 32px 80px;
    }}
    header {{
      background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
      border: 1px solid rgba(99, 102, 241, 0.3);
      border-radius: 16px;
      padding: 24px 32px;
      margin-bottom: 24px;
      box-shadow: 0 12px 36px rgba(0,0,0,0.5);
    }}
    .header-top {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
      gap: 16px;
      margin-bottom: 12px;
    }}
    h1 {{
      font-size: 24px;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 13.5px;
      margin-top: 4px;
      max-width: 980px;
    }}
    .meta-pills {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 12px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border);
      border-radius: 20px;
      font-size: 12px;
      font-family: 'JetBrains Mono', monospace;
      color: #cbd5e1;
    }}
    .pill b {{ color: #fff; }}

    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
      margin-bottom: 24px;
    }}
    .stat-card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px 20px;
      display: flex;
      flex-direction: column;
    }}
    .stat-val {{
      font-size: 24px;
      font-weight: 800;
      font-family: 'JetBrains Mono', monospace;
    }}
    .stat-lbl {{
      font-size: 11.5px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-top: 4px;
    }}

    .section-card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 22px 26px;
      margin-bottom: 24px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }}
    .sec-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .sec-title {{
      font-size: 16px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .sec-title-icon {{
      font-size: 18px;
    }}

    .navigator-grid {{
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 10px;
      scrollbar-width: thin;
    }}
    .ms-btn {{
      flex: 0 0 auto;
      background: #131c31;
      border: 1px solid #1e293b;
      border-radius: 10px;
      padding: 10px 14px;
      text-align: left;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      gap: 3px;
      transition: all 0.2s ease;
      min-width: 155px;
    }}
    .ms-btn:hover {{
      transform: translateY(-2px);
      box-shadow: 0 6px 16px rgba(0,0,0,0.4);
      border-color: var(--primary);
    }}
    .ms-badge {{
      font-size: 9.5px;
      font-weight: 800;
      letter-spacing: 0.05em;
      padding: 2px 6px;
      border-radius: 4px;
      display: inline-block;
      width: fit-content;
    }}
    .ms-time {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      color: var(--text-muted);
    }}
    .ms-name {{
      font-size: 12px;
      font-weight: 600;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    .btn-laser .ms-badge {{ background: rgba(168, 85, 247, 0.25); color: #c084fc; border: 1px solid #a855f7; }}
    .btn-anox .ms-badge {{ background: rgba(56, 189, 248, 0.25); color: #38bdf8; border: 1px solid #0284c7; }}
    .btn-dr .ms-badge {{ background: rgba(16, 185, 129, 0.25); color: #34d399; border: 1px solid #10b981; }}
    .btn-seiz .ms-badge {{ background: rgba(239, 68, 68, 0.25); color: #f87171; border: 1px solid #dc2626; }}
    .btn-neutral .ms-badge {{ background: rgba(148, 163, 184, 0.2); color: #94a3b8; border: 1px solid #475569; }}

    #mainPlot {{
      width: 100%;
      height: 780px;
    }}

    .zoom-ctrls {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    .ctrl-btn {{
      background: #1e293b;
      color: #e2e8f0;
      border: 1px solid #334155;
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .ctrl-btn:hover {{
      background: var(--primary);
      color: #07090e;
      border-color: var(--primary);
    }}

    /* High-res snapshot tab viewer */
    .snap-tabs {{
      display: flex;
      gap: 10px;
      margin-bottom: 16px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 10px;
      flex-wrap: wrap;
    }}
    .snap-tab-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 13px;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .snap-tab-btn.active {{
      background: #1e293b;
      color: var(--primary);
    }}
    #snapPlot {{
      width: 100%;
      height: 480px;
    }}

    /* Table styling */
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12.5px;
      margin-top: 10px;
    }}
    th {{
      text-align: left;
      padding: 10px 12px;
      background: #131c31;
      color: var(--text-muted);
      font-weight: 600;
      border-bottom: 1px solid var(--border);
    }}
    td {{
      padding: 9px 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      color: #e2e8f0;
    }}
    tr:hover td {{
      background: rgba(255, 255, 255, 0.02);
    }}
    .badge-laser {{
      background: rgba(168, 85, 247, 0.2);
      color: #c084fc;
      border: 1px solid rgba(168, 85, 247, 0.4);
      padding: 3px 8px;
      border-radius: 6px;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
    }}

    .notes-box {{
      background: #090d16;
      border: 1px solid #1e293b;
      border-radius: 10px;
      padding: 16px 20px;
      font-size: 13px;
      color: #cbd5e1;
      max-height: 280px;
      overflow-y: auto;
      font-family: 'Inter', sans-serif;
      line-height: 1.6;
    }}
    .notes-entry {{
      margin-bottom: 12px;
      border-bottom: 1px dashed rgba(255, 255, 255, 0.06);
      padding-bottom: 10px;
    }}
    .notes-time {{
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      color: var(--primary);
      margin-right: 8px;
    }}
  </style>
</head>
<body>

  <header>
    <div class="header-top">
      <div>
        <h1>e097_g2 Whole-Session Cardiorespiratory &amp; Opto Dynamics Dashboard</h1>
        <p class="subtitle">Complete 110-minute continuous analysis of Diaphragm EMG breathing motor patterns, cycle-by-cycle Inter-Breath Intervals (IBI), ECG Heart Rate, 9 Optogenetic Laser Stimulation trials, and live surgical protocol milestones from <code>e097.docx</code>.</p>
      </div>
      <div class="zoom-ctrls">
        <button class="ctrl-btn" onclick="resetZoom()">Full Session (110m)</button>
        <button class="ctrl-btn" onclick="jumpToEpoch(0, 1800)">First 30m</button>
        <button class="ctrl-btn" onclick="jumpToEpoch(1800, 3600)">Middle 30m</button>
        <button class="ctrl-btn" onclick="jumpToEpoch(3600, 6607)">Late / Seizure</button>
      </div>
    </div>
    <div class="meta-pills">
      <span class="pill">Animal: <b>{meta['animal_id']}</b> ({meta['sex']}, {meta['weight_g']}g)</span>
      <span class="pill">Strain: <b>{meta['strain']}</b></span>
      <span class="pill">Surgeon: <b>{meta['surgeon']}</b> ({meta['date']})</span>
      <span class="pill">Anesthesia: <b>{meta['anesthesia']}</b></span>
      <span class="pill">Target: <b>{meta['target']}</b></span>
      <span class="pill">Probe: <b>{meta['probe']}</b></span>
    </div>
  </header>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-val" style="color: var(--primary);">{meta['total_duration_min']:.1f} min</div>
      <div class="stat-lbl">Recording Duration (6,607 s)</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color: var(--emg-color);">{meta['total_breaths_detected']:,}</div>
      <div class="stat-lbl">Diaphragm Breaths Detected</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color: var(--ecg-color);">{meta['total_heartbeats_detected']:,}</div>
      <div class="stat-lbl">ECG Heartbeats Detected</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color: var(--laser-color);">9 Trains (780 Pulses)</div>
      <div class="stat-lbl">Opto Laser Stimulations</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color: #34d399;">{meta['baseline_hr_bpm']} BPM</div>
      <div class="stat-lbl">Baseline Heart Rate</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color: #fbbf24;">{meta['baseline_rr_bpm']} BPM</div>
      <div class="stat-lbl">Baseline Resp Rate ({meta['baseline_ibi_s']}s IBI)</div>
    </div>
  </div>

  <!-- Interactive Milestone Navigator -->
  <div class="section-card">
    <div class="sec-header">
      <div class="sec-title">
        <span class="sec-title-icon">🧭</span>
        <span>Surgical Protocol Epoch Jump Navigator (from e097.docx)</span>
      </div>
      <span style="font-size: 12px; color: var(--text-muted);">Click any event to immediately zoom the interactive timecourse below to that epoch:</span>
    </div>
    <div class="navigator-grid">
      {milestone_buttons}
    </div>
  </div>

  <!-- Main 4-Track Plotly Synchronized Timecourse -->
  <div class="section-card">
    <div class="sec-header">
      <div class="sec-title">
        <span class="sec-title-icon">📈</span>
        <span>Synchronized 4-Track Whole-Session Cardiorespiratory Timecourse</span>
      </div>
      <div style="font-size: 12px; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;">
        Scrub range slider below to zoom | Hover for cycle values | Double-click to reset
      </div>
    </div>
    <div id="mainPlot"></div>
  </div>

  <!-- High-Resolution 1 kHz Raw Trace Zoom Modal -->
  <div class="section-card">
    <div class="sec-header">
      <div class="sec-title">
        <span class="sec-title-icon">🔬</span>
        <span>High-Resolution Raw Waveform Zoom (1,000 Hz Sampling)</span>
      </div>
      <span style="font-size: 12px; color: var(--text-muted);">Inspect millisecond-level physiological coupling during critical perturbations:</span>
    </div>
    <div class="snap-tabs">
      <button class="snap-tab-btn active" onclick="loadSnapshot('laser1', this)">1. Laser Train 1 Apnea (850–868s / 14:10)</button>
      <button class="snap-tab-btn" onclick="loadSnapshot('anoxia1', this)">2. Anoxia #1 Gasping (1800–1835s / 30:00)</button>
      <button class="snap-tab-btn" onclick="loadSnapshot('dr2', this)">3. Diving Reflex #2 Apnea (3325–3355s / 55:25)</button>
      <button class="snap-tab-btn" onclick="loadSnapshot('seizure', this)">4. Status Epilepticus Bursts (5420–5450s / 90:20)</button>
    </div>
    <div id="snapPlot"></div>
  </div>

  <!-- Protocol Details & Surgical Log from e097.docx -->
  <div class="section-card" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 24px;">
    <div>
      <div class="sec-title" style="margin-bottom: 12px;">
        <span class="sec-title-icon">⚡</span>
        <span>Optogenetic Laser Stimulation Trials (Channel 6)</span>
      </div>
      <div style="overflow-x: auto;">
        <table>
          <thead>
            <tr>
              <th>Trial</th>
              <th>Onset</th>
              <th>Duration</th>
              <th>Pulses</th>
              <th>Freq</th>
              <th>Outcome / Observation</th>
            </tr>
          </thead>
          <tbody>
            {trains_table_rows}
          </tbody>
        </table>
      </div>
    </div>

    <div>
      <div class="sec-title" style="margin-bottom: 12px;">
        <span class="sec-title-icon">📝</span>
        <span>Live Surgical Observations &amp; Notes (from e097.docx verbatim)</span>
      </div>
      <div class="notes-box">
        <div class="notes-entry">
          <span class="notes-time">04:00</span>
          Went through a bunch of npx and files but currently on e097_g2 and it's going well with the t3 imro just shank1. One of the best VRC we have had in a while!
        </div>
        <div class="notes-entry">
          <span class="notes-time">05:00</span>
          Tissue settling / baseline data. 4-shank probe inserted pretty far, 4500 depth, so that we can delete bottom channels for motion correction if desired.
        </div>
        <div class="notes-entry">
          <span class="notes-time">14:00</span>
          Bilateral light stim (Train 1) - effective, rapidly recovered, had a few mouth breaths not captured on flow meter.
        </div>
        <div class="notes-entry">
          <span class="notes-time">17:30</span>
          A 5-second stim (Train 4). It's severe but animal seems to recover pretty fast.
        </div>
        <div class="notes-entry">
          <span class="notes-time">18:40</span>
          Another stim (Train 5). Differentiates between neurons that are "inspiratory-off" vs "pre-inspiratory".
        </div>
        <div class="notes-entry">
          <span class="notes-time">20:00</span>
          Another stim (Train 6). AP75 appears to be firmly expiratory, goes away during apnea.
        </div>
        <div class="notes-entry">
          <span class="notes-time">30:00</span>
          15 second anoxia #1. Strong gasping and we have the return of the rostral gasping group! Massive delay between gas meter and animal response.
        </div>
        <div class="notes-entry">
          <span class="notes-time">34:38</span>
          It just did a weird thing of active expiration. Strong transition in nasal airflow waveform.
        </div>
        <div class="notes-entry">
          <span class="notes-time">38:30</span>
          15 second anoxia #2. Flow waveform transitioned during gasping to become more expiratory.
        </div>
        <div class="notes-entry">
          <span class="notes-time">46:00</span>
          Diving Reflex #1 (water delivery to face).
        </div>
        <div class="notes-entry">
          <span class="notes-time">55:30</span>
          Diving Reflex #2 - definitely some good apnea in there. Lose flow measurement temporarily as mask is pulled to deliver water.
        </div>
        <div class="notes-entry">
          <span class="notes-time">1:02:00</span>
          One more long laser stim (Train 9, 10s) before seizure.
        </div>
        <div class="notes-entry">
          <span class="notes-time">1:06:30</span>
          Kainic acid injection (12 mg/kg).
        </div>
        <div class="notes-entry">
          <span class="notes-time">1:16:43</span>
          Seeing some diaphragm tonicity already.
        </div>
        <div class="notes-entry">
          <span class="notes-time">1:30:00</span>
          The breaths are HUGE now. ECG has a lot of tonicity / seizure activity.
        </div>
        <div class="notes-entry">
          <span class="notes-time">1:48:00</span>
          Terminal arrest. Significant tonicity, full hindlimb extension.
        </div>
      </div>
    </div>
  </div>

  <script>
    const timelineData = {json.dumps(tl)};
    const ibiScatter = {json.dumps(sc)};
    const snapshots = {json.dumps(snaps)};
    const laserTrains = {json.dumps(trains)};

    // Create laser vertical shading shapes
    const laserShapes = laserTrains.map(tr => ({{
      type: 'rect',
      xref: 'x',
      yref: 'paper',
      x0: tr.t_start,
      x1: tr.t_start + tr.dur,
      y0: 0,
      y1: 1,
      fillcolor: 'rgba(168, 85, 247, 0.25)',
      line: {{ width: 1, color: '#a855f7' }}
    }}));

    // 4 Synchronized Tracks
    // Track 1: Respiration Rate & Integrated EMG (Subplot 1)
    // Track 2: Inter-Breath Interval (IBI, seconds) (Subplot 2)
    // Track 3: Instantaneous Heart Rate (ECG) (Subplot 3)
    // Track 4: Laser Command State (Subplot 4)

    const traceRespRate = {{
      x: timelineData.time_s,
      y: timelineData.resp_rate_bpm,
      name: 'Resp Rate (BPM)',
      type: 'scatter',
      mode: 'lines',
      line: {{ color: '#00f5d4', width: 1.5 }},
      xaxis: 'x',
      yaxis: 'y1'
    }};

    const traceEMGEnv = {{
      x: timelineData.time_s,
      y: timelineData.emg_envelope,
      name: 'Int EMG (V)',
      type: 'scatter',
      mode: 'lines',
      line: {{ color: '#2dd4bf', width: 1, dash: 'dot' }},
      xaxis: 'x',
      yaxis: 'y1_2',
      opacity: 0.6
    }};

    const traceIBICurve = {{
      x: timelineData.time_s,
      y: timelineData.ibi_s,
      name: 'IBI Mean (s)',
      type: 'scatter',
      mode: 'lines',
      line: {{ color: '#f59e0b', width: 1.5 }},
      xaxis: 'x',
      yaxis: 'y2'
    }};

    const traceIBIScatter = {{
      x: ibiScatter.time_s,
      y: ibiScatter.ibi_s,
      name: 'Single Breath IBI',
      type: 'scatter',
      mode: 'markers',
      marker: {{
        size: 2.5,
        color: ibiScatter.ibi_s.map(v => v >= 2.0 ? '#ef4444' : (v >= 1.0 ? '#f59e0b' : '#38bdf8')),
        opacity: 0.7
      }},
      xaxis: 'x',
      yaxis: 'y2'
    }};

    const traceHR = {{
      x: timelineData.time_s,
      y: timelineData.hr_bpm,
      name: 'Heart Rate (BPM)',
      type: 'scatter',
      mode: 'lines',
      line: {{ color: '#ff0055', width: 1.5 }},
      xaxis: 'x',
      yaxis: 'y3'
    }};

    const traceLaser = {{
      x: timelineData.time_s,
      y: timelineData.laser_state,
      name: 'Laser State (V)',
      type: 'scatter',
      mode: 'lines',
      fill: 'tozeroy',
      line: {{ color: '#a855f7', width: 1.5 }},
      xaxis: 'x',
      yaxis: 'y4'
    }};

    const mainLayout = {{
      paper_bgcolor: 'transparent',
      plot_bgcolor: '#0a0e17',
      margin: {{ l: 60, r: 50, t: 30, b: 50 }},
      hovermode: 'x unified',
      showlegend: false,
      grid: {{
        rows: 4,
        columns: 1,
        pattern: 'independent',
        roworder: 'top to bottom'
      }},
      xaxis: {{
        domain: [0, 1],
        range: [0, 6607],
        showgrid: true,
        gridcolor: 'rgba(255,255,255,0.05)',
        tickfont: {{ color: '#94a3b8', family: 'JetBrains Mono' }},
        rangeslider: {{ visible: true, thickness: 0.05, bgcolor: '#0f172a' }},
        title: {{ text: 'Recording Time (Seconds)', font: {{ color: '#94a3b8', size: 12 }} }}
      }},
      // Y1: Resp Rate
      yaxis: {{
        title: {{ text: 'Resp Rate (BPM)', font: {{ color: '#00f5d4', size: 11 }} }},
        domain: [0.77, 1.0],
        showgrid: true,
        gridcolor: 'rgba(255,255,255,0.05)',
        tickfont: {{ color: '#94a3b8', family: 'JetBrains Mono' }},
        range: [0, 320]
      }},
      // Y2: IBI
      yaxis2: {{
        title: {{ text: 'IBI (Seconds)', font: {{ color: '#f59e0b', size: 11 }} }},
        domain: [0.51, 0.73],
        showgrid: true,
        gridcolor: 'rgba(255,255,255,0.05)',
        tickfont: {{ color: '#94a3b8', family: 'JetBrains Mono' }},
        range: [0, 8.0]
      }},
      // Y3: Heart Rate
      yaxis3: {{
        title: {{ text: 'Heart Rate (BPM)', font: {{ color: '#ff0055', size: 11 }} }},
        domain: [0.25, 0.47],
        showgrid: true,
        gridcolor: 'rgba(255,255,255,0.05)',
        tickfont: {{ color: '#94a3b8', family: 'JetBrains Mono' }},
        range: [50, 750]
      }},
      // Y4: Laser
      yaxis4: {{
        title: {{ text: 'Laser (V)', font: {{ color: '#a855f7', size: 11 }} }},
        domain: [0.08, 0.21],
        showgrid: true,
        gridcolor: 'rgba(255,255,255,0.05)',
        tickfont: {{ color: '#94a3b8', family: 'JetBrains Mono' }},
        range: [0, 1.2]
      }},
      shapes: [
        ...laserShapes,
        // Apnea threshold line at 2.0s
        {{
          type: 'line',
          xref: 'x',
          yref: 'y2',
          x0: 0,
          x1: 6607,
          y0: 2.0,
          y1: 2.0,
          line: {{ color: 'rgba(239, 68, 68, 0.8)', width: 1.5, dash: 'dash' }}
        }}
      ]
    }};

    Plotly.newPlot('mainPlot', [traceRespRate, traceIBIScatter, traceIBICurve, traceHR, traceLaser], mainLayout, {{ responsive: true }});

    function jumpToEpoch(tStart, tEnd) {{
      const pad = (tEnd - tStart) * 0.15;
      const x0 = Math.max(0, tStart - pad);
      const x1 = Math.min(6607, tEnd + pad);
      Plotly.relayout('mainPlot', {{
        'xaxis.range': [x0, x1]
      }});
    }}

    function resetZoom() {{
      Plotly.relayout('mainPlot', {{
        'xaxis.range': [0, 6607]
      }});
    }}

    // High-res snapshot renderer
    function loadSnapshot(key, btn) {{
      if (btn) {{
        document.querySelectorAll('.snap-tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      }}
      const snap = snapshots[key];
      if (!snap) return;

      const trEMG = {{
        x: snap.t,
        y: snap.emg,
        name: 'Diaphragm EMG',
        type: 'scatter',
        mode: 'lines',
        line: {{ color: '#00f5d4', width: 1 }},
        xaxis: 'x',
        yaxis: 'y1'
      }};

      const trIntEMG = {{
        x: snap.t,
        y: snap.int_emg,
        name: 'Int EMG',
        type: 'scatter',
        mode: 'lines',
        line: {{ color: '#f77f00', width: 1.5 }},
        xaxis: 'x',
        yaxis: 'y2'
      }};

      const trECG = {{
        x: snap.t,
        y: snap.ecg,
        name: 'ECG',
        type: 'scatter',
        mode: 'lines',
        line: {{ color: '#ff0055', width: 1 }},
        xaxis: 'x',
        yaxis: 'y3'
      }};

      const trLaser = {{
        x: snap.t,
        y: snap.laser,
        name: 'Laser Pulse',
        type: 'scatter',
        mode: 'lines',
        fill: 'tozeroy',
        line: {{ color: '#a855f7', width: 1.5 }},
        xaxis: 'x',
        yaxis: 'y4'
      }};

      const snapLayout = {{
        paper_bgcolor: 'transparent',
        plot_bgcolor: '#0a0e17',
        margin: {{ l: 60, r: 40, t: 20, b: 40 }},
        hovermode: 'x unified',
        showlegend: false,
        grid: {{ rows: 4, columns: 1, pattern: 'independent' }},
        xaxis: {{
          range: [snap.t0, snap.t1],
          showgrid: true,
          gridcolor: 'rgba(255,255,255,0.05)',
          tickfont: {{ color: '#94a3b8', family: 'JetBrains Mono' }},
          title: {{ text: 'Time (Seconds)', font: {{ color: '#94a3b8', size: 11 }} }}
        }},
        yaxis: {{ title: {{ text: 'EMG (V)', font: {{ color: '#00f5d4', size: 10 }} }}, showgrid: true, gridcolor: 'rgba(255,255,255,0.05)' }},
        yaxis2: {{ title: {{ text: 'Int EMG (V)', font: {{ color: '#f77f00', size: 10 }} }}, showgrid: true, gridcolor: 'rgba(255,255,255,0.05)' }},
        yaxis3: {{ title: {{ text: 'ECG (V)', font: {{ color: '#ff0055', size: 10 }} }}, showgrid: true, gridcolor: 'rgba(255,255,255,0.05)' }},
        yaxis4: {{ title: {{ text: 'Laser (V)', font: {{ color: '#a855f7', size: 10 }} }}, showgrid: true, gridcolor: 'rgba(255,255,255,0.05)' }}
      }};

      Plotly.newPlot('snapPlot', [trEMG, trIntEMG, trECG, trLaser], snapLayout, {{ responsive: true }});
    }}

    // Initial load
    loadSnapshot('laser1');
  </script>
</body>
</html>
"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f" Dashboard created successfully: file:///{OUT_HTML.resolve().as_posix()}")

if __name__ == "__main__":
    build_html()
