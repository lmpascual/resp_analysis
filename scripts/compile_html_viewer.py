"""
Script to compile the standalone HTML viewer application with embedded data and interactive canvas rendering.
"""

import json
from pathlib import Path

WORKSPACE = Path(r"c:\Users\lpascu\analysis_src")
DATA_JSON = WORKSPACE / "scratch" / "lp26_encoded_data.json"
OUTPUT_HTML = WORKSPACE / "outputs" / "LP26_g0_analysis_viewer.html"

with open(DATA_JSON, "r") as f:
    data_str = f.read()

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LP26 Diaphragm EMG, Integrated Signal & Opto Laser Analysis</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0a0d14;
      --bg-card: rgba(18, 24, 38, 0.75);
      --bg-card-hover: rgba(26, 35, 55, 0.85);
      --border-color: rgba(255, 255, 255, 0.08);
      --border-accent: rgba(0, 245, 212, 0.3);
      --text-main: #f0f3fa;
      --text-muted: #8b9bb4;
      --emg-color: #00f5d4;
      --integ-color: #f77f00;
      --laser-color: #7000ff;
      --laser-glow: #00d2ff;
      --ecg-color: #ff0055;
      --accent-warn: #ffb703;
      --accent-success: #06d6a0;
      --accent-danger: #ef476f;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      user-select: none;
    }

    body {
      background-color: var(--bg-dark);
      background-image: 
        radial-gradient(circle at 10% 20%, rgba(112, 0, 255, 0.07) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(0, 245, 212, 0.05) 0%, transparent 40%),
        linear-gradient(180deg, #090c13 0%, #06080d 100%);
      color: var(--text-main);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      min-height: 100vh;
      overflow-x: hidden;
      display: flex;
      flex-direction: column;
    }

    /* Glass Header */
    header {
      background: rgba(13, 17, 27, 0.8);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-color);
      padding: 14px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .header-left {
      display: flex;
      align-items: center;
      gap: 16px;
    }

    .badge-icon {
      width: 42px;
      height: 42px;
      border-radius: 10px;
      background: linear-gradient(135deg, #7000ff, #00f5d4);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 22px;
      box-shadow: 0 4px 20px rgba(112, 0, 255, 0.35);
    }

    .header-title h1 {
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: #fff;
    }

    .header-title p {
      font-size: 0.78rem;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
      margin-top: 2px;
    }

    .header-badges {
      display: flex;
      gap: 10px;
    }

    .pill {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border-color);
      padding: 5px 12px;
      border-radius: 20px;
      font-size: 0.74rem;
      font-family: 'JetBrains Mono', monospace;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .pill.red {
      background: rgba(255, 0, 85, 0.12);
      border-color: rgba(255, 0, 85, 0.35);
      color: #ff6b8b;
    }

    .pill.green {
      background: rgba(6, 214, 160, 0.12);
      border-color: rgba(6, 214, 160, 0.35);
      color: #06d6a0;
    }

    .pill.blue {
      background: rgba(0, 210, 255, 0.12);
      border-color: rgba(0, 210, 255, 0.35);
      color: #00d2ff;
    }

    /* Core Container */
    .app-container {
      flex: 1;
      padding: 18px 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      max-width: 1920px;
      margin: 0 auto;
      width: 100%;
    }

    /* Verdicts Grid */
    .verdicts-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 14px;
    }

    .verdict-card {
      background: var(--bg-card);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 16px 20px;
      position: relative;
      overflow: hidden;
      transition: all 0.25s ease;
    }

    .verdict-card:hover {
      background: var(--bg-card-hover);
      border-color: rgba(255, 255, 255, 0.16);
      transform: translateY(-2px);
    }

    .verdict-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 4px;
      height: 100%;
    }

    .card-q1::before { background: var(--accent-danger); }
    .card-q2::before { background: var(--accent-success); }
    .card-q3::before { background: var(--accent-warn); }

    .verdict-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }

    .verdict-title {
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 700;
      color: var(--text-muted);
    }

    .verdict-badge {
      font-size: 0.72rem;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 6px;
      text-transform: uppercase;
    }

    .badge-danger { background: rgba(239, 71, 111, 0.2); color: #ff5e82; }
    .badge-success { background: rgba(6, 214, 160, 0.2); color: #06d6a0; }
    .badge-warn { background: rgba(255, 183, 3, 0.2); color: #ffb703; }

    .verdict-main {
      font-size: 1.05rem;
      font-weight: 600;
      line-height: 1.35;
      margin-bottom: 10px;
      color: #fff;
    }

    .verdict-desc {
      font-size: 0.81rem;
      line-height: 1.45;
      color: #b0bed5;
    }

    .stat-row {
      display: flex;
      gap: 16px;
      margin-top: 12px;
      padding-top: 10px;
      border-top: 1px solid rgba(255, 255, 255, 0.06);
    }

    .stat-item {
      display: flex;
      flex-direction: column;
    }

    .stat-value {
      font-size: 1.15rem;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
    }

    .stat-label {
      font-size: 0.68rem;
      color: var(--text-muted);
      text-transform: uppercase;
    }

    /* Toolbar & Presets */
    .controls-toolbar {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 12px 18px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }

    .presets-group {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }

    .toolbar-label {
      font-size: 0.74rem;
      text-transform: uppercase;
      font-weight: 700;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-right: 4px;
    }

    .btn-preset {
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 0.76rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .btn-preset:hover {
      background: rgba(255, 255, 255, 0.12);
      border-color: rgba(255, 255, 255, 0.25);
      transform: translateY(-1px);
    }

    .btn-preset.active {
      background: linear-gradient(135deg, rgba(112, 0, 255, 0.3), rgba(0, 245, 212, 0.2));
      border-color: var(--emg-color);
      color: #fff;
      box-shadow: 0 0 10px rgba(0, 245, 212, 0.2);
    }

    .zoom-controls {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .zoom-btn {
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--border-color);
      color: #fff;
      width: 32px;
      height: 32px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 16px;
      cursor: pointer;
      transition: all 0.2s;
    }

    .zoom-btn:hover {
      background: rgba(255, 255, 255, 0.15);
    }

    .time-window-display {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.78rem;
      background: rgba(0, 0, 0, 0.35);
      padding: 6px 12px;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      min-width: 250px;
      text-align: center;
    }

    /* Canvas Signal Viewer Container */
    .viewer-panel {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 14px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      position: relative;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    .tracks-wrapper {
      position: relative;
      width: 100%;
      height: 600px;
      background: rgba(8, 12, 20, 0.95);
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      overflow: hidden;
      cursor: crosshair;
    }

    #mainCanvas {
      width: 100%;
      height: 100%;
      display: block;
    }

    /* Minimap Scrubber */
    .minimap-wrapper {
      height: 64px;
      width: 100%;
      background: rgba(10, 14, 22, 0.95);
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      position: relative;
      overflow: hidden;
      cursor: pointer;
    }

    #minimapCanvas {
      width: 100%;
      height: 100%;
      display: block;
    }

    .minimap-slider {
      position: absolute;
      top: 0;
      bottom: 0;
      background: rgba(0, 245, 212, 0.15);
      border: 2px solid var(--emg-color);
      border-radius: 4px;
      pointer-events: none;
      box-shadow: 0 0 12px rgba(0, 245, 212, 0.35);
    }

    /* Channel Labels Overlay */
    .track-labels {
      position: absolute;
      left: 12px;
      top: 0;
      bottom: 0;
      width: 180px;
      pointer-events: none;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      padding: 10px 0;
      z-index: 10;
    }

    .track-tag {
      background: rgba(10, 15, 25, 0.85);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      width: fit-content;
      box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }

    .track-tag .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }

    /* Diagnostic Analysis Tabs Section */
    .diagnostics-section {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .section-title {
      font-size: 0.95rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .diagnostics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
      gap: 18px;
    }

    .chart-box {
      background: rgba(10, 14, 22, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 10px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .chart-box-title {
      font-size: 0.82rem;
      font-weight: 600;
      color: #cbd5e1;
      display: flex;
      justify-content: space-between;
    }

    .chart-canvas {
      width: 100%;
      height: 180px;
      background: rgba(6, 9, 15, 0.9);
      border-radius: 6px;
    }

    /* Recommendations Table */
    .recs-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
      margin-top: 8px;
    }

    .recs-table th {
      text-align: left;
      padding: 8px 12px;
      background: rgba(255, 255, 255, 0.04);
      color: var(--text-muted);
      font-weight: 600;
      border-bottom: 1px solid var(--border-color);
    }

    .recs-table td {
      padding: 10px 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      line-height: 1.4;
    }

    .recs-table tr:hover {
      background: rgba(255, 255, 255, 0.02);
    }

    /* Pulse List Drawer */
    .pulse-table-container {
      max-height: 220px;
      overflow-y: auto;
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 8px;
      background: rgba(7, 10, 16, 0.7);
    }

    .pulse-table-container::-webkit-scrollbar {
      width: 6px;
    }
    .pulse-table-container::-webkit-scrollbar-thumb {
      background: rgba(255, 255, 255, 0.15);
      border-radius: 3px;
    }

    /* Tooltip */
    #tooltip {
      position: absolute;
      background: rgba(13, 18, 30, 0.92);
      border: 1px solid rgba(255, 255, 255, 0.2);
      padding: 8px 12px;
      border-radius: 6px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.72rem;
      pointer-events: none;
      z-index: 1000;
      display: none;
      box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
  </style>
</head>
<body>

  <!-- Header -->
  <header>
    <div class="header-left">
      <div class="badge-icon">🫁</div>
      <div class="header-title">
        <h1>LP26 Respiratory Laser Control & Artifact Diagnostics</h1>
        <p>RECORDING: Z:\\ramirezlab\\pascual\\rec_data\\LP26_g0\\LP26_g0_t0.nidq.bin | VGAT-Ai32 Animal</p>
      </div>
    </div>
    <div class="header-badges">
      <div class="pill red">⚠️ 88% Short Pulses ECG-Locked</div>
      <div class="pill green">✂️ Ti Shortened -29.6% (p < 1e-31)</div>
      <div class="pill blue">⚡ 425 Phasic Pulses (516s - 733s)</div>
    </div>
  </header>

  <div class="app-container">

    <!-- Answers to User Questions (Diagnostic Cards) -->
    <div class="verdicts-grid">
      <!-- Card 1: Question 1 Verdict -->
      <div class="verdict-card card-q1">
        <div class="verdict-header">
          <span class="verdict-title">Question 1: ECG Artifact Triggering</span>
          <span class="verdict-badge badge-danger">Confirmed: Critical Artifact</span>
        </div>
        <div class="verdict-main">
          Yes: Pulses Are Delivered on the Basis of ECG Artifacts
        </div>
        <div class="verdict-desc">
          The potentiometer threshold on the Teensy is set at <strong>~0.74 V</strong>. Cardiac R-peaks induce a sharp transient on the integrated diaphragm channel that peaks at <strong>0.75 – 0.85 V</strong>. This triggers false laser pulses firing at a fixed latency of <strong>6.25 ± 0.7 ms</strong> post R-peak. <strong>88.0% of all short pulses (<50ms)</strong> are false ECG artifact triggers!
        </div>
        <div class="stat-row">
          <div class="stat-item">
            <span class="stat-value" style="color: var(--ecg-color);">6.25 ms</span>
            <span class="stat-label">Fixed ECG Latency</span>
          </div>
          <div class="stat-item">
            <span class="stat-value" style="color: var(--accent-danger);">88.0%</span>
            <span class="stat-label">Short Pulses on ECG</span>
          </div>
          <div class="stat-item">
            <span class="stat-value" style="color: #fff;">~33 ms</span>
            <span class="stat-label">Artifact Pulse Dur</span>
          </div>
        </div>
      </div>

      <!-- Card 2: Question 2 Verdict -->
      <div class="verdict-card card-q2">
        <div class="verdict-header">
          <span class="verdict-title">Question 2: Breath Delays / Advances</span>
          <span class="verdict-badge badge-success">Confirmed: Significant Effect</span>
        </div>
        <div class="verdict-main">
          Yes: Phase Advance via Premature Inspiratory Truncation
        </div>
        <div class="verdict-desc">
          In this <code>vgat/ai32</code> animal, stimulating inhibitory neurons during inspiration <strong>prematurely terminates the diaphragm burst</strong>: Inspiratory duration (Ti) is truncated from <strong>138 ms to 97 ms (-29.6%, p=8.3e-32)</strong>, and amplitude is suppressed by <strong>-45.2% (p=2.6e-29)</strong>. <strong>50.2%</strong> of stimulated breaths show a phase advance (median <strong>-20.5 ms</strong>).
        </div>
        <div class="stat-row">
          <div class="stat-item">
            <span class="stat-value" style="color: var(--accent-success);">-29.6%</span>
            <span class="stat-label">Ti Truncation</span>
          </div>
          <div class="stat-item">
            <span class="stat-value" style="color: var(--emg-color);">50.2%</span>
            <span class="stat-label">Phase Advances</span>
          </div>
          <div class="stat-item">
            <span class="stat-value" style="color: #fff;">-45.2%</span>
            <span class="stat-label">Motor Burst Amp</span>
          </div>
        </div>
      </div>

      <!-- Card 3: Actionable Rig Fixes -->
      <div class="verdict-card card-q3">
        <div class="verdict-header">
          <span class="verdict-title">Recommended Rig Calibration</span>
          <span class="verdict-badge badge-warn">Action Required</span>
        </div>
        <div class="verdict-main">
          Raise Threshold to ~1.1 V & Add 40ms Blanking
        </div>
        <div class="verdict-desc">
          True inspiratory bursts peak at <strong>1.8 V – 3.2 V</strong>, whereas ECG artifacts peak at <strong>~0.75 V</strong>. Turning the potentiometer clockwise to set threshold from 0.74V to <strong>1.10 V</strong> will eliminate >95% of false ECG triggers while reliably catching true inspiratory peaks. Adding a 40ms debounce in firmware will reject any residual cardiac spikes.
        </div>
        <div class="stat-row">
          <div class="stat-item">
            <span class="stat-value" style="color: var(--accent-warn);">0.74 V &rarr; 1.10 V</span>
            <span class="stat-label">Recommended Thresh</span>
          </div>
          <div class="stat-item">
            <span class="stat-value" style="color: #fff;">&gt;40 ms</span>
            <span class="stat-label">Min Hold Time</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Controls Toolbar & Jump Presets -->
    <div class="controls-toolbar">
      <div class="presets-group">
        <span class="toolbar-label">Quick Jump:</span>
        <button class="btn-preset active" onclick="jumpTo(516.1, 517.6)">🔴 ECG False Trigger #1 (t=516.4s)</button>
        <button class="btn-preset" onclick="jumpTo(517.9, 519.4)">🔴 ECG False Trigger #2 (t=518.2s)</button>
        <button class="btn-preset" onclick="jumpTo(585.0, 587.2)">🟢 True Phasic Hold (t=585.3s)</button>
        <button class="btn-preset" onclick="jumpTo(588.5, 592.5)">🟢 Rhythmic Stim (t=590s)</button>
        <button class="btn-preset" onclick="jumpTo(202.0, 206.0)">⚪ Clean Baseline (t=204s)</button>
        <button class="btn-preset" onclick="jumpTo(514.0, 534.0)">🌐 Stim Sequence Start</button>
      </div>

      <div class="zoom-controls">
        <button class="zoom-btn" onclick="zoomIn()" title="Zoom In">+</button>
        <button class="zoom-btn" onclick="zoomOut()" title="Zoom Out">-</button>
        <button class="zoom-btn" onclick="panLeft()" title="Pan Left">&larr;</button>
        <button class="zoom-btn" onclick="panRight()" title="Pan Right">&rarr;</button>
        <div class="time-window-display" id="timeWindowDisplay">Viewing: 516.10s - 517.60s (1.50s)</div>
      </div>
    </div>

    <!-- Multi-Track Canvas Viewer -->
    <div class="viewer-panel">
      <div class="tracks-wrapper" id="tracksWrapper">
        <canvas id="mainCanvas"></canvas>
        <div class="track-labels">
          <div class="track-tag" style="border-left: 3px solid var(--emg-color);">
            <span class="dot" style="background: var(--emg-color);"></span>
            Raw Diaphragm EMG (Ch 0)
          </div>
          <div class="track-tag" style="border-left: 3px solid var(--integ-color);">
            <span class="dot" style="background: var(--integ-color);"></span>
            Integrated EMG & Thresh (Ch 1)
          </div>
          <div class="track-tag" style="border-left: 3px solid var(--laser-glow);">
            <span class="dot" style="background: var(--laser-glow);"></span>
            Laser Command (Ch 6)
          </div>
          <div class="track-tag" style="border-left: 3px solid var(--ecg-color);">
            <span class="dot" style="background: var(--ecg-color);"></span>
            Electrocardiogram ECG (Ch 2)
          </div>
        </div>
      </div>

      <!-- Minimap Scrubber -->
      <div class="minimap-wrapper" id="minimapWrapper">
        <canvas id="minimapCanvas"></canvas>
        <div class="minimap-slider" id="minimapSlider"></div>
      </div>
    </div>

    <!-- Scientific Diagnostic Plots -->
    <div class="diagnostics-section">
      <div class="section-title">📊 Quantitative Proof & Analytical Verification</div>
      
      <div class="diagnostics-grid">
        <!-- Plot 1: ECG-Laser Latency PSTH -->
        <div class="chart-box">
          <div class="chart-box-title">
            <span>ECG-to-Laser Latency Distribution (PSTH)</span>
            <span style="color: var(--ecg-color); font-family: monospace;">Sharp Peak at 6.25 ms</span>
          </div>
          <canvas id="chartPSTH" class="chart-canvas"></canvas>
          <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 4px;">
            Cross-correlogram showing time difference between ECG R-peak and subsequent laser pulse onset. 88% of short pulses concentrate into the single 6ms bin.
          </div>
        </div>

        <!-- Plot 2: Inspiratory Duration Ti Comparison -->
        <div class="chart-box">
          <div class="chart-box-title">
            <span>Inspiratory Burst Duration (Ti) Truncation</span>
            <span style="color: var(--accent-success); font-family: monospace;">p = 8.27e-32</span>
          </div>
          <canvas id="chartTi" class="chart-canvas"></canvas>
          <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 4px;">
            Histogram of Ti (ms) during unperturbed Control breaths (138.3ms, blue) vs Stimulated breaths (97.3ms, emerald), proving premature termination.
          </div>
        </div>

        <!-- Plot 3: Laser Pulse Table Explorer -->
        <div class="chart-box">
          <div class="chart-box-title">
            <span>Pulse Event Classification Explorer</span>
            <span style="color: var(--laser-glow); font-family: monospace;">425 Pulses Recorded</span>
          </div>
          <div class="pulse-table-container">
            <table class="recs-table" id="pulseTable">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Onset (s)</th>
                  <th>Dur (ms)</th>
                  <th>ECG Latency</th>
                  <th>Classification</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody id="pulseTableBody">
                <!-- Injected by JS -->
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

  </div>

  <div id="tooltip"></div>

  <!-- EMBEDDED DATA PAYLOAD -->
  <script>
    const RAW_DATA = """ + data_str + """;
  </script>

  <!-- INTERACTIVE VIEWER JAVASCRIPT ENGINE -->
  <script>
    // Decode Base64 to Float32Array
    function b64ToFloat32(b64) {
      const bin = atob(b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) {
        bytes[i] = bin.charCodeAt(i);
      }
      return new Float32Array(bytes.buffer);
    }

    // Unpack Overview and High-Res Blocks
    console.log("Decoding datasets...");
    const overview = {
      t: b64ToFloat32(RAW_DATA.overview.b64_t),
      c0_min: b64ToFloat32(RAW_DATA.overview.b64_c0_min),
      c0_max: b64ToFloat32(RAW_DATA.overview.b64_c0_max),
      c1_min: b64ToFloat32(RAW_DATA.overview.b64_c1_min),
      c1_max: b64ToFloat32(RAW_DATA.overview.b64_c1_max),
      c2_min: b64ToFloat32(RAW_DATA.overview.b64_c2_min),
      c2_max: b64ToFloat32(RAW_DATA.overview.b64_c2_max),
      c6_min: b64ToFloat32(RAW_DATA.overview.b64_c6_min),
      c6_max: b64ToFloat32(RAW_DATA.overview.b64_c6_max),
    };

    const blockEarly = {
      tStart: RAW_DATA.block_stim_early.t_start,
      tEnd: RAW_DATA.block_stim_early.t_end,
      fs: RAW_DATA.block_stim_early.fs || 1000.0,
      t: b64ToFloat32(RAW_DATA.block_stim_early.b64_time),
      c0: b64ToFloat32(RAW_DATA.block_stim_early.b64_ch0),
      c1: b64ToFloat32(RAW_DATA.block_stim_early.b64_ch1),
      c2: b64ToFloat32(RAW_DATA.block_stim_early.b64_ch2),
      c6: b64ToFloat32(RAW_DATA.block_stim_early.b64_ch6),
    };

    const blockMid = {
      tStart: RAW_DATA.block_stim_mid.t_start,
      tEnd: RAW_DATA.block_stim_mid.t_end,
      fs: RAW_DATA.block_stim_mid.fs || 1000.0,
      t: b64ToFloat32(RAW_DATA.block_stim_mid.b64_time),
      c0: b64ToFloat32(RAW_DATA.block_stim_mid.b64_ch0),
      c1: b64ToFloat32(RAW_DATA.block_stim_mid.b64_ch1),
      c2: b64ToFloat32(RAW_DATA.block_stim_mid.b64_ch2),
      c6: b64ToFloat32(RAW_DATA.block_stim_mid.b64_ch6),
    };

    const blockBase = {
      tStart: RAW_DATA.block_baseline.t_start,
      tEnd: RAW_DATA.block_baseline.t_end,
      fs: RAW_DATA.block_baseline.fs || 1000.0,
      t: b64ToFloat32(RAW_DATA.block_baseline.b64_time),
      c0: b64ToFloat32(RAW_DATA.block_baseline.b64_ch0),
      c1: b64ToFloat32(RAW_DATA.block_baseline.b64_ch1),
      c2: b64ToFloat32(RAW_DATA.block_baseline.b64_ch2),
      c6: b64ToFloat32(RAW_DATA.block_baseline.b64_ch6),
    };

    const highResBlocks = [blockEarly, blockMid, blockBase];

    // Viewer State
    let viewStart = 516.1;
    let viewEnd = 517.6;
    const totalDuration = RAW_DATA.overview.t_max;
    const threshV = RAW_DATA.stats.threshold_voltage_v;

    const mainCanvas = document.getElementById("mainCanvas");
    const ctx = mainCanvas.getContext("2d");
    const minimapCanvas = document.getElementById("minimapCanvas");
    const miniCtx = minimapCanvas.getContext("2d");
    const minimapSlider = document.getElementById("minimapSlider");
    const timeDisplay = document.getElementById("timeWindowDisplay");

    // Resize Canvases
    function resizeCanvases() {
      const rect = mainCanvas.parentElement.getBoundingClientRect();
      mainCanvas.width = rect.width * window.devicePixelRatio;
      mainCanvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

      const miniRect = minimapCanvas.parentElement.getBoundingClientRect();
      minimapCanvas.width = miniRect.width * window.devicePixelRatio;
      minimapCanvas.height = miniRect.height * window.devicePixelRatio;
      miniCtx.scale(window.devicePixelRatio, window.devicePixelRatio);

      renderMain();
      renderMinimap();
    }

    window.addEventListener("resize", resizeCanvases);

    // Render Main View
    function renderMain() {
      const w = mainCanvas.clientWidth;
      const h = mainCanvas.clientHeight;
      ctx.clearRect(0, 0, w, h);

      // 4 Tracks:
      // Track 1: Raw EMG (0 to h*0.25)
      // Track 2: Integrated EMG (h*0.25 to h*0.50)
      // Track 3: Laser (h*0.50 to h*0.75)
      // Track 4: ECG (h*0.75 to h)
      const trackH = h / 4;

      // Draw Dividers & Grid
      ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
      ctx.lineWidth = 1;
      for (let i = 1; i < 4; i++) {
        ctx.beginPath();
        ctx.moveTo(0, i * trackH);
        ctx.lineTo(w, i * trackH);
        ctx.stroke();
      }

      // Draw Time Grid Lines
      const duration = viewEnd - viewStart;
      let gridStep = 0.5;
      if (duration < 0.5) gridStep = 0.05;
      else if (duration < 1.5) gridStep = 0.1;
      else if (duration < 5) gridStep = 0.5;
      else if (duration < 15) gridStep = 1.0;
      else if (duration < 60) gridStep = 5.0;
      else gridStep = 20.0;

      const firstLine = Math.ceil(viewStart / gridStep) * gridStep;
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.fillStyle = "rgba(255, 255, 255, 0.3)";
      ctx.textAlign = "center";

      for (let t = firstLine; t <= viewEnd; t += gridStep) {
        const x = ((t - viewStart) / duration) * w;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
        ctx.fillText(t.toFixed(2) + "s", x, h - 6);
      }

      // Find if we have high-res block covering current window
      let activeBlock = null;
      for (const blk of highResBlocks) {
        if (viewStart >= blk.tStart && viewEnd <= blk.tEnd) {
          activeBlock = blk;
          break;
        }
      }

      // Draw Data
      if (activeBlock) {
        renderHighRes(activeBlock, w, trackH);
      } else {
        renderOverview(w, trackH);
      }

      // Draw Threshold Line on Track 2 (Integrated EMG)
      const yZero2 = 2 * trackH;
      const scale2 = trackH / 3.5; // 0 to 3.5V
      const yThresh = yZero2 - threshV * scale2;
      ctx.save();
      ctx.strokeStyle = "rgba(239, 71, 111, 0.85)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(0, yThresh);
      ctx.lineTo(w, yThresh);
      ctx.stroke();
      ctx.fillStyle = "rgba(239, 71, 111, 0.9)";
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText("THRESH: " + threshV.toFixed(2) + "V", w - 16, yThresh - 4);
      ctx.restore();

      // Draw ECG to Laser alignment indicators
      drawAlignmentGuides(w, trackH);

      // Update Header Time Display
      timeDisplay.textContent = `Viewing: ${viewStart.toFixed(2)}s - ${viewEnd.toFixed(2)}s (${(viewEnd - viewStart).toFixed(2)}s)`;
      updateMinimapSlider();
    }

    // Render High-Resolution Block (1 kHz points)
    function renderHighRes(blk, w, trackH) {
      const duration = viewEnd - viewStart;
      const iStart = Math.max(0, Math.floor((viewStart - blk.tStart) * blk.fs));
      const iEnd = Math.min(blk.t.length - 1, Math.ceil((viewEnd - blk.tStart) * blk.fs));

      // Track 1: Raw EMG (-2.5V to +2.5V)
      const yZero1 = 0.5 * trackH;
      const scale1 = (trackH * 0.45) / 2.5;
      ctx.strokeStyle = "#00f5d4";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (let i = iStart; i <= iEnd; i++) {
        const x = ((blk.t[i] - viewStart) / duration) * w;
        const y = yZero1 - blk.c0[i] * scale1;
        if (i === iStart) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Track 2: Integrated EMG (0 to 3.5V)
      const yZero2 = 2 * trackH;
      const scale2 = (trackH * 0.85) / 3.5;
      ctx.strokeStyle = "#f77f00";
      ctx.fillStyle = "rgba(247, 127, 0, 0.15)";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(((blk.t[iStart] - viewStart) / duration) * w, yZero2);
      for (let i = iStart; i <= iEnd; i++) {
        const x = ((blk.t[i] - viewStart) / duration) * w;
        const y = yZero2 - Math.max(0, blk.c1[i]) * scale2;
        ctx.lineTo(x, y);
      }
      ctx.lineTo(((blk.t[iEnd] - viewStart) / duration) * w, yZero2);
      ctx.closePath();
      ctx.fill();

      // Stroke integrated
      ctx.beginPath();
      for (let i = iStart; i <= iEnd; i++) {
        const x = ((blk.t[i] - viewStart) / duration) * w;
        const y = yZero2 - Math.max(0, blk.c1[i]) * scale2;
        if (i === iStart) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Track 3: Laser (0.45V to 0.75V)
      const yBase3 = 3 * trackH - 12;
      const scale3 = (trackH * 0.7) / 0.25;
      ctx.strokeStyle = "#00d2ff";
      ctx.fillStyle = "rgba(112, 0, 255, 0.35)";
      ctx.lineWidth = 2.0;

      ctx.beginPath();
      ctx.moveTo(((blk.t[iStart] - viewStart) / duration) * w, yBase3);
      for (let i = iStart; i <= iEnd; i++) {
        const x = ((blk.t[i] - viewStart) / duration) * w;
        const y = yBase3 - Math.max(0, blk.c6[i] - 0.48) * scale3;
        ctx.lineTo(x, y);
      }
      ctx.lineTo(((blk.t[iEnd] - viewStart) / duration) * w, yBase3);
      ctx.closePath();
      ctx.fill();

      ctx.beginPath();
      for (let i = iStart; i <= iEnd; i++) {
        const x = ((blk.t[i] - viewStart) / duration) * w;
        const y = yBase3 - Math.max(0, blk.c6[i] - 0.48) * scale3;
        if (i === iStart) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Track 4: ECG (-8V to +10V)
      const yZero4 = 3.65 * trackH;
      const scale4 = (trackH * 0.4) / 8.0;
      ctx.strokeStyle = "#ff0055";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      for (let i = iStart; i <= iEnd; i++) {
        const x = ((blk.t[i] - viewStart) / duration) * w;
        const y = yZero4 - blk.c2[i] * scale4;
        if (i === iStart) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // Render Overview (Min-Max downsampled)
    function renderOverview(w, trackH) {
      const duration = viewEnd - viewStart;
      const dt = RAW_DATA.overview.dt;
      const iStart = Math.max(0, Math.floor(viewStart / dt));
      const iEnd = Math.min(overview.t.length - 1, Math.ceil(viewEnd / dt));

      // Track 1: Raw EMG
      const yZero1 = 0.5 * trackH;
      const scale1 = (trackH * 0.45) / 2.5;
      ctx.fillStyle = "rgba(0, 245, 212, 0.4)";
      for (let i = iStart; i <= iEnd; i++) {
        const x = ((overview.t[i] - viewStart) / duration) * w;
        const yMin = yZero1 - overview.c0_max[i] * scale1;
        const yMax = yZero1 - overview.c0_min[i] * scale1;
        ctx.fillRect(x, yMin, Math.max(1, w / (iEnd - iStart)), yMax - yMin);
      }

      // Track 2: Integrated
      const yZero2 = 2 * trackH;
      const scale2 = (trackH * 0.85) / 3.5;
      ctx.fillStyle = "rgba(247, 127, 0, 0.5)";
      for (let i = iStart; i <= iEnd; i++) {
        const x = ((overview.t[i] - viewStart) / duration) * w;
        const yMin = yZero2 - Math.max(0, overview.c1_max[i]) * scale2;
        const yMax = yZero2;
        ctx.fillRect(x, yMin, Math.max(1, w / (iEnd - iStart)), yMax - yMin);
      }

      // Track 3: Laser
      const yBase3 = 3 * trackH - 12;
      const scale3 = (trackH * 0.7) / 0.25;
      ctx.fillStyle = "rgba(0, 210, 255, 0.6)";
      for (let i = iStart; i <= iEnd; i++) {
        const x = ((overview.t[i] - viewStart) / duration) * w;
        const yMin = yBase3 - Math.max(0, overview.c6_max[i] - 0.48) * scale3;
        const yMax = yBase3;
        ctx.fillRect(x, yMin, Math.max(1, w / (iEnd - iStart)), yMax - yMin);
      }

      // Track 4: ECG
      const yZero4 = 3.65 * trackH;
      const scale4 = (trackH * 0.4) / 8.0;
      ctx.fillStyle = "rgba(255, 0, 85, 0.5)";
      for (let i = iStart; i <= iEnd; i++) {
        const x = ((overview.t[i] - viewStart) / duration) * w;
        const yMin = yZero4 - overview.c2_max[i] * scale4;
        const yMax = yZero4 - overview.c2_min[i] * scale4;
        ctx.fillRect(x, yMin, Math.max(1, w / (iEnd - iStart)), yMax - yMin);
      }
    }

    // Draw alignment guides linking ECG peaks to laser pulses
    function drawAlignmentGuides(w, trackH) {
      const duration = viewEnd - viewStart;
      if (duration > 5.0) return; // only draw when zoomed in enough

      const pulses = RAW_DATA.pulses;
      ctx.save();
      for (const p of pulses) {
        if (p.off >= viewStart && p.on <= viewEnd) {
          const xOn = ((p.on - viewStart) / duration) * w;
          const xOff = ((p.off - viewStart) / duration) * w;

          // Pulse box on Track 3
          ctx.strokeStyle = p.is_short ? "rgba(255, 0, 85, 0.8)" : "rgba(6, 214, 160, 0.8)";
          ctx.lineWidth = 1;
          ctx.strokeRect(xOn, 2 * trackH + 6, xOff - xOn, trackH - 12);

          // Pulse Badge
          ctx.font = "9px 'JetBrains Mono', monospace";
          ctx.fillStyle = p.is_short ? "#ff6b8b" : "#06d6a0";
          ctx.textAlign = "left";
          const label = p.is_short ? `ECG Trigger (${p.dur_ms}ms)` : `Phasic Hold (${p.dur_ms}ms)`;
          ctx.fillText(label, xOn + 4, 2 * trackH + 20);

          // If ECG-locked, draw connection line to Track 4 (ECG)
          if (p.is_ecg && p.dt_ecg_ms !== null) {
            const tECG = p.on - p.dt_ecg_ms / 1000.0;
            const xECG = ((tECG - viewStart) / duration) * w;

            ctx.strokeStyle = "rgba(255, 0, 85, 0.4)";
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(xECG, 4 * trackH - 30);
            ctx.lineTo(xECG, 2 * trackH + 25);
            ctx.lineTo(xOn, 2 * trackH + 25);
            ctx.stroke();

            ctx.fillStyle = "rgba(255, 0, 85, 0.85)";
            ctx.fillText(`+${p.dt_ecg_ms}ms`, xECG + 2, 3.2 * trackH);
          }
        }
      }
      ctx.restore();
    }

    // Render Minimap
    function renderMinimap() {
      const w = minimapCanvas.clientWidth;
      const h = minimapCanvas.clientHeight;
      miniCtx.clearRect(0, 0, w, h);

      // Draw background stimulation block (516s to 733s)
      const stimX1 = (516.0 / totalDuration) * w;
      const stimX2 = (733.0 / totalDuration) * w;
      miniCtx.fillStyle = "rgba(112, 0, 255, 0.25)";
      miniCtx.fillRect(stimX1, 0, stimX2 - stimX1, h);

      // Draw downsampled EMG envelope
      miniCtx.fillStyle = "rgba(0, 245, 212, 0.4)";
      const n = overview.t.length;
      for (let i = 0; i < n; i++) {
        const x = (overview.t[i] / totalDuration) * w;
        const yMin = h - (Math.max(0, overview.c1_max[i]) / 3.5) * h;
        miniCtx.fillRect(x, yMin, Math.max(1, w / n), h - yMin);
      }

      // Draw Laser Pulses on Minimap
      miniCtx.fillStyle = "rgba(0, 210, 255, 0.8)";
      for (const p of RAW_DATA.pulses) {
        const px = (p.on / totalDuration) * w;
        miniCtx.fillRect(px, 0, Math.max(1.5, (p.dur_ms / 1000 / totalDuration) * w), h * 0.4);
      }

      updateMinimapSlider();
    }

    function updateMinimapSlider() {
      const w = minimapCanvas.clientWidth;
      const left = (viewStart / totalDuration) * w;
      const right = (viewEnd / totalDuration) * w;
      minimapSlider.style.left = left + "px";
      minimapSlider.style.width = Math.max(4, right - left) + "px";
    }

    // Navigation & Interaction
    function setView(start, end) {
      viewStart = Math.max(0, Math.min(totalDuration - 0.2, start));
      viewEnd = Math.max(viewStart + 0.1, Math.min(totalDuration, end));
      renderMain();
    }

    function jumpTo(start, end) {
      setView(start, end);
      // update active preset button
      document.querySelectorAll(".btn-preset").forEach(btn => btn.classList.remove("active"));
      if (event && event.target && event.target.classList.contains("btn-preset")) {
        event.target.classList.add("active");
      }
    }

    function zoomIn() {
      const dur = viewEnd - viewStart;
      const center = (viewStart + viewEnd) / 2;
      const newDur = Math.max(0.2, dur * 0.65);
      setView(center - newDur / 2, center + newDur / 2);
    }

    function zoomOut() {
      const dur = viewEnd - viewStart;
      const center = (viewStart + viewEnd) / 2;
      const newDur = Math.min(totalDuration, dur * 1.5);
      setView(center - newDur / 2, center + newDur / 2);
    }

    function panLeft() {
      const dur = viewEnd - viewStart;
      const shift = dur * 0.25;
      setView(viewStart - shift, viewEnd - shift);
    }

    function panRight() {
      const dur = viewEnd - viewStart;
      const shift = dur * 0.25;
      setView(viewStart + shift, viewEnd + shift);
    }

    // Mouse Drag Pan & Wheel Zoom
    let isDragging = false;
    let dragStartX = 0;
    let dragViewStart = 0;

    mainCanvas.addEventListener("mousedown", (e) => {
      isDragging = true;
      dragStartX = e.clientX;
      dragViewStart = viewStart;
    });

    window.addEventListener("mouseup", () => isDragging = false);

    window.addEventListener("mousemove", (e) => {
      if (!isDragging) return;
      const dx = e.clientX - dragStartX;
      const dur = viewEnd - viewStart;
      const dt = -(dx / mainCanvas.clientWidth) * dur;
      setView(dragViewStart + dt, dragViewStart + dt + dur);
    });

    mainCanvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const rect = mainCanvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseT = viewStart + (mouseX / rect.width) * (viewEnd - viewStart);

      const zoomFactor = e.deltaY < 0 ? 0.75 : 1.35;
      const newDur = Math.max(0.15, Math.min(totalDuration, (viewEnd - viewStart) * zoomFactor));

      const newStart = mouseT - (mouseX / rect.width) * newDur;
      const newEnd = newStart + newDur;
      setView(newStart, newEnd);
    }, { passive: false });

    // Minimap Click Navigation
    minimapCanvas.parentElement.addEventListener("click", (e) => {
      const rect = minimapCanvas.getBoundingClientRect();
      const clickT = ((e.clientX - rect.left) / rect.width) * totalDuration;
      const dur = viewEnd - viewStart;
      setView(clickT - dur / 2, clickT + dur / 2);
    });

    // Populate Pulse Table
    function populatePulseTable() {
      const tbody = document.getElementById("pulseTableBody");
      const pulses = RAW_DATA.pulses;
      tbody.innerHTML = "";

      pulses.slice(0, 100).forEach(p => {
        const tr = document.createElement("tr");
        const isECG = p.is_ecg;
        tr.innerHTML = `
          <td><strong>#${p.id}</strong></td>
          <td>${p.on.toFixed(3)}s</td>
          <td><span style="color:${p.is_short ? '#ff6b8b' : '#06d6a0'}">${p.dur_ms} ms</span></td>
          <td>${p.dt_ecg_ms !== null ? '+' + p.dt_ecg_ms + ' ms' : 'N/A'}</td>
          <td><span class="pill ${isECG ? 'red' : 'green'}" style="padding:2px 8px;font-size:0.68rem;">${isECG ? 'ECG Artifact' : 'Inspiratory Hold'}</span></td>
          <td><button class="btn-preset" style="padding:2px 8px;font-size:0.68rem;" onclick="jumpTo(${p.on - 0.4}, ${p.on + 0.8})">Inspect</button></td>
        `;
        tbody.appendChild(tr);
      });
    }

    // Render Diagnostic Charts
    function renderDiagnosticCharts() {
      // 1. PSTH Canvas
      const psthCanvas = document.getElementById("chartPSTH");
      const pCtx = psthCanvas.getContext("2d");
      psthCanvas.width = psthCanvas.clientWidth * window.devicePixelRatio;
      psthCanvas.height = psthCanvas.clientHeight * window.devicePixelRatio;
      pCtx.scale(window.devicePixelRatio, window.devicePixelRatio);

      const pw = psthCanvas.clientWidth;
      const ph = psthCanvas.clientHeight;
      const counts = RAW_DATA.stats.psth_counts;
      const maxCount = Math.max(...counts);

      pCtx.clearRect(0, 0, pw, ph);
      const barW = pw / counts.length;

      for (let i = 0; i < counts.length; i++) {
        const x = i * barW;
        const bH = (counts[i] / maxCount) * (ph - 30);
        const center = RAW_DATA.stats.psth_centers[i];

        // Highlight 6ms bin
        if (Math.abs(center - 6.0) < 2.0) {
          pCtx.fillStyle = "#ff0055";
        } else {
          pCtx.fillStyle = "rgba(0, 210, 255, 0.45)";
        }
        pCtx.fillRect(x, ph - 20 - bH, barW - 1, bH);
      }

      pCtx.fillStyle = "rgba(255, 255, 255, 0.4)";
      pCtx.font = "9px 'JetBrains Mono', monospace";
      pCtx.fillText("-30ms", 4, ph - 6);
      pCtx.fillText("0ms (ECG)", pw * 0.23, ph - 6);
      pCtx.fillStyle = "#ff0055";
      pCtx.fillText("6.2ms (Laser Trigger)", pw * 0.32, ph - 6);
      pCtx.fillStyle = "rgba(255, 255, 255, 0.4)";
      pCtx.fillText("+100ms", pw - 40, ph - 6);

      // 2. Ti Distribution Canvas
      const tiCanvas = document.getElementById("chartTi");
      const tCtx = tiCanvas.getContext("2d");
      tiCanvas.width = tiCanvas.clientWidth * window.devicePixelRatio;
      tiCanvas.height = tiCanvas.clientHeight * window.devicePixelRatio;
      tCtx.scale(window.devicePixelRatio, window.devicePixelRatio);

      const tw = tiCanvas.clientWidth;
      const th = tiCanvas.clientHeight;
      const ctrlVals = RAW_DATA.stats.ctrl_ti_hist_vals;
      const stimVals = RAW_DATA.stats.stim_ti_hist_vals;
      const maxTi = Math.max(...ctrlVals, ...stimVals);

      tCtx.clearRect(0, 0, tw, th);
      const nBins = ctrlVals.length;
      const tBarW = (tw - 40) / nBins;

      for (let i = 0; i < nBins; i++) {
        const x = 20 + i * tBarW;
        const cH = (ctrlVals[i] / maxTi) * (th - 35);
        const sH = (stimVals[i] / maxTi) * (th - 35);

        // Control (Blue)
        tCtx.fillStyle = "rgba(0, 210, 255, 0.45)";
        tCtx.fillRect(x, th - 20 - cH, tBarW * 0.45, cH);

        // Stimulated (Emerald)
        tCtx.fillStyle = "rgba(6, 214, 160, 0.85)";
        tCtx.fillRect(x + tBarW * 0.45, th - 20 - sH, tBarW * 0.45, sH);
      }

      tCtx.fillStyle = "rgba(255, 255, 255, 0.4)";
      tCtx.font = "9px 'JetBrains Mono', monospace";
      tCtx.fillText("50ms", 20, th - 6);
      tCtx.fillText("100ms", tw * 0.35, th - 6);
      tCtx.fillText("150ms", tw * 0.60, th - 6);
      tCtx.fillText("250ms", tw - 40, th - 6);

      // Legend
      tCtx.fillStyle = "rgba(0, 210, 255, 0.8)";
      tCtx.fillText("■ Control (138ms)", tw - 180, 16);
      tCtx.fillStyle = "rgba(6, 214, 160, 0.9)";
      tCtx.fillText("■ Stimulated (97ms)", tw - 90, 16);
    }

    // Initialize
    window.addEventListener("DOMContentLoaded", () => {
      resizeCanvases();
      populatePulseTable();
      renderDiagnosticCharts();
    });
  </script>
</body>
</html>
"""

OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Successfully generated standalone HTML viewer: {OUTPUT_HTML} (size: {OUTPUT_HTML.stat().st_size / 1024 / 1024:.2f} MB)")
