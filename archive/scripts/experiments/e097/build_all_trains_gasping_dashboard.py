"""
Build Interactive Multi-Train Eupnea-to-Gasping Transition Dashboard
Allows switching seamlessly between Laser 1, Laser 2, and Laser 3, plus side-by-side 3-train comparison.
"""

import sys
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\lpascu\analysis_src")
trans_dir = PROJECT_ROOT / "output_e097_prelaser_through_laser3_226s_motion_corrected" / "consensus_master" / "04_gasping_transition_rasters"
csv_path = trans_dir / "eupnea_gasp_transition_statistics_all_trains.csv"
html_out = trans_dir / "gasping_transition_dashboard.html"

if not csv_path.exists():
    print(f"Error: {csv_path} does not exist yet.", flush=True)
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} units from {csv_path.name}", flush=True)

# Population stats
n_total = len(df)
n_qc = int(df["passed_allen_qc"].sum())
n_gasp_l1 = int((df["gmi_laser1"] > 0.2).sum())
n_gasp_l2 = int((df["gmi_laser2"] > 0.2).sum())
n_gasp_l3 = int((df["gmi_laser3"] > 0.2).sum())
n_persistent = int((df["multi_train_progression"] == "Persistent Gasp Active").sum())
n_sensitized = int((df["multi_train_progression"] == "Sensitized Gasp Active").sum())
n_breakthrough = int((df["multi_train_progression"] == "Laser Breakthrough Dominant").sum())

units_list = []
for _, r in df.iterrows():
    cid = int(r["cluster_id"])
    units_list.append({
        "cid": cid,
        "depth": float(r["depth_um"]),
        "cell_type": str(r["cell_type"]),
        "qc": bool(r["passed_allen_qc"]),
        "total_spk": int(r["total_spikes"]),
        "r_schreiber": float(r["r_schreiber_ep1"]) if pd.notna(r["r_schreiber_ep1"]) else None,
        "progression": str(r["multi_train_progression"]),
        "l1": {
            "fr_pre": float(r["fr_pre_laser1"]),
            "fr_las": float(r["fr_las_laser1"]),
            "fr_gasp": float(r["fr_gasp_laser1"]),
            "fr_post": float(r["fr_post_laser1"]),
            "gmi": float(r["gmi_laser1"]),
            "lmi": float(r["lmi_laser1"]),
            "rec_idx": float(r["rec_laser1"]),
            "spk_gasp": float(r["mean_spk_gasp_laser1"]),
            "img": f"cards/laser1/gasp_transition_cluster_{cid:03d}.png"
        },
        "l2": {
            "fr_pre": float(r["fr_pre_laser2"]),
            "fr_las": float(r["fr_las_laser2"]),
            "fr_gasp": float(r["fr_gasp_laser2"]),
            "fr_post": float(r["fr_post_laser2"]),
            "gmi": float(r["gmi_laser2"]),
            "lmi": float(r["lmi_laser2"]),
            "rec_idx": float(r["rec_laser2"]),
            "spk_gasp": float(r["mean_spk_gasp_laser2"]),
            "img": f"cards/laser2/gasp_transition_cluster_{cid:03d}.png"
        },
        "l3": {
            "fr_pre": float(r["fr_pre_laser3"]),
            "fr_las": float(r["fr_las_laser3"]),
            "fr_gasp": float(r["fr_gasp_laser3"]),
            "fr_post": float(r["fr_post_laser3"]),
            "gmi": float(r["gmi_laser3"]),
            "lmi": float(r["lmi_laser3"]),
            "rec_idx": float(r["rec_laser3"]),
            "spk_gasp": float(r["mean_spk_gasp_laser3"]),
            "img": f"cards/laser3/gasp_transition_cluster_{cid:03d}.png"
        }
    })

units_json = json.dumps(units_list)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Multi-Train Eupnea-to-Gasping Transition Dashboard (Diaphragm I_off Aligned) — Animal e097</title>
  <style>
    :root {{
      --bg-dark: #090d16;
      --panel-bg: #111827;
      --card-bg: #1e293b;
      --border-col: #334155;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent-blue: #38bdf8;
      --accent-rose: #f43f5e;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --accent-purple: #c084fc;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg-dark);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      padding: 24px;
      line-height: 1.5;
    }}
    header {{
      background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
      padding: 28px;
      border-radius: 16px;
      border: 1px solid #312e81;
      margin-bottom: 24px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }}
    h1 {{
      font-size: 26px;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .subtitle {{
      color: #cbd5e1;
      font-size: 14.5px;
      max-width: 1100px;
    }}
    .hero-stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 14px;
      margin-top: 20px;
    }}
    .stat-pill {{
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.1);
      padding: 14px 18px;
      border-radius: 12px;
      backdrop-filter: blur(8px);
    }}
    .stat-num {{
      font-size: 26px;
      font-weight: 800;
      color: var(--accent-blue);
      line-height: 1.1;
    }}
    .stat-label {{
      font-size: 11.5px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-top: 4px;
      font-weight: 600;
    }}


    /* Train Selector Bar */
    .train-selector-bar {{
      display: flex;
      align-items: center;
      gap: 12px;
      background: var(--panel-bg);
      border: 1px solid var(--border-col);
      padding: 12px 18px;
      border-radius: 14px;
      margin-bottom: 20px;
    }}
    .train-selector-label {{
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      color: var(--text-muted);
      letter-spacing: 0.05em;
      margin-right: 8px;
    }}
    .train-tab-btn {{
      background: #0f172a;
      border: 1px solid #334155;
      color: #94a3b8;
      padding: 10px 20px;
      border-radius: 10px;
      font-size: 13.5px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s;
    }}
    .train-tab-btn:hover {{
      background: #1e293b;
      color: #fff;
    }}
    .train-tab-btn.active {{
      background: linear-gradient(135deg, #0284c7, #2563eb);
      color: #fff;
      border-color: #38bdf8;
      box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
    }}

    /* Split highlight banner */
    .split-highlight-banner {{
      background: linear-gradient(90deg, rgba(244,63,94,0.15), rgba(56,189,248,0.15));
      border: 1px solid rgba(244,63,94,0.3);
      padding: 16px 20px;
      border-radius: 12px;
      margin-bottom: 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
    }}
    .split-info {{
      font-size: 13.5px;
      color: #e2e8f0;
    }}
    .split-info strong {{
      color: #fff;
    }}
    .split-btns {{
      display: flex;
      gap: 10px;
    }}
    .btn-split {{
      background: #0f172a;
      border: 1px solid #475569;
      color: #fff;
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 12.5px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .btn-split:hover {{
      background: #334155;
      border-color: var(--accent-blue);
    }}

    /* Filters panel */
    .controls-panel {{
      background: var(--panel-bg);
      border: 1px solid var(--border-col);
      padding: 18px 24px;
      border-radius: 14px;
      margin-bottom: 24px;
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: center;
    }}
    .control-group {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .control-group label {{
      font-size: 11.5px;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--text-muted);
      letter-spacing: 0.04em;
    }}
    select, input {{
      background: #0f172a;
      border: 1px solid var(--border-col);
      color: #fff;
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 13px;
      outline: none;
      transition: border 0.2s;
    }}
    select:focus, input:focus {{
      border-color: var(--accent-blue);
    }}

    /* Grid */
    .unit-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
      gap: 20px;
    }}
    .unit-card {{
      background: var(--card-bg);
      border: 1px solid var(--border-col);
      border-radius: 14px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
      position: relative;
    }}
    .unit-card:hover {{
      transform: translateY(-3px);
      border-color: var(--accent-blue);
      box-shadow: 0 12px 24px -6px rgba(0, 0, 0, 0.6);
    }}
    .card-header {{
      padding: 12px 16px;
      background: rgba(15, 23, 42, 0.8);
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .card-title {{
      font-size: 15px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .badge {{
      font-size: 10.5px;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 9999px;
      text-transform: uppercase;
    }}
    .badge-qc {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
    .badge-nonqc {{ background: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.3); }}

    .card-img-wrap {{
      background: #000;
      cursor: pointer;
      position: relative;
      height: 250px;
      overflow: hidden;
    }}
    .card-img-wrap img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      transition: transform 0.3s;
    }}
    .card-img-wrap:hover img {{
      transform: scale(1.02);
    }}
    .zoom-hint {{
      position: absolute;
      bottom: 8px;
      right: 8px;
      background: rgba(0, 0, 0, 0.75);
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 11px;
      color: #e2e8f0;
      pointer-events: none;
    }}

    .card-body {{
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      flex-grow: 1;
    }}
    .phenotype-tag {{
      font-size: 11.5px;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 6px;
      background: #0f172a;
      border: 1px solid var(--border-col);
      color: #f1f5f9;
      display: inline-block;
    }}
    .metrics-row {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }}
    .metric-box {{
      background: #0f172a;
      padding: 6px 10px;
      border-radius: 8px;
      text-align: center;
    }}
    .m-label {{
      font-size: 10px;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
    }}
    .m-val {{
      font-size: 13.5px;
      font-weight: 700;
      color: #fff;
    }}
    .val-pos {{ color: #34d399; }}
    .val-neg {{ color: #f43f5e; }}

    .card-footer-btns {{
      display: flex;
      gap: 8px;
      margin-top: 4px;
    }}
    .btn-card-action {{
      flex: 1;
      background: #1e293b;
      border: 1px solid #475569;
      color: #cbd5e1;
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 11.5px;
      font-weight: 600;
      cursor: pointer;
      text-align: center;
      transition: all 0.2s;
    }}
    .btn-card-action:hover {{
      background: #334155;
      color: #fff;
      border-color: var(--accent-blue);
    }}

    /* Modals */
    .modal {{
      display: none;
      position: fixed;
      z-index: 1000;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.94);
      align-items: center;
      justify-content: center;
      padding: 20px;
    }}
    .modal.active {{
      display: flex;
    }}
    .modal-content {{
      max-width: 95vw;
      max-height: 95vh;
      border-radius: 12px;
      box-shadow: 0 0 40px rgba(0, 0, 0, 0.9);
    }}
    .close-btn {{
      position: absolute;
      top: 20px;
      right: 30px;
      color: #fff;
      font-size: 36px;
      font-weight: bold;
      cursor: pointer;
    }}

    /* 3-Train Comparison Modal */
    .compare-modal-content {{
      background: var(--panel-bg);
      border: 1px solid var(--border-col);
      border-radius: 16px;
      max-width: 98vw;
      max-height: 96vh;
      width: 98vw;
      padding: 24px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .compare-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
    }}
    .compare-col {{
      background: #0f172a;
      border: 1px solid var(--border-col);
      border-radius: 12px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .compare-col-header {{
      font-size: 14px;
      font-weight: 700;
      color: var(--accent-blue);
      text-align: center;
      padding-bottom: 6px;
      border-bottom: 1px solid #334155;
    }}
    .compare-col img {{
      width: 100%;
      border-radius: 8px;
      cursor: pointer;
    }}
  </style>
</head>
<body>

  <header>
    <h1>💥 Multi-Train Eupnea-to-Gasping Transition Suite — Animal e097</h1>
    <div class="subtitle">
      Single-unit respiratory dynamics evaluated across all 3 optogenetic arrest and autoresuscitation sequences
      strictly aligned to <strong>Diaphragm Inspiration Offset ($I_{{\mathrm{{off}}}}^{{\mathrm{{EMG}}}}$, Burst Termination)</strong>
      in the <strong>expanded pre-inspiratory window [-300 ms to +300 ms]</strong>:
      <strong>Laser 1 (t = 60.0 s)</strong> ➔ 
      <strong>Laser 2 (t = 120.0 s)</strong> ➔ 
      <strong>Laser 3 (t = 164.0 s)</strong>.
      Inspect pre-inspiratory ramping, in-laser breakthrough breaths, ballistic open-mouth autoresuscitation gasping, and non-stationary unit splitting.
    </div>

    <div class="hero-stats">
      <div class="stat-pill">
        <div class="stat-num">{n_total}</div>
        <div class="stat-label">Total Units Analyzed</div>
      </div>
      <div class="stat-pill">
        <div class="stat-num" style="color: var(--accent-emerald);">{n_qc}</div>
        <div class="stat-label">Passed Allen QC</div>
      </div>
      <div class="stat-pill">
        <div class="stat-num" style="color: var(--accent-rose);">{n_persistent}</div>
        <div class="stat-label">Persistent Gasp Active</div>
      </div>
      <div class="stat-pill">
        <div class="stat-num" style="color: var(--accent-purple);">{n_sensitized}</div>
        <div class="stat-label">Sensitized Gasp Active</div>
      </div>
      <div class="stat-pill">
        <div class="stat-num" style="color: var(--accent-amber);">{n_breakthrough}</div>
        <div class="stat-label">Laser Breakthrough Active</div>
      </div>
    </div>
  </header>

  <!-- Split Ground Truth Progression Banner -->
  <div class="split-highlight-banner">
    <div class="split-info">
      <strong>⚡ Benchmark Split Pair Progression (Diaphragm I_off Aligned [-300ms, +300ms] Window):</strong> 
      Cluster <strong>#378</strong> and Cluster <strong>#150</strong> (both at depth <strong>2385 µm</strong>) show the chronological handover:
      In <em>Laser 1</em>, #378 is active in pre-eupnea and fires vigorously during in-laser breakthrough breaths and gasps, while #150 is dormant;
      In <em>Laser 2</em>, #150 is first recruited during the giant autoresuscitation gasp;
      In <em>Laser 3</em>, #378 drops out upon eupnea resumption and #150 takes over permanently with a classic inspiratory ramp profile!
    </div>
    <div class="split-btns">
      <button class="btn-split" onclick="openCompareModal(378)">Compare #378 Across 3 Trains</button>
      <button class="btn-split" onclick="openCompareModal(150)">Compare #150 Across 3 Trains</button>
      <button class="btn-split" onclick="openCompareModal(13)">Compare #13 Across 3 Trains</button>
    </div>
  </div>

  <!-- Train Selector Bar -->
  <div class="train-selector-bar">
    <span class="train-selector-label">Select Laser Train:</span>
    <button class="train-tab-btn" id="btn-l1" onclick="switchTrain('l1')">
      ⚡ Laser 1 Train (t=60.0s | Oral & Nasal Gasps)
    </button>
    <button class="train-tab-btn" id="btn-l2" onclick="switchTrain('l2')">
      ⚡ Laser 2 Train (t=120.0s | Autoresuscitation Gasps)
    </button>
    <button class="train-tab-btn active" id="btn-l3" onclick="switchTrain('l3')">
      ⚡ Laser 3 Train (t=164.0s | Sustained Gasp Transition)
    </button>
  </div>

  <!-- Interactive Controls -->
  <div class="controls-panel">
    <div class="control-group">
      <label>Search Cluster ID</label>
      <input type="text" id="searchId" placeholder="e.g. 13, 378, 150, 64" oninput="applyFilters()">
    </div>

    <div class="control-group">
      <label>Progression Filter</label>
      <select id="progressionFilter" onchange="applyFilters()">
        <option value="ALL">All Multi-Train Progressions ({n_total})</option>
        <option value="PERSISTENT">Persistent Gasp Active ({n_persistent})</option>
        <option value="SENSITIZED">Sensitized Gasp Active ({n_sensitized})</option>
        <option value="HABITUATED">Habituated Gasp Active</option>
        <option value="BREAKTHROUGH">Laser Breakthrough Dominant ({n_breakthrough})</option>
        <option value="SUPPRESSED">Gasp Suppressed</option>
        <option value="STABLE">Stable / Unmodulated</option>
      </select>
    </div>


    <div class="control-group">
      <label>Cell Type</label>
      <select id="cellTypeFilter" onchange="applyFilters()">
        <option value="ALL">All Cell Types</option>
        <option value="Post-I">Pure Post-I / Expiratory Phasic</option>
        <option value="Late Inspiratory">Late Inspiratory Ramp</option>
        <option value="Off-Switch">Off-Switch Spanning</option>
      </select>
    </div>

    <div class="control-group">
      <label>Allen QC</label>
      <select id="qcFilter" onchange="applyFilters()">
        <option value="ALL">All Units</option>
        <option value="QC_ONLY">QC Passed Only ({n_qc})</option>
      </select>
    </div>

    <div class="control-group">
      <label>Sort By</label>
      <select id="sortBy" onchange="applyFilters()">
        <option value="gmi_desc">Active Train Gasp MI (High ➔ Low)</option>
        <option value="gmi_asc">Active Train Gasp MI (Low ➔ High)</option>
        <option value="depth_asc">Depth (Dorsal ➔ Ventral)</option>
        <option value="depth_desc">Depth (Ventral ➔ Dorsal)</option>
        <option value="fr_desc">Baseline Firing Rate (High ➔ Low)</option>
        <option value="schreiber_desc">Schreiber Reliability (High ➔ Low)</option>
        <option value="cid_asc">Cluster ID (1 ➔ N)</option>
      </select>
    </div>
  </div>

  <!-- Units Grid -->
  <div class="unit-grid" id="gridContainer"></div>

  <!-- Modal Single Lightbox -->
  <div class="modal" id="lightboxModal" onclick="closeModal()">
    <span class="close-btn">&times;</span>
    <img class="modal-content" id="modalImg" src="">
  </div>

  <!-- Modal 3-Train Side-by-Side Comparison -->
  <div class="modal" id="compareModal" onclick="closeCompareModal(event)">
    <div class="compare-modal-content" onclick="event.stopPropagation()">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <h2 id="compareModalTitle" style="font-size: 20px; font-weight: 800; color: #fff;"></h2>
        <span class="close-btn" style="position: static; font-size: 28px;" onclick="closeCompareModal()">&times;</span>
      </div>
      <div class="compare-grid">
        <div class="compare-col">
          <div class="compare-col-header">⚡ Laser Train 1 (t=60s | Gasp 4.15V)</div>
          <img id="cmpImgL1" src="" onclick="openModal(this.src)">
        </div>
        <div class="compare-col">
          <div class="compare-col-header">⚡ Laser Train 2 (t=120s | Gasp 5.32V)</div>
          <img id="cmpImgL2" src="" onclick="openModal(this.src)">
        </div>
        <div class="compare-col">
          <div class="compare-col-header">⚡ Laser Train 3 (t=164s | Gasp 4.90V)</div>
          <img id="cmpImgL3" src="" onclick="openModal(this.src)">
        </div>
      </div>
    </div>
  </div>

  <script>
    const allUnits = {units_json};
    let currentTrain = 'l3';

    function switchTrain(trainKey) {{
      currentTrain = trainKey;
      document.querySelectorAll('.train-tab-btn').forEach(btn => btn.classList.remove('active'));
      document.getElementById('btn-' + trainKey).classList.add('active');
      applyFilters();
    }}

    function renderUnits(units) {{
      const container = document.getElementById("gridContainer");
      container.innerHTML = "";

      if (units.length === 0) {{
        container.innerHTML = '<div style="grid-column: 1/-1; padding: 40px; text-align: center; color: #94a3b8;">No units match the selected criteria.</div>';
        return;
      }}

      units.forEach(u => {{
        const card = document.createElement("div");
        card.className = "unit-card";

        const tData = u[currentTrain];
        const qcBadge = u.qc
          ? '<span class="badge badge-qc">QC PASSED</span>'
          : '<span class="badge badge-nonqc">Non-QC</span>';

        const gmiClass = tData.gmi > 0.2 ? 'val-pos' : (tData.gmi < -0.2 ? 'val-neg' : '');
        const lmiClass = tData.lmi > 0.2 ? 'val-pos' : (tData.lmi < -0.2 ? 'val-neg' : '');

        card.innerHTML = `
          <div class="card-header">
            <div class="card-title">
              Cluster #${{u.cid}}
              <span style="font-size: 12px; color: #94a3b8; font-weight: normal;">(${{Math.round(u.depth)}} µm)</span>
            </div>
            ${{qcBadge}}
          </div>
          <div class="card-img-wrap" onclick="openModal('${{tData.img}}')">
            <img src="${{tData.img}}" alt="Cluster #${{u.cid}}" loading="lazy">
            <div class="zoom-hint">🔍 Click to Expand</div>
          </div>
          <div class="card-body">
            <div class="phenotype-tag">${{u.progression}}</div>
            <div style="font-size: 12px; color: #cbd5e1;"><strong>Cell Type:</strong> ${{u.cell_type}}</div>
            <div class="metrics-row">
              <div class="metric-box">
                <div class="m-label">${{currentTrain.toUpperCase()}} Gasp MI</div>
                <div class="m-val ${{gmiClass}}">${{tData.gmi >= 0 ? '+' : ''}}${{tData.gmi.toFixed(2)}}</div>
              </div>
              <div class="metric-box">
                <div class="m-label">${{currentTrain.toUpperCase()}} Laser MI</div>
                <div class="m-val ${{lmiClass}}">${{tData.lmi >= 0 ? '+' : ''}}${{tData.lmi.toFixed(2)}}</div>
              </div>
              <div class="metric-box">
                <div class="m-label">Pre FR</div>
                <div class="m-val">${{tData.fr_pre.toFixed(1)}} Hz</div>
              </div>
            </div>
            <div class="metrics-row">
              <div class="metric-box">
                <div class="m-label">Gasp Rate</div>
                <div class="m-val" style="color: var(--accent-rose);">${{tData.fr_gasp.toFixed(1)}} Hz</div>
              </div>
              <div class="metric-box">
                <div class="m-label">Post FR</div>
                <div class="m-val" style="color: var(--accent-emerald);">${{tData.fr_post.toFixed(1)}} Hz</div>
              </div>
              <div class="metric-box">
                <div class="m-label">Schreiber</div>
                <div class="m-val">${{u.r_schreiber !== null ? u.r_schreiber.toFixed(2) : '—'}}</div>
              </div>
            </div>
            <div class="card-footer-btns">
              <button class="btn-card-action" onclick="openCompareModal(${{u.cid}})">📊 Compare 3 Trains Side-by-Side</button>
            </div>
          </div>
        `;
        container.appendChild(card);
      }});
    }}

    function applyFilters() {{
      const search = document.getElementById("searchId").value.trim().toLowerCase();
      const progression = document.getElementById("progressionFilter").value;
      const cellType = document.getElementById("cellTypeFilter").value;
      const qcOnly = document.getElementById("qcFilter").value === "QC_ONLY";
      const sortBy = document.getElementById("sortBy").value;

      let filtered = allUnits.filter(u => {{
        if (search && !u.cid.toString().includes(search)) return false;
        if (qcOnly && !u.qc) return false;
        if (cellType !== "ALL" && !u.cell_type.includes(cellType)) return false;

        if (progression === "PERSISTENT" && !u.progression.includes("Persistent")) return false;
        if (progression === "SENSITIZED" && !u.progression.includes("Sensitized")) return false;
        if (progression === "HABITUATED" && !u.progression.includes("Habituated")) return false;
        if (progression === "BREAKTHROUGH" && !u.progression.includes("Breakthrough")) return false;
        if (progression === "SUPPRESSED" && !u.progression.includes("Suppressed")) return false;
        if (progression === "STABLE" && !u.progression.includes("Stable")) return false;

        return true;
      }});


      filtered.sort((a, b) => {{
        const aT = a[currentTrain];
        const bT = b[currentTrain];
        if (sortBy === "gmi_desc") return bT.gmi - aT.gmi;
        if (sortBy === "gmi_asc") return aT.gmi - bT.gmi;
        if (sortBy === "depth_asc") return a.depth - b.depth;
        if (sortBy === "depth_desc") return b.depth - a.depth;
        if (sortBy === "fr_desc") return bT.fr_pre - aT.fr_pre;
        if (sortBy === "schreiber_desc") return (b.r_schreiber || 0) - (a.r_schreiber || 0);
        return a.cid - b.cid;
      }});

      renderUnits(filtered);
    }}

    function openModal(imgSrc) {{
      document.getElementById("modalImg").src = imgSrc;
      document.getElementById("lightboxModal").classList.add("active");
    }}

    function closeModal() {{
      document.getElementById("lightboxModal").classList.remove("active");
    }}

    function openCompareModal(cid) {{
      const u = allUnits.find(unit => unit.cid === cid);
      if (!u) return;
      document.getElementById("compareModalTitle").innerText = `Cluster #${{u.cid}} (${{Math.round(u.depth)}} µm | ${{u.cell_type}}) — 3-Train Autoresuscitation Progression`;
      document.getElementById("cmpImgL1").src = u.l1.img;
      document.getElementById("cmpImgL2").src = u.l2.img;
      document.getElementById("cmpImgL3").src = u.l3.img;
      document.getElementById("compareModal").classList.add("active");
    }}

    function closeCompareModal() {{
      document.getElementById("compareModal").classList.remove("active");
    }}

    // Initial render
    applyFilters();
  </script>
</body>
</html>
"""

with open(html_out, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generated multi-train dashboard: {html_out} ({len(html_content)} bytes)", flush=True)
