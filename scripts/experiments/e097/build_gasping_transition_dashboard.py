"""
Build Interactive HTML Dashboard for Eupnea-to-Gasping Transition Rasters
Animal: e097 (226s Continuous Recording, Motion-Corrected Consensus Master)
Output: output_e097_prelaser_through_laser3_226s_motion_corrected/consensus_master/04_gasping_transition_rasters/gasping_transition_dashboard.html
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\lpascu\analysis_src")
trans_dir = PROJECT_ROOT / "output_e097_prelaser_through_laser3_226s_motion_corrected" / "consensus_master" / "04_gasping_transition_rasters"
csv_path = trans_dir / "eupnea_gasp_transition_statistics.csv"
html_out = trans_dir / "gasping_transition_dashboard.html"

if not csv_path.exists():
    print(f"Error: {csv_path} does not exist yet. Please wait for generation script to complete.", flush=True)
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} unit records from {csv_path.name}", flush=True)

# Compute population stats
n_total = len(df)
n_qc = int(df["passed_allen_qc"].sum())
n_gasp_rec = int((df["gasp_modulation_index"] > 0.35).sum())
n_laser_supp = int((df["laser_modulation_index"] < -0.35).sum())
n_activated = int((df["phenotype"].str.contains("Activated", na=False)).sum())
n_extinguished = int((df["phenotype"].str.contains("Extinguished", na=False)).sum())

# Prepare data for JSON injection
units_list = []
for _, r in df.iterrows():
    cid = int(r["cluster_id"])
    card_filename = f"cards/gasp_transition_cluster_{cid:03d}.png"
    units_list.append({
        "cid": cid,
        "depth": float(r["depth_um"]),
        "cell_type": str(r["cell_type"]),
        "qc": bool(r["passed_allen_qc"]),
        "total_spk": int(r["total_spikes"]),
        "r_schreiber": float(r["r_schreiber_ep1"]) if pd.notna(r["r_schreiber_ep1"]) else None,
        "fr_pre": float(r["fr_pre_eupnea_hz"]),
        "fr_las": float(r["fr_laser_hz"]),
        "fr_gasp": float(r["fr_ballistic_gasp_hz"]),
        "fr_post": float(r["fr_post_eupnea_hz"]),
        "gmi": float(r["gasp_modulation_index"]),
        "lmi": float(r["laser_modulation_index"]),
        "rec_idx": float(r["recovery_index"]),
        "mean_spk_pre": float(r["mean_spikes_pre"]),
        "mean_spk_gasp": float(r["mean_spikes_gasp"]),
        "mean_spk_post": float(r["mean_spikes_post"]),
        "phenotype": str(r["phenotype"]),
        "card_img": card_filename
    })

units_json = json.dumps(units_list)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Eupnea-to-Gasping Transition Dashboard — Animal e097</title>
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
      max-width: 1000px;
    }}
    .hero-stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
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
      background: rgba(0, 0, 0, 0.7);
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
      font-size: 12px;
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

    /* Lightbox modal */
    .modal {{
      display: none;
      position: fixed;
      z-index: 1000;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.92);
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
  </style>
</head>
<body>

  <header>
    <h1>💥 Eupnea-to-Gasping Transition Suite — Animal e097</h1>
    <div class="subtitle">
      Single-unit respiratory dynamics across the tri-phasic autoresuscitation sequence:
      <strong>Baseline Eupnea (Trial #575-586) ➔ Laser Optogenetic Arrest (t=164s) ➔ Ballistic Autoresuscitation Gasp (Flow up to 4.9V) ➔ Resumed Recovery Eupnea</strong>.
      Evaluates phase alignment to Flow Inspiration Offset ($I_{{\mathrm{{off}}}}$) across [-100 ms, +200 ms] and tracks non-stationary unit splitting.
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
        <div class="stat-num" style="color: var(--accent-rose);">{n_gasp_rec}</div>
        <div class="stat-label">Gasp-Recruited (GMI &gt; +0.35)</div>
      </div>
      <div class="stat-pill">
        <div class="stat-num" style="color: var(--accent-amber);">{n_laser_supp}</div>
        <div class="stat-label">Laser-Inhibited Units</div>
      </div>
      <div class="stat-pill">
        <div class="stat-num" style="color: var(--accent-purple);">{n_activated + n_extinguished}</div>
        <div class="stat-label">Gasp Split Exemplars</div>
      </div>
    </div>
  </header>

  <!-- Split Ground Truth Exemplar Banner -->
  <div class="split-highlight-banner">
    <div class="split-info">
      <strong>⚡ Benchmark Split Pair Discovery:</strong> Cluster <strong>#378</strong> (extinguished post-gasp) and Cluster <strong>#150</strong> (recruited precisely during ballistic gasp) reside at the identical anatomical depth (<strong>2385 µm</strong>).
      They reveal mechanical electrode displacement / template drift during violent autoresuscitation gasping.
    </div>
    <div class="split-btns">
      <button class="btn-split" onclick="filterExemplar(378)">View Cluster #378</button>
      <button class="btn-split" onclick="filterExemplar(150)">View Cluster #150</button>
      <button class="btn-split" onclick="filterExemplar(13)">View Cluster #13 (Unified)</button>
    </div>
  </div>

  <!-- Interactive Controls -->
  <div class="controls-panel">
    <div class="control-group">
      <label>Search Cluster ID</label>
      <input type="text" id="searchId" placeholder="e.g. 13, 378, 150" oninput="applyFilters()">
    </div>

    <div class="control-group">
      <label>Phenotype Filter</label>
      <select id="phenotypeFilter" onchange="applyFilters()">
        <option value="ALL">All Phenotypes ({n_total})</option>
        <option value="GASP_REC">Gasp-Recruited / Hyper-Responsive</option>
        <option value="ACTIVATED">Post-Gasp Activated (Split B)</option>
        <option value="EXTINGUISHED">Post-Gasp Extinguished (Split A)</option>
        <option value="RESPONDER">Laser-Arrested Responder</option>
        <option value="SUPPRESSED">Laser- &amp; Gasp-Suppressed</option>
        <option value="STATIONARY">Stationary Resilient Respiration</option>
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
        <option value="gmi_desc">Gasp Modulation Index (High ➔ Low)</option>
        <option value="gmi_asc">Gasp Modulation Index (Low ➔ High)</option>
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

  <!-- Modal Lightbox -->
  <div class="modal" id="lightboxModal" onclick="closeModal()">
    <span class="close-btn">&times;</span>
    <img class="modal-content" id="modalImg" src="">
  </div>

  <script>
    const allUnits = {units_json};

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

        const qcBadge = u.qc
          ? '<span class="badge badge-qc">QC PASSED</span>'
          : '<span class="badge badge-nonqc">Non-QC</span>';

        const gmiClass = u.gmi > 0.2 ? 'val-pos' : (u.gmi < -0.2 ? 'val-neg' : '');
        const lmiClass = u.lmi > 0.2 ? 'val-pos' : (u.lmi < -0.2 ? 'val-neg' : '');

        card.innerHTML = `
          <div class="card-header">
            <div class="card-title">
              Cluster #${{u.cid}}
              <span style="font-size: 12px; color: #94a3b8; font-weight: normal;">(${{Math.round(u.depth)}} µm)</span>
            </div>
            ${{qcBadge}}
          </div>
          <div class="card-img-wrap" onclick="openModal('${{u.card_img}}')">
            <img src="${{u.card_img}}" alt="Cluster #${{u.cid}}" loading="lazy">
            <div class="zoom-hint">🔍 Click to Expand</div>
          </div>
          <div class="card-body">
            <div class="phenotype-tag">${{u.phenotype}}</div>
            <div style="font-size: 12px; color: #cbd5e1;"><strong>Cell Type:</strong> ${{u.cell_type}}</div>
            <div class="metrics-row">
              <div class="metric-box">
                <div class="m-label">Gasp MI</div>
                <div class="m-val ${{gmiClass}}">${{u.gmi >= 0 ? '+' : ''}}${{u.gmi.toFixed(2)}}</div>
              </div>
              <div class="metric-box">
                <div class="m-label">Laser MI</div>
                <div class="m-val ${{lmiClass}}">${{u.lmi >= 0 ? '+' : ''}}${{u.lmi.toFixed(2)}}</div>
              </div>
              <div class="metric-box">
                <div class="m-label">Baseline FR</div>
                <div class="m-val">${{u.fr_pre.toFixed(1)}} Hz</div>
              </div>
            </div>
            <div class="metrics-row">
              <div class="metric-box">
                <div class="m-label">Gasp Rate</div>
                <div class="m-val" style="color: var(--accent-rose);">${{u.fr_gasp.toFixed(1)}} Hz</div>
              </div>
              <div class="metric-box">
                <div class="m-label">Post Rate</div>
                <div class="m-val" style="color: var(--accent-emerald);">${{u.fr_post.toFixed(1)}} Hz</div>
              </div>
              <div class="metric-box">
                <div class="m-label">Schreiber</div>
                <div class="m-val">${{u.r_schreiber !== null ? u.r_schreiber.toFixed(2) : '—'}}</div>
              </div>
            </div>
          </div>
        `;
        container.appendChild(card);
      }});
    }}

    function applyFilters() {{
      const search = document.getElementById("searchId").value.trim().toLowerCase();
      const phenotype = document.getElementById("phenotypeFilter").value;
      const cellType = document.getElementById("cellTypeFilter").value;
      const qcOnly = document.getElementById("qcFilter").value === "QC_ONLY";
      const sortBy = document.getElementById("sortBy").value;

      let filtered = allUnits.filter(u => {{
        if (search && !u.cid.toString().includes(search)) return false;
        if (qcOnly && !u.qc) return false;
        if (cellType !== "ALL" && !u.cell_type.includes(cellType)) return false;

        if (phenotype === "GASP_REC" && u.gmi <= 0.35) return false;
        if (phenotype === "ACTIVATED" && !u.phenotype.includes("Activated")) return false;
        if (phenotype === "EXTINGUISHED" && !u.phenotype.includes("Extinguished")) return false;
        if (phenotype === "RESPONDER" && !u.phenotype.includes("Responder")) return false;
        if (phenotype === "SUPPRESSED" && !u.phenotype.includes("Suppressed")) return false;
        if (phenotype === "STATIONARY" && !u.phenotype.includes("Stationary")) return false;

        return true;
      }});

      filtered.sort((a, b) => {{
        if (sortBy === "gmi_desc") return b.gmi - a.gmi;
        if (sortBy === "gmi_asc") return a.gmi - b.gmi;
        if (sortBy === "depth_asc") return a.depth - b.depth;
        if (sortBy === "depth_desc") return b.depth - a.depth;
        if (sortBy === "fr_desc") return b.fr_pre - a.fr_pre;
        if (sortBy === "schreiber_desc") return (b.r_schreiber || 0) - (a.r_schreiber || 0);
        return a.cid - b.cid;
      }});

      renderUnits(filtered);
    }}

    function filterExemplar(cid) {{
      document.getElementById("searchId").value = cid;
      document.getElementById("phenotypeFilter").value = "ALL";
      document.getElementById("cellTypeFilter").value = "ALL";
      document.getElementById("qcFilter").value = "ALL";
      applyFilters();
    }}

    function openModal(imgSrc) {{
      document.getElementById("modalImg").src = imgSrc;
      document.getElementById("lightboxModal").classList.add("active");
    }}

    function closeModal() {{
      document.getElementById("lightboxModal").classList.remove("active");
    }}

    // Initial render
    applyFilters();
  </script>
</body>
</html>
"""

with open(html_out, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generated dashboard: {html_out} (Size: {len(html_content)} bytes)", flush=True)
