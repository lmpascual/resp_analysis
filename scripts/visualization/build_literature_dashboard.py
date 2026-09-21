#!/usr/bin/env python3
"""
Build interactive HTML dashboard for e097 literature spiking diagnostics.
"""
import sys
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\lpascu\analysis_src")
out_cards_dir = PROJECT_ROOT / "output_e097_prelaser_through_laser3_226s_motion_corrected" / "consensus_master" / "03_spiking_statistics_literature_cards"

df_pairs = pd.read_csv(out_cards_dir / "population_pairwise_correlations.csv")
df_summary = pd.read_csv(out_cards_dir / "spiking_statistics_literature_summary.csv")

split_suspects = df_pairs[(df_pairs["delta_y_um"] < 60.0) & (df_pairs["r_signal"] > 0.65)].sort_values("r_signal", ascending=False)
split_suspect_cids = set(split_suspects["cid_1"].tolist() + split_suspects["cid_2"].tolist())

card_items = []
for idx_r, r in df_summary.iterrows():
    cid = int(r["cluster_id"])
    img_name = f"literature_diagnostic_cluster_{cid:03d}.png"
    if not (out_cards_dir / img_name).exists():
        continue

    card_items.append({
        "cid": cid,
        "depth": float(r["depth_um"]),
        "spikes": int(r["n_spikes"]),
        "fr": float(r["firing_rate_hz"]),
        "qc": bool(r["is_qc_passed"]),
        "cell_type": str(r["cell_type"]),
        "r_ep1": round(float(r["r_schreiber_ep1_15ms"]), 3) if not pd.isna(r["r_schreiber_ep1_15ms"]) else 0.0,
        "r_ep4": round(float(r["r_schreiber_ep4_15ms"]), 3) if not pd.isna(r["r_schreiber_ep4_15ms"]) else 0.0,
        "r_cross": round(float(r["r_schreiber_cross_1_4"]), 3) if not pd.isna(r["r_schreiber_cross_1_4"]) else 0.0,
        "sigma_star": round(float(r["sigma_star_ms"]), 1) if not pd.isna(r["sigma_star_ms"]) else "N/A",
        "phase_pref": round(float(r["phase_pref_ep1_rad"]), 2) if not pd.isna(r["phase_pref_ep1_rad"]) else "N/A",
        "phase_r": round(float(r["phase_r_ep1"]), 3) if not pd.isna(r["phase_r_ep1"]) else 0.0,
        "is_split_suspect": (cid in split_suspect_cids),
        "img": img_name
    })

print(f"Loaded {len(card_items)} rendered cards")

html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>e097 Canonical Spiking Statistics & Literature Diagnostic Gallery</title>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f2937;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --accent: #0284c7;
      --accent-glow: rgba(2, 132, 199, 0.25);
      --qc-pass: #10b981;
      --qc-fail: #ef4444;
      --split-border: #f59e0b;
    }
    body {
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    header {
      max-width: 1440px;
      margin: 0 auto 24px auto;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 20px;
    }
    h1 {
      margin: 0 0 8px 0;
      font-size: 26px;
      color: #38bdf8;
    }
    .subtitle {
      color: var(--text-muted);
      font-size: 14px;
      line-height: 1.5;
    }
    .metrics-bar {
      display: flex;
      gap: 16px;
      margin-top: 16px;
      flex-wrap: wrap;
    }
    .metric-chip {
      background: #1e293b;
      padding: 8px 14px;
      border-radius: 6px;
      border: 1px solid #334155;
      font-size: 13px;
    }
    .metric-chip strong {
      color: #38bdf8;
      margin-right: 4px;
    }
    .pop-section {
      max-width: 1440px;
      margin: 0 auto 28px auto;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 20px;
    }
    .pop-section h2 {
      margin-top: 0;
      font-size: 18px;
      color: #f8fafc;
    }
    .pop-img {
      width: 100%;
      border-radius: 6px;
      border: 1px solid #334155;
      cursor: pointer;
    }
    .controls-bar {
      max-width: 1440px;
      margin: 0 auto 24px auto;
      display: flex;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
      background: #0f172a;
      padding: 14px 18px;
      border-radius: 8px;
      border: 1px solid #1e293b;
    }
    .filter-group {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .btn-filter {
      background: #1e293b;
      color: var(--text);
      border: 1px solid #334155;
      padding: 6px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
      transition: all 0.15s ease;
    }
    .btn-filter.active {
      background: var(--accent);
      border-color: #38bdf8;
      font-weight: bold;
    }
    .search-input, .select-sort {
      background: #1e293b;
      color: var(--text);
      border: 1px solid #334155;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 13px;
    }
    .gallery-grid {
      max-width: 1440px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(440px, 1fr));
      gap: 20px;
    }
    .unit-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .unit-card:hover {
      transform: translateY(-2px);
      border-color: #38bdf8;
    }
    .unit-card.split-candidate {
      border-color: var(--split-border);
      box-shadow: 0 0 10px rgba(245, 158, 11, 0.2);
    }
    .card-header {
      padding: 10px 14px;
      background: #1a2234;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--card-border);
    }
    .card-title {
      font-size: 14px;
      font-weight: bold;
      color: #f8fafc;
    }
    .badge {
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 4px;
      font-weight: bold;
    }
    .badge-qc { background: #064e3b; color: #34d399; border: 1px solid #059669; }
    .badge-noqc { background: #450a0a; color: #f87171; border: 1px solid #b91c1c; }
    .badge-split { background: #78350f; color: #fde68a; border: 1px solid #d97706; }
    .card-body {
      padding: 10px;
    }
    .card-img {
      width: 100%;
      height: auto;
      border-radius: 4px;
      cursor: pointer;
      display: block;
    }
    .metrics-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      margin-top: 10px;
      font-size: 11.5px;
      color: #cbd5e1;
    }
    .metric-item strong {
      color: #38bdf8;
    }
    /* Modal */
    .modal {
      display: none;
      position: fixed;
      z-index: 1000;
      top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0,0,0,0.92);
      justify-content: center;
      align-items: center;
    }
    .modal.active {
      display: flex;
    }
    .modal-content {
      max-width: 95vw;
      max-height: 95vh;
      border-radius: 8px;
      border: 2px solid #38bdf8;
    }
  </style>
</head>
<body>
  <header>
    <h1>Respiratory Spiking Statistics & Diagnostic Suite (e097)</h1>
    <div class="subtitle">
      Aligned to Expiration Onset (I_off / Flow Zero-Crossing) across the 226s recording window (842 breaths).<br>
      Implements canonical metrics and visualizations from <strong>Schreiber et al. (2003)</strong> [temporal precision & reliability R_bar(sigma)],
      <strong>van Rossum (2001)</strong> [spike distance metric space via Classical MDS],
      <strong>Cohen & Kohn (2011)</strong> [signal vs noise correlation decomposition], and
      <strong>Fisher (1993)</strong> [circular respiratory phase vector statistics].
    </div>
    <div class="metrics-bar">
      <div class="metric-chip"><strong>Total Consensus Units:</strong> __TOTAL_UNITS__</div>
      <div class="metric-chip"><strong>Literature Cards Rendered:</strong> __RENDERED_COUNT__</div>
      <div class="metric-chip"><strong>Allen QC Passed:</strong> __QC_COUNT__</div>
      <div class="metric-chip"><strong>Reference Landmark:</strong> Flow Meter Zero-Crossing (100.0% detected)</div>
      <div class="metric-chip"><strong>Pairwise Relationships:</strong> __PAIRS_COUNT__</div>
    </div>
  </header>

  <section class="pop-section">
    <h2>Population Analysis: Cohen & Kohn (2011) Respiratory Structure</h2>
    <p class="subtitle" style="margin-bottom: 14px;">
      Decomposition of co-variability across respiratory units. Local units (delta_y &lt; 200 µm) exhibit structured positive noise correlations that scale linearly with signal tuning similarity (r_signal), decaying with probe somatic distance.
    </p>
    <img src="population_cohen_kohn_correlations.png" class="pop-img" onclick="openModal(this.src)" alt="Population Cohen & Kohn Analysis">
  </section>

  <div class="controls-bar">
    <div class="filter-group">
      <button class="btn-filter active" onclick="setFilter('all', this)">All Units (__RENDERED_COUNT__)</button>
      <button class="btn-filter" onclick="setFilter('qc', this)">QC Passed (__QC_RENDERED__)</button>
      <button class="btn-filter" onclick="setFilter('exp', this)">Expiratory Phasic</button>
      <button class="btn-filter" onclick="setFilter('insp', this)">Inspiratory</button>
      <button class="btn-filter" onclick="setFilter('split', this)">Split Suspects (__SPLIT_COUNT__)</button>
      <button class="btn-filter" onclick="setFilter('high_rel', this)">High Reliability (R &ge; 0.35)</button>
    </div>
    <input type="text" id="searchInput" class="search-input" placeholder="Search Cluster ID (e.g. 212, 13)..." oninput="applyFilters()">
    <select id="sortSelect" class="select-sort" onchange="applySort()">
      <option value="depth_asc">Sort: Dorsal to Ventral (Depth &uarr;)</option>
      <option value="depth_desc">Sort: Ventral to Dorsal (Depth &darr;)</option>
      <option value="cid_asc">Sort: Cluster ID (Ascending)</option>
      <option value="rel_desc">Sort: Reliability Ep1 (Highest &darr;)</option>
      <option value="spikes_desc">Sort: Spike Count (Highest &darr;)</option>
    </select>
  </div>

  <div class="gallery-grid" id="galleryGrid"></div>

  <div class="modal" id="imageModal" onclick="closeModal()">
    <img class="modal-content" id="modalImg">
  </div>

  <script>
    const items = __JSON_ITEMS__;
    let activeFilter = 'all';

    function renderCards(cardList) {
      const grid = document.getElementById('galleryGrid');
      grid.innerHTML = '';
      cardList.forEach(c => {
        const card = document.createElement('div');
        card.className = 'unit-card' + (c.is_split_suspect ? ' split-candidate' : '');
        card.innerHTML = `
          <div class="card-header">
            <span class="card-title">Cluster #${String(c.cid).padStart(3, '0')} (${Math.round(c.depth)} µm)</span>
            <div>
              ${c.qc ? '<span class="badge badge-qc">QC PASSED</span>' : '<span class="badge badge-noqc">NON-QC</span>'}
              ${c.is_split_suspect ? '<span class="badge badge-split">SPLIT CANDIDATE</span>' : ''}
            </div>
          </div>
          <div class="card-body">
            <img src="${c.img}" class="card-img" loading="lazy" onclick="openModal('${c.img}')" alt="Cluster ${c.cid}">
            <div class="metrics-grid">
              <div class="metric-item"><strong>Type:</strong> ${c.cell_type}</div>
              <div class="metric-item"><strong>FR:</strong> ${c.fr.toFixed(1)} Hz (${c.spikes} spk)</div>
              <div class="metric-item"><strong>Schreiber R (Ep1):</strong> ${c.r_ep1.toFixed(3)}</div>
              <div class="metric-item"><strong>Cross Ep1-4 R:</strong> ${c.r_cross.toFixed(3)}</div>
              <div class="metric-item"><strong>Precision &sigma;*:</strong> ${c.sigma_star} ms</div>
              <div class="metric-item"><strong>Phase Vector R:</strong> ${c.phase_r.toFixed(3)}</div>
            </div>
          </div>
        `;
        grid.appendChild(card);
      });
    }

    function setFilter(filter, el) {
      activeFilter = filter;
      document.querySelectorAll('.btn-filter').forEach(btn => btn.classList.remove('active'));
      if (el) el.classList.add('active');
      applyFilters();
    }

    function applyFilters() {
      const q = document.getElementById('searchInput').value.trim().toLowerCase();
      let filtered = items.filter(c => {
        if (q && !String(c.cid).includes(q)) return false;
        if (activeFilter === 'qc') return c.qc;
        if (activeFilter === 'exp') return c.cell_type.includes('Expiratory') || c.cell_type.includes('Post-I');
        if (activeFilter === 'insp') return c.cell_type.includes('Inspiratory');
        if (activeFilter === 'split') return c.is_split_suspect;
        if (activeFilter === 'high_rel') return c.r_ep1 >= 0.35;
        return true;
      });
      applySortToList(filtered);
    }

    function applySort() {
      applyFilters();
    }

    function applySortToList(list) {
      const sortMode = document.getElementById('sortSelect').value;
      if (sortMode === 'depth_asc') list.sort((a, b) => a.depth - b.depth);
      else if (sortMode === 'depth_desc') list.sort((a, b) => b.depth - a.depth);
      else if (sortMode === 'cid_asc') list.sort((a, b) => a.cid - b.cid);
      else if (sortMode === 'rel_desc') list.sort((a, b) => b.r_ep1 - a.r_ep1);
      else if (sortMode === 'spikes_desc') list.sort((a, b) => b.spikes - a.spikes);
      renderCards(list);
    }

    function openModal(src) {
      const m = document.getElementById('imageModal');
      const img = document.getElementById('modalImg');
      img.src = src;
      m.classList.add('active');
    }

    function closeModal() {
      document.getElementById('imageModal').classList.remove('active');
    }

    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') closeModal();
    });

    // Initial render
    applySortToList([...items]);
  </script>
</body>
</html>
"""

qc_rendered = len([c for c in card_items if c["qc"]])
split_count = len([c for c in card_items if c["is_split_suspect"]])

final_html = html_template.replace("__TOTAL_UNITS__", str(len(df_summary))) \
                          .replace("__RENDERED_COUNT__", str(len(card_items))) \
                          .replace("__QC_COUNT__", str(len(df_summary[df_summary["is_qc_passed"] == True]))) \
                          .replace("__PAIRS_COUNT__", str(len(df_pairs))) \
                          .replace("__QC_RENDERED__", str(qc_rendered)) \
                          .replace("__SPLIT_COUNT__", str(split_count)) \
                          .replace("__JSON_ITEMS__", json.dumps(card_items))

dashboard_file = out_cards_dir / "literature_spiking_statistics_dashboard.html"
with open(dashboard_file, "w", encoding="utf-8") as f:
    f.write(final_html)

print(f"[+] Successfully wrote {dashboard_file} ({len(final_html)} bytes)")
