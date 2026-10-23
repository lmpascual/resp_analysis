#!/usr/bin/env python3
"""
Render Cohen & Kohn (2011) population correlation figure and build the interactive HTML dashboard.
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(r"C:\Users\lpascu\analysis_src")
sys.path.insert(0, str(PROJECT_ROOT))
import src.pipeline.compat_patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.0

out_cards_dir = PROJECT_ROOT / "output_e097_prelaser_through_laser3_226s_motion_corrected" / "consensus_master" / "03_spiking_statistics_literature_cards"

# 1. Render Population Cohen & Kohn (2011) Figure
print("[1/2] Rendering population Cohen & Kohn figure...", flush=True)
pairs_csv = out_cards_dir / "population_pairwise_correlations.csv"
df_pairs = pd.read_csv(pairs_csv)
print(f"      Loaded {len(df_pairs)} pairwise relationships", flush=True)

fig_pop, (ax_scat, ax_decay) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'wspace': 0.28})

mask_local = df_pairs["delta_y_um"] < 200.0
mask_dist = df_pairs["delta_y_um"] >= 200.0

ax_scat.scatter(df_pairs.loc[mask_dist, "r_signal"], df_pairs.loc[mask_dist, "r_noise"], color="#94a3b8", alpha=0.45, s=20, label=r"Distant Pairs ($\Delta y \ge 200\,\mu\mathrm{m}$)")
ax_scat.scatter(df_pairs.loc[mask_local, "r_signal"], df_pairs.loc[mask_local, "r_noise"], color="#0284c7", alpha=0.75, s=28, label=r"Local Pairs ($\Delta y < 200\,\mu\mathrm{m}$)")

valid_p = df_pairs.dropna(subset=["r_signal", "r_noise"])
if len(valid_p) > 5:
    x = valid_p["r_signal"].values
    y = valid_p["r_noise"].values
    vx = np.var(x)
    slope = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (vx if vx > 1e-12 else 1.0)
    intercept = np.mean(y) - slope * np.mean(x)
    r_val = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.std(x) * np.std(y) + 1e-12)
    x_fit = np.linspace(-0.8, 1.0, 100)
    ax_scat.plot(x_fit, intercept + slope * x_fit, color="#e11d48", linewidth=2.0, label=f"Fit: slope={slope:.3f} ($r={r_val:.2f}$)")

ax_scat.axhline(0, color="#475569", linestyle="--", linewidth=1.0)
ax_scat.axvline(0, color="#475569", linestyle="--", linewidth=1.0)
ax_scat.set_xlim(-0.85, 1.05)
ax_scat.set_ylim(-0.4, 0.8)
ax_scat.set_xlabel("Signal Correlation $r_{\\mathrm{signal}}$ (Tuning Similarity)", fontsize=10, fontweight="bold")
ax_scat.set_ylabel("Noise Correlation $r_{\\mathrm{noise}}$ (Spike Count Fluctuation)", fontsize=10, fontweight="bold")
ax_scat.set_title("Cohen & Kohn (2011) Structure:\n$r_{\\mathrm{noise}}$ vs $r_{\\mathrm{signal}}$ Across Respiratory Pairs", fontsize=11, fontweight="bold")
ax_scat.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
ax_scat.grid(True, linestyle="--", alpha=0.4)

bin_edges_y = np.linspace(0, 1800, 10)
bin_centers_y = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2.0
mean_rn_binned = []
sem_rn_binned = []

for k in range(len(bin_edges_y) - 1):
    y0, y1 = bin_edges_y[k], bin_edges_y[k+1]
    sub_rn = df_pairs[(df_pairs["delta_y_um"] >= y0) & (df_pairs["delta_y_um"] < y1)]["r_noise"].dropna().values
    if len(sub_rn) > 2:
        mean_rn_binned.append(float(np.mean(sub_rn)))
        sem_rn_binned.append(float(np.std(sub_rn) / np.sqrt(len(sub_rn))))
    else:
        mean_rn_binned.append(np.nan)
        sem_rn_binned.append(np.nan)

ax_decay.errorbar(bin_centers_y, mean_rn_binned, yerr=sem_rn_binned, color="#0f766e", fmt="-o", linewidth=2.2, capsize=4, label="Mean ± SEM")
ax_decay.axhline(0, color="#475569", linestyle="--", linewidth=1.0)
ax_decay.set_xlabel("Somatic Distance Along Probe $\\Delta y$ (µm)", fontsize=10, fontweight="bold")
ax_decay.set_ylabel("Mean Noise Correlation $r_{\\mathrm{noise}}$", fontsize=10, fontweight="bold")
ax_decay.set_title("Spatial Noise Correlation Decay:\nShared Fluctuations vs Somatic Separation", fontsize=11, fontweight="bold")
ax_decay.set_ylim(-0.05, 0.35)
ax_decay.legend(loc="upper right", fontsize=8.5)
ax_decay.grid(True, linestyle="--", alpha=0.4)

plt.suptitle("Population Spiking Statistics Grounded in Cohen & Kohn (2011) — Neuropixels e097 (226s Window)", fontsize=12, fontweight="bold", y=0.98)
pop_fig_path = out_cards_dir / "population_cohen_kohn_correlations.png"
fig_pop.savefig(pop_fig_path, dpi=180)
plt.close(fig_pop)
print(f"[+] Saved population Cohen & Kohn figure to: {pop_fig_path}", flush=True)

# 2. Build Interactive HTML Dashboard
print("[2/2] Generating Interactive HTML Literature Dashboard...", flush=True)
summary_csv = out_cards_dir / "spiking_statistics_literature_summary.csv"
df_summary = pd.read_csv(summary_csv)

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
        "r_ep1": round(float(r["r_schreiber_ep1_15ms"]), 3) if not np.isnan(r["r_schreiber_ep1_15ms"]) else 0.0,
        "r_ep4": round(float(r["r_schreiber_ep4_15ms"]), 3) if not np.isnan(r["r_schreiber_ep4_15ms"]) else 0.0,
        "r_cross": round(float(r["r_schreiber_cross_1_4"]), 3) if not np.isnan(r["r_schreiber_cross_1_4"]) else 0.0,
        "sigma_star": round(float(r["sigma_star_ms"]), 1) if not np.isnan(r["sigma_star_ms"]) else "N/A",
        "phase_pref": round(float(r["phase_pref_ep1_rad"]), 2) if not np.isnan(r["phase_pref_ep1_rad"]) else "N/A",
        "phase_r": round(float(r["phase_r_ep1"]), 3) if not np.isnan(r["phase_r_ep1"]) else 0.0,
        "is_split_suspect": (cid in split_suspect_cids),
        "img": img_name
    })

print(f"      Prepared JSON for {len(card_items)} rendered unit cards", flush=True)

html_dashboard_path = out_cards_dir / "literature_spiking_statistics_dashboard.html"
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>e097 Canonical Spiking Statistics & Literature Diagnostic Gallery</title>
  <style>
    :root {{
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
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    header {{
      max-width: 1440px;
      margin: 0 auto 24px auto;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 20px;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 26px;
      color: #38bdf8;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 14px;
      line-height: 1.5;
    }}
    .metrics-bar {{
      display: flex;
      gap: 16px;
      margin-top: 16px;
      flex-wrap: wrap;
    }}
    .metric-chip {{
      background: #1e293b;
      padding: 8px 14px;
      border-radius: 6px;
      border: 1px solid #334155;
      font-size: 13px;
    }}
    .metric-chip strong {{
      color: #38bdf8;
      margin-right: 4px;
    }}
    .pop-section {{
      max-width: 1440px;
      margin: 0 auto 28px auto;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 20px;
    }}
    .pop-section h2 {{
      margin-top: 0;
      font-size: 18px;
      color: #f8fafc;
    }}
    .pop-img {{
      width: 100%;
      border-radius: 6px;
      border: 1px solid #334155;
      cursor: pointer;
    }}
    .controls-bar {{
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
    }}
    .filter-group {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .btn-filter {{
      background: #1e293b;
      color: var(--text);
      border: 1px solid #334155;
      padding: 6px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
      transition: all 0.15s ease;
    }}
    .btn-filter.active {{
      background: var(--accent);
      border-color: #38bdf8;
      font-weight: bold;
    }}
    .search-input, .select-sort {{
      background: #1e293b;
      color: var(--text);
      border: 1px solid #334155;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 13px;
    }}
    .gallery-grid {{
      max-width: 1440px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(440px, 1fr));
      gap: 20px;
    }}
    .unit-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    .unit-card:hover {{
      transform: translateY(-2px);
      border-color: #38bdf8;
    }}
    .unit-card.split-candidate {{
      border-color: var(--split-border);
      box-shadow: 0 0 10px rgba(245, 158, 11, 0.2);
    }}
    .card-header {{
      padding: 10px 14px;
      background: #1a2234;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--card-border);
    }}
    .card-title {{
      font-size: 14px;
      font-weight: bold;
      color: #f8fafc;
    }}
    .badge {{
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 4px;
      font-weight: bold;
    }}
    .badge-qc {{ background: #064e3b; color: #34d399; border: 1px solid #059669; }}
    .badge-noqc {{ background: #450a0a; color: #f87171; border: 1px solid #b91c1c; }}
    .badge-split {{ background: #78350f; color: #fde68a; border: 1px solid #d97706; }}
    .card-body {{
      padding: 10px;
    }}
    .card-img {{
      width: 100%;
      height: auto;
      border-radius: 4px;
      cursor: pointer;
      display: block;
    }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      margin-top: 10px;
      font-size: 11.5px;
      color: #cbd5e1;
    }}
    .metric-item strong {{
      color: #38bdf8;
    }}
    /* Modal */
    .modal {{
      display: none;
      position: fixed;
      z-index: 1000;
      top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0,0,0,0.92);
      justify-content: center;
      align-items: center;
    }}
    .modal.active {{
      display: flex;
    }}
    .modal-content {{
      max-width: 95vw;
      max-height: 95vh;
      border-radius: 8px;
      border: 2px solid #38bdf8;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Respiratory Spiking Statistics & Diagnostic Suite (e097)</h1>
    <div class="subtitle">
      Aligned to Expiration Onset ($I_{{\\mathrm{{off}}}}$ / Flow Zero-Crossing) across the 226s recording window (842 breaths).<br>
      Implements canonical metrics and visualizations from <strong>Schreiber et al. (2003)</strong> [temporal precision & reliability $\bar{{R}}(\sigma)$],
      <strong>van Rossum (2001)</strong> [spike distance metric space via Classical MDS],
      <strong>Cohen & Kohn (2011)</strong> [signal vs noise correlation decomposition], and
      <strong>Fisher (1993)</strong> [circular respiratory phase vector statistics].
    </div>
    <div class="metrics-bar">
      <div class="metric-chip"><strong>Total Consensus Units:</strong> {len(df_summary)}</div>
      <div class="metric-chip"><strong>Literature Cards Rendered:</strong> {len(card_items)}</div>
      <div class="metric-chip"><strong>Allen QC Passed:</strong> {len(df_summary[df_summary['is_qc_passed'] == True])}</div>
      <div class="metric-chip"><strong>Reference Landmark:</strong> Flow Meter Zero-Crossing ($100.0\%$ detected)</div>
      <div class="metric-chip"><strong>Pairwise Relationships:</strong> {len(df_pairs)}</div>
    </div>
  </header>

  <section class="pop-section">
    <h2>Population Analysis: Cohen & Kohn (2011) Respiratory Structure</h2>
    <p class="subtitle" style="margin-bottom: 14px;">
      Decomposition of co-variability across respiratory units. Local units ($\Delta y < 200\,\mu\mathrm{{m}}$) exhibit structured positive noise correlations that scale linearly with signal tuning similarity ($r_{{\\mathrm{{signal}}}}$), decaying with probe somatic distance.
    </p>
    <img src="population_cohen_kohn_correlations.png" class="pop-img" onclick="openModal(this.src)" alt="Population Cohen & Kohn Analysis">
  </section>

  <div class="controls-bar">
    <div class="filter-group">
      <button class="btn-filter active" onclick="setFilter('all', this)">All Units ({len(card_items)})</button>
      <button class="btn-filter" onclick="setFilter('qc', this)">QC Passed ({len([c for c in card_items if c['qc']])})</button>
      <button class="btn-filter" onclick="setFilter('exp', this)">Expiratory Phasic</button>
      <button class="btn-filter" onclick="setFilter('insp', this)">Inspiratory</button>
      <button class="btn-filter" onclick="setFilter('split', this)">Split Suspects ({len([c for c in card_items if c['is_split_suspect']])})</button>
      <button class="btn-filter" onclick="setFilter('high_rel', this)">High Reliability ($\bar{{R}} \ge 0.35$)</button>
    </div>
    <input type="text" id="searchInput" class="search-input" placeholder="Search Cluster ID (e.g. 212, 13)..." oninput="applyFilters()">
    <select id="sortSelect" class="select-sort" onchange="applySort()">
      <option value="depth_asc">Sort: Dorsal to Ventral (Depth ↑)</option>
      <option value="depth_desc">Sort: Ventral to Dorsal (Depth ↓)</option>
      <option value="cid_asc">Sort: Cluster ID (Ascending)</option>
      <option value="rel_desc">Sort: Reliability Ep1 (Highest ↓)</option>
      <option value="spikes_desc">Sort: Spike Count (Highest ↓)</option>
    </select>
  </div>

  <div class="gallery-grid" id="galleryGrid"></div>

  <div class="modal" id="imageModal" onclick="closeModal()">
    <img class="modal-content" id="modalImg">
  </div>

  <script>
    const items = {json.dumps(card_items)};
    let activeFilter = 'all';

    function renderCards(cardList) {{
      const grid = document.getElementById('galleryGrid');
      grid.innerHTML = '';
      cardList.forEach(c => {{
        const card = document.createElement('div');
        card.className = 'unit-card' + (c.is_split_suspect ? ' split-candidate' : '');
        card.innerHTML = `
          <div class="card-header">
            <span class="card-title">Cluster #${{String(c.cid).padStart(3, '0')}} (${{Math.round(c.depth)}} µm)</span>
            <div>
              ${{c.qc ? '<span class="badge badge-qc">QC PASSED</span>' : '<span class="badge badge-noqc">NON-QC</span>'}}
              ${{c.is_split_suspect ? '<span class="badge badge-split">SPLIT CANDIDATE</span>' : ''}}
            </div>
          </div>
          <div class="card-body">
            <img src="${{c.img}}" class="card-img" loading="lazy" onclick="openModal('${{c.img}}')" alt="Cluster ${{c.cid}}">
            <div class="metrics-grid">
              <div class="metric-item"><strong>Type:</strong> ${{c.cell_type}}</div>
              <div class="metric-item"><strong>FR:</strong> ${{c.fr.toFixed(1)}} Hz (${{c.spikes}} spk)</div>
              <div class="metric-item"><strong>Schreiber R (Ep1):</strong> ${{c.r_ep1.toFixed(3)}}</div>
              <div class="metric-item"><strong>Cross Ep1-4 R:</strong> ${{c.r_cross.toFixed(3)}}</div>
              <div class="metric-item"><strong>Precision σ*:</strong> ${{c.sigma_star}} ms</div>
              <div class="metric-item"><strong>Phase Vector R:</strong> ${{c.phase_r.toFixed(3)}}</div>
            </div>
          </div>
        `;
        grid.appendChild(card);
      }});
    }}

    function setFilter(filter, el) {{
      activeFilter = filter;
      document.querySelectorAll('.btn-filter').forEach(btn => btn.classList.remove('active'));
      if (el) el.classList.add('active');
      applyFilters();
    }}

    function applyFilters() {{
      const q = document.getElementById('searchInput').value.trim().toLowerCase();
      let filtered = items.filter(c => {{
        if (q && !String(c.cid).includes(q)) return false;
        if (activeFilter === 'qc') return c.qc;
        if (activeFilter === 'exp') return c.cell_type.includes('Expiratory') || c.cell_type.includes('Post-I');
        if (activeFilter === 'insp') return c.cell_type.includes('Inspiratory');
        if (activeFilter === 'split') return c.is_split_suspect;
        if (activeFilter === 'high_rel') return c.r_ep1 >= 0.35;
        return true;
      }});
      applySortToList(filtered);
    }}

    function applySort() {{
      applyFilters();
    }}

    function applySortToList(list) {{
      const sortMode = document.getElementById('sortSelect').value;
      if (sortMode === 'depth_asc') list.sort((a, b) => a.depth - b.depth);
      else if (sortMode === 'depth_desc') list.sort((a, b) => b.depth - a.depth);
      else if (sortMode === 'cid_asc') list.sort((a, b) => a.cid - b.cid);
      else if (sortMode === 'rel_desc') list.sort((a, b) => b.r_ep1 - a.r_ep1);
      else if (sortMode === 'spikes_desc') list.sort((a, b) => b.spikes - a.spikes);
      renderCards(list);
    }}

    function openModal(src) {{
      const m = document.getElementById('imageModal');
      const img = document.getElementById('modalImg');
      img.src = src;
      m.classList.add('active');
    }}

    function closeModal() {{
      document.getElementById('imageModal').classList.remove('active');
    }}

    document.addEventListener('keydown', e => {{
      if (e.key === 'Escape') closeModal();
    }});

    // Initial render
    applySortToList([...items]);
  </script>
</body>
</html>
"""
with open(html_dashboard_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"[+] Successfully generated dashboard at: {html_dashboard_path}", flush=True)
