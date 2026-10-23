#!/usr/bin/env python3
"""
Build interactive dashboard and render population Cohen & Kohn figure for [-50 ms, +100 ms] 6-panel literature cards.
"""
import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(r"C:\Users\lpascu\analysis_src")
sys.path.insert(0, str(PROJECT_ROOT))
import src.pipeline.compat_patch

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


out_dir = PROJECT_ROOT / "output_e097_prelaser_through_laser3_226s_motion_corrected" / "consensus_master" / "03_spiking_statistics_literature_cards_neg50_pos100"

# 1. Render Population Figure
df_pairs = pd.read_csv(out_dir / "population_pairwise_correlations.csv")
print(f"Loaded {len(df_pairs)} pairwise relationships")

fig_pop, (ax_scat, ax_decay) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'wspace': 0.28})
mask_local = df_pairs["delta_y_um"] < 200.0
mask_dist = df_pairs["delta_y_um"] >= 200.0

ax_scat.plot(df_pairs.loc[mask_dist, "r_signal"].values, df_pairs.loc[mask_dist, "r_noise"].values,
             'o', color="#94a3b8", alpha=0.45, markersize=4, label=r"Distant Pairs ($\Delta y \geq 200\,\mu\mathrm{m}$)")
ax_scat.plot(df_pairs.loc[mask_local, "r_signal"].values, df_pairs.loc[mask_local, "r_noise"].values,
             'o', color="#0284c7", alpha=0.75, markersize=5, label=r"Local Pairs ($\Delta y < 200\,\mu\mathrm{m}$)")

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
ax_scat.set_xlabel("Signal Correlation $r_{\\mathrm{signal}}$ ([-50, +100] ms PETH)", fontsize=10, fontweight="bold")
ax_scat.set_ylabel("Noise Correlation $r_{\\mathrm{noise}}$ (Spike Count Fluctuation)", fontsize=10, fontweight="bold")
ax_scat.set_title("Cohen & Kohn (2011) Structure:\n$r_{\\mathrm{noise}}$ vs $r_{\\mathrm{signal}}$ in [-50, +100] ms Window", fontsize=11, fontweight="bold")
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

plt.suptitle("Population Spiking Statistics (Cohen & Kohn 2011) — [-50 ms, +100 ms] Transition Window (e097)", fontsize=12, fontweight="bold", y=0.98)
pop_fig_path = out_dir / "population_cohen_kohn_correlations.png"
fig_pop.savefig(pop_fig_path, dpi=180)
plt.close(fig_pop)
print(f"[+] Saved population figure to: {pop_fig_path}")

# 2. Build Interactive HTML Dashboard
df_summary = pd.read_csv(out_dir / "spiking_statistics_lit_summary_neg50_pos100.csv")

split_suspects = df_pairs[(df_pairs["delta_y_um"] < 60.0) & (df_pairs["r_signal"] > 0.65)].sort_values("r_signal", ascending=False)
split_suspect_cids = set(split_suspects["cid_1"].tolist() + split_suspects["cid_2"].tolist())

card_items = []
for idx_r, r in df_summary.iterrows():
    cid = int(r["cluster_id"])
    img_name = f"lit_card_cluster_{cid:03d}.png"
    if not (out_dir / img_name).exists():
        continue

    r1 = float(r["r_ep1"]) if "r_ep1" in r and not pd.isna(r["r_ep1"]) else (float(r["r_schreiber_ep1_10ms"]) if not pd.isna(r["r_schreiber_ep1_10ms"]) else 0.0)
    r2 = float(r["r_ep2"]) if "r_ep2" in r and not pd.isna(r["r_ep2"]) else 0.0
    r3 = float(r["r_ep3"]) if "r_ep3" in r and not pd.isna(r["r_ep3"]) else 0.0
    r4 = float(r["r_ep4"]) if "r_ep4" in r and not pd.isna(r["r_ep4"]) else (float(r["r_schreiber_ep4_10ms"]) if not pd.isna(r["r_schreiber_ep4_10ms"]) else 0.0)
    r12 = float(r["r_cross_1_2"]) if "r_cross_1_2" in r and not pd.isna(r["r_cross_1_2"]) else 0.0
    r23 = float(r["r_cross_2_3"]) if "r_cross_2_3" in r and not pd.isna(r["r_cross_2_3"]) else 0.0
    r34 = float(r["r_cross_3_4"]) if "r_cross_3_4" in r and not pd.isna(r["r_cross_3_4"]) else 0.0
    r14 = float(r["r_cross_1_4"]) if "r_cross_1_4" in r and not pd.isna(r["r_cross_1_4"]) else (float(r["r_schreiber_cross_10ms"]) if not pd.isna(r["r_schreiber_cross_10ms"]) else 0.0)
    max_drop = max(0.0, r1 - r12, r2 - r23, r3 - r34)

    card_items.append({
        "cid": cid,
        "depth": float(r["depth_um"]),
        "spikes": int(r["n_spikes"]),
        "spikes_win": int(r["spikes_win"]),
        "fr": float(r["firing_rate_hz"]),
        "qc": bool(r["is_qc_passed"]),
        "cell_type": str(r["cell_type"]),
        "r_ep1": round(r1, 3),
        "r_ep2": round(r2, 3),
        "r_ep3": round(r3, 3),
        "r_ep4": round(r4, 3),
        "r_cross_1_2": round(r12, 3),
        "r_cross_2_3": round(r23, 3),
        "r_cross_3_4": round(r34, 3),
        "r_cross_1_4": round(r14, 3),
        "max_drop_val": round(max_drop, 3),
        "sigma_star": round(float(r["sigma_star_ms"]), 1) if not pd.isna(r["sigma_star_ms"]) else "N/A",
        "phase_pref": round(float(r["phase_pref_ep1"]), 2) if "phase_pref_ep1" in r and not pd.isna(r["phase_pref_ep1"]) else "N/A",
        "phase_r": round(float(r["phase_r_ep1"]), 3) if "phase_r_ep1" in r and not pd.isna(r["phase_r_ep1"]) else 0.0,
        "peak_ep1": round(float(r["peak_ep1_hz"]), 1) if not pd.isna(r["peak_ep1_hz"]) else 0.0,
        "peak_ep2": round(float(r["peak_ep2_hz"]), 1) if "peak_ep2_hz" in r and not pd.isna(r["peak_ep2_hz"]) else 0.0,
        "peak_ep3": round(float(r["peak_ep3_hz"]), 1) if "peak_ep3_hz" in r and not pd.isna(r["peak_ep3_hz"]) else 0.0,
        "peak_ep4": round(float(r["peak_ep4_hz"]), 1) if not pd.isna(r["peak_ep4_hz"]) else 0.0,
        "is_split_suspect": (cid in split_suspect_cids),
        "img": img_name
    })

print(f"Loaded {len(card_items)} rendered cards")

html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>e097 Literature Spiking Statistics [-50 ms, +100 ms]</title>
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
    <h1>Canonical Literature Spiking Diagnostics ([-50 ms, +100 ms] Transition Window)</h1>
    <div class="subtitle">
      Centered strictly on <strong>Flow Inspiration Offset / Zero-Crossing ($t = 0\\text{ ms}$)</strong> across all 842 breaths (e097).<br>
      Implements <strong>Schreiber et al. (2003)</strong> reliability &amp; precision curves, <strong>van Rossum (2001)</strong> MDS metric space, <strong>Cohen &amp; Kohn (2011)</strong> signal vs noise correlations, and <strong>Fisher (1993)</strong> polar phase tuning across ALL 4 Epochs.
    </div>
    <div class="metrics-bar">
      <div class="metric-chip"><strong>Literature Cards Rendered:</strong> __RENDERED_COUNT__</div>
      <div class="metric-chip"><strong>Allen QC Passed:</strong> __QC_RENDERED__</div>
      <div class="metric-chip"><strong>Transition Window:</strong> -50 ms to +100 ms (150 ms total)</div>
      <div class="metric-chip"><strong>QC Pop Within R:</strong> __POP_WITHIN__</div>
      <div class="metric-chip"><strong>QC Pop Transitions:</strong> __POP_TRANSITIONS__</div>
      <div class="metric-chip"><strong>Pairwise Relationships:</strong> __PAIRS_COUNT__</div>
    </div>
  </header>

  <section class="pop-section">
    <h2>Population Analysis: Cohen &amp; Kohn (2011) [-50 ms, +100 ms] Structure</h2>
    <p class="subtitle" style="margin-bottom: 14px;">
      Decomposition of co-variability strictly within the 150 ms transition window. Local units (&Delta;y &lt; 200 &mu;m) share positive trial-to-trial spike count fluctuations scaled by signal correlation (r_signal), decaying with probe somatic distance.
    </p>
    <img src="population_cohen_kohn_correlations.png" class="pop-img" onclick="openModal(this.src)" alt="Population Cohen &amp; Kohn Analysis">
  </section>

  <div class="controls-bar">
    <div class="filter-group">
      <button class="btn-filter active" onclick="setFilter('all', this)">All Units (__RENDERED_COUNT__)</button>
      <button class="btn-filter" onclick="setFilter('qc', this)">QC Passed (__QC_RENDERED__)</button>
      <button class="btn-filter" onclick="setFilter('exp', this)">Pure Post-I</button>
      <button class="btn-filter" onclick="setFilter('insp', this)">Late Inspiratory</button>
      <button class="btn-filter" onclick="setFilter('split', this)">Split Suspects (__SPLIT_COUNT__)</button>
      <button class="btn-filter" onclick="setFilter('high_rel', this)">High Reliability (R &ge; 0.35)</button>
    </div>
    <input type="text" id="searchInput" class="search-input" placeholder="Search Cluster ID (e.g. 212, 13, 150)..." oninput="applyFilters()">
    <select id="sortSelect" class="select-sort" onchange="applySort()">
      <option value="depth_asc">Sort: Dorsal to Ventral (Depth &uarr;)</option>
      <option value="depth_desc">Sort: Ventral to Dorsal (Depth &darr;)</option>
      <option value="cid_asc">Sort: Cluster ID (Ascending)</option>
      <option value="rel_desc">Sort: Ep 1 Within R (Highest &darr;)</option>
      <option value="trans_1_2_desc">Sort: Laser Train 1 Transition (R 1&rarr;2 &darr;)</option>
      <option value="trans_2_3_desc">Sort: Laser Train 2 Transition (R 2&rarr;3 &darr;)</option>
      <option value="trans_3_4_desc">Sort: Laser Train 3 Transition (R 3&rarr;4 &darr;)</option>
      <option value="trans_1_4_desc">Sort: Baseline to Recovery (R 1&rarr;4 &darr;)</option>
      <option value="drop_desc">Sort: Max Transition Drop (Largest Drift / Split &darr;)</option>
      <option value="spikes_desc">Sort: Window Spikes (Highest &darr;)</option>
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
            <div class="metrics-grid" style="margin-top:8px;">
              <div class="metric-item"><strong>Type:</strong> ${c.cell_type}</div>
              <div class="metric-item"><strong>Win Spikes:</strong> ${c.spikes_win} spk</div>
              <div class="metric-item"><strong>Precision &sigma;*:</strong> ${c.sigma_star} ms</div>
              <div class="metric-item"><strong>Phase Vector R:</strong> ${c.phase_r.toFixed(3)}</div>
            </div>
            <div style="background:#0f172a; padding:8px 10px; border-radius:6px; border:1px solid #1e293b; margin-top:8px; font-size:11.5px;">
              <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#94a3b8; font-weight:bold;">Within R:</span>
                <span><b style="color:#38bdf8">Ep1:</b> ${c.r_ep1.toFixed(2)} | <b style="color:#2dd4bf">Ep2:</b> ${c.r_ep2.toFixed(2)} | <b style="color:#fbbf24">Ep3:</b> ${c.r_ep3.toFixed(2)} | <b style="color:#f87171">Ep4:</b> ${c.r_ep4.toFixed(2)}</span>
              </div>
              <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#94a3b8; font-weight:bold;">Cross R:</span>
                <span><b>1&rarr;2:</b> ${c.r_cross_1_2.toFixed(2)} | <b>2&rarr;3:</b> ${c.r_cross_2_3.toFixed(2)} | <b>3&rarr;4:</b> ${c.r_cross_3_4.toFixed(2)} | <b>1&rarr;4:</b> ${c.r_cross_1_4.toFixed(2)}</span>
              </div>
              <div style="display:flex; justify-content:space-between;">
                <span style="color:#94a3b8; font-weight:bold;">Peak FR:</span>
                <span>${c.peak_ep1.toFixed(0)} &rarr; ${c.peak_ep2.toFixed(0)} &rarr; ${c.peak_ep3.toFixed(0)} &rarr; ${c.peak_ep4.toFixed(0)} Hz</span>
              </div>
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
        if (activeFilter === 'exp') return c.cell_type.includes('Post-I') || c.cell_type.includes('Expiratory');
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
      else if (sortMode === 'trans_1_2_desc') list.sort((a, b) => b.r_cross_1_2 - a.r_cross_1_2);
      else if (sortMode === 'trans_2_3_desc') list.sort((a, b) => b.r_cross_2_3 - a.r_cross_2_3);
      else if (sortMode === 'trans_3_4_desc') list.sort((a, b) => b.r_cross_3_4 - a.r_cross_3_4);
      else if (sortMode === 'trans_1_4_desc') list.sort((a, b) => b.r_cross_1_4 - a.r_cross_1_4);
      else if (sortMode === 'drop_desc') list.sort((a, b) => b.max_drop_val - a.max_drop_val);
      else if (sortMode === 'spikes_desc') list.sort((a, b) => b.spikes_win - a.spikes_win);
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

    applySortToList([...items]);
  </script>
</body>
</html>
"""

qc_items = [c for c in card_items if c["qc"]]
qc_count = len(qc_items)
split_count = len([c for c in card_items if c["is_split_suspect"]])

if qc_count > 0:
    pop_w1 = np.mean([c["r_ep1"] for c in qc_items])
    pop_w2 = np.mean([c["r_ep2"] for c in qc_items])
    pop_w3 = np.mean([c["r_ep3"] for c in qc_items])
    pop_w4 = np.mean([c["r_ep4"] for c in qc_items])
    pop_t12 = np.mean([c["r_cross_1_2"] for c in qc_items])
    pop_t23 = np.mean([c["r_cross_2_3"] for c in qc_items])
    pop_t34 = np.mean([c["r_cross_3_4"] for c in qc_items])
    pop_t14 = np.mean([c["r_cross_1_4"] for c in qc_items])
    pop_within_str = f"Ep1: {pop_w1:.2f} &rarr; Ep2: {pop_w2:.2f} &rarr; Ep3: {pop_w3:.2f} &rarr; Ep4: {pop_w4:.2f}"
    pop_trans_str = f"1&rarr;2: {pop_t12:.2f} | 2&rarr;3: {pop_t23:.2f} | 3&rarr;4: {pop_t34:.2f} | 1&rarr;4: {pop_t14:.2f}"
else:
    pop_within_str = "N/A"
    pop_trans_str = "N/A"

final_html = html_template.replace("__RENDERED_COUNT__", str(len(card_items))) \
                          .replace("__QC_RENDERED__", str(qc_count)) \
                          .replace("__SPLIT_COUNT__", str(split_count)) \
                          .replace("__PAIRS_COUNT__", str(len(df_pairs))) \
                          .replace("__POP_WITHIN__", pop_within_str) \
                          .replace("__POP_TRANSITIONS__", pop_trans_str) \
                          .replace("__JSON_ITEMS__", json.dumps(card_items))

dashboard_path = out_dir / "literature_spiking_statistics_dashboard.html"
with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(final_html)

print(f"[+] Successfully wrote dashboard to: {dashboard_path}")

