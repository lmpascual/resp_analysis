#!/usr/bin/env python3
"""
Generates the Unified Tripartite Cross-Sorter Concordance Validation Browser:
Brings together Lupin, Kilosort 4 (Sliding Window), and Kilosort 4 (Global Full-Probe)
into a single comparative dashboard linking directly to each sorter's cluster validation deep-dive pages.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


def generate_tripartite_browser():
    root = Path("outputs/e097/output_e097_prelaser_60s")
    comp_dir = root / "cross_sorter_comparison"
    csv_path = comp_dir / "tripartite_shared_clusters_all_3_sorters.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return
        
    df = pd.read_csv(csv_path)
    
    lupin_units_dir = root / "lupin" / "sorting_validation" / "units"
    slide_units_dir = root / "kilosort4_sliding_window" / "sorting_validation" / "units"
    glob_units_dir = root / "kilosort4_global" / "sorting_validation" / "units"
    
    # Map each cluster ID to its actual generated HTML file
    def find_unit_html(units_dir, uid):
        matches = list(units_dir.glob(f"unit_{uid:04d}_*.html"))
        if matches:
            # relative path from comp_dir
            return Path("..") / matches[0].relative_to(root)
        return None
        
    rows_data = []
    
    for idx, r in df.iterrows():
        u_lup = int(r["lupin_unit_id"])
        u_sld = int(r["ks4_slide_unit_id"])
        u_glb = int(r["ks4_global_unit_id"])
        
        html_lup = find_unit_html(lupin_units_dir, u_lup)
        html_sld = find_unit_html(slide_units_dir, u_sld)
        html_glb = find_unit_html(glob_units_dir, u_glb)
        
        # Determine classification category
        mean_jacc = float(r["mean_3way_jaccard"])
        if mean_jacc >= 0.99:
            tier_class = "tier-identical"
            tier_label = "Identical Spike Train (≥99%)"
        elif mean_jacc >= 0.90:
            tier_class = "tier-near-identical"
            tier_label = "Near-Identical (90–99%)"
        elif mean_jacc >= 0.70:
            tier_class = "tier-high"
            tier_label = "High Concordance (70–90%)"
        elif mean_jacc >= 0.50:
            tier_class = "tier-moderate"
            tier_label = "Moderate Concordance (50–70%)"
        else:
            tier_class = "tier-partial"
            tier_label = "Partial Concordance (<50%)"
            
        rows_data.append({
            "rank": idx + 1,
            "u_lup": u_lup,
            "u_sld": u_sld,
            "u_glb": u_glb,
            "mean_jacc": round(mean_jacc, 4),
            "jacc_lup_sld": round(float(r["jaccard_lupin_ks4slide"]), 4),
            "jacc_sld_glb": round(float(r["jaccard_ks4slide_ks4glob"]), 4),
            "jacc_lup_glb": round(float(r["jaccard_lupin_ks4glob"]), 4),
            "depth_um": round(float(r["depth_um"]), 1),
            "fr_lup": round(float(r["fr_lupin_hz"]), 2),
            "fr_sld": round(float(r["fr_ks4_slide_hz"]), 2),
            "fr_glb": round(float(r["fr_ks4_glob_hz"]), 2),
            "sua_lup": bool(r["sua_lupin"]),
            "sua_sld": bool(r["sua_ks4_slide"]),
            "sua_glb": bool(r["sua_ks4_glob"]),
            "all_three_sua": bool(r["all_three_sua"]),
            "tier_class": tier_class,
            "tier_label": tier_label,
            "html_lup": html_lup.as_posix() if html_lup else None,
            "html_sld": html_sld.as_posix() if html_sld else None,
            "html_glb": html_glb.as_posix() if html_glb else None,
        })
        
    n_total = len(rows_data)
    n_sua3 = sum(1 for r in rows_data if r["all_three_sua"])
    n_ident = sum(1 for r in rows_data if r["mean_jacc"] >= 0.99)
    n_high = sum(1 for r in rows_data if r["mean_jacc"] >= 0.80)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Tripartite Cross-Sorter Concordance Validation Browser | e097_g2</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f2937;
      --card-hover: #1e293b;
      --primary: #38bdf8;
      --primary-glow: rgba(56, 189, 248, 0.15);
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --purple: #a855f7;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --text-dim: #6b7280;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text-main);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      padding: 24px 36px 60px;
      line-height: 1.5;
    }}
    .header {{
      background: linear-gradient(135deg, #111827 0%, #1e1b4b 100%);
      border: 1px solid #3730a3;
      border-radius: 16px;
      padding: 28px 32px;
      margin-bottom: 28px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }}
    .title-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
      margin-bottom: 12px;
    }}
    h1 {{
      font-size: 26px;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 14px;
      max-width: 900px;
    }}
    .pipeline-nav {{
      display: flex;
      gap: 12px;
      margin-top: 18px;
      flex-wrap: wrap;
    }}
    .nav-btn {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 16px;
      border-radius: 10px;
      font-size: 13px;
      font-weight: 600;
      text-decoration: none;
      transition: all 0.2s ease;
      border: 1px solid transparent;
    }}
    .nav-lupin {{
      background: rgba(244, 63, 94, 0.15);
      color: #fb7185;
      border-color: rgba(244, 63, 94, 0.3);
    }}
    .nav-lupin:hover {{ background: rgba(244, 63, 94, 0.25); border-color: #fb7185; }}
    .nav-ks4-slide {{
      background: rgba(14, 165, 233, 0.15);
      color: #38bdf8;
      border-color: rgba(14, 165, 233, 0.3);
    }}
    .nav-ks4-slide:hover {{ background: rgba(14, 165, 233, 0.25); border-color: #38bdf8; }}
    .nav-ks4-glob {{
      background: rgba(168, 85, 247, 0.15);
      color: #c084fc;
      border-color: rgba(168, 85, 247, 0.3);
    }}
    .nav-ks4-glob:hover {{ background: rgba(168, 85, 247, 0.25); border-color: #c084fc; }}

    .stats-bar {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .stat-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 18px 20px;
      display: flex;
      flex-direction: column;
    }}
    .stat-val {{
      font-size: 28px;
      font-weight: 800;
      font-family: 'JetBrains Mono', monospace;
    }}
    .stat-label {{
      font-size: 12px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-top: 4px;
    }}

    .controls-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 20px;
      flex-wrap: wrap;
    }}
    .search-box {{
      position: relative;
      flex: 1;
      max-width: 380px;
    }}
    .search-input {{
      width: 100%;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 10px 16px;
      color: var(--text-main);
      font-size: 14px;
      outline: none;
      transition: border-color 0.2s;
    }}
    .search-input:focus {{ border-color: var(--primary); }}
    .filter-pills {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .pill {{
      padding: 8px 14px;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 20px;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .pill:hover, .pill.active {{
      background: #1f2937;
      color: var(--primary);
      border-color: var(--primary);
    }}

    .concordance-list {{
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .unit-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 20px 24px;
      transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    }}
    .unit-card:hover {{
      border-color: rgba(56, 189, 248, 0.4);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
      transform: translateY(-2px);
    }}
    .card-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      padding-bottom: 12px;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .rank-info {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .rank-badge {{
      background: #1f2937;
      color: var(--primary);
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      font-size: 14px;
      padding: 4px 10px;
      border-radius: 8px;
      border: 1px solid #374151;
    }}
    .depth-badge {{
      font-size: 13px;
      color: var(--text-muted);
      font-weight: 500;
    }}
    .jaccard-meter {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .score-badge {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 15px;
      font-weight: 800;
      padding: 4px 12px;
      border-radius: 8px;
    }}
    .score-100 {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }}
    .score-high {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #0284c7; }}
    .score-med {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #d97706; }}
    .score-low {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #dc2626; }}

    .triplet-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    .pipeline-cell {{
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 10px;
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 10px;
    }}
    .pipe-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .pipe-name {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .name-lupin {{ color: #fb7185; }}
    .name-ks4-slide {{ color: #38bdf8; }}
    .name-ks4-glob {{ color: #c084fc; }}

    .sua-tag {{
      font-size: 10px;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 4px;
      text-transform: uppercase;
    }}
    .tag-sua {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }}
    .tag-mua {{ background: rgba(156, 163, 175, 0.2); color: #9ca3af; border: 1px solid #6b7280; }}

    .unit-details {{
      font-size: 13px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .detail-row {{
      display: flex;
      justify-content: space-between;
      color: var(--text-muted);
    }}
    .detail-val {{
      font-family: 'JetBrains Mono', monospace;
      font-weight: 600;
      color: var(--text-main);
    }}

    .inspect-btn {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      text-decoration: none;
      transition: all 0.2s ease;
      background: #1e293b;
      color: #e2e8f0;
      border: 1px solid #334155;
    }}
    .inspect-btn:hover {{
      background: var(--primary);
      color: #090d16;
      border-color: var(--primary);
    }}
    .inspect-btn.disabled {{
      opacity: 0.4;
      pointer-events: none;
    }}

    .pairwise-footer {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 14px;
      padding-top: 10px;
      border-top: 1px dashed rgba(255,255,255,0.06);
      font-size: 12px;
      color: var(--text-dim);
      flex-wrap: wrap;
      gap: 8px;
    }}
    .pair-badge {{
      font-family: 'JetBrains Mono', monospace;
      color: var(--text-muted);
    }}
    .pair-val {{
      font-weight: 700;
      color: #38bdf8;
    }}
  </style>
</head>
<body>

  <div class="header">
    <div class="title-row">
      <div>
        <h1>Tripartite Cross-Sorter Concordance Validation Browser</h1>
        <p class="subtitle">Direct side-by-side comparison of the same physical single units identified across <b>Lupin</b>, <b>Kilosort 4 (Sliding-Window Consensus)</b>, and <b>Kilosort 4 (Global Full-Probe)</b> for session <code>e097_g2</code> (pre-laser 60s epoch: 796.8s–856.8s). Click any card button to open the full 30 kHz native trace validation suite for that sorter.</p>
      </div>
    </div>
    <div class="pipeline-nav">
      <span style="font-size: 13px; color: var(--text-muted); align-self: center; font-weight: 600;">Standalone Browsers:</span>
      <a href="../lupin/sorting_validation/cluster_validation_browser.html" target="_blank" class="nav-btn nav-lupin">
        <span>🔍 Lupin Browser (40 Units)</span>
      </a>
      <a href="../kilosort4_sliding_window/sorting_validation/cluster_validation_browser.html" target="_blank" class="nav-btn nav-ks4-slide">
        <span>🔍 KS4 Sliding-Window Browser (40 Units)</span>
      </a>
      <a href="../kilosort4_global/sorting_validation/cluster_validation_browser.html" target="_blank" class="nav-btn nav-ks4-glob">
        <span>🔍 KS4 Global Browser (40 Units)</span>
      </a>
    </div>
  </div>

  <div class="stats-bar">
    <div class="stat-card">
      <div class="stat-val" style="color: #38bdf8;">{n_total}</div>
      <div class="stat-label">Total Tripartite Units</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color: #10b981;">{n_sua3}</div>
      <div class="stat-label">Allen SUA in All 3 Sorters</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color: #34d399;">{n_ident}</div>
      <div class="stat-label">Near-Identical Spike Trains (≥99%)</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color: #c084fc;">{n_high}</div>
      <div class="stat-label">High Agreement (≥80%)</div>
    </div>
  </div>

  <div class="controls-row">
    <div class="search-box">
      <input type="text" id="searchInput" class="search-input" placeholder="Search by Unit ID, depth, or firing rate...">
    </div>
    <div class="filter-pills">
      <button class="pill active" onclick="filterTier('all', this)">All Clusters ({n_total})</button>
      <button class="pill" onclick="filterTier('sua', this)">Allen SUA in All 3 ({n_sua3})</button>
      <button class="pill" onclick="filterTier('tier-identical', this)">Identical ≥99% ({n_ident})</button>
      <button class="pill" onclick="filterTier('tier-near-identical', this)">Near-Identical 90–99%</button>
      <button class="pill" onclick="filterTier('tier-high', this)">High 70–90%</button>
    </div>
  </div>

  <div class="concordance-list" id="concordanceList">
"""

    for r in rows_data:
        score_class = "score-100" if r["mean_jacc"] >= 0.99 else ("score-high" if r["mean_jacc"] >= 0.80 else ("score-med" if r["mean_jacc"] >= 0.50 else "score-low"))
        sua_badge_lup = '<span class="sua-tag tag-sua">Allen SUA</span>' if r["sua_lup"] else '<span class="sua-tag tag-mua">MUA</span>'
        sua_badge_sld = '<span class="sua-tag tag-sua">Allen SUA</span>' if r["sua_sld"] else '<span class="sua-tag tag-mua">MUA</span>'
        sua_badge_glb = '<span class="sua-tag tag-sua">Allen SUA</span>' if r["sua_glb"] else '<span class="sua-tag tag-mua">MUA</span>'
        
        btn_lup = f'<a href="{r["html_lup"]}" target="_blank" class="inspect-btn">🔬 Inspect Lupin Trace</a>' if r["html_lup"] else '<span class="inspect-btn disabled">No Page</span>'
        btn_sld = f'<a href="{r["html_sld"]}" target="_blank" class="inspect-btn">🔬 Inspect KS4 Slide Trace</a>' if r["html_sld"] else '<span class="inspect-btn disabled">No Page</span>'
        btn_glb = f'<a href="{r["html_glb"]}" target="_blank" class="inspect-btn">🔬 Inspect KS4 Global Trace</a>' if r["html_glb"] else '<span class="inspect-btn disabled">No Page</span>'

        html_content += f"""
    <div class="unit-card {r["tier_class"]}" data-all-sua="{str(r["all_three_sua"]).lower()}" data-tier="{r["tier_class"]}" data-search="unit {r["u_lup"]} {r["u_sld"]} {r["u_glb"]} depth {r["depth_um"]}">
      <div class="card-top">
        <div class="rank-info">
          <span class="rank-badge">#{r["rank"]:02d}</span>
          <span class="depth-badge">📍 Probe Depth: <b>{r["depth_um"]:.0f} µm</b></span>
          <span class="depth-badge" style="color: #818cf8; font-weight: 600;">{r["tier_label"]}</span>
        </div>
        <div class="jaccard-meter">
          <span style="font-size: 13px; color: var(--text-muted); font-weight: 600;">Mean 3-Way Concordance:</span>
          <span class="score-badge {score_class}">{r["mean_jacc"] * 100:.1f}% Jaccard</span>
        </div>
      </div>

      <div class="triplet-grid">
        <!-- Lupin -->
        <div class="pipeline-cell">
          <div class="pipe-header">
            <span class="pipe-name name-lupin">Lupin (Sliding)</span>
            {sua_badge_lup}
          </div>
          <div class="unit-details">
            <div class="detail-row"><span>Unit ID:</span> <span class="detail-val">Unit {r["u_lup"]:04d}</span></div>
            <div class="detail-row"><span>Firing Rate:</span> <span class="detail-val">{r["fr_lup"]:.2f} Hz</span></div>
          </div>
          {btn_lup}
        </div>

        <!-- KS4 Sliding -->
        <div class="pipeline-cell">
          <div class="pipe-header">
            <span class="pipe-name name-ks4-slide">Kilosort 4 (Sliding)</span>
            {sua_badge_sld}
          </div>
          <div class="unit-details">
            <div class="detail-row"><span>Unit ID:</span> <span class="detail-val">Unit {r["u_sld"]:04d}</span></div>
            <div class="detail-row"><span>Firing Rate:</span> <span class="detail-val">{r["fr_sld"]:.2f} Hz</span></div>
          </div>
          {btn_sld}
        </div>

        <!-- KS4 Global -->
        <div class="pipeline-cell">
          <div class="pipe-header">
            <span class="pipe-name name-ks4-glob">Kilosort 4 (Global)</span>
            {sua_badge_glb}
          </div>
          <div class="unit-details">
            <div class="detail-row"><span>Unit ID:</span> <span class="detail-val">Unit {r["u_glb"]:04d}</span></div>
            <div class="detail-row"><span>Firing Rate:</span> <span class="detail-val">{r["fr_glb"]:.2f} Hz</span></div>
          </div>
          {btn_glb}
        </div>
      </div>

      <div class="pairwise-footer">
        <span class="pair-badge">Lupin ↔ KS4 Slide: <span class="pair-val">{r["jacc_lup_sld"]*100:.1f}%</span></span>
        <span class="pair-badge">KS4 Slide ↔ KS4 Global: <span class="pair-val">{r["jacc_sld_glb"]*100:.1f}%</span></span>
        <span class="pair-badge">Lupin ↔ KS4 Global: <span class="pair-val">{r["jacc_lup_glb"]*100:.1f}%</span></span>
      </div>
    </div>
"""

    html_content += """
  </div>

  <script>
    function filterTier(tier, btn) {
      document.querySelectorAll('.filter-pills .pill').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const cards = document.querySelectorAll('.unit-card');
      cards.forEach(card => {
        if (tier === 'all') {
          card.style.display = 'block';
        } else if (tier === 'sua') {
          card.style.display = card.getAttribute('data-all-sua') === 'true' ? 'block' : 'none';
        } else {
          card.style.display = card.getAttribute('data-tier') === tier ? 'block' : 'none';
        }
      });
    }

    document.getElementById('searchInput').addEventListener('input', function(e) {
      const q = e.target.value.toLowerCase().trim();
      document.querySelectorAll('.unit-card').forEach(card => {
        const text = card.getAttribute('data-search') + ' ' + card.innerText.toLowerCase();
        card.style.display = text.includes(q) ? 'block' : 'none';
      });
    });
  </script>
</body>
</html>
"""

    out_file = comp_dir / "tripartite_concordance_browser.html"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f" Tripartite Cross-Sorter Concordance Browser created successfully: file:///{out_file.resolve().as_posix()}")


if __name__ == "__main__":
    generate_tripartite_browser()
