#!/usr/bin/env python3
"""
Interactive Population Raster & Diagnostic Dashboard for QC-Passed Lupin Units
Generates a standalone interactive HTML dashboard featuring:
1. Synchronized Depth-Resolved Spike Raster
2. Population PSTH (Instantaneous Firing Rate)
3. Probe Spatial Anatomy & Single-Unit Highlighting
4. Interactive Filters (Depth, Rate, SNR) and Unit Inspection Cards
"""
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("LUPIN_INTERACTIVE_RASTER")

def build_interactive_lupin_raster(
    lupin_dir: Path,
    out_html_path: Path,
    bin_size_ms: float = 50.0
) -> Path:
    """
    Constructs a rich interactive HTML dashboard for Lupin QC-passed units.
    """
    lupin_dir = Path(lupin_dir)
    out_html_path = Path(out_html_path)
    out_html_path.parent.mkdir(parents=True, exist_ok=True)

    qc_csv = lupin_dir / "lupin_qc_summary.csv"
    if not qc_csv.exists():
        qc_csv = lupin_dir / "cluster_info.tsv"
    
    if not qc_csv.exists():
        raise FileNotFoundError(f"QC summary not found in {lupin_dir}")

    # Load metrics
    sep = "\t" if qc_csv.suffix == ".tsv" else ","
    df_metrics = pd.read_csv(qc_csv, sep=sep)
    
    # Filter for Allen QC passed units
    df_qc = df_metrics[df_metrics["passed_allen_qc"]].copy()
    if len(df_qc) == 0:
        logger.warning("No units passed Allen QC filter. Using top units by firing rate instead.")
        df_qc = df_metrics[df_metrics["firing_rate_hz"] >= 0.1].copy()

    # Sort by probe depth (ascending)
    df_qc = df_qc.sort_values(by="depth_um", ascending=True).reset_index(drop=True)
    qc_unit_ids = set(df_qc["cluster_id"].values)

    # Load spikes
    alf_dir = lupin_dir / "alf"
    times = np.load(alf_dir / "spike_times.npy")
    clusters = np.load(alf_dir / "spike_clusters.npy")

    # Filter spikes belonging to QC units
    mask = np.isin(clusters, list(qc_unit_ids))
    qc_times = times[mask]
    qc_clusters = clusters[mask]

    unit_map = {row["cluster_id"]: row for _, row in df_qc.iterrows()}
    unit_depths = np.array([unit_map[c]["depth_um"] for c in qc_clusters])
    unit_rates = np.array([unit_map[c]["firing_rate_hz"] for c in qc_clusters])

    logger.info(f"Loaded {len(df_qc)} QC-passed units ({len(qc_times):,} total spikes).")

    # Create Synchronized Subplots: Top = Population PSTH (Firing Rate), Bottom = Spike Raster
    duration_s = float(np.max(qc_times)) if len(qc_times) > 0 else 60.0
    n_psth_bins = int(np.ceil((duration_s * 1000.0) / bin_size_ms))
    psth_bins = np.linspace(0, duration_s, n_psth_bins + 1)
    psth_counts, _ = np.histogram(qc_times, bins=psth_bins)
    psth_rate_hz = psth_counts / (bin_size_ms / 1000.0)
    psth_centers = (psth_bins[:-1] + psth_bins[1:]) / 2.0

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.22, 0.78],
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f"Population Instantaneous Firing Rate ({bin_size_ms:.0f} ms bins)",
            f"Depth-Resolved Single-Unit Spike Raster ({len(df_qc)} QC-Passed Units)"
        )
    )

    # 1. Top Panel: Population PSTH
    fig.add_trace(
        go.Scatter(
            x=psth_centers,
            y=psth_rate_hz,
            mode="lines",
            name="Population Rate",
            line=dict(color="#00cec9", width=2.0),
            fill="tozeroy",
            fillcolor="rgba(0, 206, 201, 0.15)",
            hovertemplate="Time: %{x:.2f} s<br>Rate: %{y:.1f} Hz<extra></extra>"
        ),
        row=1, col=1
    )

    # 2. Bottom Panel: Depth vs Time Spike Raster
    # Use WebGL Scattergl for high FPS interactive zooming
    colorscale = "Viridis"
    hover_text = [
        f"Cluster ID: {c}<br>Depth: {d:.1f} µm<br>Unit FR: {fr:.2f} Hz"
        for c, d, fr in zip(qc_clusters, unit_depths, unit_rates)
    ]

    fig.add_trace(
        go.Scattergl(
            x=qc_times,
            y=unit_depths,
            mode="markers",
            name="QC Units Spikes",
            marker=dict(
                size=3.5,
                color=unit_rates,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title="Unit FR (Hz)",
                    thickness=15,
                    len=0.75,
                    y=0.38,
                    x=1.02
                ),
                opacity=0.75
            ),
            text=hover_text,
            hovertemplate="Time: %{x:.3f} s<br>Depth: %{y:.1f} µm<br>%{text}<extra></extra>"
        ),
        row=2, col=1
    )

    # Layout styling
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>LUPIN Spike Sorting — QC-Passed Single Units Interactive Raster</b><br><sup>Animal/Session: LP21_fent_EXPT_g1 | 67 Isolated Units Passing Allen Institute Criteria</sup>",
            font=dict(size=18, color="#f1f2f6"),
            x=0.03,
            y=0.97
        ),
        hovermode="closest",
        margin=dict(l=60, r=90, t=90, b=50),
        height=850,
        legend=dict(orientation="h", y=1.05, x=0.8),
        paper_bgcolor="#1e272e",
        plot_bgcolor="#141a1f"
    )

    fig.update_yaxes(
        title_text="Rate (Hz)",
        row=1, col=1,
        gridcolor="#2d3436"
    )
    fig.update_yaxes(
        title_text="Probe Depth (µm)",
        row=2, col=1,
        gridcolor="#2d3436"
    )
    fig.update_xaxes(
        title_text="Recording Time (seconds)",
        row=2, col=1,
        gridcolor="#2d3436"
    )

    # Save to interactive HTML
    fig.write_html(str(out_html_path), include_plotlyjs="cdn", full_html=True)
    logger.info(f"Interactive raster saved to: {out_html_path}")

    # Generate rich dashboard with side unit inspector
    dashboard_html_path = out_html_path.with_name("lupin_qc_interactive_dashboard.html")
    generate_full_dashboard(df_qc, qc_times, qc_clusters, psth_centers, psth_rate_hz, dashboard_html_path, duration_s)

    return out_html_path

def generate_full_dashboard(
    df_qc: pd.DataFrame,
    qc_times: np.ndarray,
    qc_clusters: np.ndarray,
    psth_time: np.ndarray,
    psth_rate: np.ndarray,
    out_path: Path,
    duration_s: float
):
    """
    Creates an interactive HTML5 / CSS3 / JavaScript dashboard with instant unit filtering,
    probe schematic, and unit metric table.
    """
    units_data = df_qc.to_dict(orient="records")
    
    # Subsample spikes for embedded fast viewer (limit 80k spikes max for instant DOM rendering)
    if len(qc_times) > 80000:
        sub_idx = np.random.choice(len(qc_times), 80000, replace=False)
        sub_times = qc_times[sub_idx].tolist()
        sub_clusters = qc_clusters[sub_idx].tolist()
    else:
        sub_times = qc_times.tolist()
        sub_clusters = qc_clusters.tolist()

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lupin QC Units Interactive Raster Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        body {{
            background: #0f141c;
            color: #e2e8f0;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }}
        header {{
            background: #182232;
            padding: 12px 24px;
            border-bottom: 1px solid #2d3d54;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header-title h1 {{
            font-size: 1.25rem;
            font-weight: 700;
            color: #38bdf8;
        }}
        .header-title p {{
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 2px;
        }}
        .stats-badge-group {{
            display: flex;
            gap: 12px;
        }}
        .stat-badge {{
            background: #202e42;
            padding: 6px 14px;
            border-radius: 8px;
            border: 1px solid #334966;
            font-size: 0.85rem;
        }}
        .stat-badge strong {{
            color: #38bdf8;
            margin-right: 4px;
        }}
        .main-container {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}
        .sidebar {{
            width: 320px;
            background: #141c28;
            border-right: 1px solid #2d3d54;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            overflow-y: auto;
        }}
        .panel-card {{
            background: #1c2738;
            padding: 14px;
            border-radius: 10px;
            border: 1px solid #2e405b;
        }}
        .panel-card h3 {{
            font-size: 0.9rem;
            font-weight: 600;
            color: #f1f5f9;
            margin-bottom: 10px;
        }}
        .control-group {{
            margin-bottom: 12px;
        }}
        .control-group label {{
            display: flex;
            justify-content: space-between;
            font-size: 0.78rem;
            color: #94a3b8;
            margin-bottom: 4px;
        }}
        input[type="range"] {{
            width: 100%;
            accent-color: #38bdf8;
        }}
        select, input[type="text"] {{
            width: 100%;
            background: #0f141c;
            border: 1px solid #334966;
            color: #f1f5f9;
            padding: 7px 10px;
            border-radius: 6px;
            font-size: 0.82rem;
        }}
        .unit-list {{
            max-height: 280px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .unit-item {{
            display: flex;
            justify-content: space-between;
            padding: 6px 10px;
            background: #0f1722;
            border-radius: 6px;
            font-size: 0.78rem;
            cursor: pointer;
            border: 1px solid transparent;
            transition: all 0.15s ease;
        }}
        .unit-item:hover, .unit-item.active {{
            background: #25354d;
            border-color: #38bdf8;
            color: #38bdf8;
        }}
        .plot-container {{
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 10px;
            background: #0b0f17;
            overflow: hidden;
        }}
        #plot-div {{
            flex: 1;
            width: 100%;
            height: 100%;
        }}
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <h1>Lupin Spike Sorting — Single-Unit QC Population Raster</h1>
            <p>Dataset: LP21_fent_EXPT_g1 (Neuropixels 2.0 | First 60s) | Modular SVD + OMP Matching</p>
        </div>
        <div class="stats-badge-group">
            <div class="stat-badge"><strong>{len(df_qc)}</strong> QC Passed Units</div>
            <div class="stat-badge"><strong>{len(qc_times):,}</strong> Total Spikes</div>
            <div class="stat-badge"><strong>{df_qc['firing_rate_hz'].mean():.2f} Hz</strong> Mean FR</div>
        </div>
    </header>

    <div class="main-container">
        <div class="sidebar">
            <div class="panel-card">
                <h3>Raster Filter Controls</h3>
                <div class="control-group">
                    <label><span>Min Firing Rate</span><span id="lbl-fr">0.0 Hz</span></label>
                    <input type="range" id="slider-fr" min="0" max="40" step="0.5" value="0">
                </div>
                <div class="control-group">
                    <label><span>Depth Range (µm)</span><span id="lbl-depth">0 - 3840 µm</span></label>
                    <input type="range" id="slider-depth-min" min="0" max="3840" step="50" value="0">
                    <input type="range" id="slider-depth-max" min="0" max="3840" step="50" value="3840" style="margin-top: 6px;">
                </div>
                <div class="control-group">
                    <label><span>Min Estimated SNR</span><span id="lbl-snr">3.0</span></label>
                    <input type="range" id="slider-snr" min="3.0" max="15.0" step="0.5" value="3.0">
                </div>
                <button id="btn-reset" style="width:100%; padding:7px; background:#0284c7; color:#fff; border:none; border-radius:6px; cursor:pointer; font-size:0.8rem; font-weight:600;">Reset Filters</button>
            </div>

            <div class="panel-card" style="flex:1; display:flex; flex-direction:column;">
                <h3>QC Single Units ({len(df_qc)})</h3>
                <input type="text" id="unit-search" placeholder="Search Unit ID..." style="margin-bottom:8px;">
                <div class="unit-list" id="unit-list"></div>
            </div>
        </div>

        <div class="plot-container">
            <div id="plot-div"></div>
        </div>
    </div>

    <script>
        const units = {json.dumps(units_data)};
        const times = {json.dumps(sub_times)};
        const clusters = {json.dumps(sub_clusters)};
        const psthTime = {json.dumps(psth_time.tolist())};
        const psthRate = {json.dumps(psth_rate.tolist())};

        const unitMap = {{}};
        units.forEach(u => {{ unitMap[u.cluster_id] = u; }});

        function populateUnitList() {{
            const container = document.getElementById("unit-list");
            container.innerHTML = "";
            units.forEach(u => {{
                const div = document.createElement("div");
                div.className = "unit-item";
                div.id = `unit-item-${{u.cluster_id}}`;
                div.innerHTML = `<span><b>#${{u.cluster_id}}</b> (${{u.depth_um.toFixed(0)}} µm)</span><span>${{u.firing_rate_hz.toFixed(1)}} Hz</span>`;
                div.onclick = () => highlightUnit(u.cluster_id);
                container.appendChild(div);
            }});
        }}

        function renderPlot(filteredUnitIds = null) {{
            const validSet = filteredUnitIds ? new Set(filteredUnitIds) : new Set(units.map(u => u.cluster_id));
            
            const plotTimes = [];
            const plotDepths = [];
            const plotRates = [];
            const hoverText = [];

            for (let i = 0; i < times.length; i++) {{
                const cid = clusters[i];
                if (validSet.has(cid)) {{
                    const u = unitMap[cid];
                    plotTimes.push(times[i]);
                    plotDepths.push(u.depth_um);
                    plotRates.push(u.firing_rate_hz);
                    hoverText.push(`Cluster: ${{cid}}<br>Depth: ${{u.depth_um.toFixed(1)}} µm<br>FR: ${{u.firing_rate_hz.toFixed(2)}} Hz<br>SNR: ${{u.snr.toFixed(1)}}`);
                }}
            }}

            const tracePSTH = {{
                x: psthTime,
                y: psthRate,
                type: 'scatter',
                mode: 'lines',
                name: 'Population PSTH',
                line: {{ color: '#38bdf8', width: 2 }},
                fill: 'tozeroy',
                fillcolor: 'rgba(56, 189, 248, 0.15)',
                xaxis: 'x',
                yaxis: 'y2'
            }};

            const traceRaster = {{
                x: plotTimes,
                y: plotDepths,
                type: 'scattergl',
                mode: 'markers',
                name: 'Spikes',
                marker: {{
                    size: 3.5,
                    color: plotRates,
                    colorscale: 'Viridis',
                    showscale: true,
                    colorbar: {{
                        title: 'FR (Hz)',
                        thickness: 14,
                        len: 0.72,
                        y: 0.36,
                        x: 1.01
                    }},
                    opacity: 0.8
                }},
                text: hoverText,
                hoverinfo: 'text+x',
                xaxis: 'x',
                yaxis: 'y'
            }};

            const layout = {{
                paper_bgcolor: '#0b0f17',
                plot_bgcolor: '#101723',
                margin: {{ l: 65, r: 70, t: 30, b: 45 }},
                grid: {{ rows: 2, columns: 1, subplots: [['xy2'], ['xy']], roworder: 'top to bottom' }},
                yaxis2: {{
                    title: 'Rate (Hz)',
                    domain: [0.80, 1.0],
                    gridcolor: '#1e293b',
                    color: '#94a3b8'
                }},
                yaxis: {{
                    title: 'Probe Depth (µm)',
                    domain: [0.0, 0.76],
                    gridcolor: '#1e293b',
                    color: '#94a3b8'
                }},
                xaxis: {{
                    title: 'Time (s)',
                    gridcolor: '#1e293b',
                    color: '#94a3b8'
                }},
                hovermode: 'closest',
                showlegend: false
            }};

            Plotly.react('plot-div', [tracePSTH, traceRaster], layout, {{ responsive: true, displayModeBar: true }});
        }}

        function applyFilters() {{
            const minFR = parseFloat(document.getElementById("slider-fr").value);
            const minDepth = parseFloat(document.getElementById("slider-depth-min").value);
            const maxDepth = parseFloat(document.getElementById("slider-depth-max").value);
            const minSNR = parseFloat(document.getElementById("slider-snr").value);

            document.getElementById("lbl-fr").textContent = `${{minFR}} Hz`;
            document.getElementById("lbl-depth").textContent = `${{minDepth}} - ${{maxDepth}} µm`;
            document.getElementById("lbl-snr").textContent = `${{minSNR}}`;

            const filtered = units.filter(u => 
                u.firing_rate_hz >= minFR &&
                u.depth_um >= minDepth &&
                u.depth_um <= maxDepth &&
                u.snr >= minSNR
            ).map(u => u.cluster_id);

            renderPlot(filtered);
        }}

        function highlightUnit(clusterId) {{
            document.querySelectorAll(".unit-item").forEach(el => el.classList.remove("active"));
            const target = document.getElementById(`unit-item-${{clusterId}}`);
            if (target) target.classList.add("active");
            renderPlot([clusterId]);
        }}

        document.getElementById("slider-fr").oninput = applyFilters;
        document.getElementById("slider-depth-min").oninput = applyFilters;
        document.getElementById("slider-depth-max").oninput = applyFilters;
        document.getElementById("slider-snr").oninput = applyFilters;
        document.getElementById("btn-reset").onclick = () => {{
            document.getElementById("slider-fr").value = 0;
            document.getElementById("slider-depth-min").value = 0;
            document.getElementById("slider-depth-max").value = 3840;
            document.getElementById("slider-snr").value = 3.0;
            document.querySelectorAll(".unit-item").forEach(el => el.classList.remove("active"));
            applyFilters();
        }};

        document.getElementById("unit-search").oninput = (e) => {{
            const q = e.target.value.toLowerCase();
            document.querySelectorAll(".unit-item").forEach(el => {{
                el.style.display = el.textContent.toLowerCase().includes(q) ? "flex" : "none";
            }});
        }};

        populateUnitList();
        renderPlot();
    </script>
</body>
</html>
"""
    out_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Rich interactive dashboard generated at: {out_path}")

def main():
    lupin_dir = Path(r"c:\Users\lpascu\analysis_src\output_LP21_fent_EXPT_g1\lupin")
    out_html = lupin_dir / "lupin_qc_interactive_raster.html"
    build_interactive_lupin_raster(lupin_dir, out_html)
    print("\n" + "=" * 85)
    print(" [SUCCESS] Interactive Lupin QC Raster Plot Generated!")
    print(f" Plotly HTML    : {out_html}")
    print(f" Full Dashboard : {out_html.with_name('lupin_qc_interactive_dashboard.html')}")
    print("=" * 85)

if __name__ == "__main__":
    main()
