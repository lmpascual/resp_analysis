#!/usr/bin/env python3
"""
Batch Dual-Sorter Pipeline across all physiological subwindows of e097 (29:00 - 36:00 / 1740s - 2160s):
- Subwindows:
    1. win1_baseline: 1740s - 1800s (60s)
    2. win2_anox1: 1800s - 1836s (36s)
    3. win3_anox2: 1836s - 1860s (24s)
    4. win4_gasp_early: 1860s - 1950s (90s)
    5. win5_gasp_late: 1950s - 2070s (120s)
    6. win6_flow_shift: 2070s - 2136s (66s)
    7. win7_recovery: 2136s - 2160s (24s)
- Dual Sorters:
    * Lupin (SVD-OMP)
    * Kilosort 4 (RTX 4000 Ada GPU)
- Permissive single-unit QC thresholds:
    * presence_ratio >= 0.05
    * isi_cutoff <= 0.15
    * min_fr >= 0.05 Hz
- Cross-sorter mutual agreement (0.4 ms coincidence window)
- Generates side-by-side comparative dashboard and master manifest for all windows.
"""

import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import spikeinterface.full as si
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import os
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# Safe stdout UTF-8 on Windows
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from src.pipeline.config import CatGTConfig, SorterConfig, LupinConfig, Kilosort4Config
from src.pipeline.processing.neural_ap.io_catgt import run_catgt
from src.pipeline.io.io_spikeglx import read_spikeglx_ap_safe
from src.pipeline.sorting.sorter_runner import run_single_sorter
from scripts.visualization.plot_interactive_sorter_raster import build_interactive_sorter_raster

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BATCH_SUBWINDOWS_SORTING")

SUBWINDOWS = [
    {
        "id": "win1_baseline",
        "name": "Baseline (Pre-Anoxia)",
        "t_start": 1740.0,
        "t_end": 1800.0,
        "duration": 60.0
    },
    {
        "id": "win2_anox1",
        "name": "Anoxic Challenge 1 (Gas Switch)",
        "t_start": 1800.0,
        "t_end": 1836.0,
        "duration": 36.0
    },
    {
        "id": "win3_anox2",
        "name": "Anoxic Challenge 2 (Apnea / HR Nadir)",
        "t_start": 1836.0,
        "t_end": 1860.0,
        "duration": 24.0
    },
    {
        "id": "win4_gasp_early",
        "name": "Early Gasping (Autoresuscitation Surge)",
        "t_start": 1860.0,
        "t_end": 1950.0,
        "duration": 90.0
    },
    {
        "id": "win5_gasp_late",
        "name": "Late Gasping (Tachypneic Transition)",
        "t_start": 1950.0,
        "t_end": 2070.0,
        "duration": 120.0
    },
    {
        "id": "win6_flow_shift",
        "name": "Active Expiration Shift",
        "t_start": 2070.0,
        "t_end": 2136.0,
        "duration": 66.0
    },
    {
        "id": "win7_recovery",
        "name": "Post-Challenge Recovery",
        "t_start": 2136.0,
        "t_end": 2160.0,
        "duration": 24.0
    }
]

def load_alf_to_sorting(alf_dir: Path, fs: float) -> si.BaseSorting:
    spike_times_s = np.load(alf_dir / "spike_times.npy")
    spike_clusters = np.load(alf_dir / "spike_clusters.npy")
    spike_times_samples = (spike_times_s * fs).astype(np.int64)
    return si.NumpySorting.from_times_and_labels(
        times_list=spike_times_samples,
        labels_list=spike_clusters,
        sampling_frequency=fs
    )

def find_raster_html(directory: Path) -> str:
    htmls = list(directory.glob("*interactive_raster.html"))
    if htmls:
        return htmls[0].name
    return ""

def process_subwindow(
    win: Dict[str, Any],
    raw_ap_dir: Path,
    out_base: Path,
    scratch_catgt_base: Path,
    scratch_sort_base: Path,
    catgt_exe: str
) -> Dict[str, Any]:
    win_id = win["id"]
    win_name = win["name"]
    t_start = win["t_start"]
    dur = win["duration"]

    win_out_dir = out_base / win_id
    win_out_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = win_out_dir / f"{win_id}_sorting_benchmark.json"

    logger.info("=" * 80)
    logger.info(f" PROCESSING SUBWINDOW: {win_id.upper()} ({win_name})")
    logger.info(f" Time Range: {t_start:.1f}s - {t_start + dur:.1f}s ({dur:.1f}s duration)")
    logger.info("=" * 80)

    # 1. CatGT Preprocessing
    if win_id == "win1_baseline":
        catgt_processed_folder = Path(r"C:\SCRATCH_neuropixels\catgt_e097_win1_baseline\catgt_e097_g2")
    else:
        catgt_scratch_top = scratch_catgt_base / win_id
        catgt_processed_folder = catgt_scratch_top / "catgt_e097_g2"
        catgt_bin = catgt_processed_folder / "e097_g2_tcat.imec0.ap.bin"
        catgt_meta = catgt_processed_folder / "e097_g2_tcat.imec0.ap.meta"

        if catgt_bin.exists() and catgt_meta.exists():
            logger.info(f"Found existing CatGT preprocessed binary at: {catgt_bin}")
        else:
            catgt_scratch_top.mkdir(parents=True, exist_ok=True)
            logger.info(f"Executing CatGT extraction for {win_id} (t={t_start:.1f}s, dur={dur:.1f}s)...")
            catgt_cfg = CatGTConfig(
                enabled=True,
                catgt_path=catgt_exe,
                run_name="e097",
                gate=2,
                trigger=0,
                probe=0,
                t_shift=True,
                apfilter="butter,12,300,9000",
                loccar_um=(75.0, 300.0),
                start_secs=t_start,
                max_secs=dur,
                dest_dir=str(catgt_scratch_top)
            )
            catgt_bin, catgt_meta = run_catgt(
                config=catgt_cfg,
                raw_dir=raw_ap_dir,
                dest_dir=catgt_scratch_top,
                start_secs=t_start,
                max_secs=dur
            )
            logger.info(f"CatGT completed: {catgt_bin}")

    # Ingest CatGT recording
    recording = read_spikeglx_ap_safe(catgt_processed_folder)
    fs = recording.get_sampling_frequency()
    logger.info(f"Recording loaded: {recording.get_num_channels()} chans, {fs:.1f} Hz, {recording.get_total_duration():.2f}s.")

    # Configurations
    sorter_cfg = SorterConfig(
        enabled_sorters=["lupin", "kilosort4"],
        do_CAR=False,
        save_phy=True,
        export_alf=True,
        isi_threshold_s=0.0015,
        qc_isi_cutoff=0.15,
        qc_presence_ratio=0.05,
        qc_amp_cutoff=0.25,
        qc_min_fr_hz=0.05,
        n_jobs=4
    )

    lupin_cfg = LupinConfig(
        enabled=True,
        sparsity_radius_um=100.0,
        clustering_method="circus",
        matching_method="circus-omp-svd",
        ms_before=1.5,
        ms_after=2.0,
        templates_from_svd=True,
        matching_amplitudes=[0.5, 2.2]
    )

    ks4_cfg = Kilosort4Config(
        enabled=True,
        Th_universal=8.5,
        Th_learned=7.5,
        do_CAR=False,
        nblocks=1,
        batch_size=60000,
        nt=61
    )

    scratch_sort_win = scratch_sort_base / win_id
    scratch_sort_win.mkdir(parents=True, exist_ok=True)

    results = {}

    # 2. RUN / LOAD LUPIN (SVD-OMP)
    lupin_out = win_out_dir / "lupin"
    if (lupin_out / "alf" / "spike_times.npy").exists():
        logger.info(f"Loading pre-computed Lupin sorting from {lupin_out / 'alf'}...")
        sorting_lupin = load_alf_to_sorting(lupin_out / "alf", fs)
        df_qc_lupin = pd.read_csv(lupin_out / "master_consensus_qc_summary.csv")
        dur_lupin = 280.0 if win_id == "win1_baseline" else 0.0
        if summary_json_path.exists():
            with open(summary_json_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
                dur_lupin = prev.get("lupin", {}).get("runtime_s", 0.0)
    else:
        logger.info(f"[{win_id}] RUNNING LUPIN (SVD-OMP) ON ALL 384 CHANNELS...")
        t0_lupin = time.time()
        sorting_lupin, df_qc_lupin, dur_lupin = run_single_sorter(
            sorter_name="lupin",
            recording=recording,
            output_dir=win_out_dir,
            scratch_dir=scratch_sort_win,
            config=sorter_cfg,
            lupin_config=lupin_cfg,
            custom_sorter_dir=lupin_out
        )
        df_qc_lupin.to_csv(lupin_out / "master_consensus_qc_summary.csv", index=False)

    n_sua_lupin = int((df_qc_lupin["passed_allen_qc"] == True).sum()) if "passed_allen_qc" in df_qc_lupin.columns else 0
    results["lupin"] = {
        "runtime_s": round(dur_lupin, 1),
        "total_units": len(df_qc_lupin),
        "sua_units": n_sua_lupin,
        "dir": str(lupin_out)
    }
    logger.info(f"[{win_id}] Lupin ready: {len(df_qc_lupin)} total units ({n_sua_lupin} clean SUA).")

    # Interactive raster for Lupin
    lupin_raster_file = find_raster_html(lupin_out)
    if not lupin_raster_file:
        try:
            build_interactive_sorter_raster(
                sorter_dir=lupin_out,
                sorter_name=f"Lupin (SVD-OMP) - {win_name}"
            )
            lupin_raster_file = find_raster_html(lupin_out)
        except Exception as e:
            logger.warning(f"Could not build Lupin interactive raster: {e}")

    # 3. RUN / LOAD KILOSORT 4 (GPU)
    ks4_out = win_out_dir / "kilosort4"
    if (ks4_out / "alf" / "spike_times.npy").exists():
        logger.info(f"Loading pre-computed Kilosort 4 sorting from {ks4_out / 'alf'}...")
        sorting_ks4 = load_alf_to_sorting(ks4_out / "alf", fs)
        df_qc_ks4 = pd.read_csv(ks4_out / "master_consensus_qc_summary.csv")
        dur_ks4 = 127.6 if win_id == "win1_baseline" else 0.0
        if summary_json_path.exists():
            with open(summary_json_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
                dur_ks4 = prev.get("kilosort4", {}).get("runtime_s", 0.0)
    else:
        logger.info(f"[{win_id}] RUNNING KILOSORT 4 (GPU) ON ALL 384 CHANNELS...")
        t0_ks4 = time.time()
        sorting_ks4, df_qc_ks4, dur_ks4 = run_single_sorter(
            sorter_name="kilosort4",
            recording=recording,
            output_dir=win_out_dir,
            scratch_dir=scratch_sort_win,
            config=sorter_cfg,
            kilosort4_config=ks4_cfg,
            custom_sorter_dir=ks4_out
        )
        df_qc_ks4.to_csv(ks4_out / "master_consensus_qc_summary.csv", index=False)

    n_sua_ks4 = int((df_qc_ks4["passed_allen_qc"] == True).sum()) if "passed_allen_qc" in df_qc_ks4.columns else 0
    results["kilosort4"] = {
        "runtime_s": round(dur_ks4, 1),
        "total_units": len(df_qc_ks4),
        "sua_units": n_sua_ks4,
        "dir": str(ks4_out)
    }
    logger.info(f"[{win_id}] Kilosort 4 ready: {len(df_qc_ks4)} total units ({n_sua_ks4} clean SUA).")

    # Interactive raster for Kilosort 4
    ks4_raster_file = find_raster_html(ks4_out)
    if not ks4_raster_file:
        try:
            build_interactive_sorter_raster(
                sorter_dir=ks4_out,
                sorter_name=f"Kilosort 4 - {win_name}"
            )
            ks4_raster_file = find_raster_html(ks4_out)
        except Exception as e:
            logger.warning(f"Could not build Kilosort 4 interactive raster: {e}")

    # 4. CROSS-SORTER MUTUAL AGREEMENT
    logger.info(f"[{win_id}] Computing cross-sorter mutual agreement consensus...")
    try:
        cmp = sc.compare_two_sorters(
            sorting1=sorting_lupin,
            sorting2=sorting_ks4,
            sorting1_name="lupin",
            sorting2_name="kilosort4",
            delta_time=0.4
        )
        agreement_matrix = cmp.agreement_scores
        matched_units_50 = 0
        matched_units_80 = 0

        for u1 in agreement_matrix.index:
            max_agr = agreement_matrix.loc[u1].max()
            if max_agr >= 0.50:
                matched_units_50 += 1
            if max_agr >= 0.80:
                matched_units_80 += 1

        pct_50 = (matched_units_50 / max(1, len(df_qc_lupin))) * 100
    except Exception as e:
        logger.warning(f"Cross-sorter comparison warning: {e}")
        matched_units_50 = 0
        matched_units_80 = 0
        pct_50 = 0.0

    results["agreement"] = {
        "matched_units_50pct": matched_units_50,
        "matched_units_80pct": matched_units_80,
        "pct_lupin_matched": round(pct_50, 1)
    }
    results["subwindow"] = win

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    # 5. Build Subwindow Side-by-Side Dashboard
    build_subwindow_dashboard(
        win=win,
        lupin_res=results["lupin"],
        ks4_res=results["kilosort4"],
        agr_res=results["agreement"],
        lupin_raster=lupin_raster_file,
        ks4_raster=ks4_raster_file,
        out_html=win_out_dir / f"{win_id}_sorter_comparison.html"
    )

    return results

def build_subwindow_dashboard(
    win: Dict[str, Any],
    lupin_res: Dict[str, Any],
    ks4_res: Dict[str, Any],
    agr_res: Dict[str, Any],
    lupin_raster: str,
    ks4_raster: str,
    out_html: Path
):
    win_name = win["name"]
    win_id = win["id"]
    t_start = win["t_start"]
    dur = win["duration"]

    lupin_link = f"lupin/{lupin_raster}" if lupin_raster else "lupin/"
    ks4_link = f"kilosort4/{ks4_raster}" if ks4_raster else "kilosort4/"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{win_name} ({win_id}) Sorter Comparison</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {{
            background: #07090e;
            color: #f1f5f9;
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 30px;
        }}
        .header {{
            background: #0f1523;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 24px 30px;
            margin-bottom: 24px;
        }}
        h1 {{
            margin: 0 0 10px 0;
            font-size: 24px;
            color: #38bdf8;
            font-weight: 800;
        }}
        .subtitle {{
            color: #94a3b8;
            font-size: 14px;
            margin: 0;
        }}
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 24px;
        }}
        .card {{
            background: #0f1523;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 20px;
        }}
        .card h3 {{
            margin: 0 0 12px 0;
            font-size: 16px;
            color: #f1f5f9;
        }}
        .metric-val {{
            font-size: 28px;
            font-weight: 800;
            margin: 6px 0;
            font-family: 'JetBrains Mono', monospace;
        }}
        .val-lupin {{ color: #00f5d4; }}
        .val-ks4 {{ color: #f59e0b; }}
        .val-consensus {{ color: #a855f7; }}
        .metric-sub {{
            font-size: 12px;
            color: #94a3b8;
        }}
        .btn {{
            display: inline-block;
            background: #1e293b;
            color: #38bdf8;
            padding: 8px 14px;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 600;
            font-size: 12px;
            margin-top: 12px;
            border: 1px solid rgba(56, 189, 248, 0.3);
            transition: all 0.2s ease;
        }}
        .btn:hover {{
            background: #38bdf8;
            color: #07090e;
        }}
        .table-container {{
            background: #0f1523;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 24px;
            margin-top: 24px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th {{
            color: #94a3b8;
            border-bottom: 1px solid rgba(255,255,255,0.12);
            padding: 10px 14px;
            text-align: left;
        }}
        td {{
            padding: 12px 14px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{win_name} ({win_id}) Spike Sorting Benchmark</h1>
        <p class="subtitle">Time Range: {t_start:.1f}s - {t_start + dur:.1f}s ({dur:.1f}s window) | Lupin (SVD-OMP) vs Kilosort 4 (RTX 4000 Ada GPU)</p>
    </div>

    <div class="cards-grid">
        <div class="card">
            <h3>Lupin (SVD-OMP)</h3>
            <div class="metric-val val-lupin">{lupin_res['total_units']} Units</div>
            <div class="metric-sub">Clean SUA: <b>{lupin_res['sua_units']}</b> | Runtime: {lupin_res['runtime_s']:.1f}s</div>
            <a href="{lupin_link}" target="_blank" class="btn">Open Lupin Raster &rarr;</a>
        </div>
        <div class="card">
            <h3>Kilosort 4 (GPU)</h3>
            <div class="metric-val val-ks4">{ks4_res['total_units']} Units</div>
            <div class="metric-sub">Clean SUA: <b>{ks4_res['sua_units']}</b> | Runtime: {ks4_res['runtime_s']:.1f}s</div>
            <a href="{ks4_link}" target="_blank" class="btn">Open Kilosort 4 Raster &rarr;</a>
        </div>
        <div class="card">
            <h3>Mutual Agreement</h3>
            <div class="metric-val val-consensus">{agr_res['matched_units_50pct']} Units</div>
            <div class="metric-sub">&ge;50% Coincident Spikes ({agr_res['pct_lupin_matched']}% of Lupin) | &ge;80% Matches: {agr_res['matched_units_80pct']}</div>
        </div>
    </div>

    <div class="table-container">
        <h3 style="margin: 0 0 16px 0; font-size: 16px; color: #38bdf8;">Subwindow Summary</h3>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Lupin (SVD-OMP)</th>
                    <th>Kilosort 4 (GPU)</th>
                    <th>Mutual Agreement</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Total Clusters</td>
                    <td><b>{lupin_res['total_units']}</b></td>
                    <td><b>{ks4_res['total_units']}</b></td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Clean Single Units (SUA)</td>
                    <td><b>{lupin_res['sua_units']}</b></td>
                    <td><b>{ks4_res['sua_units']}</b></td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Runtime</td>
                    <td>{lupin_res['runtime_s']:.1f}s</td>
                    <td>{ks4_res['runtime_s']:.1f}s</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Mutual Consensus (&ge;50%)</td>
                    <td colspan="3" style="color: #a855f7; font-weight: 700;">{agr_res['matched_units_50pct']} Units ({agr_res['pct_lupin_matched']}%)</td>
                </tr>
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Saved subwindow comparison dashboard to {out_html}")

def build_master_all_subwindows_dashboard(all_results: List[Dict[str, Any]], out_html: Path):
    rows_html = ""
    for r in all_results:
        w = r["subwindow"]
        l = r["lupin"]
        k = r["kilosort4"]
        a = r["agreement"]
        win_id = w["id"]
        rows_html += f"""
        <tr>
            <td><b>{w['id']}</b><br><span style="color:#94a3b8; font-size:11px;">{w['name']}</span></td>
            <td>{w['t_start']:.1f}s - {w['t_end']:.1f}s<br><span style="color:#94a3b8; font-size:11px;">({w['duration']:.1f}s)</span></td>
            <td><b style="color:#00f5d4;">{l['sua_units']}</b> <span style="color:#64748b; font-size:11px;">/ {l['total_units']}</span></td>
            <td><b style="color:#f59e0b;">{k['sua_units']}</b> <span style="color:#64748b; font-size:11px;">/ {k['total_units']}</span></td>
            <td><b style="color:#a855f7;">{a['matched_units_50pct']}</b> <span style="color:#64748b; font-size:11px;">({a['pct_lupin_matched']}%)</span></td>
            <td>{l['runtime_s']:.1f}s / {k['runtime_s']:.1f}s</td>
            <td>
                <a href="{win_id}/{win_id}_sorter_comparison.html" class="tbl-btn" target="_blank">Comparison &rarr;</a>
                <a href="{win_id}/lupin/" class="tbl-btn" target="_blank">Lupin</a>
                <a href="{win_id}/kilosort4/" class="tbl-btn" target="_blank">KS4</a>
            </td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>e097 Anoxic Challenge: All Subwindows Dual Spike Sorting Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {{
            background: #07090e;
            color: #f1f5f9;
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 30px;
        }}
        .header {{
            background: #0f1523;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 24px 30px;
            margin-bottom: 24px;
        }}
        h1 {{
            margin: 0 0 10px 0;
            font-size: 24px;
            color: #38bdf8;
            font-weight: 800;
        }}
        .subtitle {{
            color: #94a3b8;
            font-size: 14px;
            margin: 0;
        }}
        .table-container {{
            background: #0f1523;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 24px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th {{
            color: #94a3b8;
            border-bottom: 1px solid rgba(255,255,255,0.12);
            padding: 12px 14px;
            text-align: left;
        }}
        td {{
            padding: 14px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            vertical-align: middle;
        }}
        .tbl-btn {{
            display: inline-block;
            background: #1e293b;
            color: #38bdf8;
            padding: 5px 10px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: 600;
            font-size: 11px;
            margin: 2px;
            border: 1px solid rgba(56, 189, 248, 0.3);
            transition: all 0.2s ease;
        }}
        .tbl-btn:hover {{
            background: #38bdf8;
            color: #07090e;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>e097 Anoxic Challenge: Complete 7-Subwindow Dual Spike Sorting Dashboard</h1>
        <p class="subtitle">Spike sorting across baseline, anoxia, apnea nadir, autoresuscitation gasping, active expiration shift, and recovery (1740s - 2160s / 29:00 - 36:00)</p>
    </div>

    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Subwindow</th>
                    <th>Time Range</th>
                    <th>Lupin SUA / Total</th>
                    <th>Kilosort 4 SUA / Total</th>
                    <th>Consensus (&ge;50%)</th>
                    <th>Runtimes (L/K)</th>
                    <th>Interactive Viewers</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Saved master subwindows dashboard to {out_html}")

def main():
    parser = argparse.ArgumentParser(description="Batch Dual Spike Sorting across physiological subwindows")
    parser.add_argument("--window", type=str, default=None, help="Process specific window (e.g. win2_anox1)")
    args = parser.parse_args()

    raw_ap_dir = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_imec0")
    catgt_exe = r"C:\Users\lpascu\SpikeGLX\CatGT-win\CatGT.exe"
    out_base = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting"
    out_base.mkdir(parents=True, exist_ok=True)
    scratch_catgt_base = Path(r"C:\SCRATCH_neuropixels\catgt_e097_subwindows")
    scratch_sort_base = Path(r"C:\SCRATCH_neuropixels\sorting_working")

    windows_to_process = SUBWINDOWS
    if args.window:
        windows_to_process = [w for w in SUBWINDOWS if w["id"] == args.window]
        if not windows_to_process:
            logger.error(f"Window {args.window} not found in SUBWINDOWS list!")
            return

    all_results = []
    for w in windows_to_process:
        res = process_subwindow(
            win=w,
            raw_ap_dir=raw_ap_dir,
            out_base=out_base,
            scratch_catgt_base=scratch_catgt_base,
            scratch_sort_base=scratch_sort_base,
            catgt_exe=catgt_exe
        )
        all_results.append(res)

    # Master manifest and dashboard
    if len(windows_to_process) == len(SUBWINDOWS):
        master_json = out_base / "all_subwindows_sorting_manifest.json"
        with open(master_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4)
        build_master_all_subwindows_dashboard(all_results, out_base / "all_subwindows_sorting_dashboard.html")

if __name__ == '__main__':
    main()
