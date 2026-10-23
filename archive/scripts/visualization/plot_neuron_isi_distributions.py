#!/usr/bin/env python3
"""
Plot Inter-Spike Interval (ISI) Distributions for Each Clean Single Unit.
Generates:
1. Individual high-resolution 2-panel ISI diagnostic figures (Linear Short-Range [0-50ms] + Log-Scale Full-Range [0.1-1000ms]).
2. Multi-neuron overview grid figure showing all 37 units.
3. Interactive HTML Visual Gallery with summary metrics (Refractory violations, median ISI, firing rate).
"""
import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ISI_PLOTS")


def plot_single_neuron_isi(
    unit_id: int,
    depth_um: float,
    spk_times: np.ndarray,
    out_file: Path,
    refractory_ms: float = 1.5
):
    """
    Plots 2-panel ISI diagnostic for a single unit:
    Left: Linear histogram (0 to 50 ms) showing refractory drop and burst peak
    Right: Log-scaled histogram (0.2 ms to 2000 ms) showing firing mode distributions
    """
    if len(spk_times) < 2:
        return

    isis_s = np.diff(spk_times)
    isis_ms = isis_s * 1000.0

    n_total_spks = len(spk_times)
    n_isis = len(isis_ms)
    n_violations = np.sum(isis_ms < refractory_ms)
    violation_rate = (n_violations / n_isis) * 100.0 if n_isis > 0 else 0.0

    mean_fr = n_total_spks / (np.max(spk_times) - np.min(spk_times)) if n_total_spks > 1 else 0.0
    median_isi = np.median(isis_ms)

    fig = plt.figure(figsize=(10, 4.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1.0], wspace=0.28)

    # -------------------------------------------------------------
    # Left Panel: Short-range Linear Histogram (0 - 50 ms)
    # -------------------------------------------------------------
    ax_lin = fig.add_subplot(gs[0])
    bins_lin = np.linspace(0, 50, 101)  # 0.5 ms bins
    counts_lin, edges_lin = np.histogram(isis_ms[isis_ms <= 50.0], bins=bins_lin)

    ax_lin.bar(edges_lin[:-1], counts_lin, width=0.5, align='edge', color="#0284c7", edgecolor="#0369a1", alpha=0.85)
    
    # Refractory violation shading & line
    ax_lin.axvspan(0, refractory_ms, color="#ef4444", alpha=0.25, label=f"Refractory (< {refractory_ms:.1f} ms)")
    ax_lin.axvline(refractory_ms, color="#dc2626", linestyle="--", linewidth=1.5)

    ax_lin.set_xlim(0, 50)
    ax_lin.set_xlabel("Inter-Spike Interval (ms)", fontsize=11, fontweight="bold")
    ax_lin.set_ylabel("Spike Count", fontsize=11, fontweight="bold")
    ax_lin.set_title(
        f"Unit #{unit_id} (Depth: {depth_um:.0f} µm) — Short-Range ISI\n"
        f"ISI Violations (< {refractory_ms:.1f} ms): {violation_rate:.2f}% ({n_violations}/{n_isis})",
        fontsize=10,
        fontweight="bold"
    )
    ax_lin.grid(True, linestyle=":", alpha=0.6)
    ax_lin.legend(loc="upper right", frameon=True, fontsize=8)

    # -------------------------------------------------------------
    # Right Panel: Log-scale Full-Range Histogram (0.2 ms - 2000 ms)
    # -------------------------------------------------------------
    ax_log = fig.add_subplot(gs[1])
    log_bins = np.logspace(np.log10(0.2), np.log10(2000), 60)
    
    ax_log.hist(isis_ms, bins=log_bins, color="#0f766e", edgecolor="#115e59", alpha=0.85)
    ax_log.set_xscale("log")
    ax_log.axvline(refractory_ms, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Refractory ({refractory_ms:.1f} ms)")
    ax_log.axvline(median_isi, color="#f59e0b", linestyle=":", linewidth=1.8, label=f"Median ISI ({median_isi:.1f} ms)")

    ax_log.set_xlim(0.2, 2000)
    ax_log.set_xlabel("Inter-Spike Interval (ms, log-scale)", fontsize=11, fontweight="bold")
    ax_log.set_ylabel("Spike Count", fontsize=11, fontweight="bold")
    ax_log.set_title(
        f"Full-Range Distribution (Log Scale)\n"
        f"Firing Rate: {mean_fr:.1f} sp/s | Total Spikes: {n_total_spks}",
        fontsize=10,
        fontweight="bold"
    )
    ax_log.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_log.legend(loc="upper right", frameon=True, fontsize=8)

    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def plot_overview_grid(
    df_sua: pd.DataFrame,
    spk_times: np.ndarray,
    spk_clusters: np.ndarray,
    out_file: Path,
    refractory_ms: float = 1.5
):
    """
    Plots a multi-panel grid containing short-range ISI histograms for all clean neurons.
    """
    n_units = len(df_sua)
    cols = 6
    rows = int(np.ceil(n_units / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6), squeeze=False)
    axes_flat = axes.flatten()

    bins_lin = np.linspace(0, 50, 51)  # 1 ms bins

    for i, (_, row) in enumerate(df_sua.iterrows()):
        uid = int(row["cluster_id"])
        depth = float(row["depth_um"]) if "depth_um" in row else 0.0
        ax = axes_flat[i]

        u_spks = spk_times[spk_clusters == uid]
        if len(u_spks) > 1:
            isis_ms = np.diff(u_spks) * 1000.0
            n_isis = len(isis_ms)
            n_viol = np.sum(isis_ms < refractory_ms)
            viol_rate = (n_viol / n_isis) * 100.0 if n_isis > 0 else 0.0

            ax.hist(isis_ms[isis_ms <= 50.0], bins=bins_lin, color="#0284c7", edgecolor="none", alpha=0.8)
            ax.axvspan(0, refractory_ms, color="#ef4444", alpha=0.3)
            ax.axvline(refractory_ms, color="#dc2626", linestyle="--", linewidth=1.0)

            ax.set_title(f"Unit #{uid} ({depth:.0f}µm)\nViol: {viol_rate:.1f}% ({len(u_spks)} spk)", fontsize=8, fontweight="bold")
        else:
            ax.set_title(f"Unit #{uid} (No Spikes)", fontsize=8)

        ax.set_xlim(0, 50)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle=":", alpha=0.5)

    # Hide unused axes
    for j in range(n_units, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(f"ISI Distributions for All {n_units} Single Units (0 - 50 ms) | Red Line = {refractory_ms:.1f} ms Refractory Period", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    logger.info(f"Saved All-Units ISI Overview Grid: {out_file.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Generate ISI Distribution Figures for All Clean Neurons")
    parser.add_argument("--consensus-dir", type=str, default="'output_LP21_60s_120s'/sliding_window_pipeline/lupin/consensus_master")
    parser.add_argument("--out-dir", type=str, default="output_neuron_isi_distributions_LP21_60s")
    parser.add_argument("--refractory-ms", type=float, default=1.5, help="Refractory period cutoff in ms")
    args = parser.parse_args()

    cons_dir_raw = args.consensus_dir.strip('"')
    cons_dir = Path(cons_dir_raw)
    if not cons_dir.exists():
        alt_path = Path(f"'{cons_dir_raw}'")
        if alt_path.exists():
            cons_dir = alt_path

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 85)
    logger.info(" GENERATING INTER-SPIKE INTERVAL (ISI) DISTRIBUTIONS")
    logger.info(f" Consensus Directory : {cons_dir.resolve()}")
    logger.info(f" Output Directory    : {out_dir.resolve()}")
    logger.info("=" * 85)

    # 1. Load Clean Units
    alf_dir = cons_dir / "alf"
    spk_times = np.load(alf_dir / "spike_times.npy")
    spk_clusters = np.load(alf_dir / "spike_clusters.npy")
    df_info = pd.read_csv(cons_dir / "cluster_info.tsv", sep="\t")

    qc_col = "passed_qc" if "passed_qc" in df_info.columns else ("passed_allen_qc" if "passed_allen_qc" in df_info.columns else None)
    df_sua = df_info[df_info[qc_col] == True].copy() if qc_col else df_info.copy()

    # Sort by unit ID (cluster_id) ascending
    df_sua = df_sua.sort_values(by="cluster_id", ascending=True).reset_index(drop=True)

    sua_ids = np.array(df_sua["cluster_id"].values)
    mask_sua = np.isin(spk_clusters, sua_ids)
    spk_times = spk_times[mask_sua]
    spk_clusters = spk_clusters[mask_sua]

    # 2. Multi-panel Overview Grid
    overview_png = out_dir / "all_37_neurons_isi_overview_grid.png"
    plot_overview_grid(
        df_sua=df_sua,
        spk_times=spk_times,
        spk_clusters=spk_clusters,
        out_file=overview_png,
        refractory_ms=args.refractory_ms
    )

    # 3. Generate Individual 2-Panel Figures
    index_cards = []
    for idx, row in df_sua.iterrows():
        uid = int(row["cluster_id"])
        depth = float(row["depth_um"]) if "depth_um" in row else 0.0
        u_spks = spk_times[spk_clusters == uid]

        out_png = out_dir / f"unit_{uid:03d}_depth_{int(depth)}um_isi_distribution.png"
        plot_single_neuron_isi(
            unit_id=uid,
            depth_um=depth,
            spk_times=u_spks,
            out_file=out_png,
            refractory_ms=args.refractory_ms
        )

        n_isis = len(u_spks) - 1
        n_viol = np.sum(np.diff(u_spks) * 1000.0 < args.refractory_ms) if n_isis > 0 else 0
        viol_pct = (n_viol / n_isis) * 100.0 if n_isis > 0 else 0.0
        mean_fr = len(u_spks) / (np.max(u_spks) - np.min(u_spks)) if len(u_spks) > 1 else 0.0

        index_cards.append({
            "unit_id": uid,
            "depth_um": depth,
            "n_spikes": len(u_spks),
            "viol_pct": viol_pct,
            "mean_fr": mean_fr,
            "file_name": out_png.name
        })

    # 4. Generate Interactive HTML Visual Gallery
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Neuron ISI Distributions Gallery</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }}
        h1 {{ text-align: center; color: #38bdf8; margin-bottom: 5px; }}
        p.subtitle {{ text-align: center; color: #94a3b8; font-size: 14px; margin-bottom: 25px; }}
        .overview-box {{ max-width: 1400px; margin: 0 auto 30px auto; background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 15px; text-align: center; }}
        .overview-box img {{ width: 100%; height: auto; border-radius: 6px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 20px; max-width: 1800px; margin: 0 auto; }}
        .card {{ background: #1e293b; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5); border: 1px solid #334155; }}
        .card img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
        .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .badges {{ display: flex; gap: 8px; }}
        .badge {{ background: #0284c7; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
        .badge-viol {{ background: #16a34a; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
        .badge-viol-high {{ background: #ea580c; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Inter-Spike Interval (ISI) Distributions</h1>
    <p class="subtitle">37 Clean Single Units | Refractory Period Cutoff: {args.refractory_ms:.1f} ms</p>
    
    <div class="overview-box">
        <h3 style="color: #38bdf8; margin-top: 0;">Multi-Unit ISI Overview Grid</h3>
        <a href="all_37_neurons_isi_overview_grid.png" target="_blank">
            <img src="all_37_neurons_isi_overview_grid.png" alt="All Neurons ISI Overview Grid">
        </a>
    </div>

    <div class="grid">
"""
    for card in index_cards:
        viol_badge_cls = "badge-viol" if card["viol_pct"] < 1.0 else "badge-viol-high"
        html_content += f"""
        <div class="card">
            <div class="card-header">
                <span style="font-weight: bold; font-size: 16px;">Unit #{card['unit_id']} ({card['depth_um']:.0f} µm)</span>
                <div class="badges">
                    <span class="badge">{card['n_spikes']} spks | {card['mean_fr']:.1f} Hz</span>
                    <span class="{viol_badge_cls}">Viol: {card['viol_pct']:.2f}%</span>
                </div>
            </div>
            <a href="{card['file_name']}" target="_blank">
                <img src="{card['file_name']}" alt="Unit {card['unit_id']}">
            </a>
        </div>
"""
    html_content += """
    </div>
</body>
</html>
"""
    gallery_path = out_dir / "gallery_isi_distributions_37_neurons.html"
    gallery_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Saved ISI Visual Gallery: {gallery_path.resolve()}")

    print("\n" + "=" * 85)
    print(" [SUCCESS] ALL 37 NEURON ISI DISTRIBUTIONS GENERATED")
    print("=" * 85)
    print(f" • Output Folder       : {out_dir.resolve()}")
    print(f" • Overview Grid       : {overview_png.resolve()}")
    print(f" • Interactive Gallery : {gallery_path.resolve()}")
    print("=" * 85)


if __name__ == "__main__":
    main()
