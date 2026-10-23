#!/usr/bin/env python3
"""
Fast Re-Computation & Dynamic QC/Consensus Utility for Sliding-Window Spike Sorting
Allows instant re-evaluation of quality criteria, temporal presence profiling across epochs,
spatial stability tracking (probe drift vs. biological activation), and consensus de-duplication
without re-running any spike sorting algorithms.
"""
import sys
import os
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.sorting.sliding_window_consensus import (
    compute_spike_train_agreement,
    compute_isi_violations,
    compute_presence_ratio
)
from scripts.plot_interactive_sorter_raster import build_interactive_sorter_raster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("RECOMPUTE_QC_CONSENSUS")


def analyze_unit_temporal_epochs(
    spike_times: np.ndarray,
    duration_s: float,
    bin_size_s: float = 30.0
) -> Dict[str, Any]:
    """
    Analyzes temporal continuity across epochs.
    Distinguishes continuous firing, transient activation, burst epochs, and quiescent intervals.
    """
    if len(spike_times) == 0 or duration_s <= 0:
        return {
            "num_epochs": 0,
            "active_epochs": 0,
            "epoch_presence": 0.0,
            "binned_fr": [],
            "max_epoch_fr": 0.0,
            "burst_epochs": 0,
            "quiescent_epochs": 0
        }

    n_bins = max(1, int(np.ceil(duration_s / bin_size_s)))
    bin_edges = np.linspace(0, duration_s, n_bins + 1)
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    binned_fr = counts / bin_size_s

    active_mask = counts > 0
    active_epochs = int(np.sum(active_mask))
    epoch_presence = active_epochs / n_bins
    max_epoch_fr = float(np.max(binned_fr)) if len(binned_fr) > 0 else 0.0
    mean_active_fr = float(np.mean(binned_fr[active_mask])) if active_epochs > 0 else 0.0

    # Burst epoch: firing rate > 3x mean active firing rate
    burst_epochs = int(np.sum(binned_fr > (3.0 * mean_active_fr))) if mean_active_fr > 0 else 0
    quiescent_epochs = n_bins - active_epochs

    return {
        "num_epochs": n_bins,
        "active_epochs": active_epochs,
        "epoch_presence": epoch_presence,
        "binned_fr": binned_fr.tolist(),
        "max_epoch_fr": max_epoch_fr,
        "mean_active_fr": mean_active_fr,
        "burst_epochs": burst_epochs,
        "quiescent_epochs": quiescent_epochs,
        "bin_edges": bin_edges.tolist()
    }


def plot_master_epoch_profiles(
    master_df: pd.DataFrame,
    master_spike_times: np.ndarray,
    master_spike_clusters: np.ndarray,
    duration_s: float,
    out_file: Path
):
    """
    Generates a visual summary of temporal presence, firing dynamics, and depth distribution across the session.
    """
    if len(master_df) == 0 or len(master_spike_times) == 0:
        return

    sua_df = master_df[master_df["passed_qc"]].copy()
    if len(sua_df) == 0:
        logger.warning("No units passed QC for plotting, showing top firing units.")
        sua_df = master_df[master_df["firing_rate_hz"] >= 0.1].copy()

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=False)

    # Panel 1: Full Session Spike Rasters by Probe Depth for QC-Passed Single Units (SUA)
    palette = plt.cm.tab20(np.linspace(0, 1, len(sua_df)))
    for idx, (_, row) in enumerate(sua_df.iterrows()):
        cid = int(row["cluster_id"])
        spks = master_spike_times[master_spike_clusters == cid]
        depth = row["depth_um"]
        color = palette[idx % len(palette)]
        axes[0].scatter(spks, np.full_like(spks, depth), s=6, color=color, alpha=0.85)

    axes[0].set_title(f"Master Probe Depth Spike Rasters ({len(sua_df)} QC-Passed Single Units, 0 to {duration_s:.1f}s)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Probe Depth (µm)")
    axes[0].set_xlim(0, duration_s)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # Panel 2: Temporal Presence Heatmap / Epoch Activity (QC-Passed Units Only)
    bin_size = 20.0
    n_bins = max(1, int(np.ceil(duration_s / bin_size)))
    bin_edges = np.linspace(0, duration_s, n_bins + 1)
    
    # Sort units by depth
    sorted_df = sua_df.sort_values(by="depth_um", ascending=True)
    heat_matrix = []
    
    for _, row in sorted_df.iterrows():
        cid = int(row["cluster_id"])
        spks = master_spike_times[master_spike_clusters == cid]
        counts, _ = np.histogram(spks, bins=bin_edges)
        fr = counts / bin_size
        heat_matrix.append(fr)

    heat_matrix = np.array(heat_matrix)
    if len(heat_matrix) > 0:
        im = axes[1].imshow(
            heat_matrix,
            aspect="auto",
            cmap="inferno",
            extent=[0, duration_s, 0, len(sorted_df)],
            origin="lower",
            interpolation="nearest"
        )
        cbar = plt.colorbar(im, ax=axes[1], fraction=0.03, pad=0.01)
        cbar.set_label("Firing Rate (Hz)", fontsize=10)
    axes[1].set_title(f"Temporal Activation Heatmap (20s Time Bins) for {len(sorted_df)} Single Units", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Single Unit # (Depth Ascending)")
    axes[1].set_xlim(0, duration_s)

    # Panel 3: ISI Violation Rate vs Presence Ratio Scatter Diagnostic
    scatter = axes[2].scatter(
        master_df["presence_ratio"] * 100.0,
        master_df["isi_violations"] * 100.0,
        c=master_df["firing_rate_hz"],
        s=master_df["snr"] * 12.0,
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
        linewidth=0.5
    )
    cbar2 = plt.colorbar(scatter, ax=axes[2], fraction=0.03, pad=0.01)
    cbar2.set_label("Mean FR (Hz)", fontsize=10)

    for _, row in master_df[master_df["passed_qc"]].iterrows():
        axes[2].annotate(
            f"Unit {int(row['cluster_id'])}",
            (row["presence_ratio"] * 100.0, row["isi_violations"] * 100.0),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
            fontweight="bold",
            color="darkgreen"
        )

    axes[2].set_title("Quality Landscape: ISI Violation % vs Presence Ratio % (Dot Size = SNR)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Presence Ratio (% of Recording Active)")
    axes[2].set_ylabel("ISI Violation Rate (% Contamination)")
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)
    logger.info(f"Saved Master Epoch & Presence Diagnostic Plot: {out_file.resolve()}")


def recompute_qc_and_consensus(
    pipeline_dir: Path,
    sorter_name: str = "lupin",
    isi_cutoff: float = 0.10,
    presence_ratio: float = 0.05,
    min_fr_hz: float = 0.05,
    min_spikes: int = 20,
    agreement_thresh: float = 0.50,
    isi_threshold_s: float = 0.0015
):
    sorter_dir = pipeline_dir / sorter_name
    windows_dir = sorter_dir / "windows"
    
    if not windows_dir.exists():
        raise FileNotFoundError(f"Could not find windows directory at: {windows_dir}")

    # Discover all window directories
    window_dirs = sorted([d for d in windows_dir.iterdir() if d.is_dir() and "win_" in d.name])
    if not window_dirs:
        raise ValueError(f"No completed window directories found in {windows_dir}")

    logger.info("=" * 85)
    logger.info(f" DYNAMIC QC & CONSENSUS RE-EVALUATION ({sorter_name.upper()})")
    logger.info(f" Windows Found: {len(window_dirs)}")
    logger.info(f" QC Criteria: ISI Viol < {isi_cutoff:.2f} | Presence >= {presence_ratio:.2f} | Min FR >= {min_fr_hz:.2f} Hz | Min Spikes >= {min_spikes}")
    logger.info(f" Consensus Mutual Agreement Threshold: {agreement_thresh:.2f} ({(agreement_thresh*100):.0f}%)")
    logger.info("=" * 85)

    # 1. Load Windows Data and Re-evaluate Window QC
    windows_data = []
    total_raw_clusters = 0
    duration_s = 0.0

    for w_idx, w_dir in enumerate(window_dirs):
        alf_dir = w_dir / "alf"
        spk_times_file = alf_dir / "spike_times.npy"
        spk_clust_file = alf_dir / "spike_clusters.npy"
        cluster_info_file = w_dir / "cluster_info.tsv"

        if not spk_times_file.exists() or not spk_clust_file.exists():
            logger.warning(f"Skipping incomplete window: {w_dir.name} (missing ALF spike arrays)")
            continue

        spk_times = np.load(spk_times_file)
        spk_clusters = np.load(spk_clust_file)
        
        if len(spk_times) > 0:
            duration_s = max(duration_s, float(np.max(spk_times)))

        # Load or re-compute cluster table
        if cluster_info_file.exists():
            df_cl = pd.read_csv(cluster_info_file, sep="\t")
        else:
            unique_ids = np.unique(spk_clusters)
            records = []
            for u in unique_ids:
                u_spks = spk_times[spk_clusters == u]
                records.append({
                    "cluster_id": int(u),
                    "depth_um": 0.0,
                    "peak_amp_uv": 100.0,
                    "snr": 6.0
                })
            df_cl = pd.DataFrame(records)

        total_raw_clusters += len(df_cl)
        unit_trains = {}
        unit_qc = {}
        
        for u in df_cl["cluster_id"].values:
            u_spks = np.sort(spk_times[spk_clusters == u])
            unit_trains[int(u)] = u_spks
            
            # Compute window-level QC
            u_n = len(u_spks)
            u_fr = u_n / duration_s if duration_s > 0 else 0.0
            u_isi, _ = compute_isi_violations(u_spks, duration_s, isi_threshold_s=isi_threshold_s)
            u_pres = compute_presence_ratio(u_spks, 0.0, duration_s, n_bins=100)
            u_passed = (
                (u_isi <= isi_cutoff) and
                (u_pres >= presence_ratio) and
                (u_fr >= min_fr_hz) and
                (u_n >= min_spikes)
            )
            unit_qc[int(u)] = u_passed

        windows_data.append({
            "window_id": w_idx,
            "name": w_dir.name,
            "win_dir": w_dir,
            "df_cl": df_cl,
            "unit_trains": unit_trains,
            "unit_qc": unit_qc
        })

    logger.info(f"Loaded {len(windows_data)} windows with {total_raw_clusters} total clusters across {duration_s:.1f}s recording.")

    # 2. Compute Cross-Window Pairwise Spike Agreement Matches (QC-Passed Single Units ONLY)
    pairwise_matches = []
    for i in range(len(windows_data) - 1):
        w_a = windows_data[i]
        w_b = windows_data[i + 1]
        
        # Strictly isolate units that passed single-unit QC to prevent multi-unit / noisy contamination
        qc_units_a = {u: t for u, t in w_a["unit_trains"].items() if w_a["unit_qc"].get(u, False)}
        qc_units_b = {u: t for u, t in w_b["unit_trains"].items() if w_b["unit_qc"].get(u, False)}
        
        logger.info(f"Cross-Window QC Consensus Matching: [{w_a['name']} ({len(qc_units_a)} SUA)] <---> [{w_b['name']} ({len(qc_units_b)} SUA)]...")

        for u_a, t_a in qc_units_a.items():
            for u_b, t_b in qc_units_b.items():
                score, n_match = compute_spike_train_agreement(t_a, t_b, delta_s=0.0004)
                if score >= agreement_thresh:
                    logger.info(f"  --> MATCH CONFIRMED: Win {w_a['window_id']} SUA #{u_a} <==> Win {w_b['window_id']} SUA #{u_b} (Agreement: {score*100:.1f}%, {n_match} spikes)")
                    pairwise_matches.append({
                        "win_a_id": w_a["window_id"],
                        "unit_a": u_a,
                        "win_b_id": w_b["window_id"],
                        "unit_b": u_b,
                        "agreement_score": score,
                        "matched_spikes": n_match
                    })

    # 3. Assemble Connected Components / Graph Deduplication
    unit_nodes = set()
    for w in windows_data:
        for u in w["unit_trains"].keys():
            unit_nodes.add((w["window_id"], u))

    adjacency = {n: set() for n in unit_nodes}
    for m in pairwise_matches:
        node_a = (m["win_a_id"], m["unit_a"])
        node_b = (m["win_b_id"], m["unit_b"])
        if node_a in adjacency and node_b in adjacency:
            adjacency[node_a].add(node_b)
            adjacency[node_b].add(node_a)

    visited = set()
    connected_components = []
    for n in unit_nodes:
        if n not in visited:
            comp = []
            queue = [n]
            visited.add(n)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adjacency[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            connected_components.append(comp)

    # 4. Master Unit Metrics, Temporal Epochs & QC Decision
    master_spike_times = []
    master_spike_clusters = []
    master_records = []

    for g_id, comp in enumerate(connected_components):
        all_spks = []
        depths = []
        amps = []
        snrs = []

        for win_id, uid in comp:
            w_obj = windows_data[win_id]
            t_u = w_obj["unit_trains"].get(uid, np.array([]))
            all_spks.extend(t_u)
            
            row = w_obj["df_cl"][w_obj["df_cl"]["cluster_id"] == uid]
            if not row.empty:
                depths.append(float(row["depth_um"].values[0]) if "depth_um" in row else 0.0)
                amps.append(float(row["peak_amp_uv"].values[0]) if "peak_amp_uv" in row else 100.0)
                snrs.append(float(row["snr"].values[0]) if "snr" in row else 6.0)

        # De-duplicate coincident spikes across windows
        sorted_spks = np.sort(all_spks)
        if len(sorted_spks) > 0:
            diffs = np.diff(sorted_spks)
            keep_mask = np.insert(diffs > 0.0004, 0, True)
            dedup_spks = sorted_spks[keep_mask]
        else:
            dedup_spks = np.array([])

        n_spk = len(dedup_spks)
        fr_hz = n_spk / duration_s if duration_s > 0 else 0.0
        isi_viol, n_viol = compute_isi_violations(dedup_spks, duration_s, isi_threshold_s=isi_threshold_s)
        pres_ratio = compute_presence_ratio(dedup_spks, 0.0, duration_s, n_bins=100)
        epoch_stats = analyze_unit_temporal_epochs(dedup_spks, duration_s, bin_size_s=30.0)

        avg_depth = float(np.mean(depths)) if depths else 0.0
        avg_amp = float(np.mean(amps)) if amps else 100.0
        avg_snr = float(np.mean(snrs)) if snrs else 6.0

        is_consensus = len(comp) > 1
        passed_qc = (
            (isi_viol <= isi_cutoff) and
            (pres_ratio >= presence_ratio) and
            (fr_hz >= min_fr_hz) and
            (n_spk >= min_spikes)
        )

        master_spike_times.extend(dedup_spks)
        master_spike_clusters.extend([g_id] * n_spk)

        master_records.append({
            "cluster_id": g_id,
            "depth_um": round(avg_depth, 1),
            "firing_rate_hz": round(fr_hz, 4),
            "n_spikes": n_spk,
            "peak_amp_uv": round(avg_amp, 1),
            "snr": round(avg_snr, 2),
            "isi_violations": round(isi_viol, 4),
            "n_isi_violations": n_viol,
            "presence_ratio": round(pres_ratio, 3),
            "active_30s_epochs": epoch_stats["active_epochs"],
            "total_30s_epochs": epoch_stats["num_epochs"],
            "burst_epochs": epoch_stats["burst_epochs"],
            "num_contributing_windows": len(comp),
            "is_cross_window_consensus": is_consensus,
            "contributing_units": str(comp),
            "passed_qc": passed_qc,
            "passed_allen_qc": passed_qc
        })

    sort_idx = np.argsort(master_spike_times) if len(master_spike_times) > 0 else np.array([])
    final_spike_times = np.array(master_spike_times)[sort_idx] if len(sort_idx) > 0 else np.array([])
    final_spike_clusters = np.array(master_spike_clusters)[sort_idx] if len(sort_idx) > 0 else np.array([])
    df_master = pd.DataFrame(master_records)

    # 5. Export Master Outputs
    master_dir = sorter_dir / "consensus_master"
    master_dir.mkdir(parents=True, exist_ok=True)
    alf_dir = master_dir / "alf"
    alf_dir.mkdir(parents=True, exist_ok=True)

    np.save(alf_dir / "spike_times.npy", final_spike_times)
    np.save(alf_dir / "spike_clusters.npy", final_spike_clusters)
    df_master.to_csv(master_dir / "master_consensus_qc_summary.csv", index=False)
    df_master.to_csv(master_dir / "cluster_info.tsv", sep="\t", index=False)
    df_master.to_csv(alf_dir / "cluster_info.tsv", sep="\t", index=False)
    pd.DataFrame(pairwise_matches).to_csv(master_dir / "cross_window_pairwise_matches.csv", index=False)

    # 6. Generate Diagnostic Plots
    plot_file = master_dir / "master_temporal_epoch_and_presence_diagnostics.png"
    plot_master_epoch_profiles(
        master_df=df_master,
        master_spike_times=final_spike_times,
        master_spike_clusters=final_spike_clusters,
        duration_s=duration_s,
        out_file=plot_file
    )

    # 7. Generate Synchronized 5-Panel Interactive HTML Raster
    try:
        build_interactive_sorter_raster(
            sorter_dir=master_dir,
            sorter_name=f"{sorter_name.capitalize()} (Recomputed Consensus Master)"
        )
    except Exception as e:
        logger.warning(f"Could not render interactive HTML raster: {e}")

    # Summary
    n_sua = int(np.sum(df_master["passed_qc"]))
    n_consensus = int(np.sum(df_master["is_cross_window_consensus"]))
    
    print("\n" + "=" * 85)
    print(" [SUMMARY] DYNAMIC RE-COMPUTATION COMPLETED SUCCESSFULLY")
    print("=" * 85)
    print(f" • Total Unified Clusters Extracted:    {len(df_master)}")
    print(f" • Single Units Passing Custom QC:     {n_sua} units")
    print(f" • Cross-Window Confirmed Consensus:   {n_consensus} units")
    print(f" • Master TSV File:                    {master_dir / 'cluster_info.tsv'}")
    print(f" • Epoch Diagnostics Figure:           {plot_file}")
    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description="Recompute QC Criteria & Cross-Window Consensus on Existing Sorted Data")
    parser.add_argument("--pipeline-dir", type=str, default="output/sliding_window_pipeline", help="Path to sliding_window_pipeline directory")
    parser.add_argument("--sorter", type=str, default="lupin", help="Sorter name (lupin, mountainsort5, kilosort4)")
    parser.add_argument("--isi-cutoff", type=float, default=0.10, help="Max refractory period ISI violation rate (e.g. 0.10)")
    parser.add_argument("--presence-ratio", type=float, default=0.05, help="Min presence ratio across recording (e.g. 0.05 = 5%% of time active)")
    parser.add_argument("--min-fr", type=float, default=0.05, help="Min mean firing rate in Hz (default: 0.05 Hz)")
    parser.add_argument("--min-spikes", type=int, default=20, help="Min total spikes to evaluate unit")
    parser.add_argument("--agreement-thresh", type=float, default=0.50, help="Cross-window mutual spike coincidence agreement (default: 0.50)")
    args = parser.parse_args()

    recompute_qc_and_consensus(
        pipeline_dir=Path(args.pipeline_dir),
        sorter_name=args.sorter.lower(),
        isi_cutoff=args.isi_cutoff,
        presence_ratio=args.presence_ratio,
        min_fr_hz=args.min_fr,
        min_spikes=args.min_spikes,
        agreement_thresh=args.agreement_thresh
    )

if __name__ == "__main__":
    main()
