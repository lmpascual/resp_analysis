"""
Cross-Window Consensus & Unit Agreement Matcher
Compares spike trains in overlapping channel regions across adjacent sliding windows.
Units with Agreement Score >= threshold (default 0.50) are flagged as High-Quality Consensus Units.
Merges de-duplicated spikes into a unified master Phy / ALF sorting output.
"""
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import pandas as pd
import spikeinterface.full as si

from .postprocessing import export_to_phy_bundle, compute_isi_violations, compute_presence_ratio

logger = logging.getLogger("SLIDING_CONSENSUS")

def compute_spike_train_agreement(
    times_a: np.ndarray,
    times_b: np.ndarray,
    delta_s: float = 0.0004
) -> Tuple[float, int]:
    """
    Computes symmetric agreement score between two spike trains:
    Agreement = N_matched / (N_A + N_B - N_matched)
    where spikes match if |t_A - t_B| <= delta_s.
    """
    n_a = len(times_a)
    n_b = len(times_b)
    if n_a == 0 or n_b == 0:
        return 0.0, 0

    idx_b = np.searchsorted(times_b, times_a)
    matched = 0
    used_b = set()

    for i, t in enumerate(times_a):
        # Check closest in b
        b_candidates = []
        if idx_b[i] < n_b:
            b_candidates.append(idx_b[i])
        if idx_b[i] > 0:
            b_candidates.append(idx_b[i] - 1)

        for b_cand in b_candidates:
            if b_cand not in used_b and abs(times_b[b_cand] - t) <= delta_s:
                matched += 1
                used_b.add(b_cand)
                break

    denom = (n_a + n_b - matched)
    score = matched / denom if denom > 0 else 0.0
    return float(score), matched

def compute_sliding_window_consensus(
    window_results: List[Dict[str, Any]],
    recording: si.BaseRecording,
    out_dir: Path,
    agreement_threshold: float = 0.50,
    sorter_name: str = "lupin",
    pipeline_config: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Evaluates cross-window unit agreement across all adjacent sliding window pairs.
    Merges matching units into master consensus units, applies configured QC criteria,
    and writes out run_journal.md, Phy bundle, and ALF arrays.
    """
    import datetime
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    duration_s = recording.get_total_duration()

    # Extract QC thresholds from config or use standard defaults
    qc_isi_cutoff = 0.10
    qc_presence_ratio = 0.90
    qc_amp_cutoff = 0.10
    qc_min_fr_hz = 0.10
    isi_threshold_s = 0.0010

    if pipeline_config is not None:
        qc_isi_cutoff = getattr(pipeline_config.sorters, "qc_isi_cutoff", 0.10)
        qc_presence_ratio = getattr(pipeline_config.sorters, "qc_presence_ratio", 0.90)
        qc_amp_cutoff = getattr(pipeline_config.sorters, "qc_amp_cutoff", 0.10)
        qc_min_fr_hz = getattr(pipeline_config.sorters, "qc_min_fr_hz", 0.10)
        isi_threshold_s = getattr(pipeline_config.sorters, "isi_threshold_s", 0.0010)

    logger.info("=" * 80)
    logger.info(" COMPUTING CROSS-WINDOW CONSENSUS & UNIT AGREEMENT")
    logger.info(f" Total Windows: {len(window_results)} | Agreement Threshold: {agreement_threshold:.2f}")
    logger.info(f" Single-Unit QC Criteria: ISI Viol < {qc_isi_cutoff:.2f} | Presence > {qc_presence_ratio:.2f} | Amp Cutoff < {qc_amp_cutoff:.2f} | Min FR > {qc_min_fr_hz:.2f} Hz")
    logger.info("=" * 80)

    # 1. Load all window spike trains and QC summaries
    windows_data = []
    for wr in window_results:
        w_info = wr["window"]
        w_dir = wr["win_dir"]
        alf_dir = w_dir / "alf"
        spk_t = np.load(alf_dir / "spike_times.npy")
        spk_c = np.load(alf_dir / "spike_clusters.npy")
        qc_df = wr["qc_df"]

        unit_trains = {}
        for uid in qc_df["cluster_id"].values:
            unit_trains[uid] = np.sort(spk_t[spk_c == uid])

        # Identify units that passed single-unit QC in this window
        qc_passing_uids = set()
        if "passed_allen_qc" in qc_df.columns:
            qc_passing_uids = set(qc_df[qc_df["passed_allen_qc"]]["cluster_id"].values)
        elif "passed_qc" in qc_df.columns:
            qc_passing_uids = set(qc_df[qc_df["passed_qc"]]["cluster_id"].values)
        else:
            qc_passing_uids = set(qc_df["cluster_id"].values)

        windows_data.append({
            "window_id": w_info["window_id"],
            "name": w_info["name"],
            "start_ch": w_info["start_ch"],
            "end_ch": w_info["end_ch"],
            "qc_df": qc_df,
            "unit_trains": unit_trains,
            "qc_passing_uids": qc_passing_uids
        })

    # 2. Pairwise Agreement Matching across Adjacent Windows (QC Single Units ONLY)
    pairwise_matches = []
    for i in range(len(windows_data) - 1):
        w_a = windows_data[i]
        w_b = windows_data[i + 1]

        # Only evaluate units that passed single-unit QC to prevent multi-unit contamination
        qc_trains_a = {u: t for u, t in w_a["unit_trains"].items() if u in w_a["qc_passing_uids"]}
        qc_trains_b = {u: t for u, t in w_b["unit_trains"].items() if u in w_b["qc_passing_uids"]}

        logger.info(f"Matching Overlap between {w_a['name']} ({len(qc_trains_a)} SUA) and {w_b['name']} ({len(qc_trains_b)} SUA)...")
        for u_a, t_a in qc_trains_a.items():
            if len(t_a) < 10:
                continue
            for u_b, t_b in qc_trains_b.items():
                if len(t_b) < 10:
                    continue
                score, n_match = compute_spike_train_agreement(t_a, t_b)
                if score >= agreement_threshold:
                    logger.info(f"  --> MATCH FOUND: Win {w_a['window_id']} SUA #{u_a} <--> Win {w_b['window_id']} SUA #{u_b} (Agreement: {score*100:.1f}%, {n_match} spikes)")
                    pairwise_matches.append({
                        "win_a_id": w_a["window_id"],
                        "unit_a": u_a,
                        "win_b_id": w_b["window_id"],
                        "unit_b": u_b,
                        "agreement_score": score,
                        "matched_spikes": n_match
                    })

    # 3. Cluster Connected Components into Master Global Units
    unit_nodes = set()
    for w in windows_data:
        for u in w["unit_trains"].keys():
            unit_nodes.add((w["window_id"], u))

    adjacency: Dict[Tuple[int, Any], Set[Tuple[int, Any]]] = {n: set() for n in unit_nodes}
    match_scores_map: Dict[Tuple[Tuple[int, Any], Tuple[int, Any]], float] = {}

    for m in pairwise_matches:
        node_a = (m["win_a_id"], m["unit_a"])
        node_b = (m["win_b_id"], m["unit_b"])
        adjacency[node_a].add(node_b)
        adjacency[node_b].add(node_a)
        match_scores_map[(node_a, node_b)] = m["agreement_score"]
        match_scores_map[(node_b, node_a)] = m["agreement_score"]

    visited = set()
    global_clusters = []

    for node in unit_nodes:
        if node not in visited:
            component = []
            queue = [node]
            visited.add(node)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adjacency[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            global_clusters.append(component)

    # 4. Merge Spike Trains and Construct Master QC Summary
    master_spike_times = []
    master_spike_clusters = []
    master_cluster_records = []

    for g_id, comp in enumerate(global_clusters):
        all_spikes_list = []
        depths_list = []
        amps_list = []
        snrs_list = []
        is_sua_list = []

        for win_id, u_id in comp:
            w_obj = windows_data[win_id]
            t_spks = w_obj["unit_trains"][u_id]
            all_spikes_list.extend(t_spks)
            
            u_row = w_obj["qc_df"][w_obj["qc_df"]["cluster_id"] == u_id].iloc[0]
            depths_list.append(u_row["depth_um"])
            amps_list.append(u_row["peak_amp_uv"])
            snrs_list.append(u_row["snr"])
            is_sua_list.append(bool(u_row["passed_allen_qc"]))

        # De-duplicate merged spikes (within 0.4 ms)
        sorted_spks = np.sort(all_spikes_list)
        if len(sorted_spks) > 0:
            diffs = np.diff(sorted_spks)
            keep_mask = np.insert(diffs > 0.0004, 0, True)
            dedup_spks = sorted_spks[keep_mask]
        else:
            dedup_spks = np.array([])

        # Master cluster metrics
        n_spk = len(dedup_spks)
        fr_hz = n_spk / duration_s if duration_s > 0 else 0.0
        isi_viol, n_viol = compute_isi_violations(dedup_spks, duration_s, isi_threshold_s=isi_threshold_s)
        pres_ratio = compute_presence_ratio(dedup_spks, 0.0, duration_s)
        avg_depth = float(np.mean(depths_list))
        avg_amp = float(np.mean(amps_list))
        avg_snr = float(np.mean(snrs_list))

        is_consensus = len(comp) > 1
        passed_allen_qc = (isi_viol < qc_isi_cutoff) and (pres_ratio >= qc_presence_ratio) and (fr_hz >= qc_min_fr_hz)

        master_spike_times.extend(dedup_spks)
        master_spike_clusters.extend([g_id] * n_spk)

        master_cluster_records.append({
            "cluster_id": g_id,
            "depth_um": avg_depth,
            "firing_rate_hz": fr_hz,
            "n_spikes": n_spk,
            "peak_amp_uv": avg_amp,
            "snr": avg_snr,
            "isi_violations": isi_viol,
            "n_isi_violations": n_viol,
            "presence_ratio": pres_ratio,
            "num_contributing_windows": len(comp),
            "is_cross_window_consensus": is_consensus,
            "contributing_units": str(comp),
            "passed_allen_qc": passed_allen_qc
        })

    # Sort master arrays
    sort_idx = np.argsort(master_spike_times) if len(master_spike_times) > 0 else np.array([])
    final_spike_times = np.array(master_spike_times)[sort_idx] if len(sort_idx) > 0 else np.array([])
    final_spike_clusters = np.array(master_spike_clusters)[sort_idx] if len(sort_idx) > 0 else np.array([])
    df_master = pd.DataFrame(master_cluster_records)

    # 5. Export Master ALF and Phy Bundle
    alf_dir = out_dir / "alf"
    alf_dir.mkdir(parents=True, exist_ok=True)
    np.save(alf_dir / "spike_times.npy", final_spike_times)
    np.save(alf_dir / "spike_clusters.npy", final_spike_clusters)
    df_master.to_csv(out_dir / "master_consensus_qc_summary.csv", index=False)
    df_master.to_csv(out_dir / "cluster_info.tsv", sep="\t", index=False)
    df_master.to_csv(alf_dir / "cluster_info.tsv", sep="\t", index=False)

    # Export Pairwise Matches
    pd.DataFrame(pairwise_matches).to_csv(out_dir / "cross_window_pairwise_matches.csv", index=False)

    n_total_master = len(df_master)
    n_sua_master = int(np.sum(df_master["passed_allen_qc"]))
    n_consensus = int(np.sum(df_master["is_cross_window_consensus"]))

    # 6. Generate Run Journal (run_journal.md & run_journal.json)
    journal_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "sorter": sorter_name,
        "total_recording_channels": recording.get_num_channels(),
        "sampling_frequency_hz": float(recording.sampling_frequency),
        "duration_seconds": float(duration_s),
        "windows_evaluated": len(window_results),
        "agreement_threshold": agreement_threshold,
        "qc_criteria": {
            "qc_isi_cutoff": qc_isi_cutoff,
            "qc_presence_ratio": qc_presence_ratio,
            "qc_amp_cutoff": qc_amp_cutoff,
            "qc_min_fr_hz": qc_min_fr_hz,
            "isi_threshold_s": isi_threshold_s
        },
        "results": {
            "total_unified_units": n_total_master,
            "cross_window_confirmed_units": n_consensus,
            "passing_single_units_sua": n_sua_master,
            "mua_noise_units": n_total_master - n_sua_master
        },
        "pairwise_matches_count": len(pairwise_matches)
    }

    with open(out_dir / "run_journal.json", "w") as f:
        json.dump(journal_data, f, indent=4)

    journal_md = f"""# Spike Sorting Pipeline Run Journal
**Execution Timestamp**: {journal_data['timestamp']}
**Sorter**: {sorter_name.upper()}

## 1. Recording Metadata
- **Total Probe Channels**: {recording.get_num_channels()}
- **Sampling Frequency**: {recording.sampling_frequency:.1f} Hz
- **Total Duration**: {duration_s:.2f} seconds

## 2. Sliding-Window & Consensus Parameters
- **Windows Processed**: {len(window_results)}
- **Cross-Window Agreement Threshold**: {agreement_threshold*100:.0f}%
- **Max ISI Refractory Violation Cutoff**: {qc_isi_cutoff:.2f}
- **Min Presence Ratio**: {qc_presence_ratio:.2f}
- **Min Firing Rate**: {qc_min_fr_hz:.2f} Hz

## 3. Results Summary
- **Total Unified Master Units**: {n_total_master}
- **Cross-Window Confirmed Units**: {n_consensus}
- **Single Units Passing QC (SUA)**: {n_sua_master}
- **Multi-Unit / Noise (MUA)**: {n_total_master - n_sua_master}
- **Pairwise Overlap Matches**: {len(pairwise_matches)}
"""
    with open(out_dir / "run_journal.md", "w") as f:
        f.write(journal_md)

    logger.info("=" * 80)
    logger.info(f" MASTER CONSENSUS SORTING SUMMARY ({sorter_name.upper()})")
    logger.info(f" Total Unified Units:            {n_total_master}")
    logger.info(f" High-Quality Single Units (SUA): {n_sua_master}")
    logger.info(f" Cross-Window Confirmed Units:   {n_consensus}")
    logger.info(f" Run Journal Saved to:           {out_dir / 'run_journal.md'}")
    logger.info("=" * 80)

    return {
        "master_qc_df": df_master,
        "pairwise_matches": pairwise_matches,
        "n_total": n_total_master,
        "n_sua": n_sua_master,
        "n_consensus": n_consensus,
        "alf_dir": alf_dir,
        "journal_path": out_dir / "run_journal.md"
    }

    return {
        "master_qc_df": df_master,
        "pairwise_matches": pairwise_matches,
        "n_total": n_total_master,
        "n_sua": n_sua_master,
        "n_consensus": n_consensus,
        "alf_dir": alf_dir
    }
