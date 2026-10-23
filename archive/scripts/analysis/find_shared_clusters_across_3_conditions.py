#!/usr/bin/env python3
"""
Three-Way Cross-Sorter Cluster Agreement & Candidate Discovery:
Finds clusters shared across:
1. Lupin (Sliding-Window Consensus Master)
2. Kilosort 4 (Sliding-Window Consensus Master)
3. Kilosort 4 (Global Full-Probe 384-ch)

Quantifies similarity using:
- Jaccard Spike Coincidence Agreement (tolerance delta_t = 0.5 ms)
- Spike Recall & Precision
- Depth Proximity (um)
- Firing Rate Consistency
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def compute_fast_coincidences(spk1: np.ndarray, spk2: np.ndarray, delta_s: float = 0.0005):
    """
    Computes bipartite spike coincidences within delta_s (0.5 ms).
    Returns (n_coinc, jaccard, recall_1_in_2, recall_2_in_1).
    """
    n1, n2 = len(spk1), len(spk2)
    if n1 == 0 or n2 == 0:
        return 0, 0.0, 0.0, 0.0
    
    # Sort just in case
    spk1 = np.sort(spk1)
    spk2 = np.sort(spk2)

    # Two-pointer matching
    i, j = 0, 0
    n_coinc = 0
    while i < n1 and j < n2:
        dt = spk1[i] - spk2[j]
        if abs(dt) <= delta_s:
            n_coinc += 1
            i += 1
            j += 1
        elif dt < -delta_s:
            i += 1
        else:
            j += 1

    union = n1 + n2 - n_coinc
    jaccard = n_coinc / union if union > 0 else 0.0
    rec1 = n_coinc / n1 if n1 > 0 else 0.0
    rec2 = n_coinc / n2 if n2 > 0 else 0.0
    return n_coinc, jaccard, rec1, rec2


def load_sorter_data(base_dir: Path, name: str):
    alf_dir = base_dir / "alf"
    st = np.load(alf_dir / "spike_times.npy")
    sc = np.load(alf_dir / "spike_clusters.npy")
    
    tsv_path = base_dir / "cluster_info.tsv"
    df = pd.read_csv(tsv_path, sep="\t")
    
    # Standardize column names
    col_map = {}
    for c in df.columns:
        if c in ["depth", "depth_um"]:
            col_map[c] = "depth_um"
        elif c in ["firing_rate", "firing_rate_hz"]:
            col_map[c] = "firing_rate_hz"
        elif c in ["passed_qc", "passed_allen_qc"]:
            col_map[c] = "passed_allen_qc"
    df = df.rename(columns=col_map)
    
    # Group spike times by cluster
    uids = np.unique(sc)
    spk_by_cluster = {}
    for u in uids:
        spk_by_cluster[int(u)] = st[sc == u]
        
    print(f"[{name}] Loaded {len(df)} clusters ({len(st):,} spikes, {df.get('passed_allen_qc', pd.Series([True]*len(df))).sum()} Allen SUA).")
    return {
        "name": name,
        "df": df,
        "spk": spk_by_cluster,
        "base_dir": base_dir
    }


def match_pair(cond_a, cond_b, delta_s: float = 0.0005):
    """
    Finds best matching clusters between condition A and condition B.
    Evaluates spike coincidences across all candidate pairs satisfying mathematical
    Jaccard bounds (min(N1, N2) / max(N1, N2) >= 0.10).
    """
    df_a = cond_a["df"].set_index("cluster_id")
    df_b = cond_b["df"].set_index("cluster_id")
    spk_a = cond_a["spk"]
    spk_b = cond_b["spk"]
    
    matches = []
    
    for uid_a, s_a in spk_a.items():
        n_a = len(s_a)
        if n_a < 10:
            continue
        depth_a = float(df_a.loc[uid_a, "depth_um"]) if uid_a in df_a.index and "depth_um" in df_a.columns else 0.0
        sua_a = bool(df_a.loc[uid_a, "passed_allen_qc"]) if uid_a in df_a.index and "passed_allen_qc" in df_a.columns else False
        fr_a = float(df_a.loc[uid_a, "firing_rate_hz"]) if uid_a in df_a.index and "firing_rate_hz" in df_a.columns else n_a/60.0
        
        best_b = None
        best_jacc = 0.0
        best_coinc = 0
        best_rec_a = 0.0
        best_rec_b = 0.0
        
        for uid_b, s_b in spk_b.items():
            n_b = len(s_b)
            if n_b < 10:
                continue
            if min(n_a, n_b) / max(n_a, n_b) < 0.10:
                continue
                
            n_c, jacc, rec_a, rec_b = compute_fast_coincidences(s_a, s_b, delta_s=delta_s)
            if jacc > best_jacc:
                best_jacc = jacc
                best_b = uid_b
                best_coinc = n_c
                best_rec_a = rec_a
                best_rec_b = rec_b
                
        if best_b is not None and best_jacc >= 0.10:
            sua_b = bool(df_b.loc[best_b, "passed_allen_qc"]) if best_b in df_b.index and "passed_allen_qc" in df_b.columns else False
            fr_b = float(df_b.loc[best_b, "firing_rate_hz"]) if best_b in df_b.index and "firing_rate_hz" in df_b.columns else len(spk_b[best_b])/60.0
            depth_b = float(df_b.loc[best_b, "depth_um"]) if best_b in df_b.index and "depth_um" in df_b.columns else 0.0
            
            matches.append({
                f"unit_{cond_a['name']}": uid_a,
                f"unit_{cond_b['name']}": best_b,
                "jaccard_agreement": round(best_jacc, 4),
                f"recall_{cond_a['name']}_in_{cond_b['name']}": round(best_rec_a, 4),
                f"recall_{cond_b['name']}_in_{cond_a['name']}": round(best_rec_b, 4),
                f"depth_{cond_a['name']}_um": round(depth_a, 1),
                f"depth_{cond_b['name']}_um": round(depth_b, 1),
                f"fr_{cond_a['name']}_hz": round(fr_a, 2),
                f"fr_{cond_b['name']}_hz": round(fr_b, 2),
                f"sua_{cond_a['name']}": sua_a,
                f"sua_{cond_b['name']}": sua_b,
                "both_sua": (sua_a and sua_b)
            })
            
    df_matches = pd.DataFrame(matches)
    if not df_matches.empty:
        df_matches = df_matches.sort_values(by="jaccard_agreement", ascending=False).reset_index(drop=True)
    return df_matches


def main():
    root = Path("outputs/e097/output_e097_prelaser_60s")
    
    dir_lupin = root / "lupin" / "consensus_master"
    dir_ks4_slide = root / "kilosort4_sliding_window" / "consensus_master"
    dir_ks4_glob = root / "kilosort4_global"
    
    print("=" * 80)
    print(" 3-WAY SPIKE SORTING CONSENSUS & SHARED CLUSTER ANALYSIS")
    print("=" * 80)
    
    data_lupin = load_sorter_data(dir_lupin, "Lupin")
    data_ks4_slide = load_sorter_data(dir_ks4_slide, "KS4_Slide")
    data_ks4_glob = load_sorter_data(dir_ks4_glob, "KS4_Global")
    
    # 1. Pairwise Matching
    print("\n--- Computing Pairwise Matches ---")
    
    # A. KS4_Slide vs KS4_Global
    t0 = time.time()
    m_slide_glob = match_pair(data_ks4_slide, data_ks4_glob)
    print(f"KS4_Slide vs KS4_Global: {len(m_slide_glob)} candidate matches found (Jaccard >= 0.10) in {time.time()-t0:.2f}s.")
    
    # B. Lupin vs KS4_Slide
    t0 = time.time()
    m_lupin_slide = match_pair(data_lupin, data_ks4_slide)
    print(f"Lupin vs KS4_Slide: {len(m_lupin_slide)} candidate matches found (Jaccard >= 0.10) in {time.time()-t0:.2f}s.")

    # C. Lupin vs KS4_Global
    t0 = time.time()
    m_lupin_glob = match_pair(data_lupin, data_ks4_glob)
    print(f"Lupin vs KS4_Global: {len(m_lupin_glob)} candidate matches found (Jaccard >= 0.10) in {time.time()-t0:.2f}s.")
    
    # 2. Tripartite Consensus Discovery (Shared across ALL THREE)
    print("\n--- Tripartite Shared Clusters (Lupin + KS4_Slide + KS4_Global) ---")
    tripartite_rows = []
    
    # Use KS4_Slide as reference
    slide_to_glob = {row["unit_KS4_Slide"]: row for _, row in m_slide_glob.iterrows()}
    lupin_to_slide = {row["unit_KS4_Slide"]: row for _, row in m_lupin_slide.iterrows()}
    
    for uid_slide, row_glob in slide_to_glob.items():
        if uid_slide in lupin_to_slide:
            row_lupin = lupin_to_slide[uid_slide]
            uid_glob = row_glob["unit_KS4_Global"]
            uid_lupin = row_lupin["unit_Lupin"]
            
            jacc_slide_glob = row_glob["jaccard_agreement"]
            jacc_lupin_slide = row_lupin["jaccard_agreement"]
            
            # Direct coincidence between Lupin and Global
            spk_l = data_lupin["spk"][uid_lupin]
            spk_g = data_ks4_glob["spk"][uid_glob]
            _, jacc_lupin_glob, rec_lg, rec_gl = compute_fast_coincidences(spk_l, spk_g)
            
            mean_jaccard = round(float(np.mean([jacc_slide_glob, jacc_lupin_slide, jacc_lupin_glob])), 4)
            min_jaccard = round(float(np.min([jacc_slide_glob, jacc_lupin_slide, jacc_lupin_glob])), 4)
            
            tripartite_rows.append({
                "lupin_unit_id": int(uid_lupin),
                "ks4_slide_unit_id": int(uid_slide),
                "ks4_global_unit_id": int(uid_glob),
                "mean_3way_jaccard": mean_jaccard,
                "min_3way_jaccard": min_jaccard,
                "jaccard_lupin_ks4slide": jacc_lupin_slide,
                "jaccard_ks4slide_ks4glob": jacc_slide_glob,
                "jaccard_lupin_ks4glob": round(jacc_lupin_glob, 4),
                "depth_um": row_glob["depth_KS4_Slide_um"],
                "fr_lupin_hz": row_lupin["fr_Lupin_hz"],
                "fr_ks4_slide_hz": row_glob["fr_KS4_Slide_hz"],
                "fr_ks4_glob_hz": row_glob["fr_KS4_Global_hz"],
                "sua_lupin": row_lupin["sua_Lupin"],
                "sua_ks4_slide": row_glob["sua_KS4_Slide"],
                "sua_ks4_glob": row_glob["sua_KS4_Global"],
                "all_three_sua": (row_lupin["sua_Lupin"] and row_glob["sua_KS4_Slide"] and row_glob["sua_KS4_Global"])
            })
            
    df_tripartite = pd.DataFrame(tripartite_rows)
    if not df_tripartite.empty:
        df_tripartite = df_tripartite.sort_values(by="mean_3way_jaccard", ascending=False).reset_index(drop=True)
        
    print(f"Total Tripartite Candidate Clusters (present in all 3): {len(df_tripartite)}")
    high_conf = df_tripartite[df_tripartite["mean_3way_jaccard"] >= 0.50]
    very_high = df_tripartite[df_tripartite["mean_3way_jaccard"] >= 0.70]
    all_sua_high = df_tripartite[(df_tripartite["all_three_sua"]) & (df_tripartite["mean_3way_jaccard"] >= 0.50)]
    print(f"  * High Agreement (Mean Jaccard >= 0.50): {len(high_conf)}")
    print(f"  * Very High Agreement (Mean Jaccard >= 0.70): {len(very_high)}")
    print(f"  * All 3 Sorters Classified as Allen SUA (Jaccard >= 0.50): {len(all_sua_high)}")
    
    # Save CSV outputs
    out_dir = root / "cross_sorter_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_tripartite.to_csv(out_dir / "tripartite_shared_clusters_all_3_sorters.csv", index=False)
    m_slide_glob.to_csv(out_dir / "pairwise_matches_ks4_slide_vs_ks4_global.csv", index=False)
    m_lupin_slide.to_csv(out_dir / "pairwise_matches_lupin_vs_ks4_slide.csv", index=False)
    m_lupin_glob.to_csv(out_dir / "pairwise_matches_lupin_vs_ks4_global.csv", index=False)
    
    print(f"\nSaved cross-matching tables to: {out_dir}")
    
    # Display top 15 tripartite units
    print("\n--- TOP 15 SHARED UNITS ACROSS ALL 3 CONDITIONS ---")
    cols_show = ["lupin_unit_id", "ks4_slide_unit_id", "ks4_global_unit_id", "depth_um", 
                 "mean_3way_jaccard", "fr_lupin_hz", "fr_ks4_slide_hz", "fr_ks4_glob_hz", "all_three_sua"]
    print(df_tripartite[cols_show].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
