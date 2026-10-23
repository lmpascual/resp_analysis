#!/usr/bin/env python3
"""
Computes cross-sorter mutual agreement between Kilosort 4 and Lupin on the intact 420s recording.
Matching coincidence window = 0.4 ms.
Saves outputs/e097/intact_window/cross_sorter_agreement.json.
"""
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.comparison as sc

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("COMPARE_INTACT_SORTERS")

def main():
    out_base = REPO_ROOT / "outputs" / "e097" / "intact_window"
    ks4_alf = out_base / "kilosort4" / "alf"
    lupin_alf = out_base / "lupin" / "alf"
    
    fs = 29999.82421875
    
    logger.info("Loading Kilosort 4 spikes...")
    ks4_spk_t = np.load(ks4_alf / "spike_times.npy")
    ks4_spk_c = np.load(ks4_alf / "spike_clusters.npy")
    sort_ks4 = si.NumpySorting.from_times_and_labels(
        times_list=(ks4_spk_t * fs).astype(np.int64),
        labels_list=ks4_spk_c,
        sampling_frequency=fs
    )
    
    logger.info("Loading Lupin spikes...")
    lup_spk_t = np.load(lupin_alf / "spike_times.npy")
    lup_spk_c = np.load(lupin_alf / "spike_clusters.npy")
    sort_lup = si.NumpySorting.from_times_and_labels(
        times_list=(lup_spk_t * fs).astype(np.int64),
        labels_list=lup_spk_c,
        sampling_frequency=fs
    )
    
    logger.info(f"Loaded {len(sort_ks4.get_unit_ids())} KS4 units and {len(sort_lup.get_unit_ids())} Lupin units.")
    
    # Load QC
    ks4_qc = pd.read_csv(out_base / "kilosort4" / "kilosort4_qc_summary.csv")
    lup_qc = pd.read_csv(out_base / "lupin" / "lupin_qc_summary.csv")
    
    ks4_sua = int((ks4_qc["passed_allen_qc"] == True).sum())
    lup_sua = int((lup_qc["passed_allen_qc"] == True).sum())
    
    logger.info(f"KS4 SUA: {ks4_sua} / {len(ks4_qc)} | Lupin SUA: {lup_sua} / {len(lup_qc)}")
    
    logger.info("Comparing sorters with 0.4 ms delta_time...")
    cmp = sc.compare_two_sorters(
        sorting1=sort_lup,
        sorting2=sort_ks4,
        sorting1_name="lupin",
        sorting2_name="kilosort4",
        delta_time=0.4
    )
    
    agr = cmp.agreement_scores
    max_agr_lup = agr.max(axis=1) # best match in KS4 for each Lupin unit
    
    matched_50 = int((max_agr_lup >= 0.50).sum())
    matched_80 = int((max_agr_lup >= 0.80).sum())
    pct_lup_50 = float(matched_50 / max(1, len(sort_lup.get_unit_ids())) * 100.0)
    pct_lup_80 = float(matched_80 / max(1, len(sort_lup.get_unit_ids())) * 100.0)
    
    # Also evaluate among QC-passed single units
    lup_qc_ids = set(lup_qc[lup_qc["passed_allen_qc"] == True]["cluster_id"].values)
    lup_qc_indices = [i for i, uid in enumerate(sort_lup.get_unit_ids()) if uid in lup_qc_ids]
    if lup_qc_indices:
        max_agr_lup_qc = max_agr_lup.iloc[lup_qc_indices] if hasattr(max_agr_lup, 'iloc') else max_agr_lup[lup_qc_indices]
        matched_qc_50 = int((max_agr_lup_qc >= 0.50).sum())
        matched_qc_80 = int((max_agr_lup_qc >= 0.80).sum())
        pct_qc_50 = float(matched_qc_50 / len(lup_qc_indices) * 100.0)
    else:
        matched_qc_50, matched_qc_80, pct_qc_50 = 0, 0, 0.0
        
    agreement_res = {
        "ks4_total_units": len(sort_ks4.get_unit_ids()),
        "ks4_sua_units": ks4_sua,
        "lupin_total_units": len(sort_lup.get_unit_ids()),
        "lupin_sua_units": lup_sua,
        "matched_units_50pct": matched_50,
        "matched_units_80pct": matched_80,
        "pct_lupin_matched_50pct": pct_lup_50,
        "pct_lupin_matched_80pct": pct_lup_80,
        "qc_matched_units_50pct": matched_qc_50,
        "qc_matched_units_80pct": matched_qc_80,
        "pct_qc_lupin_matched_50pct": pct_qc_50
    }
    
    out_json = out_base / "cross_sorter_agreement.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(agreement_res, f, indent=2)
        
    logger.info(f"Saved cross-sorter agreement to {out_json}:")
    logger.info(f"  Matched >= 50%: {matched_50} units ({pct_lup_50:.1f}% of all Lupin units)")
    logger.info(f"  Matched >= 80%: {matched_80} units ({pct_lup_80:.1f}% of all Lupin units)")
    logger.info(f"  QC-passed matched >= 50%: {matched_qc_50} units ({pct_qc_50:.1f}% of QC Lupin units)")

if __name__ == '__main__':
    main()
