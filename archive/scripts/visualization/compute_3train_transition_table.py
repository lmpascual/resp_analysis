#!/usr/bin/env python3
"""
Compute 3-Train Eupnea-to-Gasping Transition Statistics Table across Laser 1, Laser 2, and Laser 3
Strictly aligned to physiological Diaphragm I_off and using true Diaphragm Gasp & Breakthrough classification.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\lpascu\analysis_src")
sys.path.insert(0, str(PROJECT_ROOT))
import src.pipeline.compat_patch

from src.pipeline.processors.proc_diaphragm_transition import (
    extract_diaphragm_envelope,
    segment_diaphragm_breaths,
    build_train_transition_data
)

print("Computing 3-Train Transition Statistics Table (Diaphragm I_off Aligned)...", flush=True)

# 1. Load NIDAQ Telemetry
nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
fs = 10000.0; n_ch = 9; start_t = 796.827; dur = 226.243
raw = np.memmap(nidq_file, dtype="int16", mode="r")
slice_data = raw[int(start_t*fs)*n_ch : int((start_t+dur)*fs)*n_ch].reshape(-1, n_ch)
emg_v = slice_data[:, 0].astype(np.float32) * (10.0 / 32768.0)
flow_v = slice_data[:, 5].astype(np.float32) * (10.0 / 32768.0)
t_vec = np.arange(len(emg_v)) / fs

# 2. Extract Diaphragm Landmarking
env = extract_diaphragm_envelope(emg_v, fs=fs)
t_on, t_off, t_pk, pks = segment_diaphragm_breaths(env, t_vec, fs=fs)

td_l1 = build_train_transition_data("laser1", "Laser 1", 60.0, 2.0, 4.0, 12.0, 67.2, t_vec, emg_v, flow_v, env, t_on, t_off, t_pk, fs=fs)
td_l2 = build_train_transition_data("laser2", "Laser 2", 120.0, 2.0, 4.0, 10.0, 125.5, t_vec, emg_v, flow_v, env, t_on, t_off, t_pk, fs=fs)
td_l3 = build_train_transition_data("laser3", "Laser 3", 164.0, 2.5, 4.0, 11.0, 170.2, t_vec, emg_v, flow_v, env, t_on, t_off, t_pk, fs=fs)

train_datas = [td_l1, td_l2, td_l3]

# 3. Load Spikes
cons_dir = PROJECT_ROOT / "output_e097_prelaser_through_laser3_226s_motion_corrected" / "consensus_master"
spk_times = np.load(cons_dir / "alf" / "spike_times.npy")
spk_clusters = np.load(cons_dir / "alf" / "spike_clusters.npy")
df_summary = pd.read_csv(cons_dir / "03_spiking_statistics_literature_cards_neg50_pos100" / "spiking_statistics_lit_summary_neg50_pos100.csv")
unique_cids = sorted(list(np.unique(spk_clusters)))

win_dur_s = 0.600  # [-300ms, +300ms]
records = []
eps = 1e-5

for cid in unique_cids:
    c_spks = spk_times[spk_clusters == cid]
    total_spks = len(c_spks)
    match = df_summary[df_summary["cluster_id"] == cid]
    if len(match) > 0:
        row_m = match.iloc[0]
        depth = float(row_m["depth_um"])
        cell_type = str(row_m["cell_type"])
        is_qc = bool(row_m["is_qc_passed"])
        r_schreiber = float(row_m.get("r_schreiber_ep1_10ms", np.nan))
    else:
        depth = 0.0
        cell_type = "Unclassified"
        is_qc = False
        r_schreiber = np.nan

    rec = {
        "cluster_id": cid,
        "depth_um": depth,
        "cell_type": cell_type,
        "passed_allen_qc": is_qc,
        "total_spikes": total_spks,
        "r_schreiber_ep1": round(r_schreiber, 3) if not np.isnan(r_schreiber) else np.nan
    }

    gmi_list = []
    lmi_list = []

    for td in train_datas:
        k = td.train_key
        # Trials by state
        trs_eup = [tr for tr in td.trials if tr.state == "PRE_EUPNEA"]
        trs_las = [tr for tr in td.trials if tr.state == "LASER_BREAKTHROUGH"]
        trs_gasp = [tr for tr in td.trials if tr.state == "DIAPHRAGM_GASP"]
        trs_post = [tr for tr in td.trials if tr.state == "POST_EUPNEA"]

        spk_eup = [len(c_spks[(c_spks >= tr.t_off - 0.30) & (c_spks <= tr.t_off + 0.30)]) for tr in trs_eup]
        spk_las = [len(c_spks[(c_spks >= tr.t_off - 0.30) & (c_spks <= tr.t_off + 0.30)]) for tr in trs_las]
        spk_gasp = [len(c_spks[(c_spks >= tr.t_off - 0.30) & (c_spks <= tr.t_off + 0.30)]) for tr in trs_gasp]
        spk_post = [len(c_spks[(c_spks >= tr.t_off - 0.30) & (c_spks <= tr.t_off + 0.30)]) for tr in trs_post]

        fr_eup = (np.mean(spk_eup) / win_dur_s) if len(spk_eup) > 0 else 0.0
        fr_las = (np.mean(spk_las) / win_dur_s) if len(spk_las) > 0 else 0.0
        fr_gasp = (np.mean(spk_gasp) / win_dur_s) if len(spk_gasp) > 0 else 0.0
        fr_post = (np.mean(spk_post) / win_dur_s) if len(spk_post) > 0 else 0.0

        gmi = float((fr_gasp - fr_eup) / (fr_gasp + fr_eup + eps))
        lmi = float((fr_las - fr_eup) / (fr_las + fr_eup + eps))
        rec_idx = float((fr_post - fr_eup) / (fr_post + fr_eup + eps))

        rec[f"fr_pre_{k}"] = round(fr_eup, 2)
        rec[f"fr_las_{k}"] = round(fr_las, 2)
        rec[f"fr_gasp_{k}"] = round(fr_gasp, 2)
        rec[f"fr_post_{k}"] = round(fr_post, 2)
        rec[f"gmi_{k}"] = round(gmi, 3)
        rec[f"lmi_{k}"] = round(lmi, 3)
        rec[f"rec_{k}"] = round(rec_idx, 3)
        rec[f"mean_spk_gasp_{k}"] = round(float(np.mean(spk_gasp)), 2) if len(spk_gasp) > 0 else 0.0

        gmi_list.append(gmi)
        lmi_list.append(lmi)

    rec["mean_gmi"] = round(float(np.mean(gmi_list)), 3)
    rec["mean_lmi"] = round(float(np.mean(lmi_list)), 3)
    rec["is_gasp_activated"] = bool(rec["mean_gmi"] >= 0.25)
    rec["is_laser_breakthrough_active"] = bool(rec["mean_lmi"] >= 0.20 or any(rec[f"fr_las_{td.train_key}"] >= 5.0 for td in train_datas))

    # Multi-Train Progression Classification
    g1, g2, g3 = rec["gmi_laser1"], rec["gmi_laser2"], rec["gmi_laser3"]
    if g1 > 0.2 and g2 > 0.2 and g3 > 0.2:
        prog = "Persistent Gasp Active"
    elif g3 > g1 + 0.25 and g3 > 0.2:
        prog = "Sensitized Gasp Active"
    elif g1 > g3 + 0.25 and g1 > 0.2:
        prog = "Habituated Gasp Active"
    elif rec["is_laser_breakthrough_active"] and not rec["is_gasp_activated"]:
        prog = "Laser Breakthrough Dominant"
    elif rec["mean_gmi"] <= -0.25:
        prog = "Gasp Suppressed"
    else:
        prog = "Stable / Unmodulated"

    rec["multi_train_progression"] = prog
    records.append(rec)

df_out = pd.DataFrame(records)
# Compatibility alias for laser1/2/3 keys
for old_k, new_k in [("l1", "laser1"), ("l2", "laser2"), ("l3", "laser3")]:
    for col_prefix in ["fr_pre", "fr_las", "fr_gasp", "fr_post", "gmi", "lmi", "rec", "mean_spk_gasp"]:
        df_out[f"{col_prefix}_{old_k}"] = df_out[f"{col_prefix}_{new_k}"]
        if col_prefix == "rec":
            df_out[f"rec_idx_{old_k}"] = df_out[f"rec_{new_k}"]

out_csv = cons_dir / "04_gasping_transition_rasters" / "eupnea_gasp_transition_statistics_all_trains.csv"
df_out.to_csv(out_csv, index=False)
print(f"Saved updated statistics table for all {len(df_out)} units to {out_csv}")
print("Summary Statistics:")
print(f"  Gasp Activated Units (mean GMI >= 0.25): {df_out['is_gasp_activated'].sum()}/{len(df_out)}")
print(f"  Laser Breakthrough Active Units: {df_out['is_laser_breakthrough_active'].sum()}/{len(df_out)}")
print("Progression Breakdown:")
print(df_out['multi_train_progression'].value_counts())

