#!/usr/bin/env python3
"""
Exploration and Canonical Resolution of 6-State and 7-State Poisson HMMs
on the Intact 420.0s Window (e097, 120 Top QC Units, 959 Breaths).
"""
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from hmmlearn import hmm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.e097.run_intact_window_dual_sorting_and_hmm_pipeline import (
    INTACT_T_START, INTACT_DURATION, extract_intact_breaths, load_intact_spikes_and_qc
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EXPLORE_HMM_ORDERS")

def fit_and_characterize(spike_mat, n_states, dt=0.010, breath_data=None):
    n_bins, n_units = spike_mat.shape
    logger.info(f"Fitting {n_states}-state Poisson HMM on {n_bins} bins x {n_units} units...")
    
    model = hmm.PoissonHMM(n_components=n_states, n_iter=120, tol=1e-3, random_state=42, verbose=False)
    model.fit(spike_mat)
    
    state_seq = model.predict(spike_mat)
    post_probs = model.predict_proba(spike_mat)
    ll = float(model.score(spike_mat))
    
    # AIC and BIC
    # Poisson HMM parameters:
    # transition matrix: n_states * (n_states - 1)
    # initial probabilities: n_states - 1
    # emission parameters: n_states * n_units
    n_params = n_states * (n_states - 1) + (n_states - 1) + n_states * n_units
    aic = 2 * n_params - 2 * ll
    bic = n_params * np.log(n_bins) - 2 * ll
    
    rates_hz = model.lambdas_ / dt # (n_states, n_units)
    mean_rates = rates_hz.mean(axis=1) # (n_states,)
    occupancy_pct = np.array([np.mean(state_seq == s) * 100.0 for s in range(n_states)])
    
    # Temporal alignment relative to I_off and I_on
    state_on_pref = np.zeros(n_states)
    state_off_pref = np.zeros(n_states)
    
    if breath_data is not None:
        valid_on = breath_data["valid_onsets"]
        valid_off = breath_data["valid_offsets"]
        
        # Check occupancy in [-50ms, +50ms] of I_on
        on_bins = []
        for t in valid_on:
            b = int(t / dt)
            if 0 <= b < n_bins:
                on_bins.extend(range(max(0, b - 5), min(n_bins, b + 5)))
        on_bins = np.unique(on_bins)
        
        # Check occupancy in [-50ms, +50ms] of I_off
        off_bins = []
        for t in valid_off:
            b = int(t / dt)
            if 0 <= b < n_bins:
                off_bins.extend(range(max(0, b - 5), min(n_bins, b + 5)))
        off_bins = np.unique(off_bins)
        
        for s in range(n_states):
            state_on_pref[s] = np.mean(state_seq[on_bins] == s) * 100.0
            state_off_pref[s] = np.mean(state_seq[off_bins] == s) * 100.0
            
    return {
        "n_states": n_states,
        "log_likelihood": ll,
        "aic": aic,
        "bic": bic,
        "n_params": n_params,
        "mean_rates_hz": mean_rates.tolist(),
        "occupancy_pct": occupancy_pct.tolist(),
        "state_on_pref": state_on_pref.tolist(),
        "state_off_pref": state_off_pref.tolist(),
        "transmat": model.transmat_.tolist(),
        "model": model,
        "state_seq": state_seq,
        "lambdas": model.lambdas_
    }

def main():
    ks4_dir = REPO_ROOT / "outputs" / "e097" / "intact_window" / "kilosort4"
    nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
    
    logger.info("Loading spikes and extracting breaths...")
    spk_times, spk_clusters, depths, df_qc = load_intact_spikes_and_qc(ks4_dir, max_units=120)
    breath_data = extract_intact_breaths(nidq_file, start_t=INTACT_T_START, dur=INTACT_DURATION)
    
    dt = 0.010
    time_bins = np.arange(0, INTACT_DURATION + dt, dt)
    n_bins = len(time_bins) - 1
    cids = df_qc["cluster_id"].values
    n_units = len(cids)
    
    spike_mat = np.zeros((n_bins, n_units), dtype=np.int32)
    for u_idx, cid in enumerate(cids):
        ts = spk_times[spk_clusters == cid]
        ts_valid = ts[(ts >= 0) & (ts <= INTACT_DURATION)]
        counts, _ = np.histogram(ts_valid, bins=time_bins)
        spike_mat[:, u_idx] = counts

    results = {}
    for k in [5, 6, 7]:
        res = fit_and_characterize(spike_mat, n_states=k, dt=dt, breath_data=breath_data)
        results[k] = res
        logger.info(f"\n--- {k}-State Model ---")
        logger.info(f"Log-Likelihood: {res['log_likelihood']:.1f}, AIC: {res['aic']:.1f}, BIC: {res['bic']:.1f}")
        for s in range(k):
            logger.info(f"  Raw State {s}: Rate = {res['mean_rates_hz'][s]:.1f} Hz, Occ = {res['occupancy_pct'][s]:.1f}%, On-Pref = {res['state_on_pref'][s]:.1f}%, Off-Pref = {res['state_off_pref'][s]:.1f}%")
        
    out_json = REPO_ROOT / "outputs" / "e097" / "intact_window" / "hmm_model_order_comparison.json"
    clean_res = {
        str(k): {
            "n_states": v["n_states"],
            "log_likelihood": v["log_likelihood"],
            "aic": v["aic"],
            "bic": v["bic"],
            "n_params": v["n_params"],
            "mean_rates_hz": v["mean_rates_hz"],
            "occupancy_pct": v["occupancy_pct"],
            "state_on_pref": v["state_on_pref"],
            "state_off_pref": v["state_off_pref"],
            "transmat": v["transmat"]
        } for k, v in results.items()
    }
    with open(out_json, "w") as f:
        json.dump(clean_res, f, indent=2)
    logger.info(f"Saved model order comparison to {out_json}")

if __name__ == '__main__':
    main()
