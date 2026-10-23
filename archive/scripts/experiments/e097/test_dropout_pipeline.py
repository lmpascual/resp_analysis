"""Quick test of drop-out and subsampling pipeline components."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.metrics import adjusted_rand_score, f1_score, confusion_matrix, mutual_info_score
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(r"c:\Users\lpascu\resp_analysis")
base = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / "win1_baseline" / "kilosort4"
spk_times = np.load(base / "alf" / "spike_times.npy")
spk_clusters = np.load(base / "alf" / "spike_clusters.npy")
qc_df = pd.read_csv(base / "kilosort4_qc_summary.csv")
qc_df = qc_df[qc_df["passed_allen_qc"] == True].sort_values(by="depth_um", ascending=False).head(120)
cids = qc_df["cluster_id"].values
depths = qc_df["depth_um"].values

dt = 0.010
dur = 60.0
time_bins = np.arange(0, dur + dt, dt)
counts = np.zeros((len(time_bins)-1, len(cids)), dtype=np.int32)
for i, cid in enumerate(cids):
    c_spk = spk_times[spk_clusters == cid]
    counts[:, i] = np.histogram(c_spk, bins=time_bins)[0]

pop_dir = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / "win1_baseline" / "population_dynamics"
true_states = np.load(pop_dir / "kilosort4_consolidated_states.npy")

print(f"Counts: {counts.shape}, True states: {true_states.shape}, unique: {np.unique(true_states)}")

def align_hungarian(pred, true, n_states=5):
    C = confusion_matrix(pred, true, labels=list(range(n_states)))
    row_ind, col_ind = linear_sum_assignment(-C)
    remap = {r: c for r, c in zip(row_ind, col_ind)}
    aligned = np.array([remap.get(s, s) for s in pred])
    ari = adjusted_rand_score(true, aligned)
    f1_m = f1_score(true, aligned, average="macro")
    f1_per = f1_score(true, aligned, average=None, labels=list(range(n_states)))
    return aligned, ari, f1_m, f1_per

# Test subsampling with N=30, 10, 5
for N in [60, 30, 15, 5]:
    subset = np.random.choice(counts.shape[1], N, replace=False)
    m = hmm.PoissonHMM(n_components=5, n_iter=40, random_state=42, tol=1e-3)
    m.fit(counts[:, subset])
    p_states = m.predict(counts[:, subset])
    aligned, ari, f1_m, f1_per = align_hungarian(p_states, true_states)
    print(f"N={N:2d} -> ARI: {ari:.3f}, Macro F1: {f1_m:.3f}, Per-State F1: {np.round(f1_per, 2)}")
