"""Population dynamics of reviewed single units across eupnea, anoxia, gasping and recovery (e097 anoxia #1).

The defaults read the 2026-10-20 snapshot of the intact_window_v2 KS4 spikes (outputs/e097/population_dynamics_v2/
snapshot, with the redo's removal of #435's 6 stray spikes applied). On 2026-10-21 a run on the finished redo's
kilosort4/alf (--spikes-dir/--units/--review pointing there, --removed '') gave identical results; use that form after
any later re-curation.

Run in the `cibrrig` env (hmmlearn + Nick Bush's cibrrig, so breaths, phase and PCA preprocessing are the
Bush & Ramirez code itself):

    C:/Users/lpascu/AppData/Local/miniforge3/envs/cibrrig/python.exe population_dynamics.py ^
        --out C:/Users/lpascu/resp_analysis/outputs/e097/population_dynamics_v2

Breaths and phase (Bush & Ramirez 2024 Nat Neurosci; Bush, Oliveira, Glovak & Ramirez 2025): diaphragm EMG
-> cibrrig `nidq_utils.filt_int_ds_dia` (EKG removal, 50 ms median filter of the rectified EMG, find_peaks /
peak_widths) -> onsets and offsets -> `physiology.compute_dia_phase`: phi = 0 at inspiration onset, pi at
inspiration offset, -pi just after offset, linear in between (1 ms resolution). Gasps: 2024 rule, breath-to-
breath interval median-filtered over 7 breaths, gasping from > 1 s until < 0.85 s.

Conditions (per breath cycle, by the onset that starts it): by default baseline, anoxia, recovery and post-recovery
from the config epochs; with --conditions-file, the spans in that epochs file (epoch_change_points.py splits the
recovery at change points in breathing and population state). Gasping = the Bush-rule gasp cycles in both cases.
Cycles outside every condition (e.g. between anoxia and the first gasp) belong to no condition but still enter the
HMM fit, which doesn't depend on the conditions at all.

Single units: phase curves with cibrrig `singlecell.get_phase_curve` (spikes / time per phase bin, per breath),
modulation and preferred phase with `extract_resp_modulation.get_vector_means` (vector-mean length L and angle).
2025 classes: tonic if L < 0.2, otherwise inspiratory if the preferred phase >= 0, else expiratory.

Poisson HMM, directly on rate vectors (no PCA): spike counts in HMM_BIN_S bins, Gaussian-smoothed with
--hmm-sigma-ms, fitted with hmmlearn's PoissonHMM over the whole window for K = 3..8 states (best of
--n-restarts, emission rates initialised by k-means). hmmlearn scores emissions with scipy's integer Poisson pmf,
so the smoothed (non-integer) counts use the continuous Poisson log-likelihood x log(lambda) - lambda -
lgamma(x + 1); the M-step is unchanged.

State colours across K: every model is matched one-to-one to the reference model (--ref-k, default 6: K = 8 has
redundant states) with the Hungarian method on a similarity that adds the correlation of sqrt emission rates, the
cosine of the states' P(state | phi) profiles in each condition, and a penalty for a different mean rate. So a colour
means "most like this state of the reference model". States without a partner (larger models), or whose match scores
< MATCH_MIN_SIM, are drawn in grey. The reference states keep the colours they get when matched to the largest
model (coloured in order of phase), so colours don't change with the choice of reference. Per-state figures (state
cards, state strips over PETHs) use the reference model. Functional labels (conditions listed are those where the state
takes >= 2 % of the time, as runs in time order): where the state sits in the breath cycle (vector
mean of P(state | phi) over all conditions; "not phase-locked" if its length is < 0.25) and the conditions it occurs
in (listed when it is absent from some; "all but gasping" when only gasping lacks it).

PCA as Bush & Ramirez: 5 ms counts, Gaussian sigma 10 ms, square root, PCA fitted on the baseline epoch and applied
to the whole window; speed = Euclidean step in PC1-3. The inspiration-off point (2025) is the mean PC position in the
bin just after each baseline inspiration offset.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.special import gammaln

# cibrrig was written against older numpy / scikit-learn: alias np.trapz and accept with_mean=0
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import sklearn.preprocessing as _skp  # noqa: E402


class _StandardScaler(_skp.StandardScaler):
    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        super().__init__(copy=copy, with_mean=bool(with_mean), with_std=bool(with_std))


_skp.StandardScaler = _StandardScaler

from cibrrig.analysis import singlecell  # noqa: E402
from cibrrig.postprocess.extract_resp_modulation import get_vector_means  # noqa: E402
from cibrrig.preprocess import nidq_utils, physiology  # noqa: E402
from hmmlearn import hmm  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

log = logging.getLogger("population_dynamics")

OUT = Path("C:/Users/lpascu/resp_analysis/outputs/e097")
HMM_BIN_S = 0.005
PCA_BIN_S, PCA_SIGMA_S = 0.005, 0.010        # Bush & Ramirez 2024/2025
PHASE_BINS = 36                             # HMM state profiles over phi
CURVE_BINS = 50                             # single-unit phase curves
TONIC_MAX_L = 0.2                           # 2025 classification
GASP_ON_S, GASP_OFF_S, GASP_MEDIAN_N = 1.0, 0.85, 7
NIDQ_FS = 10000.0
CONDITIONS = [("baseline", "baseline"), ("anoxia", "anoxia"), ("gasping", None), ("recovery", "recovery"),
              ("post-recovery", "active_expiration")]   # (name, config epoch; gasping = Bush-rule gasps)
MATCH_MIN_SIM = 1.5                         # of 2 (+ mean-rate penalty); below it a state stays unmatched (grey)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
           "#9a9893", "#52514e", "#c3c2b7", "#77756f", "#2f2e2b"]       # greys: unmatched states
N_CATEGORICAL = 8
COND_COLOURS = {"baseline": "#2a78d6", "anoxia": "#eda100", "gasping": "#e34948", "recovery": "#1baf7a",
                "post-recovery": "#6250d6"}
# fallback colours for conditions from an epochs file that names none (the e097 file sets its own: gray baseline,
# reds for anoxia/gasping, greens for recovery, yellow for slow breathing; see epoch_change_points.COLOURS)
COND_PALETTE = ["#6a6a66", "#f2918a", "#d93636", "#77c06a", "#1c9157", "#eda100", "#0f6b3a"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    snap = OUT / "population_dynamics_v2" / "snapshot"
    p.add_argument("--spikes-dir", default=str(snap), help="folder with spike_times.npy (s) and spike_clusters.npy")
    p.add_argument("--units", default=str(snap / "units.tsv"))
    p.add_argument("--review", default=str(snap / "kilosort4_unit_review_snapshot.tsv"))
    p.add_argument("--removed", default=str(snap / "removed_spikes_redo_20261020.tsv"),
                   help="removed_spikes.tsv to apply ('' for none)")
    p.add_argument("--config", default=str(snap / "config_used.yaml"))
    p.add_argument("--nidq-cache", default=str(OUT / "intact_window" / "intact_nidq_cache.npz"),
                   help="npz with emg_v: raw diaphragm EMG at 10 kHz from the start of the window")
    p.add_argument("--time-offset-s", type=float, default=1740.0)
    p.add_argument("--conditions-file", default=None,
                   help="epochs JSON from epoch_change_points.py (raw-run time); default: config epochs")
    p.add_argument("--hmm-sigma-ms", type=float, default=5.0)
    p.add_argument("--k-min", type=int, default=3)
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--ref-k", type=int, default=6,
                   help="reference model: colours are matched to it, and per-state figures use it")
    p.add_argument("--n-restarts", type=int, default=10)
    p.add_argument("--n-iter", type=int, default=300)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--figures-only", action="store_true", help="redraw figures from <out>/cache/ctx.pkl")
    p.add_argument("--out", required=True)
    return p.parse_args()


# ----------------------------------------------------------------------------------------------- data

def load_units(a):
    st = np.load(Path(a.spikes_dir) / "spike_times.npy")
    sc = np.load(Path(a.spikes_dir) / "spike_clusters.npy")
    review = pd.read_csv(a.review, sep="\t")
    units = pd.read_csv(a.units, sep="\t").set_index("unit_id")
    ids = np.sort(review.loc[review["verdict"] == "single", "unit_id"].to_numpy())
    removed = pd.read_csv(a.removed, sep="\t") if a.removed else pd.DataFrame(columns=["unit_id", "time_s"])
    spikes = {}
    for u in ids:
        t = np.sort(st[sc == u])
        rm = removed.loc[removed["unit_id"] == u, "time_s"].to_numpy()
        if rm.size:
            t = t[np.abs(t[:, None] - rm[None, :]).min(axis=1) > 1e-4]
        spikes[int(u)] = t
    info = units.loc[ids, ["label", "depth_um", "peak_channel_index"]].reset_index()
    info["n_spikes"] = [spikes[int(u)].size for u in ids]
    return spikes, info


def breaths_and_phase(cache: str, duration: float):
    emg = np.load(cache)["emg_v"].astype(np.float64)
    dia_df, dia_sub, sr_sub, _, _, _ = nidq_utils.filt_int_ds_dia(emg - emg.mean(), NIDQ_FS, ds_factor=10)
    dia_df = dia_df.reset_index(drop=True)
    ons, offs = dia_df["on_sec"].to_numpy(), dia_df["off_sec"].to_numpy()
    phi_t, phi = physiology.compute_dia_phase(ons, offs, t_start=0, t_stop=duration, dt=1e-3)
    # 2024 gasp rule on breath-to-breath intervals
    ibi = np.r_[np.diff(ons), np.nan]
    med = pd.Series(ibi).rolling(GASP_MEDIAN_N, center=True, min_periods=1).median().to_numpy()
    gasp, on = np.zeros(ons.size, bool), False
    for i, v in enumerate(med):
        if not on and v > GASP_ON_S:
            on = True
        elif on and v < GASP_OFF_S:
            on = False
        gasp[i] = on
    gasp[-1] = False
    dia_df["ibi_s"], dia_df["ibi_median7_s"], dia_df["gasp"] = ibi, med, gasp
    return dia_df, phi_t, phi, dia_sub, sr_sub


def set_conditions(path, offset_s):
    """Replace the default conditions with those of an epochs file (epoch_change_points.py): list of {name,
    start_raw_s, end_raw_s} or {name, rule: gasp_flag}; entries with include: false are left unassigned. Spans are
    converted to file time (CONDITIONS then holds (name, (start_s, end_s)) or (name, None) for the gasp rule)."""
    global CONDITIONS, COND_COLOURS
    spec = json.loads(Path(path).read_text(encoding="utf-8"))["conditions"]
    conds = [c for c in spec if c.get("include", True)]
    CONDITIONS = [(c["name"], None if c.get("rule") == "gasp_flag" else
                   (c["start_raw_s"] - offset_s, c["end_raw_s"] - offset_s)) for c in conds]
    COND_COLOURS = {c["name"]: c.get("colour", COND_PALETTE[i % len(COND_PALETTE)]) for i, c in enumerate(conds)}
    return spec


def assign_conditions(dia_df, epochs):
    """Condition of each breath cycle (onset i to onset i+1). A condition's span is a config epoch name or a
    (start_s, end_s) pair in file time; None means the Bush-rule gasp cycles."""
    ons = dia_df["on_sec"].to_numpy()
    cond = np.full(ons.size, "", dtype=object)
    for name, ep in CONDITIONS:
        if ep is None:
            cond[dia_df["gasp"].to_numpy()] = name
        else:
            t0, t1 = epochs[ep] if isinstance(ep, str) else ep
            cond[(ons >= t0) & (ons < t1) & ~dia_df["gasp"].to_numpy()] = name
    cond[-1] = ""
    return cond


def bin_counts(spikes, ids, dt, n_bins):
    out = np.zeros((n_bins, len(ids)), dtype=np.float64)
    for j, u in enumerate(ids):
        k = np.floor(spikes[u] / dt).astype(np.int64)
        k = k[(k >= 0) & (k < n_bins)]
        out[:, j] = np.bincount(k, minlength=n_bins)
    return out


def bin_labels(dia_df, cond, phi_t, phi, dt, n_bins):
    """Per bin: phase (NaN outside the first..last onset), cycle index, condition."""
    centres = (np.arange(n_bins) + 0.5) * dt
    ons = dia_df["on_sec"].to_numpy()
    ph = phi[np.clip(np.searchsorted(phi_t, centres), 0, phi.size - 1)].astype(np.float64)
    cyc = np.searchsorted(ons, centres, side="right") - 1
    valid = (cyc >= 0) & (cyc < ons.size - 1)
    ph[~valid] = np.nan
    bcond = np.where(valid, cond[np.clip(cyc, 0, ons.size - 1)], "")
    return centres, ph, cyc, bcond.astype(str)


# ------------------------------------------------------------------------------------------ single units

def phase_tuning(spikes, ids, dia_df, cond, phi_t, phi):
    """Per condition: phase curves (bins x units), preferred phase, modulation L, class and mean rate."""
    curves, rows = {}, []
    for name, _ in CONDITIONS:
        idx = np.flatnonzero(cond == name)
        if idx.size < 2:
            continue
        sub = dia_df.iloc[idx.min():idx.max() + 2]        # consecutive cycles plus the closing onset
        rates = np.full((CURVE_BINS, len(ids)), np.nan)
        for j, u in enumerate(ids):
            ts = spikes[u]
            ts = ts[(ts >= sub["on_sec"].iloc[0]) & (ts < sub["on_sec"].iloc[-1])]
            pc = singlecell.get_phase_curve(ts, sub, phi_t=phi_t, phi=phi, nbins=CURVE_BINS)
            rates[:, j] = pc["rate_mean"]
            bins = pc["bins"]
        theta, L = get_vector_means(bins, np.nan_to_num(rates))
        curves[name] = rates
        t0, t1 = sub["on_sec"].iloc[0], sub["on_sec"].iloc[-1]
        for j, u in enumerate(ids):
            n = int(((spikes[u] >= t0) & (spikes[u] < t1)).sum())
            cls = "tonic" if not L[j] >= TONIC_MAX_L else ("inspiratory" if theta[j] >= 0 else "expiratory")
            rows.append({"unit_id": u, "condition": name, "n_cycles": int(idx.size), "rate_hz": n / (t1 - t0),
                         "preferred_phase": theta[j], "modulation_L": L[j], "class": cls})
    return bins, curves, pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ HMM

class RatePoissonHMM(hmm.PoissonHMM):
    """Poisson emissions on non-negative real-valued (smoothed) counts."""

    def _compute_log_likelihood(self, X):
        lam = np.maximum(self.lambdas_, 1e-8)
        return X @ np.log(lam).T - lam.sum(axis=1)[None, :] - gammaln(X + 1).sum(axis=1, keepdims=True)

    def _compute_likelihood(self, X):
        return np.exp(self._compute_log_likelihood(X))

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        self.lambdas_ = np.maximum(self.lambdas_, 1e-8)


def fit_one(X, k, seed, n_iter):
    """One EM run, emission rates initialised from k-means on a random 20k-bin subset."""
    rng = np.random.default_rng(seed)
    sub = X[rng.choice(X.shape[0], min(20000, X.shape[0]), replace=False)]
    m = RatePoissonHMM(n_components=k, n_iter=n_iter, tol=0.1, random_state=seed, init_params="st")
    m.lambdas_ = np.maximum(KMeans(k, n_init=1, random_state=seed).fit(sub).cluster_centers_, 1e-3)
    m.fit(X)
    return k, seed, m, m.score(X)


def fit_hmms(X, a):
    jobs = [(k, s) for k in range(a.k_min, a.k_max + 1) for s in range(a.n_restarts)]
    res = Parallel(n_jobs=a.n_jobs)(delayed(fit_one)(X, k, s, a.n_iter) for k, s in jobs)
    best, restarts = {}, []
    for k, s, m, ll in res:
        restarts.append({"k": k, "seed": s, "log_likelihood": ll, "converged": bool(m.monitor_.converged),
                         "n_iter": int(m.monitor_.iter)})
        if k not in best or ll > best[k][1]:
            best[k] = (m, ll)
    return {k: v[0] for k, v in best.items()}, pd.DataFrame(restarts)


def circ_vector(profile, centres):
    z = np.sum(profile * np.exp(1j * centres)) / max(np.sum(profile), 1e-12)
    return float(np.angle(z)), float(np.abs(z))


def state_features(m, X, ph, bcond):
    """Emission rates, occupancy over phi per condition and over conditions, dwell time, labels."""
    post = m.predict_proba(X)
    seq = m.predict(X)
    k = m.n_components
    edges = np.linspace(-np.pi, np.pi, PHASE_BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    pb = np.clip(np.digitize(ph, edges) - 1, 0, PHASE_BINS - 1)
    prof = {}
    for name in [n for n, _ in CONDITIONS] + ["all"]:
        sel = ((bcond == name) if name != "all" else (bcond != "")) & np.isfinite(ph)
        P = np.full((PHASE_BINS, k), np.nan)
        for b in range(PHASE_BINS):
            s = sel & (pb == b)
            if s.any():
                P[b] = post[s].mean(axis=0)
        prof[name] = P
    occ_cond = np.array([[post[bcond == name, s].mean() if (bcond == name).any() else np.nan
                          for name, _ in CONDITIONS] for s in range(k)])
    occ_all = post.mean(axis=0)
    change = np.flatnonzero(np.diff(seq)) + 1
    bounds = np.r_[0, change, seq.size]
    run_state, run_len = seq[bounds[:-1]], np.diff(bounds)
    dwell = np.array([run_len[run_state == s].mean() * HMM_BIN_S * 1e3 if (run_state == s).any() else np.nan
                      for s in range(k)])
    rows = []
    for s in range(k):
        th, L = circ_vector(np.nan_to_num(prof["all"][:, s]), centres)
        enr = occ_cond[s] / max(occ_all[s], 1e-12)
        rows.append({"state": s, "rate_mean_hz": m.lambdas_[s].mean() / HMM_BIN_S, "occupancy": occ_all[s],
                     "dwell_ms": dwell[s], "phase": th, "phase_locking": L,
                     **{f"occ_{n}": occ_cond[s, i] for i, (n, _) in enumerate(CONDITIONS)},
                     **{f"enrich_{n}": enr[i] for i, (n, _) in enumerate(CONDITIONS)}})
    feats = pd.DataFrame(rows)
    feats["label"] = [functional_label(r) for _, r in feats.iterrows()]
    return {"post": post, "seq": seq, "profiles": prof, "phase_centres": centres, "feats": feats,
            "rates_hz": m.lambdas_ / HMM_BIN_S}


def functional_label(r):
    """Where in the breath cycle (P(state | phi) over all conditions) and in which conditions the state occurs."""
    if r["phase_locking"] >= 0.25:
        th = r["phase"]
        part = ("early I" if th < np.pi / 2 else "late I") if th >= 0 else \
            ("post-I / early E" if th < -np.pi / 2 else "late E / pre-I")
    else:
        part = "not phase-locked"
    names = [n for n, _ in CONDITIONS]
    present = [n for n in names if r[f"occ_{n}"] >= 0.02]
    if len(present) == len(names) or not present:
        return part
    if "gasping" in names and set(present) == set(names) - {"gasping"}:
        return part + " | all but gasping"
    return part + " | " + condition_span_text(present, names)


def condition_span_text(sel, names):
    """Conditions as runs in time order: ['baseline', 'anoxia', 'early recovery'] -> 'baseline–anoxia, early
    recovery'."""
    idx = sorted(names.index(x) for x in sel)
    runs, start = [], idx[0]
    for a, b in zip(idx, idx[1:] + [None]):
        if b != a + 1:
            runs.append(names[start] if start == a else f"{names[start]}–{names[a]}")
            start = b
    return ", ".join(runs)


def occupancy_vector(f, s):
    """P(state | phi) in each condition, one block per condition (when in the cycle and in which condition)."""
    return np.concatenate([np.nan_to_num(f["profiles"][n][:, s]) for n, _ in CONDITIONS]) / np.sqrt(PHASE_BINS)


def similarity(fa, fb):
    ka, kb = len(fa["feats"]), len(fb["feats"])
    S = np.zeros((ka, kb))
    for i in range(ka):
        oa = occupancy_vector(fa, i)
        for j in range(kb):
            ob = occupancy_vector(fb, j)
            em = np.corrcoef(np.sqrt(fa["rates_hz"][i]), np.sqrt(fb["rates_hz"][j]))[0, 1]
            oc = oa @ ob / max(np.linalg.norm(oa) * np.linalg.norm(ob), 1e-12)
            mag = abs(np.log2(fa["rates_hz"][i].mean() / fb["rates_hz"][j].mean()))
            S[i, j] = em + oc - 0.5 * mag
    return S


def match_colours(F, ks, ref):
    """Colour index per state per model. Every model is matched one-to-one to the reference model `ref`; a state
    without a partner, or whose match scores < MATCH_MIN_SIM, gets a grey (index N_CATEGORICAL and up). The
    reference states take the colours they get when matched to the largest model, whose states are coloured in order
    of phase, so the colours stay the same whichever K is the reference. (Chaining K -> K+1 instead lets a colour
    drift between functions.)"""
    big = ks[-1]
    big_col = np.empty(big, int)
    big_col[np.argsort(F[big]["feats"]["phase"].to_numpy())] = np.arange(big)
    if ref == big:
        ref_col = big_col
    else:
        S = similarity(F[ref], F[big])
        r, c = linear_sum_assignment(-S)
        ref_col = np.empty(ref, int)
        ref_col[r] = big_col[c]
    colours, sims = {ref: ref_col}, {}
    for k in ks:
        if k == ref:
            continue
        S = similarity(F[k], F[ref])
        r, c = linear_sum_assignment(-S)
        col = np.full(k, -1)
        col[r] = ref_col[c]
        col[r[S[r, c] < MATCH_MIN_SIM]] = -1
        unmatched = np.flatnonzero(col < 0)
        col[unmatched] = N_CATEGORICAL + np.arange(unmatched.size)
        colours[k] = col
        sims[k] = (S, r, c)
    return colours, sims


# ------------------------------------------------------------------------------------------------ PCA

def pca_projection(spikes, ids, n_bins, fit_mask, n_comp=10):
    counts = bin_counts(spikes, ids, PCA_BIN_S, n_bins)
    Z = np.sqrt(gaussian_filter1d(counts, PCA_SIGMA_S / PCA_BIN_S, axis=0))
    pca = PCA(min(n_comp, len(ids))).fit(Z[fit_mask])
    return Z, pca, pca.transform(Z)


def principal_angle_similarity(Z, bcond, n=3):
    loads = {}
    for name, _ in CONDITIONS:
        s = bcond == name
        if s.sum() > 100:
            loads[name] = PCA(n).fit(Z[s]).components_.T
    names = list(loads)
    M = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            M[i, j] = np.linalg.svd(loads[a].T @ loads[b], compute_uv=False).mean()
    return names, M


# ---------------------------------------------------------------------------------------------- main

def main():
    a = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(exist_ok=True)
    import figures_population as figs
    if a.figures_only:
        with open(out / "cache" / "ctx.pkl", "rb") as f:
            ctx = pickle.load(f)
        ctx["out"] = out / "figures"
        figs.make_all(ctx)
        return
    cfg = yaml.safe_load(Path(a.config).read_text())
    epochs = {e["name"]: (e["start_s"], e["end_s"]) for e in cfg["epochs"]}
    if a.conditions_file:
        set_conditions(a.conditions_file, a.time_offset_s)
        epochs["baseline"] = dict(CONDITIONS)["baseline"]
        log.info("conditions from %s: %s", a.conditions_file, [n for n, _ in CONDITIONS])

    spikes, info = load_units(a)
    ids = info["unit_id"].astype(int).tolist()
    duration = float(len(np.load(a.nidq_cache)["emg_v"]) / NIDQ_FS)
    dia_df, phi_t, phi, dia_sub, sr_sub = breaths_and_phase(a.nidq_cache, duration)
    cond = assign_conditions(dia_df, epochs)
    dia_df["condition"] = cond
    dia_df.to_csv(out / "breaths.tsv", sep="\t", index=False, float_format="%.4f")
    log.info("%d units, %d breaths, %d Bush-rule gasps (%.1f-%.1f s)", len(ids), len(dia_df), dia_df["gasp"].sum(),
             dia_df.loc[dia_df["gasp"], "on_sec"].min(), dia_df.loc[dia_df["gasp"], "on_sec"].max())

    # single units
    curve_bins, curves, tuning = phase_tuning(spikes, ids, dia_df, cond, phi_t, phi)
    base = tuning[tuning["condition"] == "baseline"].set_index("unit_id")
    tonic = base["modulation_L"] < TONIC_MAX_L
    order_units = list(base.assign(t=tonic.astype(int)).sort_values(["t", "preferred_phase"]).index)
    info = info.merge(base[["preferred_phase", "modulation_L", "class"]], left_on="unit_id", right_index=True)
    tuning.to_csv(out / "unit_phase_tuning.tsv", sep="\t", index=False, float_format="%.4f")

    # HMM on smoothed rate vectors
    n_bins = int(round(duration / HMM_BIN_S))
    counts = bin_counts(spikes, ids, HMM_BIN_S, n_bins)
    X = gaussian_filter1d(counts, a.hmm_sigma_ms / 1e3 / HMM_BIN_S, axis=0) if a.hmm_sigma_ms > 0 else counts
    centres, ph, cyc, bcond = bin_labels(dia_df, cond, phi_t, phi, HMM_BIN_S, n_bins)
    ks = list(range(a.k_min, a.k_max + 1))
    log.info("fitting Poisson HMMs K=%s on %s smoothed counts (%d restarts)", ks, X.shape, a.n_restarts)
    models, restarts = fit_hmms(X, a)
    restarts.to_csv(out / "hmm_restarts.tsv", sep="\t", index=False, float_format="%.2f")
    F = {k: state_features(models[k], X, ph, bcond) for k in ks}
    ref_k = a.ref_k if a.ref_k in ks else ks[-1]
    colours, sims = match_colours(F, ks, ref_k)
    sel_rows, states = [], []
    for k in ks:
        m = models[k]
        ll = m.score(X)
        p = k * len(ids) + k * (k - 1) + (k - 1)
        sel_rows.append({"k": k, "log_likelihood": ll, "n_params": p, "aic": 2 * p - 2 * ll,
                         "bic": p * np.log(n_bins) - 2 * ll})
        f = F[k]["feats"].copy()
        f.insert(0, "k", k)
        f.insert(2, "colour", colours[k])
        sim = np.full(k, np.nan if k != ref_k else 2.0)
        if k in sims:
            S, r, c_ = sims[k]
            sim[r] = S[r, c_]
        f[f"similarity_to_k{ref_k}"] = sim
        states.append(f)
    sel = pd.DataFrame(sel_rows)
    states = pd.concat(states)
    sel.to_csv(out / "hmm_model_selection.tsv", sep="\t", index=False, float_format="%.2f")
    states.to_csv(out / "hmm_states.tsv", sep="\t", index=False, float_format="%.4f")
    np.savez_compressed(out / "hmm_fits.npz", unit_ids=np.array(ids), bin_s=HMM_BIN_S, sigma_ms=a.hmm_sigma_ms,
                        **{f"k{k}_states": F[k]["seq"].astype(np.int8) for k in ks},
                        **{f"k{k}_colours": colours[k] for k in ks},
                        **{f"k{k}_rates_hz": F[k]["rates_hz"] for k in ks},
                        **{f"k{k}_transmat": models[k].transmat_ for k in ks})

    # PCA (Bush & Ramirez)
    base_fit = (centres >= epochs["baseline"][0]) & (centres < epochs["baseline"][1])
    Z, pca, P = pca_projection(spikes, ids, n_bins, base_fit)
    speed = np.r_[0, np.linalg.norm(np.diff(P[:, :3], axis=0), axis=1)] / PCA_BIN_S
    off_bins = np.floor(dia_df["off_sec"].to_numpy() / PCA_BIN_S).astype(int) + 1
    ioff_sel = off_bins[(cond == "baseline") & (off_bins < n_bins)]
    ioff_point = P[ioff_sel, :3].mean(axis=0)
    pa_names, pa_M = principal_angle_similarity(Z, bcond)

    ctx = dict(out=out / "figures", time_offset_s=a.time_offset_s, ids=ids, info=info, order_units=order_units, spikes=spikes, dia_df=dia_df,
               dia_sub=dia_sub, sr_sub=sr_sub, phi_t=phi_t, phi=phi, epochs=epochs, cond=cond, centres=centres, ph=ph,
               bcond=bcond, X=X, ks=ks, ref_k=ref_k, F=F, colours=colours, sims=sims, sel=sel, models=models,
               curve_bins=curve_bins, curves=curves, tuning=tuning, Z=Z, pca=pca, P=P, speed=speed,
               ioff_point=ioff_point, pa_names=pa_names, pa_M=pa_M, conditions=[c for c, _ in CONDITIONS],
               palette=PALETTE, cond_colours=COND_COLOURS, hmm_bin_s=HMM_BIN_S, pca_bin_s=PCA_BIN_S)
    with open(out / "cache" / "ctx.pkl", "wb") as f:
        pickle.dump(ctx, f)
    figs.make_all(ctx)

    summary = {
        "n_units": len(ids), "units": ids, "n_breaths": int(len(dia_df)),
        "gasps": {"n": int(dia_df["gasp"].sum()), "first_on_s": float(dia_df.loc[dia_df["gasp"], "on_sec"].min()),
                  "last_on_s": float(dia_df.loc[dia_df["gasp"], "on_sec"].max())},
        "cycles_per_condition": pd.Series(cond[cond != ""]).value_counts().to_dict(),
        "breath_stats": dia_df[dia_df["condition"] != ""].groupby("condition")[["ibi_s", "duration_sec", "amp_z"]]
        .median().round(4).to_dict(orient="index"),
        "unit_classes": tuning.groupby(["condition", "class"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "pca_baseline_explained_variance": pca.explained_variance_ratio_.round(4).tolist(),
        "principal_angle_mean_cosine": {f"{x}|{y}": float(pa_M[i, j]) for i, x in enumerate(pa_names)
                                        for j, y in enumerate(pa_names) if i < j},
        "hmm_selection": sel.round(2).to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log.info("done: %s", out)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
