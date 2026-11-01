"""Is the gasping population rotation absent, or has it moved out of the eupneic subspace?

Bush et al. 2024 fit PCA on the 100% O2 period and apply that decomposition to the whole recording
(Methods, "PCA decompositions"), so the paper's gasping latents (Fig 7e-g: ballistic, all-or-none,
pre- and post-inspiratory activity lost) describe gasping seen through a eupnea-defined basis. Their
own Extended Data Fig 8 fits PCA per condition and finds gasps the LEAST similar to O2 of every
condition tested, and ED Fig 10e has 39% of expiratory units becoming inspiratory during the gasp --
enough sign flips for the eupneic PCs to cancel rotational structure when applied to gasping data.

Per epoch this script:
  * builds the phase-averaged mean cycle;
  * fits the basis NATIVELY on that epoch's bins, and separately projects the same mean cycle through
    the BASELINE basis (what population_dynamics.py and Bush et al. do);
  * measures rotation in the leading jPC plane (Churchland et al. 2012, Nature 487:51) rather than
    in PC1/PC2 -- e097's PC1 holds ~0.5 of the variance, so a PC1/PC2 view flattens even the eupneic
    cycle into a line and cannot adjudicate anything;
  * cross-validates across cycles: the basis and the jPC plane are fit on odd-numbered cycles and the
    rotation is measured on the even-numbered ones, so a rotation that is noise in a thin average
    does not survive;
  * reports how much of each epoch's variance the baseline basis captures, and the principal angle
    spectrum against baseline.

Normalisations: the Bush & Ramirez one (sqrt of smoothed counts, no per-unit scaling) hands e097's
PC2 almost wholly to one tonic unit (#89, loading 0.997; analysis record F18), so the soft-normalised
version (Churchland: Hz / (range + 5)) is the primary and the sqrt one is reported next to it.

A per-unit circular shuffle of the mean cycle was tried as a null and is NOT used: rolling each unit
independently spreads unit phases uniformly, which is the configuration that produces a perfect
circle, so the "null" scored higher than every real epoch. Cross-validation across cycles replaces it.

  cibrrig python gasping_native_rotation.py [--runs <run> ...] [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

PCA_BIN_S = 0.005
N_PHASE = 50          # phase bins of the mean cycle
N_PCS = 6             # jPCA is defined on an even number of components
MIN_BINS = 400
MIN_CYCLES = 8        # cross-validation needs cycles on both sides of the split
SOFT_HZ = 5.0
RUNS = [r"C:\Users\lpascu\resp_analysis\outputs\e097\population_dynamics_v2",
        r"C:\Users\lpascu\resp_analysis\outputs\e097\population_dynamics_anoxia2"]
BASELINE = "baseline"


# ------------------------------------------------------------------ loading

class _Stub:
    """ctx.pkl pickles a RatePoissonHMM from __main__; the HMM is never used here."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return _Stub


def load_ctx(run) -> dict:
    with open(Path(run) / "cache" / "ctx.pkl", "rb") as f:
        return _Unpickler(f).load()


# ------------------------------------------------------------------ features

def rates_hz(Z):
    """ctx['Z'] is sqrt(gaussian-smoothed counts); undo the sqrt and the bin width."""
    return np.asarray(Z, dtype=float) ** 2 / PCA_BIN_S


def soft_normalise(R):
    rng = R.max(axis=0) - R.min(axis=0)
    return R / (rng + SOFT_HZ)


def phase_bins(ph, sel, n_phase=N_PHASE):
    ok = sel & np.isfinite(ph)
    idx = np.floor((ph[ok] + np.pi) / (2 * np.pi) * n_phase).astype(int)
    return ok, np.clip(idx, 0, n_phase - 1)


def mean_cycle(F, ph, sel, keep=None, n_phase=N_PHASE):
    """Phase-averaged cycle (n_phase x n_units); keep is an optional per-bin mask applied on top."""
    ok, idx = phase_bins(ph, sel, n_phase)
    Fok = F[ok]
    if keep is not None:
        k = keep[ok]
        Fok, idx = Fok[k], idx[k]
    out = np.zeros((n_phase, F.shape[1]))
    for b in range(n_phase):
        m = idx == b
        if not m.any():
            return None
        out[b] = Fok[m].mean(axis=0)
    return out


def cycle_index(centres, sel, onsets):
    """Per-bin index of the cycle it falls in (-1 before the first onset)."""
    return np.searchsorted(onsets, centres[sel], side="right") - 1


# ------------------------------------------------------------------ rotation metrics

def _skew_design(X):
    """Design matrix mapping the k(k-1)/2 free entries of a skew M onto vec(X @ M)."""
    T, k = X.shape
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    D = np.zeros((T * k, len(pairs)))
    for p, (i, j) in enumerate(pairs):
        col = np.zeros((T, k))
        col[:, j] = X[:, i]
        col[:, i] = -X[:, j]
        D[:, p] = col.ravel()
    return D, pairs


def _dcirc(X):
    return (np.roll(X, -1, axis=0) - np.roll(X, 1, axis=0)) / 2.0


def skew_fit(X):
    """Least-squares skew-symmetric M for dX ~ X M on a closed cycle, plus R^2 against unconstrained."""
    Xc = X - X.mean(axis=0)
    dX = _dcirc(Xc)
    sst = float(((dX - dX.mean(axis=0)) ** 2).sum())
    D, pairs = _skew_design(Xc)
    beta, *_ = np.linalg.lstsq(D, dX.ravel(), rcond=None)
    M = np.zeros((X.shape[1], X.shape[1]))
    for p, (i, j) in enumerate(pairs):
        M[i, j], M[j, i] = beta[p], -beta[p]
    r2_skew = 1.0 - float(((dX.ravel() - D @ beta) ** 2).sum()) / sst if sst > 0 else np.nan
    m_full, *_ = np.linalg.lstsq(Xc, dX, rcond=None)
    r2_full = 1.0 - float(((dX - Xc @ m_full) ** 2).sum()) / sst if sst > 0 else np.nan
    return M, r2_skew, r2_full


def jpc_plane(X):
    """Leading jPC plane: the eigenvector pair of the skew fit with the largest |imaginary| eigenvalue."""
    M, r2_skew, r2_full = skew_fit(X)
    w, V = np.linalg.eig(M)
    v = V[:, int(np.argmax(np.abs(w.imag)))]
    u1, u2 = np.real(v), np.imag(v)
    n1 = np.linalg.norm(u1)
    if n1 == 0:
        return None, r2_skew, r2_full
    u1 = u1 / n1
    u2 = u2 - (u2 @ u1) * u1
    n2 = np.linalg.norm(u2)
    if n2 == 0:
        return None, r2_skew, r2_full
    return np.c_[u1, u2 / n2], r2_skew, r2_full


def rotation_index(Y2):
    """Enclosed area of the closed trajectory, normalised so a circle = 1 and a retraced line = 0."""
    x, y = Y2[:, 0] - Y2[:, 0].mean(), Y2[:, 1] - Y2[:, 1].mean()
    area = 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
    w = np.linalg.eigvalsh(np.cov(np.c_[x, y].T))
    denom = 2 * np.pi * float(np.sqrt(max(w[0], 0.0) * max(w[1], 0.0)))
    return area / denom if denom > 0 else np.nan


def measure(cycle, mean_, comp):
    """Rotation of a mean cycle seen through a basis, in that basis's leading jPC plane."""
    Y = (cycle - mean_) @ comp.T
    X = Y[:, :N_PCS]
    plane, r2_skew, r2_full = jpc_plane(X)
    ri = np.nan if plane is None else rotation_index((X - X.mean(axis=0)) @ plane)
    return {"rotation_index_jpc": ri, "rotation_index_pc12": rotation_index(Y[:, :2]),
            "jpca_r2_skew": r2_skew, "jpca_r2_full": r2_full,
            "jpca_skew_frac": r2_skew / r2_full if r2_full and r2_full > 0 else np.nan}


def cycle_geometry(cycle):
    """Rank and enclosed area of a mean cycle in its OWN leading plane.

    A ballistic out-and-back is rank 1 -- every unit rises and falls together, so one component
    carries the cycle. A rotation needs two comparable components in quadrature. Taking the plane
    from the cycle's own covariance avoids the leading PC of the binned data, which in e097 holds
    ~0.5 of the variance (much of it not respiratory) and flattens any cycle into a line.
    """
    C = cycle - cycle.mean(axis=0)
    p = PCA(min(N_PCS, C.shape[1])).fit(C)
    ev = p.explained_variance_
    return {"lambda2_over_1": float(ev[1] / ev[0]), "lambda3_over_1": float(ev[2] / ev[0]),
            "rotation_index_cycle": float(rotation_index(p.transform(C)[:, :2]))}, p


def cross_validated_cycle_ri(F, ph, sel, cyc):
    """Fit the cycle plane on odd cycles, measure the even cycles' mean cycle in it."""
    odd, even = (cyc % 2 == 1), (cyc % 2 == 0)
    odd_bins, even_bins = np.zeros(len(F), bool), np.zeros(len(F), bool)
    idx = np.where(sel)[0]
    odd_bins[idx[odd & (cyc >= 0)]] = True
    even_bins[idx[even & (cyc >= 0)]] = True
    c_tr, c_te = mean_cycle(F, ph, odd_bins), mean_cycle(F, ph, even_bins)
    if c_tr is None or c_te is None:
        return np.nan
    _, p = cycle_geometry(c_tr)
    return float(rotation_index(p.transform(c_te - c_te.mean(axis=0))[:, :2]))


def cross_validated_ri(F, ph, sel, cyc, n_comp):
    """Fit basis + jPC plane on odd cycles, measure the rotation of the even cycles' mean."""
    odd, even = (cyc % 2 == 1), (cyc % 2 == 0)
    odd_bins, even_bins = np.zeros(len(F), bool), np.zeros(len(F), bool)
    idx = np.where(sel)[0]
    odd_bins[idx[odd & (cyc >= 0)]] = True
    even_bins[idx[even & (cyc >= 0)]] = True
    if odd_bins.sum() < MIN_BINS // 2 or even_bins.sum() < MIN_BINS // 2:
        return np.nan
    c_tr = mean_cycle(F, ph, odd_bins)
    c_te = mean_cycle(F, ph, even_bins)
    if c_tr is None or c_te is None:
        return np.nan
    p = PCA(n_comp).fit(F[odd_bins])
    X_tr = ((c_tr - p.mean_) @ p.components_.T)[:, :N_PCS]
    plane, _, _ = jpc_plane(X_tr)
    if plane is None:
        return np.nan
    X_te = ((c_te - p.mean_) @ p.components_.T)[:, :N_PCS]
    return rotation_index((X_te - X_te.mean(axis=0)) @ plane)


def principal_angles(A, B):
    return np.clip(np.linalg.svd(A.T @ B, compute_uv=False), -1, 1)


# ------------------------------------------------------------------ per-run analysis

def analyse(ctx, norm):
    Z = np.asarray(ctx["Z"], dtype=float)
    bcond = np.asarray(ctx["bcond"])
    ph = np.asarray(ctx["ph"], dtype=float)
    centres = np.asarray(ctx["centres"], dtype=float)
    dia = ctx["dia_df"]
    base_sel = bcond == BASELINE
    if not base_sel.any():
        return None

    F = soft_normalise(rates_hz(Z)) if norm == "softnorm" else Z
    n_comp = min(10, F.shape[1])
    base_pca = PCA(n_comp).fit(F[base_sel])
    base_sub = PCA(3).fit(F[base_sel]).components_.T

    rows = []
    for ep in [e for e in dict.fromkeys(bcond.tolist()) if e]:
        sel = bcond == ep
        onsets = np.sort(dia.loc[dia["condition"] == ep, "on_sec"].to_numpy())
        if sel.sum() < MIN_BINS or len(onsets) < MIN_CYCLES:
            continue
        cycle = mean_cycle(F, ph, sel)
        if cycle is None:
            continue
        nat_pca = PCA(n_comp).fit(F[sel])
        nat, base = measure(cycle, nat_pca.mean_, nat_pca.components_), \
            measure(cycle, base_pca.mean_, base_pca.components_)
        cyc = cycle_index(centres, sel, onsets)
        cv = cross_validated_ri(F, ph, sel, cyc, n_comp)
        geom, _ = cycle_geometry(cycle)
        geom["cv_rotation_index_cycle"] = cross_validated_cycle_ri(F, ph, sel, cyc)

        Xc = F[sel] - F[sel].mean(axis=0)
        tot = float((Xc ** 2).sum())
        cos_t = principal_angles(PCA(3).fit(F[sel]).components_.T, base_sub)
        rows.append({
            "epoch": ep, "n_bins": int(sel.sum()), "n_cycles": int(len(onsets)),
            "native": nat, "baseline_basis": base, "cv_rotation_index_native": float(cv),
            "cycle_geometry": geom,
            "var_captured_top3_baseline_basis": float(((Xc @ base_pca.components_[:3].T) ** 2).sum()) / tot,
            "var_captured_top3_native_basis": float(((Xc @ nat_pca.components_[:3].T) ** 2).sum()) / tot,
            "principal_angle_cos_vs_baseline": [float(x) for x in cos_t],
            "principal_angle_mean_cos": float(cos_t.mean()),
            "_cycle": cycle, "_nat_pca": nat_pca,
        })
    return {"rows": rows, "base_pca": base_pca}


# ------------------------------------------------------------------ figure

def figure(res, run_name, norm, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = res["rows"]
    if not rows:
        return
    fig, ax = plt.subplots(2, len(rows), figsize=(2.5 * len(rows), 5.8), squeeze=False)
    for j, r in enumerate(rows):
        # top: the cycle in its own coordinates; bottom: the same cycle through the eupneic basis
        for i, (lab, C) in enumerate([
                ("native", r["_cycle"]),
                ("baseline basis",
                 (r["_cycle"] - res["base_pca"].mean_) @ res["base_pca"].components_.T)]):
            C = C - C.mean(axis=0)
            P2 = PCA(min(N_PCS, C.shape[1])).fit(C).transform(C)[:, :2]
            a = ax[i][j]
            a.plot(np.r_[P2[:, 0], P2[0, 0]], np.r_[P2[:, 1], P2[0, 1]], "-", lw=1.6, color="#2a78d6")
            a.scatter(P2[0, 0], P2[0, 1], s=18, color="#e34948", zorder=3)
            a.set_aspect("equal", "datalim")
            a.set_xticks([]), a.set_yticks([])
            a.set_title(f"{r['epoch']} (n={r['n_cycles']})\n{lab}\n"
                        f"RI={rotation_index(P2):.2f}", fontsize=7)
    fig.suptitle(f"{run_name} — phase-averaged cycle in its own leading plane, {norm} "
                 f"(RI 1=circle 0=line; red dot = phase -\u03c0)", fontsize=9)
    fig.tight_layout()
    fig.savefig(Path(out) / f"{run_name}_{norm}_mean_cycle.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", nargs="+", default=RUNS)
    p.add_argument("--out", default=r"C:\Users\lpascu\resp_analysis\outputs\e097\gasping_native_rotation")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for run in a.runs:
        name = Path(run).name
        if not (Path(run) / "cache" / "ctx.pkl").exists():
            print(f"skip {name}: no ctx.pkl")
            continue
        ctx = load_ctx(run)
        summary[name] = {}
        for norm in ("softnorm", "sqrt"):
            res = analyse(ctx, norm)
            if res is None:
                continue
            figure(res, name, norm, out)
            summary[name][norm] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in res["rows"]]
            print(f"\n=== {name} / {norm} ===")
            print(f"{'epoch':<16}{'cyc':>5}{'L2/L1':>7}{'L3/L1':>7}{'RIcyc':>7}{'RIcyc cv':>10}"
                  f"{'var base':>10}{'var nat':>9}{'cos∠':>7}")
            for r in res["rows"]:
                g = r["cycle_geometry"]
                print(f"{r['epoch']:<16}{r['n_cycles']:>5}"
                      f"{g['lambda2_over_1']:>7.2f}{g['lambda3_over_1']:>7.2f}"
                      f"{g['rotation_index_cycle']:>7.2f}{g['cv_rotation_index_cycle']:>10.2f}"
                      f"{r['var_captured_top3_baseline_basis']:>10.2f}"
                      f"{r['var_captured_top3_native_basis']:>9.2f}"
                      f"{r['principal_angle_mean_cos']:>7.2f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {out / 'summary.json'}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
