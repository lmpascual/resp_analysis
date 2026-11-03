"""Match units across several window sorts of one run when the tissue moved between the windows (e038, e040).

match_sorts.py compares templates in place, which is right for e097 (no movement between its windows). In e038 the
tissue moved tens of um between windows, and each motion-corrected window has its own reference depth, so the same
neuron sits at a shifted position in another window. This uses anchor_motion.py's measure: cosine of the two units'
whole templates (samples x the unit's 12 largest channels) with one of them shifted along the probe by whole rows
(Neuropixels 1.0: 20 um, 2 channels). For each pair of windows:
  1. every unit (not noise) against units of the other window within 200 um, best shift in +-10 rows;
  2. mutual-best pairs with cosine >= PROFILE_MIN_COS give the template displacement profile between the windows
     (median in 300 um bins, interpolated);
  3. (with --firing-condition) firing candidates and drift hypotheses, below;
  4. the best cosine again, shifts limited to within +-20 um of the template profile at the unit's depth, or of a
     drift hypothesis where one exists; one-to-one assignment (Hungarian, a unit may stay unmatched: leaving one
     unmatched costs (1 - MATCH_MIN_COS) / 2). Accepted: cosine >= MATCH_MIN_COS; "a neighbour looks alike" if the
     runner-up on either side is within ALIKE_MARGIN; "review: shift from firing hypothesis" if the accepted shift is
     only consistent with a firing hypothesis, not with the template profile.

Firing candidates (never accepted on their own; the firing changes are what the analyses measure). Per window, each
unit's firing in one condition of its population_dynamics run (--firing-condition, e.g. baseline): phase curve,
onset PETH, ACG, rate, CV2, L (firing_features.py). For every unit of A that is phase-locked (L >= 0.2) and fires
>= FIRING_MIN_RATE Hz, the B unit within FIRING_SEARCH_UM with the highest combined firing score is its firing
candidate, and its depth difference an implied displacement. Benchmark (e097, outputs/e097/firing_benchmark): for such
units the true partner ranks first among ~39 neighbours 76-85 % of the time between similar states up to ~8 min and
one anoxia apart (62 % across two anoxias), but only 30 % across gasping, and tonic or sparse units are unreliable; so
compare similar conditions (e.g. each window's pre-challenge baseline). Drift hypothesis per 300 um depth bin: the mode
of the implied displacements (5 um grid, Gaussian 10 um, weighted by the score margin over the runner-up), kept if
>= HYP_MIN_SUPPORT candidates lie within +-15 um of it and they are >= HYP_MIN_FRAC of the bin's candidates. A
hypothesis gives the constrained template search (4) a second centre: the new anatomical window.

Outputs: matched_pairs.tsv (accepted), review_candidates.tsv (to inspect by eye, with a figure each in review/),
firing_candidates.tsv, drift_hypotheses.tsv, tracks.tsv, summary.json, match_windows.png.
Review reasons: "firing candidate, template-supported" (firing top-1 has no accepted template partner, but its
template at the hypothesis shift reaches REVIEW_MIN_COS), "firing disagrees with template match" (firing top-1 is
another unit, with a clear firing margin, and its template also reaches REVIEW_MIN_COS), "shift from firing hypothesis"
(accepted only thanks to a hypothesis). Checked on e097 (2026-10-30): with the second window's templates moved +60 um,
the firing hypotheses alone recover +60 um in every depth bin with candidates (5-22 supporting units each); the 4
firing-vs-template disagreements were all neighbouring neurons with similar firing and different waveforms.

    python match_windows.py <outputs>/grp/e038/baseline <outputs>/grp/e038/anoxia1 ... ^
        --firing-condition baseline --out <outputs>/grp/e038/matched_units
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from firing_features import (ACG_EDGES, PETH_EDGES, combined_score, cycle_spans, features,  # noqa: E402
                             phase_from_breaths, similarity)  # noqa: E402

log = logging.getLogger("match_windows")

N_CH = 12
MAX_ROWS = 10
ROW_UM = 20.0
N_CHANNELS = 384
SEARCH_UM = 200.0
PROFILE_MIN_COS = 0.95
PROFILE_BIN_UM = 300.0
PROFILE_TOL_UM = 20.0
MATCH_MIN_COS = 0.92
ALIKE_MARGIN = 0.02
LABELS = ("good", "mua", "non_soma")
FIRING_MIN_RATE = 5.0
FIRING_SEARCH_UM = 250.0
HYP_TOL_UM = 15.0
HYP_MIN_SUPPORT = 3
HYP_MIN_FRAC = 0.4
REVIEW_MIN_COS = 0.85
REVIEW_FIRING_MARGIN = 0.5
MAX_REVIEW_FIGURES = 80


class Window:
    def __init__(self, d: Path, sorter: str, labels, name=None):
        import spikeinterface.core as si
        self.dir, self.name = d, name or d.name
        units = pd.read_csv(d / sorter / "units.tsv", sep="\t").set_index("unit_id")
        an = si.load_sorting_analyzer(d / sorter / "analyzer", load_extensions=False)
        dense = an.get_extension("templates").get_data(operator="average").astype(np.float32)
        idx = [int(re.findall(r"\d+", str(c))[-1]) for c in an.channel_ids]
        self.T = np.zeros((dense.shape[0], dense.shape[1], N_CHANNELS), np.float32)
        self.T[:, :, idx] = dense
        self.ids = np.array(an.unit_ids)
        keep = units.reindex(self.ids)["label"].isin(labels).to_numpy()
        self.T, self.ids = self.T[keep], self.ids[keep]
        self.units = units.loc[self.ids]
        self.depth = self.units["depth_um"].to_numpy().astype(float)
        st = np.load(d / sorter / "alf" / "spike_times.npy")
        sc = np.load(d / sorter / "alf" / "spike_clusters.npy").astype(np.int64)
        order = np.argsort(sc, kind="stable")
        b = np.searchsorted(sc[order], self.ids)
        e = np.searchsorted(sc[order], self.ids, side="right")
        self.spk = {int(u): np.sort(st[order[i:j]]) for u, i, j in zip(self.ids, b, e)}
        dur = float(st.max()) if st.size else 1.0
        self.rate = np.array([self.spk[int(u)].size / dur for u in self.ids])
        self.F = None
        del an
        self._top()

    def _top(self):
        ptp = self.T.max(axis=1) - self.T.min(axis=1)
        self.top = np.sort(np.argsort(-ptp, axis=1)[:, :N_CH], axis=1)

    def shift(self, rows: int):
        """Test only: move every template `rows` rows up the probe (and the depths with it)."""
        self.T = np.roll(self.T, 2 * rows, axis=2)
        if rows > 0:
            self.T[:, :, :2 * rows] = 0
        elif rows < 0:
            self.T[:, :, 2 * rows:] = 0
        self.depth = self.depth + rows * ROW_UM
        self._top()

    def attach_firing(self, run: Path, condition: str):
        d = pd.read_csv(run / "breaths.tsv", sep="\t")
        ons, offs = d["on_sec"].to_numpy(), d["off_sec"].to_numpy()
        phi_t, phi = phase_from_breaths(ons, offs)
        spans = cycle_spans(ons, d["condition"].fillna("").to_numpy() == condition)
        if not len(spans):
            raise ValueError(f"{self.name}: no '{condition}' cycles in {run}")
        self.F = {int(u): features(self.spk[int(u)], spans, ons, phi_t, phi) for u in self.ids}
        self.firing_condition = condition


def cos(a, b):
    n = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / n) if n > 0 else 0.0


def shifted_cos(A, i, B, j, rows):
    """Best cosine of A unit i (its top channels) with B unit j shifted by k rows, over k in rows."""
    ch = A.top[i]
    best, bk = -1.0, 0
    for k in rows:
        c2 = ch + 2 * k
        if c2.min() < 0 or c2.max() >= N_CHANNELS:
            continue
        c = cos(A.T[i][:, ch], B.T[j][:, c2])
        if c > best:
            best, bk = c, k
    return best, bk


def profile_from(pairs: pd.DataFrame, value="shift_um", depth="depth_a_um"):
    if len(pairs) < 3:
        return None, pd.DataFrame()
    edges = np.arange(0, 4000 + PROFILE_BIN_UM, PROFILE_BIN_UM)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        g = pairs[(pairs[depth] >= a) & (pairs[depth] < b)]
        if len(g) >= 2:
            rows.append({"depth_um": (a + b) / 2, "shift_um": float(g[value].median()), "n": len(g)})
    P = pd.DataFrame(rows)
    if P.empty:
        return None, P
    return (lambda y: np.interp(y, P["depth_um"], P["shift_um"])), P


def firing_candidates(A: Window, B: Window):
    rows = []
    for i, u in enumerate(A.ids):
        fa = A.F[int(u)]
        if fa["rate_hz"] < FIRING_MIN_RATE or fa["L"] < 0.2:
            continue
        js = np.flatnonzero(np.abs(B.depth - A.depth[i]) <= FIRING_SEARCH_UM)
        sc = []
        for j in js:
            fb = B.F[int(B.ids[j])]
            sc.append(combined_score(similarity(fa, fb), fa["L"], fb["L"]) if fb["n_spikes"] >= 20 else -np.inf)
        if not len(sc) or not np.isfinite(np.max(sc)):
            continue
        o = np.argsort(sc)[::-1]
        j1 = js[o[0]]
        margin = sc[o[0]] - sc[o[1]] if len(o) > 1 and np.isfinite(sc[o[1]]) else np.nan
        rows.append({"window_a": A.name, "unit_a": int(u), "window_b": B.name, "unit_b": int(B.ids[j1]),
                     "depth_a_um": A.depth[i], "depth_b_um": B.depth[j1], "implied_shift_um": B.depth[j1] - A.depth[i],
                     "firing_score": sc[o[0]], "firing_margin": margin, "n_compared": int(np.isfinite(sc).sum()),
                     "rate_a_hz": fa["rate_hz"], "rate_b_hz": B.F[int(B.ids[j1])]["rate_hz"], "L_a": fa["L"],
                     "L_b": B.F[int(B.ids[j1])]["L"]})
    return pd.DataFrame(rows)


def drift_hypotheses(C: pd.DataFrame):
    rows = []
    if C.empty:
        return pd.DataFrame(rows)
    grid = np.arange(-FIRING_SEARCH_UM, FIRING_SEARCH_UM + 1, 5.0)
    edges = np.arange(0, 4000 + PROFILE_BIN_UM, PROFILE_BIN_UM)
    for a, b in zip(edges[:-1], edges[1:]):
        g = C[(C["depth_a_um"] >= a) & (C["depth_a_um"] < b)]
        if len(g) < HYP_MIN_SUPPORT:
            continue
        w = np.clip(g["firing_margin"].fillna(0.1).to_numpy(), 0.05, None)
        dens = (w[None, :] * np.exp(-0.5 * ((grid[:, None] - g["implied_shift_um"].to_numpy()[None, :]) / 10.0) ** 2)).sum(1)
        mode = float(grid[np.argmax(dens)])
        sup = int((np.abs(g["implied_shift_um"] - mode) <= HYP_TOL_UM).sum())
        rows.append({"depth_um": (a + b) / 2, "hypothesis_shift_um": mode, "support": sup, "n_candidates": len(g),
                     "kept": bool(sup >= HYP_MIN_SUPPORT and sup / len(g) >= HYP_MIN_FRAC)})
    return pd.DataFrame(rows)


def match_pair(A: Window, B: Window):
    allrows = range(-MAX_ROWS, MAX_ROWS + 1)
    near = np.abs(A.depth[:, None] - B.depth[None, :]) <= SEARCH_UM
    C = np.full(near.shape, -1.0)
    K = np.zeros(near.shape, int)
    for i, j in zip(*np.nonzero(near)):
        C[i, j], K[i, j] = shifted_cos(A, i, B, j, allrows)
    bi, bj = C.argmax(axis=1), C.argmax(axis=0)
    mutual = [(i, bi[i]) for i in range(C.shape[0]) if C[i, bi[i]] >= PROFILE_MIN_COS and bj[bi[i]] == i]
    prof_pairs = pd.DataFrame([{"depth_a_um": A.depth[i], "shift_um": K[i, j] * ROW_UM} for i, j in mutual])
    prof, P = profile_from(prof_pairs)
    FC, H = pd.DataFrame(), pd.DataFrame()
    hyp = None
    if A.F is not None and B.F is not None:
        FC = firing_candidates(A, B)
        H = drift_hypotheses(FC)
        Hk = H[H["kept"]] if len(H) else H
        if len(Hk):
            hyp = lambda y, Hk=Hk: np.interp(y, Hk["depth_um"], Hk["hypothesis_shift_um"])  # noqa: E731
    # constrained pass: around the template profile and/or the firing hypothesis
    tol = int(np.ceil(PROFILE_TOL_UM / ROW_UM))
    C2 = np.full(near.shape, -1.0)
    K2 = np.zeros(near.shape, int)
    centres_t = np.round(prof(A.depth) / ROW_UM).astype(int) if prof else None
    centres_f = np.round(hyp(A.depth) / ROW_UM).astype(int) if hyp else None
    for i, j in zip(*np.nonzero(near)):
        rows = set()
        if centres_t is not None:
            rows |= set(range(centres_t[i] - tol, centres_t[i] + tol + 1))
        if centres_f is not None:
            rows |= set(range(centres_f[i] - tol, centres_f[i] + tol + 1))
        if not rows:
            rows = set(allrows)
        C2[i, j], K2[i, j] = shifted_cos(A, i, B, j, sorted(rows))
    reject = (1 - MATCH_MIN_COS) / 2
    nA, nB = C2.shape
    cost = np.full((nA + nB, nB + nA), 0.0)
    cost[:nA, :nB] = np.where(C2 >= MATCH_MIN_COS, 1 - C2, 10.0)
    cost[:nA, nB:] = np.where(np.eye(nA) > 0, reject, 10.0)
    cost[nA:, :nB] = np.where(np.eye(nB) > 0, reject, 10.0)
    r, c = linear_sum_assignment(cost)
    rows = []
    for i, j in zip(r, c):
        if i < nA and j < nB and C2[i, j] >= MATCH_MIN_COS:
            ra = np.sort(np.delete(C2[i], j))[-1] if nB > 1 else -1
            rb = np.sort(np.delete(C2[:, j], i))[-1] if nA > 1 else -1
            margin = C2[i, j] - max(ra, rb)
            s_um = K2[i, j] * ROW_UM
            by_t = centres_t is not None and abs(K2[i, j] - centres_t[i]) <= tol
            verdict = "match" if margin >= ALIKE_MARGIN else "match; a neighbour looks alike"
            if not by_t and centres_f is not None:
                verdict = "review: shift from firing hypothesis"
            rows.append({"window_a": A.name, "unit_a": int(A.ids[i]), "window_b": B.name, "unit_b": int(B.ids[j]),
                         "cosine": round(float(C2[i, j]), 4), "shift_um": s_um,
                         "template_profile_um": round(float(prof(A.depth[i])), 1) if prof else np.nan,
                         "firing_hypothesis_um": round(float(hyp(A.depth[i])), 1) if hyp else np.nan,
                         "margin": round(float(margin), 4), "verdict": verdict,
                         "depth_a_um": A.depth[i], "depth_b_um": B.depth[j],
                         "label_a": A.units.loc[A.ids[i], "label"], "label_b": B.units.loc[B.ids[j], "label"],
                         "rate_a_hz": round(float(A.rate[i]), 3), "rate_b_hz": round(float(B.rate[j]), 3)})
    M = pd.DataFrame(rows)
    # review list
    rev = []
    idxA = {int(u): i for i, u in enumerate(A.ids)}
    idxB = {int(u): j for j, u in enumerate(B.ids)}
    matchedA = dict(zip(M["unit_a"], M["unit_b"])) if len(M) else {}
    for m in M[M["verdict"].str.startswith("review")].itertuples() if len(M) else []:
        rev.append({"reason": "shift from firing hypothesis", "unit_a": m.unit_a, "unit_b": m.unit_b,
                    "template_cosine": m.cosine, "shift_um": m.shift_um})
    for f in FC.itertuples() if len(FC) else []:
        i, j = idxA[f.unit_a], idxB[f.unit_b]
        centre = int(round((hyp(A.depth[i]) if hyp else f.implied_shift_um) / ROW_UM))
        cc, kk = shifted_cos(A, i, B, j, range(centre - tol, centre + tol + 1))
        if f.unit_a not in matchedA and cc >= REVIEW_MIN_COS and f.unit_b not in set(matchedA.values()):
            rev.append({"reason": "firing candidate, template-supported", "unit_a": f.unit_a, "unit_b": f.unit_b,
                        "template_cosine": round(cc, 4), "shift_um": kk * ROW_UM, "firing_margin": f.firing_margin})
        elif (f.unit_a in matchedA and matchedA[f.unit_a] != f.unit_b and np.isfinite(f.firing_margin)
              and f.firing_margin >= REVIEW_FIRING_MARGIN and cc >= REVIEW_MIN_COS):
            # a firing look-alike with a clearly different template is another neuron (e097: 4 of 4, cosine <= 0.42);
            # only an ambiguous template makes the disagreement worth an eye
            rev.append({"reason": "firing disagrees with template match", "unit_a": f.unit_a, "unit_b": f.unit_b,
                        "template_cosine": round(cc, 4), "shift_um": kk * ROW_UM, "firing_margin": f.firing_margin,
                        "template_match_b": matchedA[f.unit_a]})
    R = pd.DataFrame(rev)
    if len(R):
        R.insert(0, "window_b", B.name)
        R.insert(0, "window_a", A.name)
        R["your_verdict"] = ""
    return M, P, FC, H, R


def review_figure(A: Window, B: Window, r, path: Path):
    i = int(np.flatnonzero(A.ids == r["unit_a"])[0])
    j = int(np.flatnonzero(B.ids == r["unit_b"])[0])
    ch = A.top[i]
    k = int(r["shift_um"] // ROW_UM)
    ch_b = np.clip(ch + 2 * k, 0, N_CHANNELS - 1)
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6), gridspec_kw={"width_ratios": [2.4, 1, 1, 1]})
    ta, tb = A.T[i][:, ch], B.T[j][:, ch_b]
    n = ta.shape[0]
    for c in range(len(ch)):
        x = np.arange(n) + c * (n + 8)
        ax[0].plot(x, ta[:, c], color="#2a78d6", lw=1)
        ax[0].plot(x, tb[:, c], color="#d93636", lw=1)
    ax[0].set_title(f"templates: {A.name} #{r['unit_a']} (blue, {A.depth[i]:.0f} um) vs {B.name} #{r['unit_b']} "
                    f"(red, shifted {r['shift_um']:+.0f} um); cos {r['template_cosine']:.2f}", fontsize=8)
    ax[0].set_xticks([])
    if A.F is not None and B.F is not None:
        fa, fb = A.F[int(r["unit_a"])], B.F[int(r["unit_b"])]
        ph = np.linspace(-np.pi, np.pi, len(fa["curve"]) + 1)
        ph = (ph[:-1] + ph[1:]) / 2
        ax[1].plot(ph, fa["curve"], color="#2a78d6"); ax[1].plot(ph, fb["curve"], color="#d93636")
        ax[1].set_title(f"phase curve ({A.firing_condition})", fontsize=8); ax[1].set_xlabel("phi (0 = onset)")
        pe = (PETH_EDGES[:-1] + PETH_EDGES[1:]) / 2 * 1e3
        ax[2].plot(pe, fa["peth"], color="#2a78d6"); ax[2].plot(pe, fb["peth"], color="#d93636")
        ax[2].axvline(0, color="0.5", lw=0.5); ax[2].set_title("onset PETH (Hz)", fontsize=8); ax[2].set_xlabel("ms")
        ac = (ACG_EDGES[:-1] + ACG_EDGES[1:]) / 2 * 1e3
        ax[3].plot(ac, fa["acg"], color="#2a78d6"); ax[3].plot(ac, fb["acg"], color="#d93636")
        ax[3].set_title(f"ACG; rate {fa['rate_hz']:.1f} vs {fb['rate_hz']:.1f} Hz", fontsize=8); ax[3].set_xlabel("ms")
    fig.suptitle(f"{r['reason']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def nearest_other(W: Window):
    out = []
    for i in range(len(W.ids)):
        js = np.flatnonzero((np.abs(W.depth - W.depth[i]) <= 100) & (np.arange(len(W.ids)) != i))
        out.append(max([shifted_cos(W, i, W, j, [0])[0] for j in js], default=np.nan))
    return np.array(out)


def tracks(windows, pairs: pd.DataFrame):
    """Connected components of accepted pairs, split where one window would hold two units."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    acc = pairs[~pairs["verdict"].str.startswith("review")].sort_values("cosine", ascending=False) if len(pairs) else pairs
    members = {}
    for r in acc.itertuples():
        a, b = (r.window_a, r.unit_a), (r.window_b, r.unit_b)
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        wa = {w for w, _ in members.get(ra, {a})}
        wb = {w for w, _ in members.get(rb, {b})}
        if wa & wb:                         # would put two units of one window in one track: skip the weaker link
            continue
        parent[rb] = ra
        members[ra] = members.get(ra, {a}) | members.pop(rb, {b})
    rows = []
    for t, (root, mem) in enumerate(sorted(members.items(), key=lambda kv: -len(kv[1]))):
        row = {"track": t, "n_windows": len(mem)}
        for w, u in mem:
            row[w] = u
        rows.append(row)
    cols = ["track", "n_windows"] + [w.name for w in windows]
    return pd.DataFrame(rows).reindex(columns=cols)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("windows", nargs="+", help="npxsort output folders, in time order")
    p.add_argument("--sorter", default="kilosort4")
    p.add_argument("--firing-condition", default=None,
                   help="condition of each window's population_dynamics run used for firing candidates (e.g. baseline)")
    p.add_argument("--run", action="append", default=[],
                   help="window_name=population run folder (default: <window>/population_dynamics)")
    p.add_argument("--simulate-shift-rows", type=int, default=0,
                   help="test: move the last window's templates this many 20 um rows (the answer is then known)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    W = [Window(Path(d), a.sorter, LABELS) for d in a.windows]
    if a.simulate_shift_rows:
        W[-1].name += f"_shift{a.simulate_shift_rows * ROW_UM:+.0f}um"
        W[-1].shift(a.simulate_shift_rows)
    if a.firing_condition:
        runs = dict(x.split("=", 1) for x in a.run)
        for w in W:
            w.attach_firing(Path(runs.get(w.dir.name, w.dir / "population_dynamics")), a.firing_condition)
    calib = {w.name: nearest_other(w) for w in W}
    allp, allfc, allh, allr, profs = [], [], [], [], {}
    (out / "review").mkdir(exist_ok=True)
    n_fig = 0
    for A, B in itertools.combinations(W, 2):
        M, P, FC, H, R = match_pair(A, B)
        kept = H[H["kept"]][["depth_um", "hypothesis_shift_um", "support"]].values.tolist() if len(H) else "-"
        log.info("%s - %s: %d accepted (%d clean), %d to review; template profile %s; firing hypotheses %s",
                 A.name, B.name, int((~M["verdict"].str.startswith("review")).sum()) if len(M) else 0,
                 int((M.get("verdict", pd.Series(dtype=str)) == "match").sum()), len(R),
                 P[["depth_um", "shift_um"]].round(0).values.tolist() if len(P) else "-", kept)
        allp.append(M)
        profs[f"{A.name}|{B.name}"] = P
        if len(FC):
            allfc.append(FC)
        if len(H):
            allh.append(H.assign(window_a=A.name, window_b=B.name))
        if len(R):
            allr.append(R)
            for r in R.to_dict(orient="records"):
                if n_fig < MAX_REVIEW_FIGURES:
                    review_figure(A, B, r, out / "review" / f"{A.name}__{B.name}__{r['unit_a']}_{r['unit_b']}.png")
                    n_fig += 1
    pairs = pd.concat(allp, ignore_index=True) if allp else pd.DataFrame()
    pairs.to_csv(out / "matched_pairs.tsv", sep="\t", index=False)
    if allfc:
        pd.concat(allfc).to_csv(out / "firing_candidates.tsv", sep="\t", index=False, float_format="%.3f")
    if allh:
        pd.concat(allh).to_csv(out / "drift_hypotheses.tsv", sep="\t", index=False, float_format="%.1f")
    Rall = pd.concat(allr, ignore_index=True) if allr else pd.DataFrame()
    Rall.to_csv(out / "review_candidates.tsv", sep="\t", index=False, float_format="%.3f")
    T = tracks(W, pairs)
    T.to_csv(out / "tracks.tsv", sep="\t", index=False)
    summary = {"match_min_cos": MATCH_MIN_COS, "firing_condition": a.firing_condition,
               "nearest_other_unit_cos_p95": {k: float(np.nanpercentile(v, 95)) for k, v in calib.items()},
               "units_per_window": {w.name: int(len(w.ids)) for w in W},
               "good_units_per_window": {w.name: int((w.units["label"] == "good").sum()) for w in W},
               "accepted_per_pair": {f"{r[0]}|{r[1]}": int((~g["verdict"].str.startswith("review")).sum())
                                     for r, g in pairs.groupby(["window_a", "window_b"])} if len(pairs) else {},
               "to_review": int(len(Rall)),
               "template_profiles": {k: v.round(1).to_dict(orient="records") for k, v in profs.items()},
               "tracks_by_n_windows": T["n_windows"].value_counts().sort_index().to_dict() if len(T) else {}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for k, P in profs.items():
        if len(P):
            axes[0].plot(P["shift_um"], P["depth_um"], ".-", label=f"{k} templates", lw=0.8)
    for H in allh:
        Hk = H[H["kept"]]
        if len(Hk):
            axes[0].plot(Hk["hypothesis_shift_um"], Hk["depth_um"], "x--", lw=0.6,
                         label=f"{H['window_a'].iloc[0]}|{H['window_b'].iloc[0]} firing")
    axes[0].set_xlabel("shift of window B vs A (um)")
    axes[0].set_ylabel("depth (um)")
    axes[0].legend(fontsize=6)
    axes[0].set_title("displacement between windows", fontsize=9)
    allc = np.concatenate([v[np.isfinite(v)] for v in calib.values()])
    axes[1].hist(allc, bins=np.linspace(0.5, 1, 51), alpha=0.6, label="nearest other unit, same window")
    if len(pairs):
        axes[1].hist(pairs["cosine"], bins=np.linspace(0.5, 1, 51), alpha=0.6, label="accepted cross-window pairs")
    axes[1].axvline(MATCH_MIN_COS, color="k", lw=0.8)
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel("template cosine")
    fig.tight_layout()
    fig.savefig(out / "match_windows.png", dpi=130)
    print(json.dumps({k: summary[k] for k in ("units_per_window", "accepted_per_pair", "to_review",
                                               "tracks_by_n_windows")}, indent=1, default=float))


if __name__ == "__main__":
    main()
