"""Interactive 3D view of the K = 6 HMM states in PC1-PC3, with every epoch of one or more anoxia trials togglable.

Writes one HTML page (template: states_pca_3d_viewer.html next to this file; plotly.js is inlined from the cibrrig
env's plotly package, so the page opens offline). Per trial and epoch (pre-gasp and post-gasp as their own epochs, as
in transition_windows.py): up to --n-max 5 ms bins coloured by Viterbi state, each state's mean position in that epoch
(states with more than 20 bins), and the epoch's mean cycle (PCs averaged in 50 phase bins, each vertex coloured by the
most common state in its phase bin, inspiration onset marked); plus each trial's baseline inspiration-off point.
Colour by state, epoch or trial; camera buttons look straight down each PC plane (orthographic) to match the
hmm_states_pca_K6_by_epoch*.png figures.

  cibrrig python states_pca_3d.py [--runs <run> <run> ...] [--out <html>] [--n-max 4000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import population_dynamics as pdm  # noqa: E402
import figures_population as fp  # noqa: E402
import transition_windows as tw  # noqa: E402
import transition_slides as tsl  # noqa: E402
from figures_population import state_colour  # noqa: E402

RUNS = [pdm.OUT / "population_dynamics_v2", pdm.OUT / "population_dynamics_anoxia2"]
TRIAL_COLOURS = ["#2f6db5", "#d0743c", "#3a9a5b", "#8a4fb0"]
CYCLE_BINS = 50
MIN_BINS = 20               # a state's mean is drawn only with more bins than this in the epoch (as in the 2D figures)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", nargs="+", default=[str(r) for r in RUNS],
                   help="population_dynamics.py output folders projected on the same PCA axes (first = reference)")
    p.add_argument("--out", default="", help="default <first run>/figures/hmm_states_pca_3d.html")
    p.add_argument("--n-max", type=int, default=4000, help="5 ms bins drawn per epoch (random, seed 0)")
    return p.parse_args()


def r3(a):
    """Rounded to 3 decimals, NaN (an empty phase bin) as None: JSON has no NaN."""
    return [None if np.isnan(v) else v for v in np.round(np.asarray(a, dtype=float), 3).tolist()]


def trial_data(c, n_max, rng):
    k = c["ref_k"]
    cond = tw.label_transitions(c["cond"])[0]
    bcond = tw.bin_conditions(c, cond)
    P, seq, ph, off = c["P"][:, :3], c["F"][k]["seq"], c["ph"], c["time_offset_s"]
    assert len(seq) == len(P) == len(ph), "states, PCs and phase are not on the same bins"
    ons = c["dia_df"]["on_sec"].to_numpy()
    colours = {**c["cond_colours"], **tw.TRANSITION_COLOURS}
    t = np.asarray(c["centres"]) + off
    epochs = []
    for name in tw.epoch_order(c, cond):
        idx = np.flatnonzero(bcond == name)
        sub = idx if idx.size <= n_max else np.sort(rng.choice(idx, n_max, replace=False))
        b = np.flatnonzero(cond == name)
        means = [[s, *r3(P[idx[seq[idx] == s]].mean(axis=0)), int((seq[idx] == s).sum())]
                 for s in range(k) if (seq[idx] == s).sum() > MIN_BINS]
        sel = bcond == name
        _, cyc = fp.phase_binned(P, ph, sel, nbins=CYCLE_BINS)
        _, dom = fp.phase_binned(np.eye(k)[seq], ph, sel, nbins=CYCLE_BINS)     # state fractions per phase bin
        dom = np.where(np.isnan(dom).all(axis=1), -1, np.nanargmax(np.nan_to_num(dom, nan=-1), axis=1))
        epochs.append({
            "name": name, "colour": colours.get(name, "#cccccc"), "n_breaths": int(b.size),
            "t0": round(float(ons[b[0]] + off), 1), "t1": round(float(ons[min(b[-1] + 1, ons.size - 1)] + off), 1),
            "n_bins": int(idx.size), "x": r3(P[sub, 0]), "y": r3(P[sub, 1]), "z": r3(P[sub, 2]),
            "s": seq[sub].astype(int).tolist(), "t": np.round(t[sub], 3).tolist(), "means": means,
            "cycle": {"x": r3(cyc[:, 0]), "y": r3(cyc[:, 1]), "z": r3(cyc[:, 2]), "s": dom.astype(int).tolist(),
                      "onset": CYCLE_BINS // 2},     # phase bins run -pi..pi; 0 = inspiration onset
        })
    return {"name": c.get("trial", "anoxia #1"), "ioff": r3(c["ioff_point"][:3]), "epochs": epochs}


def main():
    a = parse_args()
    runs = [Path(r) for r in a.runs]
    rng = np.random.default_rng(0)
    trials, c0 = [], None
    for i, run in enumerate(runs):
        c = tw.load_context(run)
        c0 = c0 or c
        trials.append({**trial_data(c, a.n_max, rng), "colour": TRIAL_COLOURS[i % len(TRIAL_COLOURS)]})
    k = c0["ref_k"]
    f = c0["F"][k]["feats"]
    data = {
        "k": k, "pca_fit": c0.get("pca_fit") or f"{trials[0]['name']}'s baseline", "n_max": a.n_max,
        "state_colours": [state_colour(c0, k, s) for s in range(k)],
        "state_names": [tsl.state_name(f.loc[s, "label"]) for s in range(k)],
        "state_order": [int(s) for s in np.argsort(c0["colours"][k])],
        "limits": [list(tw.PC_LIMITS[d]) for d in range(3)], "trials": trials,
    }
    import plotly.offline
    template = (HERE / "states_pca_3d_viewer.html").read_text(encoding="utf-8")
    html = (template.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
            .replace("__PLOTLY__", plotly.offline.get_plotlyjs()))
    out = Path(a.out) if a.out else runs[0] / "figures" / "hmm_states_pca_3d.html"
    out.write_text(html, encoding="utf-8")
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB; " +
          ", ".join(f"{t['name']}: {len(t['epochs'])} epochs" for t in trials) + ")")


if __name__ == "__main__":
    main()
