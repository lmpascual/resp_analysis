"""Gasping events in one long recording compared with each other: anoxic, spontaneous and terminal gasping.

Reads a population_dynamics.py run over the whole sort (cache/ctx.pkl), so every event is described in the same
units, the same baseline-fitted PCA space and the same HMM states. An event is a run of Bush-rule gasps with gaps
< --gap-s; its type comes from --type "name|start|end" spans (run time; first match wins, otherwise "other").

Per event (event_table.tsv):
- breathing: gasps, duration, median gasp interval, inspiratory duration and amplitude (z) of the gasps;
- population: summed rate of the baseline-active units relative to baseline, in the 60 s before the first gasp
  (split in 20 s thirds), during the gasps (gasp cycles), and 0-60 s after the last gasp;
- HMM (reference K): share of the gasping time in each state;
- PCA: mean position of the inter-gasp interval (gasp offset + 0.3 s to the next onset - 0.3 s) in PC1-3 and its
  median speed, the gasp's excursion (max PC1 in onset +-0.5 s minus the inter-gasp mean).
Fingerprints (fingerprints.npz, similarity figure): per event, each unit's mean rate around its gasps (onset -0.5 to
+1.0 s, 50 ms bins), and each unit's inter-gasp rate. Similarity between events = Pearson r of the sqrt
fingerprints. The test of "same as anoxic gasping": the mean similarity of spontaneous events to anoxic ones against
the anoxic events' similarity to each other, and a label permutation (all events of the two types shuffled).

    python gasp_events.py --run <out>/grp/e038/population_dynamics_long --type "anoxic|4850|5400" ^
        --type "spontaneous|5400|8400" --type "terminal|8400|9000" --out <out>/grp/e038/gasp_events
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import population_dynamics as pdm  # noqa: E402
from recovery_compare import load_context  # noqa: E402

FP_T = np.arange(-0.5, 1.0 + 1e-9, 0.05)             # gasp fingerprint bin edges around onset (s)


def events_from_gasps(d, gap_s):
    ons = d["on_sec"].to_numpy()
    g = np.flatnonzero(d["gasp"].to_numpy().astype(bool))
    if not g.size:
        return []
    br = np.flatnonzero(np.diff(ons[g]) > gap_s)
    starts, ends = np.r_[0, br + 1], np.r_[br, g.size - 1]
    return [g[s:e + 1] for s, e in zip(starts, ends)]


def rate_in(spikes, ids, spans):
    dur = sum(b - a for a, b in spans)
    return np.array([sum(((spikes[u] >= a) & (spikes[u] < b)).sum() for a, b in spans) / max(dur, 1e-9)
                     for u in ids])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--type", action="append", default=[], help="name|start run s|end run s")
    p.add_argument("--gap-s", type=float, default=15.0, help="gasps further apart than this start a new event")
    p.add_argument("--min-gasps", type=int, default=3)
    p.add_argument("--min-rate", type=float, default=0.5, help="baseline-active units: >= this (Hz)")
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    c = load_context(Path(a.run))
    off, d, spikes, ids_all = c["time_offset_s"], c["dia_df"], c["spikes"], c["ids"]
    ons, offs = d["on_sec"].to_numpy(), d["off_sec"].to_numpy()
    types = [(n, float(s) - off, float(e) - off) for n, s, e in (t.split("|") for t in a.type)]
    base = d["condition"].to_numpy() == "baseline"
    bi = np.flatnonzero(base[:-1])
    b_spans = list(zip(ons[bi], ons[bi + 1]))
    rb_all = rate_in(spikes, ids_all, b_spans)
    keep = rb_all >= a.min_rate
    ids = [u for u, k in zip(ids_all, keep) if k]
    rb = rb_all[keep]
    col = {u: j for j, u in enumerate(ids_all)}
    k = c["ref_k"]
    seq, colours = c["F"][k]["seq"], c["colours"][k]
    dt, P, speed = c["hmm_bin_s"], c["P"], c["speed"]
    feats = c["F"][k]["feats"]
    state_names = [f"s{i}_{feats.iloc[i]['label']}" for i in range(k)]

    rows, fps, igs = [], [], []
    for ev in events_from_gasps(d, a.gap_s):
        if ev.size < a.min_gasps:
            continue
        t_first, t_last_end = ons[ev[0]], ons[min(ev[-1] + 1, ons.size - 1)]
        ty = next((n for n, s, e in types if s <= t_first < e), "other")
        gspans = [(ons[i], ons[min(i + 1, ons.size - 1)]) for i in ev]
        r = {"event": len(rows) + 1, "type": ty, "first_gasp_raw_s": t_first + off, "last_gasp_end_raw_s": t_last_end + off,
             "n_gasps": int(ev.size), "duration_s": t_last_end - t_first,
             "median_gasp_interval_s": float(np.median(np.diff(ons[ev]))) if ev.size > 1 else np.nan,
             "gasp_ti_ms": float(np.median(d["duration_sec"].to_numpy()[ev]) * 1e3),
             "gasp_amp_z": float(np.median(d["amp_z"].to_numpy()[ev]))}
        for lab, (t0, t1) in {"pre_60_40": (t_first - 60, t_first - 40), "pre_40_20": (t_first - 40, t_first - 20),
                               "pre_20_0": (t_first - 20, t_first), "post_0_60": (t_last_end, t_last_end + 60)}.items():
            r[f"pop_rate_{lab}"] = rate_in(spikes, ids, [(max(t0, 0), t1)]).sum() / rb.sum()
        rg = rate_in(spikes, ids, gspans)
        r["pop_rate_gasping"] = rg.sum() / rb.sum()
        r["frac_units_silent_gasping"] = float(np.mean(rg < 0.1 * rb))
        bins = np.concatenate([np.arange(int(a0 / dt), int(a1 / dt)) for a0, a1 in gspans])
        bins = bins[bins < seq.size]
        occ = np.bincount(seq[bins], minlength=k) / max(bins.size, 1)
        for i in range(k):
            r[f"hmm_{state_names[i]}"] = occ[i]
        # inter-gasp intervals and the gasps in PC space
        ig = [(offs[i] + 0.3, ons[i + 1] - 0.3) for i in ev[:-1] if ons[i + 1] - 0.3 > offs[i] + 0.3]
        igb = np.concatenate([np.arange(int(a0 / c["pca_bin_s"]), int(a1 / c["pca_bin_s"])) for a0, a1 in ig]) \
            if ig else np.array([], int)
        igb = igb[igb < P.shape[0]]
        if igb.size:
            m = P[igb, :3].mean(axis=0)
            r.update({"intergasp_pc1": m[0], "intergasp_pc2": m[1], "intergasp_pc3": m[2],
                      "intergasp_speed": float(np.median(speed[igb]))})
            exc = [P[int((ons[i] - 0.5) / c["pca_bin_s"]):int((ons[i] + 0.5) / c["pca_bin_s"]), 0].max() - m[0]
                   for i in ev if (ons[i] + 0.5) / c["pca_bin_s"] < P.shape[0]]
            r["gasp_pc1_excursion"] = float(np.median(exc))
        # fingerprints
        fp = np.zeros((len(ids), FP_T.size - 1))
        for j, u in enumerate(ids):
            s = spikes[u]
            for i in ev:
                fp[j] += np.histogram(s - ons[i], FP_T)[0]
        fps.append(fp / (ev.size * np.diff(FP_T)[None, :]))
        igs.append(rate_in(spikes, ids, ig) if ig else np.full(len(ids), np.nan))
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "event_table.tsv", sep="\t", index=False, float_format="%.4f")
    fps, igs = np.array(fps), np.array(igs)
    np.savez_compressed(out / "fingerprints.npz", unit_ids=np.array(ids), fingerprints=fps, intergasp_rates=igs,
                        baseline_rates=rb, bins_s=FP_T, events=tab["event"].to_numpy(), types=tab["type"].to_numpy())

    # similarity between events
    n = len(tab)
    S = np.full((n, n), np.nan)
    Si = np.full((n, n), np.nan)
    for i, j in itertools.product(range(n), range(n)):
        S[i, j] = np.corrcoef(np.sqrt(fps[i]).ravel(), np.sqrt(fps[j]).ravel())[0, 1]
        ok = np.isfinite(igs[i]) & np.isfinite(igs[j])
        Si[i, j] = np.corrcoef(np.sqrt(igs[i][ok] / rb[ok]), np.sqrt(igs[j][ok] / rb[ok]))[0, 1] if ok.sum() > 3 else np.nan
    ty = tab["type"].to_numpy()
    tests = {}
    rng = np.random.default_rng(0)
    for name, M in (("gasp_fingerprint", S), ("intergasp_rates", Si)):
        res = {}
        for other in sorted(set(ty) - {"anoxic"}):
            sel = np.flatnonzero((ty == "anoxic") | (ty == other))
            lab = ty[sel]
            M2 = M[np.ix_(sel, sel)]

            def stat(lb):
                aa = [M2[x, y] for x in range(len(lb)) for y in range(x + 1, len(lb)) if lb[x] == lb[y] == "anoxic"]
                ao = [M2[x, y] for x in range(len(lb)) for y in range(len(lb)) if lb[x] == "anoxic" and lb[y] == other]
                oo = [M2[x, y] for x in range(len(lb)) for y in range(x + 1, len(lb)) if lb[x] == lb[y] == other]
                within = np.nanmean(aa + oo) if (aa + oo) else np.nan
                return within - np.nanmean(ao), np.nanmean(aa) if aa else np.nan, np.nanmean(ao), np.nanmean(oo) if oo else np.nan
            obs = stat(lab)
            null = np.array([stat(rng.permutation(lab))[0] for _ in range(a.n_perm)])
            res[other] = {"anoxic_vs_anoxic_r": obs[1], f"anoxic_vs_{other}_r": obs[2], f"{other}_vs_{other}_r": obs[3],
                          "within_minus_between": obs[0],
                          "perm_p": float((np.sum(null >= obs[0]) + 1) / (a.n_perm + 1)) if np.isfinite(obs[0])
                          else np.nan,
                          "n_events": {t: int((lab == t).sum()) for t in ("anoxic", other)}}
        tests[name] = res
    (out / "event_similarity_tests.json").write_text(json.dumps(tests, indent=1, default=float))
    np.savez_compressed(out / "event_similarity.npz", gasp_fingerprint=S, intergasp_rates=Si)

    # figures: similarity matrices, and the fingerprints themselves
    names = [f"{r.event}: {r.type} {r.first_gasp_raw_s:.0f} s ({r.n_gasps})" for r in tab.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, M, t in ((axes[0], S, "gasp fingerprints (units x time around gasp onset)"),
                     (axes[1], Si, "inter-gasp firing (rate / baseline, units)")):
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=90, fontsize=7); ax.set_yticklabels(names, fontsize=7)
        ax.set_title(t, fontsize=9)
        for i, j in itertools.product(range(n), range(n)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5, color="w" if M[i, j] < 0.6 else "k")
    fig.colorbar(im, ax=axes, shrink=0.6, label="Pearson r (sqrt rates)")
    fig.suptitle(f"{c.get('title', '')}: similarity of gasping events ({len(ids)} baseline-active units)", fontsize=10)
    fig.savefig(out / "event_similarity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    order = np.argsort(-np.nanmax(np.nan_to_num(fps.mean(axis=0)), axis=1))
    fig, axes = plt.subplots(1, n, figsize=(1.6 * n + 1, 8), sharey=True)
    vmax = np.nanpercentile(np.sqrt(fps), 99)
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.imshow(np.sqrt(fps[i][order]), aspect="auto", vmin=0, vmax=vmax, cmap="magma",
                  extent=(FP_T[0], FP_T[-1], len(ids), 0))
        ax.axvline(0, color="c", lw=0.6)
        ax.set_title(names[i], fontsize=6)
        ax.set_xlabel("s from gasp onset", fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("unit (sorted by peak rate)")
    fig.suptitle("gasp-aligned firing per event (sqrt Hz)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "event_fingerprints.png", dpi=150)
    plt.close(fig)
    print(tab[["event", "type", "first_gasp_raw_s", "n_gasps", "pop_rate_pre_20_0", "pop_rate_gasping",
               "pop_rate_post_0_60"]].to_string(index=False))
    print(json.dumps(tests, indent=1, default=float))


if __name__ == "__main__":
    main()
