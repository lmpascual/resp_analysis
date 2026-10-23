"""e097 anoxia #2 in the population space of anoxia #1, ready for transition_windows.py / transition_slides.py.

Units: the anoxia #2 (B) partners of the reviewed anoxia #1 singles, from match_sorts.py's matched_pairs.tsv (baseline
templates, cosine >= 0.99 and r >= 0.90), labelled by their anoxia #1 ids so the two trials line up. A unit without a
partner (#422) is left out.
- HMM: no new fit. Anoxia #2 is decoded with the anoxia #1 K = 6 model (posteriors and Viterbi path), so a state means
  the same thing in both trials. A missing unit's factor is dropped from the Poisson emissions (exact for independent
  units).
- PCA: anoxia #2 is projected onto the anoxia #1 axes (fit on anoxia #1's baseline); a missing unit sits at its anoxia #1
  baseline mean. The inspiration-off point is anoxia #2's own baseline mean at inspiration offset.
- Breaths, phi, gasps, phase curves: as population_dynamics.py (cibrrig), from a NIDQ cache of this window.
- Conditions: baseline (60 s before the logged N2 switch, 38:30 = 2310 s), anoxia (15 s N2 + 10 s), gasping (Bush &
  Ramirez rule), and recovery epochs from change points in breathing + population state (epoch_change_points.py's
  method: 5 segments, >= 15 breaths each; the first is the post-gasp transition and is left unassigned).

Writes <out>/cache/ctx.pkl (same keys transition_windows.py reads from a population_dynamics run), breaths.tsv,
epochs_anoxia2.json/.png, unit_map.tsv.

  cibrrig python anoxia2_dynamics.py [--out <folder>]
  cibrrig python transition_windows.py --run <folder> --nidq-cache <folder>/cache/nidq_cache.npz
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import population_dynamics as pdm  # noqa: E402
import epoch_change_points as ecp  # noqa: E402
import transition_windows as tw  # noqa: E402

RUN_A = pdm.OUT / "population_dynamics_v2"
SORT_B = pdm.OUT / "anoxia2_window_v2" / "kilosort4" / "alf"
PAIRS = pdm.OUT / "anoxia_recovery" / "matched_units" / "matched_pairs.tsv"
NIDQ_BIN = Path(r"Z:/ramirezlab/budde/npx/animal_files/e097_g2/e097_g2_t0.nidq.bin")
OFFSET_B, DURATION_B = 2250.0, 360.0          # raw-run start of the anoxia #2 sort window (match_sorts --offset-b-s)
N2_ON, N2_OFF = 2310.0, 2325.0                # experiment log: anoxia #2 38:30-38:45
NAMES_B = ["post-gasp transition", "early recovery", "recovery 3", "recovery 4", "recovery 5"]
COLOURS_B = {**ecp.COLOURS, "recovery 3": "#1c9157", "recovery 4": "#3f8f5a", "recovery 5": "#0f6b3a"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-a", default=str(RUN_A))
    p.add_argument("--spikes-dir", default=str(SORT_B))
    p.add_argument("--pairs", default=str(PAIRS))
    p.add_argument("--n-segments", type=int, default=5)
    p.add_argument("--out", default=str(pdm.OUT / "population_dynamics_anoxia2"))
    return p.parse_args()


def nidq_cache(path):
    """Diaphragm EMG (XA0) and airflow (XA5) at 10 kHz from int(OFFSET_B * 10 kHz), as the anoxia #1 cache."""
    if not path.exists():
        s0, n = int(OFFSET_B * pdm.NIDQ_FS), int(DURATION_B * pdm.NIDQ_FS)
        raw = np.memmap(NIDQ_BIN, dtype="int16", mode="r")
        blk = np.asarray(raw[s0 * 9:(s0 + n) * 9]).reshape(-1, 9)
        np.savez(path, emg_v=blk[:, 0].astype(np.float32) * (10 / 32768),
                 flow_v=blk[:, 5].astype(np.float32) * (10 / 32768))
    return path


def matched_units(c, pairs_path, spikes_dir):
    pairs = pd.read_csv(pairs_path, sep="\t")
    pairs = pairs[pairs["match_class"].str.startswith("match")].set_index("unit_a")
    st = np.load(Path(spikes_dir) / "spike_times.npy")
    sc = np.load(Path(spikes_dir) / "spike_clusters.npy")
    keep = [u for u in c["ids"] if u in pairs.index]
    spikes = {u: np.sort(st[sc == int(pairs.loc[u, "unit_b"])]) for u in keep}
    umap = pd.DataFrame({"unit_a": keep, "unit_b": [int(pairs.loc[u, "unit_b"]) for u in keep],
                         "match_class": [pairs.loc[u, "match_class"] for u in keep],
                         "footprint_cosine": [pairs.loc[u, "footprint_cosine"] for u in keep],
                         "label_b": [pairs.loc[u, "label_b"] for u in keep],
                         "n_spikes_b": [spikes[u].size for u in keep]})
    return keep, spikes, umap


def recovery_epochs(dia_df, seq, k, dt, n_seg):
    """epoch_change_points.py's segmentation on the breaths after the last gasp (features: log IBI, Ti, state mix)."""
    ons = dia_df["on_sec"].to_numpy()
    gasp = dia_df["gasp"].to_numpy().astype(bool)
    first, last = int(np.flatnonzero(gasp).max()) + 1, len(dia_df) - 2
    idx = np.arange(first, last + 1)
    mix = np.stack([np.bincount(seq[int(ons[i] / dt):int(ons[i + 1] / dt)], minlength=k) for i in idx]).astype(float)
    mix /= np.maximum(mix.sum(axis=1, keepdims=True), 1)
    roll = lambda x, f: pd.Series(x).rolling(ecp.SMOOTH, center=True, min_periods=1).agg(f).to_numpy()  # noqa: E731
    breath = np.column_stack([roll(np.log(dia_df["ibi_s"].to_numpy()[idx]), "median"),
                              roll(dia_df["duration_sec"].to_numpy()[idx], "median")])
    pop = np.column_stack([roll(mix[:, s], "mean") for s in range(k) if mix[:, s].max() > 0.02])
    z = lambda X: (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9) / np.sqrt(X.shape[1])  # noqa: E731
    result = {}
    for key, Z in {"breathing": z(breath), "population": z(pop), "both": np.hstack([z(breath), z(pop)])}.items():
        starts, cost = ecp.segment(Z, n_seg, ecp.MIN_SEG)
        result[key] = {"start_breath": [int(idx[s]) for s in starts],
                       "start_raw_s": [round(float(ons[idx[s]] + OFFSET_B), 2) for s in starts], "cost": round(cost, 2)}
        print(f"{key:<11} starts (raw s): {result[key]['start_raw_s']}")
    starts = result["both"]["start_breath"] + [last + 1]
    rec = []
    for m, nm in enumerate(NAMES_B[:n_seg]):
        i0, i1 = starts[m], starts[m + 1]
        end = ons[i1] if i1 < len(dia_df) else ons[-1]
        rec.append({"name": nm, "start_raw_s": round(float(ons[i0] + OFFSET_B), 3),
                    "end_raw_s": round(float(end + OFFSET_B), 3), "n_breaths": int(i1 - i0),
                    "median_rate_per_min": round(float(60 / np.median(dia_df["ibi_s"].to_numpy()[i0:i1])), 1),
                    "source": "change point in breathing + population state (anoxia2_dynamics.py)",
                    **({"include": False} if nm in ecp.UNASSIGNED else {})})
    return idx, mix, result, rec


def main():
    a = parse_args()
    out = Path(a.out)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    c = tw.load_context(Path(a.run_a))
    k, dt = c["ref_k"], c["hmm_bin_s"]
    keep, spikes, umap = matched_units(c, a.pairs, a.spikes_dir)
    umap.to_csv(out / "unit_map.tsv", sep="\t", index=False, float_format="%.4f")
    print(f"{len(keep)} of {len(c['ids'])} anoxia #1 singles matched; missing: {sorted(set(c['ids']) - set(keep))}")
    cache = nidq_cache(out / "cache" / "nidq_cache.npz")
    dia_df, phi_t, phi, dia_sub, sr_sub = pdm.breaths_and_phase(str(cache), DURATION_B)

    # decode with the anoxia #1 model, dropping the factors of units without a partner
    cols = [c["ids"].index(u) for u in keep]
    n_bins = int(round(DURATION_B / dt))
    counts = pdm.bin_counts(spikes, keep, dt, n_bins)
    sigma_ms = float(np.load(Path(a.run_a) / "hmm_fits.npz")["sigma_ms"])
    X = gaussian_filter1d(counts, sigma_ms / 1e3 / dt, axis=0) if sigma_ms > 0 else counts
    m = copy.deepcopy(c["models"][k])
    m.lambdas_ = m.lambdas_[:, cols]
    m.n_features = len(cols)
    post, seq = m.predict_proba(X), m.predict(X)

    # conditions: fixed spans, Bush-rule gasps, recovery change points
    idx, mix, cp, rec = recovery_epochs(dia_df, seq, k, dt, a.n_segments)
    fixed = [{"name": "baseline", "start_raw_s": N2_ON - 60, "end_raw_s": N2_ON,
              "source": "60 s before the logged N2 switch (38:30)"},
             {"name": "anoxia", "start_raw_s": N2_ON, "end_raw_s": N2_OFF + 10,
              "source": "logged 15 s N2 (38:30-38:45) + 10 s"},
             {"name": "gasping", "rule": "gasp_flag",
              "source": "Bush & Ramirez 2024: 7-breath median IBI > 1 s until < 0.85 s"}]
    conditions = [{**x, **({"colour": COLOURS_B[x["name"]]} if x["name"] in COLOURS_B else {})} for x in fixed + rec]
    epochs_path = out / "epochs_anoxia2.json"
    epochs_path.write_text(json.dumps({"recording": "e097_g2 anoxia #2 window (raw-run time)", "ref_k": k,
                                       "decoded_with": "anoxia #1 K = 6 model", "min_segment_breaths": ecp.MIN_SEG,
                                       "smooth_breaths": ecp.SMOOTH, "conditions": conditions, "change_points": cp},
                                      indent=1), encoding="utf-8")
    ecp.plot(dia_df.assign(amp_z=(dia_df["amp"] - dia_df["amp"].mean()) / dia_df["amp"].std()), idx, mix,
             {f"k{k}_colours": c["colours"][k]}, k, cp, rec, epochs_path.with_suffix(".png"), offset_s=OFFSET_B,
             trial="anoxia #2 (anoxia #1 model)")
    pdm.set_conditions(epochs_path, OFFSET_B)
    cond = pdm.assign_conditions(dia_df, {})
    dia_df["condition"] = cond
    dia_df.to_csv(out / "breaths.tsv", sep="\t", index=False, float_format="%.4f")
    centres, ph, cyc, bcond = pdm.bin_labels(dia_df, cond, phi_t, phi, dt, n_bins)
    curve_bins, curves, tuning = pdm.phase_tuning(spikes, keep, dia_df, cond, phi_t, phi)

    # PCA on the anoxia #1 axes; a missing unit sits at its anoxia #1 baseline mean (contributes 0)
    pca = c["pca"]
    full = np.tile(pca.mean_, (n_bins, 1))
    Zb = np.sqrt(gaussian_filter1d(pdm.bin_counts(spikes, keep, pdm.PCA_BIN_S, n_bins),
                                   pdm.PCA_SIGMA_S / pdm.PCA_BIN_S, axis=0))
    full[:, cols] = Zb
    P = pca.transform(full)
    speed = np.r_[0, np.linalg.norm(np.diff(P[:, :3], axis=0), axis=1)] / pdm.PCA_BIN_S
    off_bins = np.floor(dia_df["off_sec"].to_numpy() / pdm.PCA_BIN_S).astype(int) + 1
    ioff_point = P[off_bins[(cond == "baseline") & (off_bins < n_bins)], :3].mean(axis=0)

    feats = c["F"][k]["feats"]                       # anoxia #1 state labels: the states are anoxia #1's
    F = {k: {"post": post, "seq": seq, "feats": feats, "rates_hz": c["F"][k]["rates_hz"][:, cols],
             "profiles": c["F"][k]["profiles"], "phase_centres": c["F"][k]["phase_centres"]}}
    info = c["info"].set_index("unit_id").loc[keep].reset_index()
    info["n_spikes"] = [spikes[u].size for u in keep]
    ctx = dict(out=out / "figures", time_offset_s=OFFSET_B, ids=keep, info=info,
               order_units=[u for u in c["order_units"] if u in keep], spikes=spikes, dia_df=dia_df, dia_sub=dia_sub,
               sr_sub=sr_sub, phi_t=phi_t, phi=phi, epochs={}, cond=cond, centres=centres, ph=ph, bcond=bcond, X=X,
               ks=[k], ref_k=k, F=F, colours={k: c["colours"][k]}, sims={}, sel=None, models={k: m},
               curve_bins=curve_bins, curves=curves, tuning=tuning, Z=Zb, pca=pca, P=P, speed=speed,
               ioff_point=ioff_point, pa_names=[], pa_M=None, conditions=[n for n, _ in pdm.CONDITIONS],
               palette=c["palette"], cond_colours=pdm.COND_COLOURS, hmm_bin_s=dt, pca_bin_s=pdm.PCA_BIN_S,
               trial="anoxia #2", pca_fit="anoxia #1's baseline", unit_map=umap)
    with open(out / "cache" / "ctx.pkl", "wb") as f:
        pickle.dump(ctx, f)
    g = dia_df[dia_df["gasp"]]
    print(f"{len(dia_df)} breaths; {len(g)} Bush-rule gasps {g['on_sec'].min() + OFFSET_B:.1f}-"
          f"{g['on_sec'].max() + OFFSET_B:.1f} s")
    for r in rec:
        print(f"{r['name']:<22} {r['start_raw_s']:8.1f}-{r['end_raw_s']:8.1f} s  {r['n_breaths']:4d} breaths  "
              f"{r['median_rate_per_min']:.0f}/min")
    occ = pd.Series(seq).value_counts(normalize=True).sort_index()
    print("state occupancy (anoxia #1 states):", {int(s): round(v, 3) for s, v in occ.items()})


if __name__ == "__main__":
    main()
