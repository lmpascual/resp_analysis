"""Why does e097 show a gasping rotation when m2020-19 g3/g4 largely don't?

`gasping_native_rotation.py` measures each epoch's phase-averaged cycle in its OWN basis and scores
it by enclosed area (RI: 1 = circle, 0 = a retraced line). In e097 the gasping cycle rotates
(RI 1.13); on our sorts of the 2024 paper's own animals m2020-19 g3/g4 it reads 0.05-0.11. Either
e097 is special, or the measure is. These are the controls that decide between them.

Each control is a separate --test, and each either survived or did not (analysis record, notes §11):

  recruitment    gasp-recruited units per recording, by the paper's ED Fig 10g firing-rate bins,
                 with probe depth.  NOT SUPPORTED as the explanation: g3 has MORE gasp-recruited
                 units than e097 (9 vs 3) and comparable depth coverage.
  probe          e097 restricted to one end of the probe, or with gasp-recruited units dropped.
                 Dropping them RAISES RI to 1.20 and both halves rotate, so they are not the
                 carrier.
  power          is RI stable at 16-28 units?  RULED OUT: e097 subsampled to g4's 16 units gives
                 median RI 1.09, only 8% of draws below 0.3; drop-one jackknife spans 1.04-1.19.
  coherence      do units shift preferred phase by a COMMON amount (reorganisation, ordering
                 survives) or every which way (scrambling)?  REFUTED, in the wrong direction:
                 e097 R = 0.21 is the LEAST coherent and rotates most.
  tiling         do units still spread round the cycle during gasping, or bunch at one phase?
                 Bunching is the mechanism behind rank-1.  NOT SUPPORTED: R 0.23-0.30 everywhere.
  modulation     restrict to respiratory-modulated units.  PARTLY SUPPORTED and the most useful:
                 g4's gasping RI goes 0.11 (all 16 units) -> 1.04 (9 phasic) -> 1.63 (4 strongly
                 phasic).  Including weakly modulated units dilutes a cycle measure -- which was an
                 analysis choice of mine, and is also what B24/B25 do.
  modulation-cv  the honest version of the above: units selected on ODD gasps, rotation measured on
                 EVEN gasps, so selection cannot manufacture the cycle.  Leaves a GRADIENT rather
                 than a contradiction -- e097 1.04 > g4 0.70 > g3 0.38.

Two different modulation measures appear here and MUST NOT be mixed: `modulation_L` from the
pipeline's tuning table is occupancy-corrected, while `cycle_resultant_L` is recomputed from the
mean cycle. They disagree on counts (3 vs 16 phasic units in g3). --test modulation uses the
pipeline column; --test modulation-cv uses the recomputed one throughout, including for selection.

Still untested and not separable with these data: inspired O2 (e097 is room air, B24 is 100% O2)
and sorting quality (e097 is manually checked with minimal drift; confidence in m2020-19 is lower
even after waveform anchoring). Both predict the observed gradient.

  cibrrig python gasping_rotation_controls.py [--test <name> ...] [--runs <label>=<run> ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gasping_native_rotation import (N_PHASE, cycle_geometry, cycle_index,  # noqa: E402
                                     load_ctx, mean_cycle, rates_hz, soft_normalise)

OUT = r"C:\Users\lpascu\resp_analysis\outputs"
RUNS = {
    "e097 anoxia1": rf"{OUT}\e097\population_dynamics_v2",
    "e097 anoxia2": rf"{OUT}\e097\population_dynamics_anoxia2",
    "g3 anchor": rf"{OUT}\bush2024\m2020-19_g3\population_dynamics_anchor",
    "g4 anchor": rf"{OUT}\bush2024\m2020-19_g4\population_dynamics_anchor",
    "g3 uncorrected": rf"{OUT}\bush2024\m2020-19_g3\population_dynamics_uncorrected",
    "g4 uncorrected": rf"{OUT}\bush2024\m2020-19_g4\population_dynamics_uncorrected",
}
# the three the notes compare; the uncorrected sorts are only wanted for the sorting contrasts
PRIMARY = ["e097 anoxia1", "g3 anchor", "g4 anchor"]
# B24 ED Fig 10g categories, on the gasping/baseline firing-rate ratio
RECRUIT_BINS = [(-np.inf, 0.2, "eupnea only"), (0.2, 0.7, "gasp decreasing"), (0.7, 1.5, "stable"),
                (1.5, 5.0, "gasp increasing"), (5.0, np.inf, "gasp only")]
TONIC_L = 0.2       # the pipeline's phasic/tonic threshold
N_DRAWS = 200


# ------------------------------------------------------------------ shared helpers

def prep(run):
    """Soft-normalised rates plus the masks every control needs."""
    ctx = load_ctx(run)
    F = soft_normalise(rates_hz(ctx["Z"]))
    bcond = np.asarray(ctx["bcond"])
    return ctx, F, np.asarray(ctx["ph"], float), bcond == "gasping", bcond == "baseline"


def ri(F, ph, sel, cols=None):
    """Rotation index of the mean cycle over `cols`, in that cycle's own leading plane."""
    cols = np.arange(F.shape[1]) if cols is None else np.asarray(cols)
    if len(cols) < 4:
        return np.nan
    c = mean_cycle(F[:, cols], ph, sel)
    return np.nan if c is None else cycle_geometry(c)[0]["rotation_index_cycle"]


def fr_ratio(ctx):
    """Per-unit gasping/baseline firing rate, in ctx['ids'] order."""
    w = ctx["tuning"].pivot_table(index="unit_id", columns="condition", values="rate_hz")
    if "gasping" not in w or "baseline" not in w:
        return None
    return ((w["gasping"] + 1e-9) / (w["baseline"] + 1e-9)).reindex(ctx["ids"])


def depths(ctx):
    return ctx["info"].set_index("unit_id").reindex(ctx["ids"])["depth_um"].to_numpy(float)


def pipeline_mod_L(ctx, cond="gasping"):
    """The tuning table's occupancy-corrected modulation_L, in ctx['ids'] order."""
    t = ctx["tuning"]
    return (t[t["condition"] == cond].set_index("unit_id")["modulation_L"]
            .reindex(ctx["ids"]).to_numpy(float))


def cycle_resultant_L(cycle, n_phase=N_PHASE):
    """Per-unit resultant length recomputed from the mean cycle -- NOT the pipeline's column."""
    centres = np.arange(n_phase) * 2 * np.pi / n_phase - np.pi + np.pi / n_phase
    p = np.clip(cycle, 0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs((p * np.exp(1j * centres)[:, None]).sum(axis=0) / p.sum(axis=0))


def concentration(ph, w=None):
    """Circular resultant of preferred phases; high = bunched at one phase, low = tiled.

    Units with no modulation_L are dropped rather than poisoning the weighted sum with NaN.
    """
    ph = np.asarray(ph, float)
    if w is None:
        ok = np.isfinite(ph)
        return float(np.abs(np.mean(np.exp(1j * ph[ok])))) if ok.any() else np.nan
    w = np.asarray(w, float)
    ok = np.isfinite(ph) & np.isfinite(w)
    if not ok.any() or np.sum(w[ok]) == 0:
        return np.nan
    return float(np.abs(np.sum(w[ok] * np.exp(1j * ph[ok])) / np.sum(w[ok])))


def odd_even_masks(F, ph, sel, centres, onsets):
    """Bin masks for odd- and even-numbered cycles of an epoch."""
    cyc = cycle_index(centres, sel, onsets)
    idx = np.where(sel)[0]
    odd, even = np.zeros(len(F), bool), np.zeros(len(F), bool)
    odd[idx[(cyc % 2 == 1) & (cyc >= 0)]] = True
    even[idx[(cyc % 2 == 0) & (cyc >= 0)]] = True
    return odd, even


# ------------------------------------------------------------------ controls

def test_recruitment(runs):
    """A. Does e097 simply sample gasp-recruited units the other recordings miss?"""
    for name in runs:
        ctx, F, ph, g, b = prep(RUNS[name])
        r, d = fr_ratio(ctx), depths(ctx)
        print(f"\n{name}  (n={len(ctx['ids'])}, depth {np.nanmin(d):.0f}-{np.nanmax(d):.0f} um)")
        if r is None:
            print("  no gasping tuning rows")
            continue
        rv = r.to_numpy()
        for lo, hi, lab in RECRUIT_BINS:
            m = (rv >= lo) & (rv < hi)
            if m.sum():
                print(f"    {lab:<17} n={m.sum():>3}   depth median {np.nanmedian(d[m]):>7.0f} um"
                      f"   [{np.nanmin(d[m]):.0f}-{np.nanmax(d[m]):.0f}]")


def test_probe(runs):
    """B. Is e097's gasping rotation carried by one end of the probe, or by recruited units?"""
    name = "e097 anoxia1"
    ctx, F, ph, g, b = prep(RUNS[name])
    d, r = depths(ctx), fr_ratio(ctx).to_numpy()
    allc = np.arange(F.shape[1])
    med = np.nanmedian(d)
    subsets = {
        "all units": allc,
        f"shallow half (<{med:.0f} um, near tip)": allc[d < med],
        f"deep half (>={med:.0f} um)": allc[d >= med],
        "drop gasp-recruited (ratio>=1.5)": allc[r < 1.5],
        "gasp-recruited only (ratio>=1.5)": allc[r >= 1.5],
        "drop gasp-suppressed (ratio<0.7)": allc[r >= 0.7],
    }
    print(f"\n{name}")
    for lab, cols in subsets.items():
        print(f"  {lab:<40} n={len(cols):>3}   gasping RI {ri(F, ph, g, cols):>6.2f}"
              f"   baseline RI {ri(F, ph, b, cols):>6.2f}")


def test_power(runs):
    """C. Is RI stable at these unit counts, or is e097 a lucky draw?"""
    name = "e097 anoxia1"
    ctx, F, ph, g, b = prep(RUNS[name])
    allc = np.arange(F.shape[1])
    d, r = depths(ctx), fr_ratio(ctx).to_numpy()
    rng = np.random.default_rng(0)
    print(f"\n{name} gasping, {N_DRAWS} draws per size")
    for n in (16, 20, 24, 28):
        if n > len(allc):
            continue
        v = np.array([ri(F, ph, g, rng.choice(allc, n, replace=False)) for _ in range(N_DRAWS)])
        print(f"  n={n:>3}   median {np.nanmedian(v):>5.2f}   IQR [{np.nanpercentile(v, 25):.2f},"
              f" {np.nanpercentile(v, 75):.2f}]   5-95% [{np.nanpercentile(v, 5):.2f},"
              f" {np.nanpercentile(v, 95):.2f}]   frac<0.3 {np.mean(v < 0.3):.2f}")
    jk = np.array([ri(F, ph, g, np.delete(allc, i)) for i in range(len(allc))])
    w = int(np.nanargmin(jk))
    print(f"\n  drop-one jackknife: RI range {np.nanmin(jk):.2f}-{np.nanmax(jk):.2f}; most "
          f"influential unit {ctx['ids'][w]} (depth {d[w]:.0f} um, ratio {r[w]:.2f}) -> {jk[w]:.2f}")


def test_coherence(runs):
    """D. Do units shift preferred phase by a common amount, or every which way?"""
    print(f"\n{'run':<17}{'n':>4}{'|shift|':>9}{'exp->insp':>12}{'insp->exp':>11}{'coherence R':>13}")
    print("-" * 66)
    for name in runs:
        ctx = load_ctx(RUNS[name])
        w = ctx["tuning"].pivot_table(index="unit_id", columns="condition",
                                      values="preferred_phase")
        if "gasping" not in w or "baseline" not in w:
            continue
        p = w[["baseline", "gasping"]].dropna()
        delta = np.angle(np.exp(1j * (p["gasping"] - p["baseline"]).to_numpy()))
        R = float(np.abs(np.mean(np.exp(1j * delta))))
        e2i = int(((p["baseline"] < 0) & (p["gasping"] >= 0)).sum())
        i2e = int(((p["baseline"] >= 0) & (p["gasping"] < 0)).sum())
        n_e, n_i = int((p["baseline"] < 0).sum()), int((p["baseline"] >= 0).sum())
        print(f"{name:<17}{len(p):>4}{np.median(np.abs(delta)):>9.2f}"
              f"{f'{e2i}/{n_e}':>12}{f'{i2e}/{n_i}':>11}{R:>13.2f}")
    print("\n  R near 1: every unit shifts by about the same amount -> the population reorganises")
    print("            coherently and an orderly phase ordering, hence a rotation, survives.")
    print("  R near 0: units shift every which way -> phase ordering is scrambled, no rotation.")


def test_tiling(runs):
    """E. Do units still tile the cycle during gasping, or bunch at one phase (rank-1)?"""
    print(f"\n{'run':<17}{'cond':<10}{'n':>4}{'tiling R':>10}{'R weighted':>12}{'mean modL':>11}")
    print("-" * 64)
    for name in runs:
        t = load_ctx(RUNS[name])["tuning"]
        for cond in ("baseline", "gasping"):
            s = t[t["condition"] == cond].dropna(subset=["preferred_phase"])
            if not len(s):
                continue
            print(f"{name:<17}{cond:<10}{len(s):>4}"
                  f"{concentration(s['preferred_phase']):>10.2f}"
                  f"{concentration(s['preferred_phase'], s['modulation_L']):>12.2f}"
                  f"{s['modulation_L'].mean():>11.2f}")
        print()
    print("  If bunching explained the difference, gasping tiling R would be LOW in e097 (units")
    print("  still spread round the cycle) and HIGH in m2020-19 (units co-active at the gasp).")


def test_modulation(runs):
    """F. Does restricting to respiratory-modulated units recover a rotation?"""
    print(f"\n{'run':<15}{'subset':<34}{'n':>4}{'gasp RI':>9}{'base RI':>9}{'mean modL':>11}")
    print("-" * 82)
    for name in runs:
        ctx, F, ph, g, b = prep(RUNS[name])
        modg = pipeline_mod_L(ctx)
        allc = np.arange(F.shape[1])
        med = np.nanmedian(modg)
        for lab, cols in [("all units", allc),
                          (f"best-modulated half (L>={med:.2f})", allc[modg >= med]),
                          (f"worst-modulated half (L<{med:.2f})", allc[modg < med]),
                          (f"phasic only (L>={TONIC_L})", allc[modg >= TONIC_L]),
                          ("strongly phasic (L>=0.4)", allc[modg >= 0.4])]:
            if len(cols) < 4:
                print(f"{name:<15}{lab:<34}{len(cols):>4}{'--':>9}{'--':>9}{'(too few)':>11}")
                continue
            print(f"{name:<15}{lab:<34}{len(cols):>4}{ri(F, ph, g, cols):>9.2f}"
                  f"{ri(F, ph, b, cols):>9.2f}{np.nanmean(modg[cols]):>11.2f}")
        print()


def test_modulation_cv(runs):
    """G. Select phasic units on ODD gasps, measure the rotation on EVEN gasps.

    Selecting units by modulation and then measuring the cycle on the same gasps could manufacture
    the rotation that --test modulation recovers. Here selection and measurement use disjoint
    cycles, so it cannot. Selection uses `cycle_resultant_L`, not the pipeline column.
    """
    print(f"\n{'run':<15}{'n sel (odd)':>12}{'RI even, selected':>20}{'RI even, all':>14}")
    print("-" * 61)
    for name in runs:
        ctx, F, ph, g, b = prep(RUNS[name])
        dia = ctx["dia_df"]
        onsets = np.sort(dia.loc[dia["condition"] == "gasping", "on_sec"].to_numpy())
        odd, even = odd_even_masks(F, ph, g, np.asarray(ctx["centres"], float), onsets)
        c_odd, c_even = mean_cycle(F, ph, odd), mean_cycle(F, ph, even)
        if c_odd is None or c_even is None:
            print(f"{name:<15}{'--':>12}{'(empty phase bin)':>20}")
            continue
        keep = np.where(cycle_resultant_L(c_odd) >= TONIC_L)[0]   # selection: ODD gasps only
        ri_all = cycle_geometry(c_even)[0]["rotation_index_cycle"]
        if len(keep) < 4:
            print(f"{name:<15}{len(keep):>12}{'(too few)':>20}{ri_all:>14.2f}")
            continue
        ri_sel = cycle_geometry(c_even[:, keep])[0]["rotation_index_cycle"]
        print(f"{name:<15}{len(keep):>12}{ri_sel:>20.2f}{ri_all:>14.2f}")


TESTS = {
    "recruitment": test_recruitment,
    "probe": test_probe,
    "power": test_power,
    "coherence": test_coherence,
    "tiling": test_tiling,
    "modulation": test_modulation,
    "modulation-cv": test_modulation_cv,
}
# the sorting contrasts are the only ones that want the uncorrected sorts alongside the anchored
WANTS_UNCORRECTED = {"coherence", "tiling"}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--test", nargs="+", default=list(TESTS), choices=list(TESTS))
    p.add_argument("--runs", nargs="+", metavar="LABEL=PATH",
                   help="override the run table, e.g. 'g5 anchor=D:/out/.../population_dynamics'")
    a = p.parse_args()
    if a.runs:
        RUNS.clear()
        RUNS.update(dict(r.split("=", 1) for r in a.runs))
        PRIMARY[:] = list(RUNS)

    for name in a.test:
        runs = [r for r in RUNS if r in PRIMARY or name in WANTS_UNCORRECTED]
        print("\n" + "#" * 78)
        print(f"# {name}  -- {TESTS[name].__doc__.splitlines()[0]}")
        print("#" * 78)
        TESTS[name]([r for r in runs if (Path(RUNS[r]) / "cache" / "ctx.pkl").exists()])


if __name__ == "__main__":
    main()
