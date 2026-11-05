"""The e097 population figure (SfN 2026 poster) from the curated runs: npx_pipeline's PF8 drawn on the units reviewed single.

npx_pipeline (Ryan's project) reran this chain on the Kilosort 4 units bombcell labels good, without the manual review,
and redrew the poster's panels as PF8 (npxpipe/plotting_pf.py, pf8_population). This script draws the same panels from
population_dynamics_v2 (anoxia #1) and population_dynamics_anoxia2 (anoxia #2, decoded with the anoxia #1 model on the
anoxia #1 PCA axes), so the units are the 28 reviewed single. Changes from PF8:
- the raster units are given (--units, any number; PF8 picks two by a rule). The default is one example of each kind
  of change a phase screen finds (phase_screen: every unit's preferred phase, L and rate per epoch against baseline);
  unit_raster_gallery.png draws every unit, grouped by what the screen finds, to choose from;
- the unit rasters have the state raster's rows (all 959 breaths) and window (-300 to 300 ms from inspiration onset;
  PF8's state raster: -400 to 400 ms), so their rows line up with it; each row shows every spike in the window, the
  previous and next breaths' spikes included, with the other breaths' onsets in grey and no offset ticks, so the
  bursts stand alone (PF8: 0.3 s before onset to the next onset, offsets and next onset ticked, on the last 15 breaths
  of the anoxia to the first 15 of early recovery);
- the trajectories and the state panels are in PC1-PC3, where these units' breath loop lies (PF8: trajectories also in
  PC1-PC2 and PC2-PC3, states in PC1-PC2; --states-plane 1,2 for PF8's state plane).
--crop draws a second version on fewer breaths (crop_breaths): from the second half of the baseline to the first half
of the epoch after early recovery (mid recovery in anoxia #1, recovery 3 in anoxia #2), so the gasping rows take a
bigger share of the rasters. Every panel, the phase screen and the gallery use only those breaths, except the state
panels, which also draw the epochs left out after them whole (slow breathing and later recovery); the PCA and the HMM
stay as fit on the whole run. Its files end in _crop.

  top left      mean breath trajectory of every epoch, both trials, in PC1-PC3
  top right     the K = 6 HMM states by epoch (anoxia #1)
  bottom        example stretches of three epochs; the K = 6 state around every breath's onset; the unit rasters

Writes into --out (default <anoxia #1 run>/figures): population_figure.png (300 dpi), .svg (text kept as text, Arial
embedded), _poster.png (300 dpi across a 48 in poster side) and population_figure_caption.txt;
unit_raster_gallery.png (every unit's raster), unit_phase_screen.tsv (the screen per unit) and
unit_phase_screen_epochs.tsv (each unit's phase, L, rate and spikes per epoch). Draws in about 20 s.

  cibrrig python population_figure.py [--units 350,510,331,362,459,422] [--states-plane 1,3] [--crop] [--out <folder>]
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import math
import re
import sys
from pathlib import Path

import textwrap

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import population_dynamics as pdm  # noqa: E402
import transition_windows as tw  # noqa: E402
import figures_population as fp  # noqa: E402
import figures_signatures as fsig  # noqa: E402
import transition_slides as tsl  # noqa: E402
from figures_population import save_fig, state_colour  # noqa: E402

import matplotlib  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

RUNS = {"anoxia #1": pdm.OUT / "population_dynamics_v2", "anoxia #2": pdm.OUT / "population_dynamics_anoxia2"}
# examples of the changes the phase screen finds (unit_raster_gallery.png has every unit): #350 moves from mid-E into
# pre-I and stays there; #510 moves from mid-E into pre-I after the gasps and back; the inspiratory core #331, #362
# (both fire earlier around gasping) and #459 (no change); #422 is silent at baseline and recruited around onset
UNITS = (350, 510, 331, 362, 459, 422)
UNIT_WIDTH = 0.85                    # width ratio of one unit raster in the bottom row
FONT = "Arial"
DPI = 300
POSTER_IN, POSTER_DPI = 48.0, 300   # the poster PNG holds POSTER_DPI across a 48 in side (a 36 x 48 in poster)

FIGSIZE = (26.0, 17.0)
TRAJ_BINS = 50                       # 25 inspiration + 25 expiration (her mean_cycle)
TRAJ_PLANES = ((0, 2),)              # PC1-PC3, the plane of the breath loop (PF8: also PC1-PC2 and PC2-PC3)
TRAJ_BOX = 1.0                       # trajectory panel height / width
TRAJ_THICK, TRAJ_LW = ("baseline", "gasping"), (1.4, 3.2)
TRAJ_GREENS = ("#9fd38f", "#0f6b3a")   # every epoch after the post-gasp on one ramp, early lightest
# the plane with the breath loop: PC1-PC3 for these units, as on the poster (PF8 draws PC1-PC2, where the loop lies for
# the bombcell-good units; here PC2 is mostly unit #89's step and flattens the loop)
STATES_PLANE = (0, 2)
STATES_MAX_POINTS, STATES_MIN_BINS, STATE_DOT = 6000, 20, 110
EXAMPLES = ("baseline", "gasping", "later recovery")
EXAMPLE_DEFAULT_S = (0.15, 1.15)     # (before, after) the example breath's onset; tw.EXAMPLE_S for the transitions
STATE_RASTER_MS = (-300.0, 300.0)
UNIT_RASTER_MS = STATE_RASTER_MS     # the unit rasters share the state raster's window and rows, so they line up
GALLERY_MAX_BREATHS = 40             # gallery rows per epoch, evenly spaced (her cycles_of); shorter epochs whole
LABEL_GAP_PT = 27                    # height of a two-line epoch label beside the unit rasters

# The phase screen behind the gallery, fixed before looking at the units. An epoch's phase (the mean breath phase of
# its spikes, epoch_tuning) counts when the unit has >= tw.MIN_SPIKES spikes and L >= pdm.TONIC_MAX_L there and at
# baseline; it is a shift when it moves >= SHIFT_DEG.
SHIFT_DEG = 30
ZONES = ((-180, -120, "post-I"), (-120, -60, "mid-E"), (-60, 0, "pre-I"), (0, 90, "early I"), (90, 180, "late I"))
SILENT_FRACTION = 0.1                # silent: under this fraction of a baseline rate >= SILENT_MIN_HZ, in a recovery epoch
SILENT_MIN_HZ = 2.0
KINDS = ("→ pre-I", "→ inspiration", "→ expiration", "earlier", "later", "recruited", "silent in recovery",
         "gains modulation", "loses modulation")
SHORT = {"baseline": "base", "anoxia": "anoxia", "pre-gasp": "pre-G", "gasping": "gasp", "post-gasp": "post-G",
         "early recovery": "early", "mid recovery": "mid", "slow breathing": "slow", "later recovery": "later"}
INK, INK2, GREY, TICK = "#0b0b0b", "#52514e", "#e3e2dd", "#8f8d86"
FS = dict(tick=11, label=13, title=14, small=10)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--units", default=",".join(map(str, UNITS)), help="units (anoxia #1 ids) for the rasters")
    p.add_argument("--states-plane", default="{},{}".format(*(i + 1 for i in STATES_PLANE)),
                   help="the two PCs of the state panels (PF8: 1,2)")
    p.add_argument("--crop", action="store_true",
                   help="second half of the baseline to the first half of the mid recovery only; files end in _crop")
    p.add_argument("--out", default="", help="default <anoxia #1 run>/figures")
    a = p.parse_args()
    a.units = [int(u) for u in a.units.split(",")]
    a.states_plane = tuple(int(i) - 1 for i in a.states_plane.split(","))
    a.out = Path(a.out or RUNS["anoxia #1"] / "figures")
    return a


def epoch_tuning(c, cond):
    """Her phase_tuning in every epoch (pre- and post-gasp included), with each unit's spikes in the epoch's cycles (as
    transition_windows.zoom_context counts them) and their mean phase: phase_deg, the circular mean of the breath phase
    (her phi) at each of those spikes, and the class from it. Her preferred phase (rate_phase_deg) is the vector mean of
    the rate-per-phase curve, which weights the brief inspiration up against the long expirations of the gasping and the
    recovery: #510, firing just before onset after the gasps (76 % of its post-gasp spikes in -60 to 0 deg), sits at +4
    deg by it and at -11 deg by its spikes."""
    eps = tw.epoch_order(c, cond)
    saved, pdm.CONDITIONS = pdm.CONDITIONS, [(n, None) for n in eps]     # phase_tuning only reads the names
    try:
        tu = pdm.phase_tuning(c["spikes"], c["ids"], c["dia_df"], cond, c["phi_t"], c["phi"])[2]
    finally:
        pdm.CONDITIONS = saved
    ons = c["dia_df"]["on_sec"].to_numpy()
    nxt = np.r_[ons[1:], np.inf]
    phi_t, phi = np.asarray(c["phi_t"]), np.asarray(c["phi"])
    n, mean = {}, {}
    for e in eps:
        i = np.flatnonzero(cond == e)
        for u in c["ids"]:
            ts = c["spikes"][u]
            t = np.concatenate([ts[a:b] for a, b in zip(np.searchsorted(ts, ons[i]), np.searchsorted(ts, nxt[i]))])
            n[(u, e)] = t.size
            p = phi[np.clip(np.searchsorted(phi_t, t), 0, phi.size - 1)]
            mean[(u, e)] = np.degrees(np.angle(np.exp(1j * p).sum())) if t.size else np.nan
    key = list(zip(tu["unit_id"], tu["condition"]))
    tu["n_spikes"] = [n[k] for k in key]
    tu["unit_id"] = tu["unit_id"].astype(int)
    tu["rate_phase_deg"] = wrap(np.degrees(tu["preferred_phase"]))
    tu["phase_deg"] = wrap(np.array([mean[k] for k in key]))
    tu["class"] = np.where(~(tu["modulation_L"] >= pdm.TONIC_MAX_L), "tonic",
                           np.where(tu["phase_deg"] >= 0, "inspiratory", "expiratory"))
    return tu


def crop_breaths(cond, epochs):
    """--crop: cond with the first half of the baseline, the second half of the epoch after early recovery and every
    breath after it unlabelled (""), as the run's last breath is (each half keeps the middle breath)."""
    cond = cond.copy()
    b = np.flatnonzero(cond == "baseline")
    cond[b[:b.size // 2]] = ""
    m = np.flatnonzero(cond == epochs[epochs.index("early recovery") + 1])
    cond[m[(m.size + 1) // 2]:] = ""
    return cond


def trial_data(run, tuning=False, crop=False):
    """One trial's arrays as PF8 reads them (npx_pipeline tools/pf8_extract.py), with pre- and post-gasp labelled;
    with tuning, every unit's phase tuning per epoch; with crop, only crop_breaths' breaths labelled (cond_all: all;
    states_*: crop's breaths and, whole, the epochs it leaves out after them, for the state panels)."""
    c = tw.load_context(run)
    cond_all = np.asarray(tw.label_transitions(c["cond"])[0]).astype(str)
    cond = crop_breaths(cond_all, tw.epoch_order(c, cond_all)) if crop else cond_all
    rows = np.flatnonzero(cond != "") if crop else np.arange(cond.size)
    after = ~np.isin(cond_all, tw.epoch_order(c, cond)) & (np.arange(cond.size) > rows[-1])
    states_cond = np.where(after, cond_all, cond)
    k = c["ref_k"]
    F = c["F"][k]
    labels = F["feats"]["label"]
    return dict(
        tuning=epoch_tuning(c, cond) if tuning else None,
        P=np.asarray(c["P"][:, :3]), ph=np.asarray(c["ph"]), seq=np.asarray(F["seq"]), post=np.asarray(F["post"]),
        on_sec=c["dia_df"]["on_sec"].to_numpy(float), off_sec=c["dia_df"]["off_sec"].to_numpy(float), cond=cond,
        cond_all=cond_all, rows=rows, states_cond=states_cond, states_bcond=tw.bin_conditions(c, states_cond),
        states_epochs=tw.epoch_order(c, states_cond), bcond=tw.bin_conditions(c, cond), spikes={int(u): np.sort(c["spikes"][u]) for u in c["ids"]},
        dia_sub=np.asarray(c["dia_sub"]), ioff_point=np.asarray(c["ioff_point"]), time_offset_s=c["time_offset_s"],
        hmm_bin_s=c["hmm_bin_s"], sr_sub=c["sr_sub"], ref_k=k, ids=[int(u) for u in c["ids"]],
        order_units=[int(u) for u in c["order_units"]], epochs=tw.epoch_order(c, cond),
        cond_colours={**c["cond_colours"], **tw.TRANSITION_COLOURS},
        state_colours=[state_colour(c, k, s) for s in range(k)],
        state_names=[tsl.state_name(labels[s]) for s in range(k)],
        state_cycle_order=[int(s) for s in fsig.cycle_order(c, k, list(range(k)))],
        state_legend_order=[int(s) for s in np.argsort(c["colours"][k])], info=c["info"])


def wrap(deg):
    """Degrees into [-180, 180)."""
    return (np.asarray(deg) + 180) % 360 - 180


def zone(deg):
    return next(name for lo, hi, name in ZONES if lo <= deg < hi)


def phase_screen(z):
    """Per unit, how each later epoch differs from baseline by the screen above (SHIFT_DEG, ZONES, the silence rule);
    its group is the first of KINDS it shows, else "no change"."""
    tu = z["tuning"].set_index(["unit_id", "condition"])
    tonic, insp = pdm.TONIC_MAX_L, {"early I", "late I"}
    rec = recovery_epochs(z)
    rows = []
    for u in z["ids"]:
        b = tu.loc[(u, z["epochs"][0])]
        b_ok = b["n_spikes"] >= tw.MIN_SPIKES
        b_phasic = b_ok and b["modulation_L"] >= tonic
        found = {}
        for e in z["epochs"][1:]:
            r = tu.loc[(u, e)]
            ok = r["n_spikes"] >= tw.MIN_SPIKES
            phasic = ok and r["modulation_L"] >= tonic
            name = SHORT.get(e, e)
            if b_phasic and phasic:
                d = float(wrap(r["phase_deg"] - b["phase_deg"]))
                if abs(d) >= SHIFT_DEG:
                    zb, ze = zone(b["phase_deg"]), zone(r["phase_deg"])
                    if ze == "pre-I" and zb != "pre-I":
                        kind = "→ pre-I"
                    elif (ze in insp) != (zb in insp):
                        kind = "→ inspiration" if ze in insp else "→ expiration"
                    else:
                        kind = "earlier" if d < 0 else "later"
                    found.setdefault(kind, []).append(f"{name} {r['phase_deg']:+.0f}°")
            elif b_ok and ok and b_phasic != phasic:
                found.setdefault("gains modulation" if phasic else "loses modulation", []).append(
                    f"{name} L {r['modulation_L']:.2f}")
            if not b_ok and ok:
                found.setdefault("recruited", []).append(f"{name} {r['rate_hz']:.0f} Hz")
            if e in rec and b["rate_hz"] >= SILENT_MIN_HZ and r["rate_hz"] < SILENT_FRACTION * b["rate_hz"]:
                found.setdefault("silent in recovery", []).append(f"{name} {r['rate_hz']:.1f} Hz")
        kinds = [k for k in KINDS if k in found]
        rows.append(dict(
            unit_id=u, group=kinds[0] if kinds else "no change", rank=KINDS.index(kinds[0]) if kinds else len(KINDS),
            baseline_class=b["class"] if b_ok else "silent", baseline_phase_deg=float(np.round(b["phase_deg"])),
            baseline_L=round(float(b["modulation_L"]), 2), baseline_rate_hz=round(float(b["rate_hz"]), 1),
            changes="; ".join(f"{k}: {', '.join(found[k])}" for k in kinds)))
    info = z["info"].assign(unit_id=z["info"]["unit_id"].astype(int))[["unit_id", "depth_um"]]
    return pd.DataFrame(rows).merge(info, on="unit_id")


def mean_cycle(z, epoch, dims, nbins):
    """Her mean_cycle: the PCs of every 5 ms bin of the epoch averaged in nbins equal bins of breath phase, closed."""
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    b = np.clip(np.digitize(z["ph"], edges) - 1, 0, nbins - 1)
    sel = z["bcond"] == epoch
    m = np.full((nbins, len(dims)), np.nan)
    for i in range(nbins):
        s = sel & (b == i)
        if s.any():
            m[i] = z["P"][s][:, list(dims)].mean(axis=0)
    return np.vstack([m, m[:1]])


def recovery_epochs(z):
    return z["epochs"][z["epochs"].index("post-gasp") + 1:]


def epoch_colour(z, epoch):
    rec = recovery_epochs(z)
    if epoch not in rec:
        return z["cond_colours"].get(epoch, "#cccccc")
    a, b = (np.array(mcolors.to_rgb(c)) for c in TRAJ_GREENS)
    return mcolors.to_hex(a + (b - a) * rec.index(epoch) / max(len(rec) - 1, 1))


def traj_limits(data):
    """Per PC: the range of every epoch's mean cycle in both trials, 8 % margin."""
    m = np.vstack([mean_cycle(z, e, (0, 1, 2), TRAJ_BINS) for z in data.values() for e in z["epochs"]])
    lo, hi = np.nanmin(m, axis=0), np.nanmax(m, axis=0)
    pad = 0.08 * (hi - lo)
    return {i: (lo[i] - pad[i], hi[i] + pad[i]) for i in range(3)}


def epoch_span(z, epoch):
    """(first onset, next onset after the last breath, breaths) of the epoch in raw-run time."""
    ons, b = z["on_sec"], np.flatnonzero(z["cond"] == epoch)
    return ons[b[0]] + z["time_offset_s"], ons[min(b[-1] + 1, ons.size - 1)] + z["time_offset_s"], b.size


def panel_trajectories(fig, spec, data):
    """One column per trial, one row per plane in TRAJ_PLANES: every epoch's mean cycle."""
    lim = traj_limits(data)
    grid = spec.subgridspec(len(TRAJ_PLANES) + 1, len(data), height_ratios=[1] * len(TRAJ_PLANES) + [0.16],
                            hspace=0.3, wspace=0.25)
    shown = []
    for c, (label, z) in enumerate(data.items()):
        for r, (a, b) in enumerate(TRAJ_PLANES):
            ax = fig.add_subplot(grid[r, c])
            for epoch in z["epochs"]:
                m = mean_cycle(z, epoch, (a, b), TRAJ_BINS)
                thick = epoch in TRAJ_THICK
                ax.plot(m[:, 0], m[:, 1], color=epoch_colour(z, epoch), lw=TRAJ_LW[thick], zorder=3 if thick else 2)
                if epoch not in shown:
                    shown.append(epoch)
            ax.set_xlim(lim[a])
            ax.set_ylim(lim[b])
            ax.set_box_aspect(TRAJ_BOX)
            ax.set_anchor("S")
            ax.set_xlabel(f"PC{a + 1}", fontsize=FS["label"])
            ax.set_ylabel(f"PC{b + 1}", fontsize=FS["label"])
            ax.tick_params(labelsize=FS["tick"])
            ax.set_title(f"{label} · PC{a + 1}–PC{b + 1}", loc="left", fontsize=FS["title"], fontweight="bold")
    lax = fig.add_subplot(grid[-1, :])
    lax.axis("off")
    z = next(iter(data.values()))
    rec = recovery_epochs(z)
    keys = [e for e in shown if e in z["epochs"] and e not in rec]
    handles = [Line2D([], [], color=epoch_colour(z, e), lw=TRAJ_LW[e in TRAJ_THICK]) for e in keys]
    names = list(keys)
    if rec:
        handles.append(tuple(Line2D([], [], color=epoch_colour(z, e), lw=TRAJ_LW[0]) for e in rec))
        names.append(f"{rec[0].split()[0]} → {rec[-1]}")
    lax.legend(handles, names, loc="center", ncol=len(names), frameon=False, fontsize=FS["tick"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})


def panel_states(fig, spec, z, lim, plane):
    """Her fig_states_pca_epochs: 5 ms bins by Viterbi state, one panel per epoch, over the baseline bins in grey."""
    (dx, dy), k, order = plane, z["ref_k"], z["epochs"]
    ncol = 3
    nrow = int(np.ceil(len(order) / ncol))
    grid = spec.subgridspec(nrow + 1, ncol, height_ratios=[1] * nrow + [0.32], hspace=0.5, wspace=0.12)
    rng = np.random.default_rng(0)
    cols = np.array(z["state_colours"])
    base = np.flatnonzero(z["bcond"] == "baseline")
    base = base if base.size <= STATES_MAX_POINTS else np.sort(rng.choice(base, STATES_MAX_POINTS, replace=False))
    P, seq = z["P"], z["seq"]
    first = None
    for i, epoch in enumerate(order):
        ax = fig.add_subplot(grid[i // ncol, i % ncol], sharex=first, sharey=first)
        first = first or ax
        idx = np.flatnonzero(z["bcond"] == epoch)
        sub = idx if idx.size <= STATES_MAX_POINTS else np.sort(rng.choice(idx, STATES_MAX_POINTS, replace=False))
        if epoch != "baseline":
            ax.scatter(P[base, dx], P[base, dy], s=2, color=GREY, edgecolor="none", zorder=1, rasterized=True)
        ax.scatter(P[sub, dx], P[sub, dy], c=cols[seq[sub]], s=3, alpha=0.6, edgecolor="none", zorder=2, rasterized=True)
        for s in range(k):
            m = seq[idx] == s
            if m.sum() > STATES_MIN_BINS:
                ax.scatter(P[idx[m], dx].mean(), P[idx[m], dy].mean(), color=cols[s], s=STATE_DOT, edgecolor=INK,
                           lw=0.9, zorder=4)
        ax.scatter(*z["ioff_point"][[dx, dy]], color=INK, marker="X", s=70, zorder=5)
        t0, t1, n = epoch_span(z, epoch)
        ax.add_patch(Rectangle((0, 1.0), 1, 0.07, transform=ax.transAxes, color=z["cond_colours"].get(epoch, "#cccccc"),
                               clip_on=False))
        ax.set_title(f"{i + 1}. {epoch}", loc="left", fontsize=FS["label"], fontweight="bold", pad=18)
        ax.text(1.0, 1.09, f"{t0:.0f}–{t1:.0f} s · {n} breaths", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=FS["small"], color=INK2)
        ax.set_xlim(lim[dx])
        ax.set_ylim(lim[dy])
        ax.tick_params(labelsize=FS["small"], labelleft=i % ncol == 0, labelbottom=i >= len(order) - ncol)
        if i % ncol == 0:
            ax.set_ylabel(f"PC{dy + 1}", fontsize=FS["label"])
        if i >= len(order) - ncol:
            ax.set_xlabel(f"PC{dx + 1}", fontsize=FS["label"])
    lax = fig.add_subplot(grid[-1, :])
    lax.axis("off")
    handles = [Line2D([], [], marker="o", ls="", color=cols[s], mec=INK, ms=10) for s in z["state_legend_order"]]
    handles += [Line2D([], [], marker="X", ls="", color=INK, ms=10), Line2D([], [], marker="o", ls="", color=GREY, ms=7)]
    names = ([z["state_names"][s] for s in z["state_legend_order"]]
             + ["baseline inspiration-off point", "baseline bins (reference)"])
    lax.legend(handles, names, loc="center", ncol=4, frameon=False, fontsize=FS["tick"])


def example_onset(z, epoch):
    """Her example_breath: the epoch's most typical breath (state mix closest to the epoch's mean, middle half)."""
    k, dt, seq, ons = z["ref_k"], z["hmm_bin_s"], z["seq"], z["on_sec"]
    idx = np.flatnonzero(z["cond"] == epoch)
    mix = np.stack([np.bincount(seq[int(ons[i] / dt):int(ons[i + 1] / dt)], minlength=k) for i in idx]).astype(float)
    mix /= np.maximum(mix.sum(axis=1, keepdims=True), 1)
    dist = np.abs(mix - mix.mean(axis=0)).sum(axis=1)
    middle = np.arange(len(idx) // 4, max(len(idx) // 4 + 1, 3 * len(idx) // 4))
    return float(ons[idx[middle[int(np.argmin(dist[middle]))]]])


def panel_examples(fig, spec, z):
    """Her fig_state_examples_epochs, left half: every unit's spikes (baseline phase order) over P(state), ∫dia below;
    an example epoch the breaths leave out (--crop) is replaced by the last epoch drawn."""
    dt, off, nu = z["hmm_bin_s"], z["time_offset_s"], len(z["order_units"])
    examples = [e if e in z["epochs"] else z["epochs"][-1] for e in EXAMPLES]
    grid = spec.subgridspec(len(examples), 1, hspace=0.55)
    windows = []
    for r, epoch in enumerate(examples):
        before, after = tw.EXAMPLE_S.get(epoch, EXAMPLE_DEFAULT_S)
        on = example_onset(z, epoch)
        t0, t1 = max(0.0, on - before), min(z["dia_sub"].size / z["sr_sub"], on + after)
        cell = grid[r].subgridspec(2, 1, height_ratios=[4, 0.9], hspace=0.06)
        ax = fig.add_subplot(cell[0])
        axd = fig.add_subplot(cell[1], sharex=ax)
        b0, b1 = int(t0 / dt), int(t1 / dt)
        tt = (np.arange(b0, b1) + 0.5) * dt - t0
        post = z["post"][b0:b1]
        for s in z["state_cycle_order"]:
            if post[:, s].max() < 0.05:
                continue
            col = np.array(mcolors.to_rgb(z["state_colours"][s]))
            ax.fill_between(tt, 0, post[:, s], color=col, alpha=0.35, lw=0)
            ax.plot(tt, post[:, s], color=col * 0.75, lw=1.5)
        for y, u in enumerate(z["order_units"]):
            ts = z["spikes"].get(u)
            if ts is None:
                continue
            ts = ts[(ts >= t0) & (ts < t1)] - t0
            ax.vlines(ts, 1 - (y + 0.85) / nu, 1 - (y + 0.15) / nu, color=INK, lw=0.7)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, t1 - t0)
        ax.set_ylabel("P(state)", fontsize=FS["small"])
        ax.tick_params(labelbottom=False, labelsize=FS["small"])
        ax.add_patch(Rectangle((0, 1.0), 1, 0.06, transform=ax.transAxes, color=z["cond_colours"].get(epoch, "#cccccc"),
                               clip_on=False))
        ax.set_title(f"{z['epochs'].index(epoch) + 1}. {epoch}", loc="left", fontsize=FS["label"], fontweight="bold",
                     pad=26)
        e0, e1, n = epoch_span(z, epoch)
        ax.annotate(f"{e0:.0f}–{e1:.0f} s · {n} breaths · example {t0 + off:.2f}–{t1 + off:.2f} s", xy=(0, 1),
                    xycoords="axes fraction", xytext=(0, 8), textcoords="offset points", va="bottom",
                    fontsize=FS["small"], color=INK2)
        i0, i1 = int(t0 * z["sr_sub"]), int(t1 * z["sr_sub"])
        axd.plot(np.arange(i0, i1) / z["sr_sub"] - t0, z["dia_sub"][i0:i1], color=INK, lw=0.8)
        axd.set_yticks([])
        axd.set_ylabel("∫dia", fontsize=FS["small"])
        axd.tick_params(labelsize=FS["small"])
        if r == len(examples) - 1:
            axd.set_xlabel("time from window start (s)", fontsize=FS["label"])
        windows.append((epoch, t0 + off, t1 + off))
    return windows


def panel_state_raster(ax, z):
    """Her fig_breath_raster_wide, onset half: the Viterbi state in each 5 ms bin around each breath's onset, one row
    per breath in z["rows"] (every breath, or --crop's); dots: onsets (black) and offsets (white) of the previous, this
    and the next breath."""
    dt, seq = z["hmm_bin_s"], z["seq"]
    lo, hi = STATE_RASTER_MS
    n0, n1 = int(round(-lo / 1e3 / dt)), int(round(hi / 1e3 / dt))
    ons, offs, idx = z["on_sec"], z["off_sec"], z["rows"]
    rows = np.full((idx.size, n0 + n1), -1)
    for r, i in enumerate(idx):
        b = int(np.floor(ons[i] / dt))
        if b - n0 >= 0 and b + n1 <= seq.size:
            rows[r] = seq[b - n0:b + n1]
    cols = np.array([mcolors.to_rgb(c) for c in z["state_colours"]] + [mcolors.to_rgb(GREY)])
    n = idx.size
    ax.imshow(cols[rows], aspect="auto", interpolation="nearest", extent=[lo, hi, n, 0], rasterized=True)
    y = np.arange(n) + 0.5
    for times, colour in ((ons, INK), (offs, "white")):
        for shift in (-1, 0, 1):
            j = idx + shift
            x = (times[np.clip(j, 0, ons.size - 1)] - ons[idx]) * 1e3
            x[(j < 0) | (j >= ons.size)] = np.nan
            m = (x > lo) & (x < hi) & (np.abs(x) > 1e-6)
            ax.scatter(x[m], y[m], s=0.7, color=colour, lw=0, alpha=0.85, rasterized=True)
    ax.axvline(0, color=INK, lw=1.4, ls=(0, (4, 3)))
    ax.set_xlim(lo, hi)
    ax.set_ylim(n, 0)
    ax.set_xlabel("time from inspiration onset (ms)", fontsize=FS["label"])
    ax.set_ylabel("breath (recording order)", fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])
    epoch_bar(ax, z, idx, x0=1.01)
    shown = f"all {n}" if n == ons.size else f"{n} of {ons.size}"
    ax.set_title(f"K = {z['ref_k']} HMM state, {shown} breaths", loc="left", fontsize=FS["label"])


def epoch_bar(ax, z, idx, x0, labels=False):
    """A bar right of the axes in each row's epoch colour; with labels, each block's name, span and breaths, spread
    apart where blocks are too short to hold them (a leader line back to the block)."""
    cond = z["cond"][idx]
    change = np.flatnonzero(cond[1:] != cond[:-1]) + 1
    ons, off = z["on_sec"], z["time_offset_s"]
    tr = ax.get_yaxis_transform()
    named = []
    for a, b in zip(np.r_[0, change], np.r_[change, cond.size]):
        name = cond[a]
        ax.add_patch(Rectangle((x0, a), 0.035, b - a, transform=tr, clip_on=False, lw=0,
                               color=z["cond_colours"].get(name, "white") if name else "white"))
        if name:
            total = int((z["cond_all"] == name).sum())
            count = f"{b - a} of {total}" if b - a < total else f"{total}"
            named.append((a, b, f"{name}\n{ons[idx[a]] + off:.0f}–{ons[idx[b - 1]] + off:.0f} s ({count})"))
    if not labels or not named:
        return
    fig = ax.get_figure()
    gap = LABEL_GAP_PT * cond.size / (ax.get_position().height * fig.get_figheight() * 72)   # in rows
    want = np.array([(a + b) / 2 for a, b, _ in named])
    y = want.copy()
    for i in range(1, y.size):
        y[i] = max(y[i], y[i - 1] + gap)
    y[-1] = min(y[-1], cond.size - gap / 2)
    for i in range(y.size - 2, -1, -1):
        y[i] = min(y[i], y[i + 1] - gap)
    for (a, b, text), yi, wi in zip(named, y, want):
        ax.text(x0 + 0.07, yi, text, transform=tr, va="center", fontsize=FS["small"], clip_on=False)
        if abs(yi - wi) > 0.5:
            ax.plot([x0 + 0.037, x0 + 0.065], [wi, yi], transform=tr, color=INK2, lw=0.6, clip_on=False)


def raster_breaths(z, cap=None):
    """The unit raster rows in recording order: the state raster's (every breath, or --crop's); with cap, each epoch's
    breaths, at most cap of them evenly spaced across it (her cycles_of)."""
    if cap is None:
        return z["rows"]
    return np.concatenate([fp.cycles_of(z, e, cap) for e in z["epochs"]])


def in_window(times, centres, lo, hi):
    """Every time within [centre + lo, centre + hi) of each centre, relative to it, and its row (0, 1, ...)."""
    a, b = np.searchsorted(times, centres + lo), np.searchsorted(times, centres + hi)
    x = np.concatenate([times[i:j] - t for i, j, t in zip(a, b, centres)])
    return x, np.repeat(np.arange(centres.size) + 0.5, b - a)


def unit_label(r):
    """A screen row in a few words: 'expiratory → pre-I', 'inspiratory, fires earlier', 'silent at baseline, recruited'."""
    base = "silent at baseline" if r["baseline_class"] == "silent" else r["baseline_class"]
    if r["group"].startswith("→"):
        return f"{base} {r['group']}"
    return f"{base}, {'fires ' if r['group'] in ('earlier', 'later') else ''}{r['group']}"


def panel_unit_raster(ax, z, u, labels=True, label=None, cap=None):
    """Unit u's spikes on the raster breaths (raster_breaths(z, cap)), all of them from UNIT_RASTER_MS[0] to [1] around
    the breath's onset (the previous and next breaths' spikes included), one row-high tick each; grey: the other
    breaths' inspiration onsets (no offsets, so the bursts stand alone). labels: name each epoch beside its bar; label:
    words after the unit id in the title."""
    lo, hi = np.array(UNIT_RASTER_MS) / 1e3
    idx = raster_breaths(z, cap)
    ons = z["on_sec"][idx]
    x, y = in_window(z["on_sec"], ons, lo, hi)
    keep = np.abs(x) > 1e-9                                     # its own onset is the dashed line
    ax.vlines(x[keep] * 1e3, y[keep] - 0.5, y[keep] + 0.5, color=TICK, lw=1.8)
    x, y = in_window(z["spikes"][u], ons, lo, hi)
    ax.vlines(x * 1e3, y - 0.45, y + 0.45, color=INK, lw=1.4, zorder=3)
    ax.axvline(0, color=INK2, lw=0.9, ls=(0, (4, 3)))
    ax.set_ylim(idx.size, 0)
    ax.set_xlim(*UNIT_RASTER_MS)
    ax.set_xlabel("time from inspiration onset (ms)", fontsize=FS["label"])
    ax.set_ylabel("breath (recording order)", fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])
    ax.set_title(f"#{u}", loc="left", fontsize=FS["label"], fontweight="bold", pad=18 if label else 6)
    if label:
        ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                    va="bottom", fontsize=FS["small"], color=INK2)
    epoch_bar(ax, z, idx, x0=1.01, labels=labels)
    return idx


def draw_gallery(z, scr):
    """Every unit's raster (the unit rasters' window, at most GALLERY_MAX_BREATHS breaths per epoch so each stays
    legible), grouped by the phase screen (KINDS order), within a group by baseline phase; each titled with its
    baseline tuning and the screen's findings."""
    scr = scr.sort_values(["rank", "baseline_phase_deg"])
    ncol = 7
    nrow = int(np.ceil(len(scr) / ncol))
    fig = plt.figure(figsize=(3.6 * ncol, 6.4 * nrow + 1.2))
    grid = fig.add_gridspec(nrow, ncol, hspace=0.3, wspace=0.32, left=0.04, right=0.98, top=0.93, bottom=0.05)
    for i, r in enumerate(scr.itertuples()):
        ax = fig.add_subplot(grid[i // ncol, i % ncol])
        panel_unit_raster(ax, z, r.unit_id, labels=False, cap=GALLERY_MAX_BREATHS)
        ax.set_title("", loc="left")
        ax.tick_params(labelsize=FS["small"])
        if i % ncol:
            ax.set_ylabel("")
        if i + ncol < len(scr):
            ax.set_xlabel("")
        else:
            ax.set_xlabel("ms from inspiration onset", fontsize=FS["small"])
        base = (f"{r.baseline_class} {r.baseline_phase_deg:+.0f}°, L {r.baseline_L:.2f}" if r.baseline_class != "silent"
                else "silent")
        lines = [ln for part in (r.changes.split("; ") if r.changes else ["no change"])
                 for ln in textwrap.wrap(part, 44)]
        ax.annotate("\n".join(lines), xy=(0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                    va="bottom", fontsize=FS["small"] - 1, color=INK2)
        ax.annotate(f"#{r.unit_id} · {base} · {r.baseline_rate_hz:g} Hz · {r.depth_um:.0f} µm", xy=(0, 1),
                    xycoords="axes fraction", xytext=(0, 6 + 11 * len(lines)), textcoords="offset points",
                    va="bottom", fontsize=FS["small"], fontweight="bold", color=INK)
    handles = [Rectangle((0, 0), 1, 1, color=z["cond_colours"].get(e, "#cccccc")) for e in z["epochs"]]
    fig.legend(handles, z["epochs"], loc="lower center", ncol=len(handles), frameon=False, fontsize=FS["small"])
    note = crop_note(z)
    fig.suptitle(
        f"e097 anoxia #1{f' ({note})' if note else ''}: " +
        "every unit's spikes {:.0f} to {:.0f} ms around inspiration onset, up to {} breaths per epoch, "
        "grouped by how its breath phase changes from baseline\n".format(*UNIT_RASTER_MS, GALLERY_MAX_BREATHS) +
        f"Screen: a phase (the mean breath phase of the epoch's spikes) counts with ≥ {tw.MIN_SPIKES} spikes and L ≥ {pdm.TONIC_MAX_L} (in the epoch and at "
        f"baseline) and shifts when it moves ≥ {SHIFT_DEG}°; zones post-I −180 to −120°, mid-E −120 to −60°, "
        f"pre-I −60 to 0°, early I 0 to 90°, late I 90 to 180°; silent in recovery: under "
        f"{SILENT_FRACTION:.0%} of a baseline rate ≥ {SILENT_MIN_HZ:g} Hz; recruited: under {tw.MIN_SPIKES} spikes at "
        "baseline", x=0.04, ha="left", fontsize=FS["label"])
    return fig


def draw(data, units, plane, labels):
    z = data["anoxia #1"]
    missing = [u for u in units if u not in z["spikes"]]
    if missing:
        raise SystemExit(f"units {missing} are not among anoxia #1's {len(z['ids'])} units: {z['ids']}")
    fig = plt.figure(figsize=FIGSIZE)
    outer = fig.add_gridspec(2, 1, height_ratios=(1.0, 0.95), hspace=0.16, left=0.04, right=0.97, top=0.97, bottom=0.05)
    top = outer[0].subgridspec(1, 2, width_ratios=(1.25, 1.0), wspace=0.12)
    bottom = outer[1].subgridspec(1, 3, width_ratios=(1.3, 0.8, UNIT_WIDTH * len(units) + 0.45), wspace=0.3)
    panel_trajectories(fig, top[0], data)
    # the states' 5 ms bins spread wider than the mean cycles: their own range, 0.1th to 99.9th percentile
    lo, hi = np.percentile(z["P"], 0.1, axis=0), np.percentile(z["P"], 99.9, axis=0)
    zs = {**z, "cond": z["states_cond"], "bcond": z["states_bcond"], "epochs": z["states_epochs"]}
    panel_states(fig, top[1], zs, {i: (lo[i], hi[i]) for i in range(3)}, plane)
    windows = panel_examples(fig, bottom[0], z)
    panel_state_raster(fig.add_subplot(bottom[1]), z)
    # the rasters share their rows, so only the last names the epochs (in the extra width at its right)
    cells = bottom[2].subgridspec(1, len(units) + 1, width_ratios=[UNIT_WIDTH] * len(units) + [0.45], wspace=0.22)
    rows, axes = [], []
    for j, u in enumerate(units):
        ax = fig.add_subplot(cells[j])
        rows.append(panel_unit_raster(ax, z, u, labels=j == len(units) - 1, label=labels.get(u)))
        ax.set_xlabel("")
        if j:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        axes.append(ax)
    # one x label under all the unit rasters, centred on them
    p0, p1 = axes[0].get_position(), axes[-1].get_position()
    axes[0].set_xlabel("time from inspiration onset (ms)", fontsize=FS["label"])
    axes[0].xaxis.set_label_coords(((p0.x0 + p1.x1) / 2 - p0.x0) / p0.width, -0.04)
    return fig, windows, rows[0]


def crop_note(z):
    """--crop's breaths in a few words ("" when every breath is drawn)."""
    if z["rows"].size == z["on_sec"].size:
        return ""
    return (f"breaths {z['rows'][0] + 1}–{z['rows'][-1] + 1} of {z['on_sec'].size}: the second half of the "
            f"{z['epochs'][0]} to the first half of the {z['epochs'][-1]}")


def crop_caption(data):
    """The caption's account of --crop's breaths, per trial."""
    def half(z, e):
        return f"{(z['cond'] == e).sum()} of {(z['cond_all'] == e).sum()} breaths"
    def dropped(z):
        return " and ".join(e for e in dict.fromkeys(z["cond_all"]) if e and e not in z["epochs"])
    trials = "; ".join(f"{label}, the second half of the {z['epochs'][0]} ({half(z, z['epochs'][0])}) to the first half "
                       f"of the {z['epochs'][-1]} ({half(z, z['epochs'][-1])}), leaving out {dropped(z)}"
                       for label, z in data.items())
    z = data["anoxia #1"]
    return (f"This version draws {z['rows'].size} of the {z['on_sec'].size} breaths of anoxia #1 (breaths "
            f"{z['rows'][0] + 1}–{z['rows'][-1] + 1}) and the same stretch of anoxia #2: {trials}. The other breaths "
            f"are left out of every panel and of the phase screen, except the state panels, which also show the "
            f"{dropped(z)} whole; the PCA axes and the HMM are those fit on the whole run.")


def caption(data, units, plane, windows, rows, scr):
    z = data["anoxia #1"]
    s = scr.set_index("unit_id")
    spans = "; ".join("{} {:.0f} - {:.0f} s ({} breaths)".format(e, *epoch_span(z, e)) for e in z["epochs"])
    shown = "; ".join(
        f"#{u} (" + ("silent at baseline" if s.loc[u, "baseline_class"] == "silent" else
                     f"{s.loc[u, 'baseline_class']} at baseline, {s.loc[u, 'baseline_phase_deg']:+.0f}°, L "
                     f"{s.loc[u, 'baseline_L']:.2f}")
        + f", {s.loc[u, 'depth_um']:.0f} µm; {s.loc[u, 'changes'] or 'no change'})" for u in units)
    tag = "_crop" if crop_note(z) else ""
    slow = (" (so slow breathing is green here but amber in the other panels)" if "slow breathing" in z["epochs"]
            else "")
    return [
        "e097 population analysis (the SfN 2026 poster), npx_pipeline's PF8 drawn on the curated units: the "
        f"{len(z['ids'])} units reviewed single in the anoxia #1 window. Anoxia #2 uses their matched partners in its own "
        f"sort ({len(data['anoxia #2']['ids'])} units) and is decoded with the anoxia #1 HMM on the anoxia #1 PCA axes.",
        *([crop_caption(data)] if tag else []),
        "Top left: the mean breath trajectory of every epoch in {}, the PCs of all its 5 ms bins (spike counts smoothed 10 ms, "
        "square root, PCA fit on the baseline: Bush & Ramirez 2024) averaged in 50 equal bins of breath phase (25 across "
        "inspiration, 25 across expiration); each axis spans that PC's range over every trajectory of both trials. Thick: "
        "baseline and gasping; greens: the recovery epochs, early lightest{}. Top right: the K = 6 Poisson HMM states in "
        "PC{}–PC{}, one panel per epoch (small dots: 5 ms bins coloured by Viterbi state, at most 6000 per panel; big "
        "dots: each state's mean position in that epoch, states with over 20 bins; X: the baseline inspiration-off "
        "point; grey: baseline bins).".format(
            ", ".join(f"PC{a + 1}–PC{b + 1}" for a, b in TRAJ_PLANES), slow, *(p + 1 for p in plane)),
        "Bottom, left to right: example stretches of three epochs (each unit's spikes, one row per unit in baseline-phase "
        "order, over the probability of each state; the integrated diaphragm below; the example is the epoch's most "
        "typical breath by state mix); the Viterbi state from {:.0f} to {:.0f} ms around every breath's inspiration "
        "onset, one row per breath drawn, the epoch in the bar at the right; and the spikes of {} units on the same {} breaths "
        "in the same window, so each row lines up with the state raster's: every spike from {:.0f} to {:.0f} ms around "
        "that breath's onset, the previous and next breaths' spikes included (grey ticks: the other breaths' inspiration "
        "onsets; inspiration offsets are not marked).".format(*STATE_RASTER_MS, len(units), rows.size, *UNIT_RASTER_MS),
        f"Epochs (anoxia #1, raw-run time; breaths drawn): {spans}.",
        f"Units shown, with the phase screen's findings (unit_phase_screen{tag}.tsv; every unit in "
        f"unit_raster_gallery{tag}.png; a phase is the circular mean of the breath phase at each of the epoch's spikes, "
        f"0° at inspiration onset, so it places a unit where its raster shows it: the vector mean of the rate-per-phase "
        f"curve weights the brief inspiration up and put #510's post-gasp burst, 76 % of it before onset, at +4°): "
        f"{shown}. Example windows: " + "; ".join(f"{e} {a:.2f} - {b:.2f} s" for e, a, b in windows) + ".",
        "How it was made: analysis/population/population_figure.py, from the population_dynamics.py runs "
        "population_dynamics_v2 and population_dynamics_anoxia2 (cache/ctx.pkl), the panels ported from npx_pipeline's "
        "plotting_pf.pf8_population (her mean_cycle, example_breath, fig_states_pca_epochs, fig_breath_raster_wide and "
        "unit_transition_raster). Changed from PF8: the units are chosen by hand; the unit rasters share the state "
        "raster's rows (every breath drawn) and window ({:.0f} to {:.0f} ms), with every spike in it (PF8: 0.3 s before "
        "onset to the next onset, on the last 15 breaths of the anoxia to the first 15 of early recovery); the "
        "trajectories are drawn in {} only (PF8: PC1–PC3, PC1–PC2 and PC2–PC3); and the state "
        "panels are drawn in PC{}–PC{} (PF8: PC1–PC2, the plane of the breath loop for its bombcell-good units).".format(
            *UNIT_RASTER_MS, ", ".join(f"PC{a + 1}–PC{b + 1}" for a, b in TRAJ_PLANES), *(p + 1 for p in plane)),
    ]


def embed_font(svg):
    """The SVG carries FONT as a WOFF2 subset of the characters its text uses, so the text stays text and stays Arial."""
    from fontTools import subset
    from matplotlib import font_manager

    doc = svg.read_text(encoding="utf-8")
    chars = {ch for t in re.findall(r"<text\b[^>]*>(.*?)</text>", doc, re.S)
             for ch in html.unescape(re.sub(r"<[^>]+>", "", t))}
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.name_IDs = ["*"]
    opts.drop_tables += ["meta"]
    font = subset.load_font(font_manager.findfont(FONT, fallback_to_default=False), opts)
    sub = subset.Subsetter(opts)
    sub.populate(text="".join(sorted(chars)) or " ")
    sub.subset(font)
    buf = io.BytesIO()
    subset.save_font(font, buf, opts)
    face = (f"@font-face {{ font-family: '{FONT}'; font-style: normal; font-weight: normal; "
            f"src: url(data:font/woff2;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}) format('woff2'); }}\n")
    doc, n = re.subn(r"(<style[^>]*>)", lambda m: m.group(1) + face, doc, count=1)
    if not n:
        raise ValueError(f"{svg} has no <style> element to hold the font")
    svg.write_text(doc, encoding="utf-8")


def save_all(fig, out, tag=""):
    stem = out / f"population_figure{tag}"
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        paths = [save_fig(fig, stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight"),
                 save_fig(fig, stem.with_suffix(".svg"), bbox_inches="tight")]
    embed_font(paths[1])
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    long_in = max(box.width, box.height) + 2 * matplotlib.rcParams["savefig.pad_inches"]
    paths.append(save_fig(fig, out / f"population_figure{tag}_poster.png", bbox_inches="tight",
                          dpi=math.ceil(POSTER_IN * POSTER_DPI / long_in), pil_kwargs={"dpi": (POSTER_DPI, POSTER_DPI)}))
    return paths


def main():
    a = parse_args()
    sys.stdout.reconfigure(errors="replace")             # a Windows console has no "→"
    a.out.mkdir(parents=True, exist_ok=True)
    from matplotlib import font_manager
    font_manager.findfont(FONT, fallback_to_default=False)
    data = {label: trial_data(run, tuning=label == "anoxia #1", crop=a.crop) for label, run in RUNS.items()}
    tag = "_crop" if a.crop else ""
    z = data["anoxia #1"]
    scr = phase_screen(z)
    with matplotlib.rc_context({"font.family": FONT}):
        fig, windows, rows = draw(data, a.units, a.states_plane, {r["unit_id"]: unit_label(r) for _, r in scr.iterrows()})
        paths = save_all(fig, a.out, tag)
        plt.close(fig)
        fig = draw_gallery(z, scr)
        paths.append(save_fig(fig, a.out / f"unit_raster_gallery{tag}.png", dpi=200, bbox_inches="tight"))
        plt.close(fig)
    scr.sort_values(["rank", "baseline_phase_deg"]).drop(columns="rank").to_csv(
        a.out / f"unit_phase_screen{tag}.tsv", sep="\t", index=False, float_format="%.3f")
    cols = ["unit_id", "condition", "n_cycles", "n_spikes", "rate_hz", "phase_deg", "rate_phase_deg", "modulation_L",
            "class"]
    z["tuning"][cols].to_csv(
        a.out / f"unit_phase_screen_epochs{tag}.tsv", sep="\t", index=False, float_format="%.3f")
    text = "\n\n".join(caption(data, a.units, a.states_plane, windows, rows, scr)) + "\n"
    (a.out / f"population_figure{tag}_caption.txt").write_text(text, encoding="utf-8")
    for p in paths:
        print(p)
    print(text)


if __name__ == "__main__":
    main()
