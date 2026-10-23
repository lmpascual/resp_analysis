"""Slide-sized transition figures and the deck text for them (read by make_unit_slides.js).

Run after transition_windows.py, which writes the tables and the per-unit transition PETH / raster PNGs:

  cibrrig python transition_slides.py [--run <population_dynamics output>]

Writes, under <run>/transitions/:
  slides/*.png                  12.3 x 5.75 in, 10.5-12.5 pt text: the two transitions on a time axis, breath-by-breath
                                breathing and population measures, HMM state sequences, unit phase per window and per
                                breath
  units_slides/transitions.json  slide titles, speaker notes, and per-unit drafted slide text
and points <run>/figures/units_slides/slides.json at transitions.json (figures_signatures.write_slide_manifest keeps
the link when it rewrites slides.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import transition_windows as tw  # noqa: E402
import figures_population as fp  # noqa: E402
from figures_population import INK, INK2, GRID, save_fig, state_colour  # noqa: E402

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

W, H = 12.3, 5.75
FONT = {"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10}
N_STRIP = 10                 # breaths of anoxia / early recovery either side in the breath strips
REGIONS = [(-180, -120, "early E (post-I)"), (-120, -45, "mid E"), (-45, 0, "late E / pre-I"), (0, 90, "early I"),
           (90, 181, "late I")]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", default=str(tw.RUN))
    p.add_argument("--no-manifest", action="store_true", help="figures only, no deck text")
    p.add_argument("--deck-manifest", default=str(tw.RUN / "figures" / "units_slides" / "slides.json"),
                   help="the deck's slides.json (anoxia #1 run) that links the transition sections")
    return p.parse_args()


def state_name(label):
    """'late E / pre-I | baseline–anoxia, slow breathing' -> 'late E / pre-I (baseline)'."""
    part, _, where = label.partition(" | ")
    where = (where.replace("early recovery–later recovery", "recovery")
             .replace("baseline–anoxia, slow breathing", "baseline").replace("all but gasping", ""))
    return f"{part} ({where})" if where else part


def state_legend(fig, z, y=0.0, ncol=6, fontsize=9.5):
    k = z["ref_k"]
    order = np.argsort(z["colours"][k])
    f = z["F"][k]["feats"]
    handles = [Rectangle((0, 0), 1, 1, color=state_colour(z, k, s)) for s in order]
    fig.legend(handles, [state_name(f.loc[s, "label"]) for s in order], loc="lower center", bbox_to_anchor=(0.5, y),
               ncol=ncol, frameon=False, fontsize=fontsize, handlelength=1.1, columnspacing=1.2,
               title="HMM states (K = 6)", title_fontsize=fontsize)


def events(z, T, beats, pre, post, g):
    """Raw times of the moments named on the timeline slides."""
    off = z["time_offset_s"]
    on_raw = T["on_raw_s"].to_numpy()
    gs_ = T[f"state{z['gasp_state']}_frac"].to_numpy()
    tb, rr = beats[1:] + off, np.diff(beats)
    med = pd.Series(rr).rolling(10, min_periods=3).median().shift(1).to_numpy()
    def first(times, mask):                       # None when the event doesn't happen in this trial
        i = np.flatnonzero(mask)
        return float(times[i[0]]) if i.size else None

    halve = first(tb, (tb > on_raw[pre[0]] - 10) & (rr > 1.6 * med))
    take = first(on_raw[pre[0]:g[0] + 1], gs_[pre[0]:g[0] + 1] > 0.5)
    switch = first(on_raw[g[-1] + 1:], gs_[g[-1] + 1:] < 0.5)
    hr_med = pd.Series(60 / rr).rolling(9, center=True, min_periods=5).median().to_numpy()    # ignore single beats
    hr_up = first(tb, (tb > (switch or on_raw[post[0]])) & (hr_med > 500))
    lead = T["I_state_entry_ms_from_onset"].to_numpy()          # first post-gasp breath with I state 50 ms early
    i_lead = first(on_raw[post], lead[post] < -50)
    keep = lambda evs: [(t, s) for t, s in evs if t is not None]  # noqa: E731
    return {"into": keep([(halve, "heart rate halves"), (take, "gasp state takes over")]),
            "out": keep([(switch, "expiration switches"), (hr_up, "heart rate steps up")]),
            "through": keep([(i_lead, "I state leads diaphragm"), (switch, "expiration switches"),
                             (hr_up, "heart rate steps up")])}


def slide_timeline(z, T, beats, side, pre, post, g, ev, path):
    off, sr, dt, k = z["time_offset_s"], z["sr_sub"], z["hmm_bin_s"], z["ref_k"]
    on_raw = T["on_raw_s"].to_numpy()
    if side == "into":
        t0, t1, marks = on_raw[pre[0]] - 8, on_raw[g[0]] + 5, [on_raw[pre[0]], on_raw[g[0]]]
    elif side == "through":           # all of gasping, post-gasp and the first 10 s of early recovery
        t0, t1 = on_raw[g[0]] - 3, on_raw[post[-1] + 1] + 10
        marks = [on_raw[g[0]], on_raw[post[0]], on_raw[post[-1] + 1]]
    else:
        t0, t1, marks = on_raw[g[-1]] - 5, on_raw[post[-1] + 1] + 8, [on_raw[post[0]], on_raw[post[-1] + 1]]
    rows = ["cond", "dia", "hr", "rate", "state", "raster", "gl"]
    labels = {"dia": "∫ dia (z)", "hr": "heart\n(bpm)", "rate": "breaths\n(/min)", "state": "HMM",
              "raster": f"{len(z['order_units'])} units", "gl": "gasp-\nlikeness"}
    seq = z["F"][k]["seq"]
    scol = np.array([matplotlib.colors.to_rgb(state_colour(z, k, s)) for s in range(k)])
    tb, hr = beats[1:] + off, 60 / np.diff(beats)
    with plt.rc_context(FONT):
        fig = plt.figure(figsize=(W, H))
        gsp = fig.add_gridspec(len(rows), 1, height_ratios=[0.17, 0.7, 0.55, 0.55, 0.17, 1.9, 0.5], hspace=0.13,
                               left=0.1, right=0.99, top=0.95, bottom=0.2)
        ax = {}
        for r, name in enumerate(rows):
            ax[name] = fig.add_subplot(gsp[r], sharex=ax.get("dia"))
        idx = np.flatnonzero((on_raw >= t0 - 3) & (on_raw <= t1 + 3))
        tw.condition_strip(ax["cond"], z, idx, x=on_raw[idx], lim=(t0, t1), counts=False, fontsize=10.5)
        ax["cond"].set_xlim(t0, t1)
        i0, i1 = int((t0 - off) * sr), int((t1 - off) * sr)
        ax["dia"].plot(np.arange(i0, i1) / sr + off, z["dia_sub"][i0:i1], color=INK, lw=0.7)
        ax["dia"].set_ylim(-0.5, 2.1 * np.nanmax(z["dia_sub"][i0:i1]))      # headroom for the event labels
        m = (tb >= t0) & (tb <= t1)
        ax["hr"].plot(tb[m], hr[m], ".", color=INK, ms=2.5)
        ax["hr"].set_ylim(0, 750)
        sel = idx[(on_raw[idx] >= t0) & (on_raw[idx] <= t1)]
        cols = [z["cond_colours"].get(c_, "white") for c_ in z["cond"][sel]]
        y = T["rate_per_min"].to_numpy()[sel]
        ax["rate"].plot(on_raw[sel], y, color=GRID, lw=0.8)
        ax["rate"].scatter(on_raw[sel], y, c=cols, s=22, lw=0.3, edgecolor=INK2, zorder=2)
        b0, b1 = int((t0 - off) / dt), int((t1 - off) / dt)
        ax["state"].imshow(scol[seq[b0:b1]][None], aspect="auto", extent=[t0, t1, 0, 1], interpolation="nearest")
        ev_sp = [z["spikes"][u][(z["spikes"][u] >= t0 - off) & (z["spikes"][u] < t1 - off)] + off
                 for u in z["order_units"]]
        ax["raster"].eventplot(ev_sp, lineoffsets=np.arange(len(ev_sp)), linelengths=0.8, linewidths=0.6, colors=INK)
        ax["raster"].set_ylim(len(ev_sp) - 0.5, -0.5)
        gl = T["gasp_likeness"].to_numpy()
        for i in sel:
            e = on_raw[i + 1] if i + 1 < on_raw.size else on_raw[i]
            ax["gl"].plot([on_raw[i], e], [gl[i]] * 2, color=z["cond_colours"].get(z["cond"][i], INK2), lw=2.4)
        ax["gl"].set_ylim(0.2, 0.85)
        ax["gl"].axhline(0.5, color=GRID, lw=0.7)
        for name in rows[1:]:
            a = ax[name]
            for x in marks:
                a.axvline(x, color=INK2, lw=0.8, ls=(0, (4, 3)))
            for x, _ in ev[side]:
                a.axvline(x, color="#2f2e2b", lw=1.0, ls=":")
            if name in ("state", "raster"):
                a.set_yticks([])
            if name != "gl":
                a.tick_params(labelbottom=False)
            a.set_ylabel(labels[name], rotation=0, ha="right", va="center", labelpad=6)
        top = ax["dia"].get_ylim()[1]
        pad = 0.004 * (t1 - t0)
        for j, (x, text) in enumerate(ev[side]):
            right_edge = x > t1 - 0.2 * (t1 - t0)          # keep labels near the right edge inside the axes
            ax["dia"].text(x - pad if right_edge else x + pad, top if j % 2 == 0 else 0.62 * top, text,
                           ha="right" if right_edge else "left", va="top", fontsize=10.5, color=INK,
                           bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax["gl"].set_xlim(t0, t1)
        ax["gl"].set_xlabel("raw-run time (s)   ·   dashed: window boundaries   ·   dotted: events named on top")
        state_legend(fig, z, y=0.0)
        save_fig(fig, path, dpi=200)
        plt.close(fig)


def slide_strip(z, T, pre, post, g, rows, path, right=None):
    """Breath-by-breath strips (dots in condition colours) from late anoxia to early recovery; `right` = (x column,
    y column, x label, y label, log, legend location) for a scatter beside them."""
    idx = np.arange(pre[0] - N_STRIP, post[-1] + 1 + N_STRIP)
    x = idx - pre[0]
    k = z["ref_k"]
    cols = np.array([z["cond_colours"].get(c_, "white") for c_ in z["cond"][idx]])
    base = T[T["condition"] == "baseline"]
    with plt.rc_context(FONT):
        fig = plt.figure(figsize=(W, H))
        gsp = fig.add_gridspec(len(rows) + 1, 1, height_ratios=[0.22] + [1] * len(rows), hspace=0.16, left=0.115,
                               right=0.69 if right else 0.99, top=0.95, bottom=0.12 if rows[-1] != "mix" else 0.2)
        ax0 = fig.add_subplot(gsp[0])
        tw.condition_strip(ax0, z, idx, shift=-N_STRIP, fontsize=10)
        ax0.set_xlim(x[0] - 0.5, x[-1] + 0.5)
        for r, row in enumerate(rows):
            ax = fig.add_subplot(gsp[r + 1], sharex=ax0)
            if row == "mix":
                bottom = np.zeros(idx.size)
                for s in np.argsort(z["colours"][k]):
                    v = np.nan_to_num(T[f"state{s}_frac"].to_numpy()[idx])
                    ax.bar(x, v, bottom=bottom, width=0.9, color=state_colour(z, k, s), lw=0)
                    bottom += v
                ax.set_ylim(0, 1)
                ax.set_ylabel("HMM state\nmix", rotation=0, ha="right", va="center")
            else:
                col, lab = row
                yv = T[col].to_numpy()[idx]
                lo, hi = np.nanpercentile(base[col], [10, 90])
                ax.axhspan(lo, hi, color=GRID, alpha=0.6, lw=0)
                ax.plot(x, yv, color=INK2, lw=0.6, zorder=1)
                ax.scatter(x, yv, c=cols, s=24, lw=0.3, edgecolor=INK2, zorder=2)
                if col in ("te_ms", "ti_ms"):
                    ax.set_yscale("log")
                    ax.set_ylim((60, 250) if col == "ti_ms" else (80, 4000))
                    ax.set_yticks([60, 100, 200] if col == "ti_ms" else [100, 300, 1000, 3000])
                    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                if col == "gasp_likeness":
                    ax.axhline(0.5, color=INK2, lw=0.6, ls=":")
                if col.startswith("I_state"):
                    ax.axhline(0, color=INK2, lw=0.6, ls=":")
                ax.set_ylabel(lab, rotation=0, ha="right", va="center")
            for b_ in (0, g[0] - pre[0], post[0] - pre[0], post[-1] + 1 - pre[0]):
                ax.axvline(b_ - 0.5, color=INK2, lw=0.7, ls=(0, (4, 3)))
            if r < len(rows) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(f"breath (0 = first pre-gasp breath, {T['on_raw_s'][pre[0]]:.1f} s); grey band = "
                              "baseline 10th–90th percentile")
        if right:
            xc, yc, xl, yl, logs, loc = right
            a = fig.add_axes([0.775, 0.2, 0.215, 0.72])
            Tn = T.assign(ibi_next=T["ibi_s"].shift(-1))
            for name in tw.ZOOM:
                s_ = Tn[Tn["condition"] == name]
                big = name in tw.TRANSITION_COLOURS or name == "gasping"
                a.scatter(s_[xc], s_[yc], s=26 if big else 10, color=z["cond_colours"][name], lw=0.3,
                          edgecolor=INK2 if big else "none", alpha=1.0 if big else 0.5, label=name)
            for name in tw.TRANSITION_COLOURS:
                s_ = Tn[Tn["condition"] == name]
                a.plot(s_[xc], s_[yc], color=z["cond_colours"][name], lw=0.8)
            if logs:
                a.set_xscale("log")
                a.set_yscale("log")
            a.set_xlabel(xl)
            a.set_ylabel(yl)
            a.legend(frameon=False, fontsize=9.5, loc=loc, handletextpad=0.2, borderaxespad=0.2)
        if rows[-1] == "mix":
            state_legend(fig, z, y=0.0, ncol=3, fontsize=9.5)
        save_fig(fig, path, dpi=200)
        plt.close(fig)


def slide_states(z, idx, title, path):
    k = z["ref_k"]
    rgb = np.array([matplotlib.colors.to_rgb(state_colour(z, k, s)) for s in range(k)] + [(1, 1, 1)])
    d = z["dia_df"]
    ons, offs = d["on_sec"].to_numpy(), d["off_sec"].to_numpy()
    with plt.rc_context(FONT):
        fig = plt.figure(figsize=(W, H))
        gsp = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.08, left=0.085, right=0.86, top=0.95,
                               bottom=0.25)
        ax = fig.add_subplot(gsp[0])
        ax.imshow(rgb[tw.state_image(z, idx, "time")], aspect="auto", interpolation="nearest",
                  extent=[-tw.CYCLE_PRE_S, tw.CYCLE_POST_S, idx.size, 0])
        yy = np.arange(idx.size) + 0.5
        ax.plot(offs[idx] - ons[idx], yy, "|", color=INK, ms=6, mew=1.3)
        ax.plot(ons[idx + 1] - ons[idx], yy, "|", color=INK, ms=10, mew=1.8)
        ax.axvline(0, color=INK, lw=0.9)
        ax.set_xlim(-tw.CYCLE_PRE_S, min(2.45, np.max(ons[idx + 1] - ons[idx]) + 0.1))
        ax.set_yticks(yy[::3])
        ax.set_yticklabels([f"{t:.1f}" for t in (ons[idx] + z["time_offset_s"])[::3]])
        ax.set_ylabel("breath onset (raw s)")
        ax.set_xlabel("s from inspiration onset  (short tick: offset, long tick: next onset)")
        ax = fig.add_subplot(gsp[1])
        ax.imshow(rgb[tw.state_image(z, idx, "phase")], aspect="auto", interpolation="nearest",
                  extent=[0, 2 * np.pi, idx.size, 0])
        ax.axvline(np.pi, color=INK, lw=0.9, ls=(0, (4, 3)))
        ax.set_xticks([0, np.pi, 2 * np.pi])
        ax.set_xticklabels(["onset", "offset", "next onset"])
        ax.set_yticks([])
        ax.set_xlabel("cycle phase")
        x0 = 2 * np.pi * 1.03
        for name, a, b in tw.blocks(z["cond"], idx):
            ax.add_patch(Rectangle((x0, a), 0.3, b - a, color=z["cond_colours"].get(name, "white"), clip_on=False))
            ax.text(x0 + 0.42, (a + b) / 2, f"{name}\n({b - a} breaths)", va="center", fontsize=10.5, clip_on=False)
        ax.set_xlim(0, 2 * np.pi)
        state_legend(fig, z, y=0.0, ncol=3, fontsize=10)
        save_fig(fig, path, dpi=200)
        plt.close(fig)


def slide_unit_phase(z, path):
    """Units across, windows down: preferred φ (direction; onset right, inspiration up) and L (length)."""
    tu = z["tuning"].set_index(["unit_id", "condition"])
    units = z["order_units"]
    R = 0.42
    short = {"inspiratory": "ins", "expiratory": "exp", "tonic": "ton"}
    with plt.rc_context(FONT):
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0.13, 0.2, 0.86, 0.72])
        for y, name in enumerate(tw.ZOOM):
            for x, u in enumerate(units):
                ax.add_patch(plt.Circle((x, y), R, fill=False, color=GRID, lw=0.7))
                th, L = tu.loc[(u, name), "preferred_phase"], tu.loc[(u, name), "modulation_L"]
                if not np.isfinite(L * th) or tu.loc[(u, name), "n_spikes"] < tw.MIN_SPIKES:
                    ax.text(x, y, "–", ha="center", va="center", color=INK2, fontsize=9)
                    continue
                tonic = not L >= 0.2
                col = z["cond_colours"][name]
                ax.plot([x, x + R * L * np.cos(th)], [y, y - R * L * np.sin(th)], color=col, lw=1.0 if tonic else 2.6,
                        solid_capstyle="round")
                ax.plot(x + R * L * np.cos(th), y - R * L * np.sin(th), "o", color=col, ms=2.5 if tonic else 4.5)
        ax.set_xlim(-0.6, len(units) - 0.4)
        ax.set_ylim(len(tw.ZOOM) - 0.5, -0.55)
        ax.set_aspect("equal")
        ax.set_yticks(range(len(tw.ZOOM)))
        ax.set_yticklabels(tw.ZOOM, fontsize=11)
        for y, name in enumerate(tw.ZOOM):
            ax.add_patch(Rectangle((-0.03, y - 0.35), 0.011, 0.7, color=z["cond_colours"][name],
                                   transform=ax.get_yaxis_transform(), clip_on=False))
        ax.tick_params(axis="y", pad=30, length=0)
        ax.set_xticks(range(len(units)))
        ax.set_xticklabels([f"#{u} {short.get(tu.loc[(u, 'baseline'), 'class_min_spikes'], 'sil')}" for u in units],
                           rotation=90, fontsize=9.5)
        for s in ("left", "bottom", "right", "top"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="x", length=0)
        fig.text(0.13, 0.955, "Direction = preferred phase (onset at right, offset at left, inspiration up, expiration "
                 f"down); length = modulation L (circle = 1); thin = tonic (L < 0.2); – = < {tw.MIN_SPIKES} spikes. "
                 "Units in baseline φ order, class at baseline.", fontsize=10, color=INK2, va="top", wrap=True)
        tab = z["tuning"]
        parts = []
        for name in tw.ZOOM:
            v = tab.loc[tab["condition"] == name, "class_min_spikes"].value_counts()
            parts.append(f"{name} {v.get('inspiratory', 0)} / {v.get('expiratory', 0)} / {v.get('tonic', 0)} / "
                         f"{v.get(f'< {tw.MIN_SPIKES} spikes', 0)}")
        fig.text(0.13, 0.03, f"Units inspiratory / expiratory / tonic / < {tw.MIN_SPIKES} spikes:   "
                 + "   ·   ".join(parts[:3]) + "\n" + "   ·   ".join(parts[3:]), fontsize=10.5, color=INK,
                 linespacing=1.5)
        save_fig(fig, path, dpi=200)
        plt.close(fig)


def slide_unit_phase_by_breath(z, sides, path):
    """One continuous strip of breaths, from the start of the eupnoea -> gasping range to the end of the gasping ->
    recovery range (every breath in between, all the gasps included)."""
    units = z["order_units"]
    ons = z["dia_df"]["on_sec"].to_numpy()
    ranges = list(sides.values())
    idx = np.arange(ranges[0][0], ranges[-1][-1] + 1)
    with plt.rc_context(FONT):
        fig = plt.figure(figsize=(W, H))
        gsp = fig.add_gridspec(2, 1, height_ratios=[0.09, 1], hspace=0.05, left=0.07, right=0.89, top=0.9,
                               bottom=0.17)
        ax = fig.add_subplot(gsp[1])
        ax.imshow(tw.breath_phase_image(z, idx), aspect="auto", interpolation="nearest",
                  extent=[-0.5, idx.size - 0.5, len(units), 0])
        ax.set_yticks(np.arange(len(units)) + 0.5)
        ax.set_yticklabels([f"#{u}" for u in units], fontsize=7.5)
        ax.set_xticks(np.arange(idx.size)[::5])
        ax.set_xticklabels([f"{t:.0f}" for t in ons[idx][::5] + z["time_offset_s"]], fontsize=9.5)
        ax.set_xlabel("breath onset (raw s), one column per breath")
        top = fig.add_subplot(gsp[0], sharex=ax)
        tw.condition_strip(top, z, idx, fontsize=10)
        top.set_xlim(-0.5, idx.size - 0.5)
        top.set_title("late anoxia → gasping → early recovery", loc="left", fontsize=12, pad=20)
        cax = fig.add_axes([0.905, 0.3, 0.013, 0.5])
        sm = matplotlib.cm.ScalarMappable(cmap=tw.PHASE_CMAP, norm=matplotlib.colors.Normalize(-np.pi, np.pi))
        cb = fig.colorbar(sm, cax=cax)
        cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        cb.set_ticklabels(["offset", "mid E", "onset", "mid I", "offset"])
        cax.set_title("spike φ", fontsize=10, loc="left")
        fig.text(0.07, 0.03, "Colour: circular mean phase of the unit's spikes in that breath; paler = 1–5 spikes, "
                 "white = none. Units in baseline φ order, tonic last.", fontsize=10, color=INK2)
        save_fig(fig, path, dpi=200)
        plt.close(fig)


# ------------------------------------------------------------------------------------------- slide text

def region(deg):
    return next(name for lo, hi, name in REGIONS if lo <= deg < hi)


def unit_story(tu, u):
    """Draft headline and bullets for a unit's transition slides, from transition_unit_tuning (≥ MIN_SPIKES)."""
    r = {n: tu.loc[(u, n)] for n in tw.ZOOM}
    ok = {n: r[n]["n_spikes"] >= tw.MIN_SPIKES for n in tw.ZOOM}
    deg = {n: float(np.degrees(r[n]["preferred_phase"])) for n in tw.ZOOM}

    def where(n):
        return region(deg[n]) if ok[n] and r[n]["modulation_L"] >= 0.2 else None

    def desc(n):
        x = r[n]
        if not ok[n]:
            return f"{x['rate_hz']:.1f} Hz, too few spikes"
        if x["modulation_L"] < 0.2:
            return f"{x['rate_hz']:.1f} Hz, tonic"
        return f"{x['rate_hz']:.1f} Hz, {where(n)} ({deg[n]:+.0f}°, L {x['modulation_L']:.2f})"

    def short(n):                  # for the headline
        if not ok[n]:
            return "near-silent"
        return f"{where(n)} ({deg[n]:+.0f}°)" if where(n) else "tonic"

    steps = [short(n) for n in ("baseline", "gasping", "post-gasp")]
    if len({where(n) for n in ("baseline", "gasping", "post-gasp")} - {None}) == 1 and all(
            where(n) for n in ("baseline", "gasping", "post-gasp")):
        head = f"{where('baseline')} at baseline, in gasping and after"
    else:
        head = f"{steps[0]} at baseline → {steps[1]} in gasping → {steps[2]} after"
    head = head[0].upper() + head[1:]
    return head, [f"{n}: {desc(n)}" for n in tw.ZOOM]


FIGURES = [
    ("timeline_into.png", "Into gasping: the population changes before breathing and the heart",
     "Raw-run time from 8 s before the first pre-gasp breath to 5 s into gasping. Rows: integrated diaphragm, every "
     "heartbeat (60 / R-R), breathing rate per breath, K = 6 HMM state, spikes of all 28 units, gasp-likeness per "
     "breath. Population rate falls and gasp-likeness rises from ~1822.3 s while breathing is still eupnoeic; at "
     "1823.8 s expiration moves to the post-I / late-E states of recovery; the heart rate halves within one beat at "
     "~1824.5 s; after the first pre-gasp breath (1825.2 s) a 2.35 s apnoea; from the second pre-gasp breath "
     "(1827.6 s) the gasp state fills expiration, 9 breaths before the Bush rule's first gasp."),
    ("timeline_out.png", "Out of gasping: breathing speeds up while expiration stays in the gasp state",
     "From 5 s before the last gasp to 8 s into early recovery. Over 11 post-gasp breaths the rate rises 73 → 98/min "
     "while the gasp state still fills 55–82 % of each cycle; at 1870.3 s expiration switches to the recovery post-I "
     "state in one breath (rate steps 94 → 113/min); the heart steps up 8.6 s later."),
    ("timeline_through.png", "From the first gasp to early recovery on one time axis",
     "From 3 s before the first Bush-rule gasp to 10 s into early recovery. The slowest gasps are mid-gasping; from "
     "~1855 s gasps speed up and firing rises while the gasp state still fills expiration. The population "
     "inspiratory state starts > 50 ms before the diaphragm from the second post-gasp breath (1863.5 s), expiration "
     "switches at 1870.3 s, and the heart steps up at 1879.0 s."),
    ("breathing.png", "Breath by breath: pre-gasp breaths are gasp-shaped; the heart rate halves in one beat",
     "Dots in condition colours, from 10 breaths before the first pre-gasp breath to 10 into early recovery; grey "
     "band = baseline 10th–90th percentile; dashed = window boundaries. Pre-gasp breaths have gasp-like Ti (96 ms), "
     "fast rise and duty 0.11 at IBIs of 0.71–1.08 s; they move more air than baseline or gasps. Right: breath-to-"
     "breath timing (IBI n vs n+1)."),
    ("population.png", "Population: gasp mode starts 9 breaths before the gasp rule; pre-I returns before "
     "expiration recovers",
     "Population rate, gasp-likeness (distance to the baseline vs gasp mean cycle; 0.5 = halfway), when the "
     "population inspiratory state starts relative to the diaphragm onset, and the K = 6 state mix per breath. In "
     "post-gasp the I state starts up to 150 ms before the diaphragm while the gasp state still fills expiration. "
     "Right: gasp-likeness against breathing rate — post-gasp breaths speed up at gasp-like population patterns."),
    ("states_into.png", "HMM states into gasping: recovery-type expiratory states, then the gasp state",
     "Each row is one breath (recording order), coloured by the K = 6 Viterbi state from 0.3 s before onset to the "
     "next onset (left) and over the cycle phase (right)."),
    ("states_out.png", "HMM states out of gasping: pre-I grows, then expiration switches in one breath",
     "As the previous slide, from the last gasps into early recovery. Early-I (pink) appears in late expiration and "
     "grows breath by breath; at 1870.3 s the gasp state (orange) gives way to the recovery post-I state (blue)."),
    ("unit_phase.png", "Units: those that return after gasping fire just before inspiration",
     "Preferred phase and modulation of every unit in the six windows (cibrrig phase curves; ≥ 20 spikes). #312, "
     "#350 and #363 fire mid-expiration at baseline (−64° to −103°) and late E / pre-I after gasping (−7° to −13°); "
     "#510 moves to onset; the inspiratory core (#331, #459, #362) keeps its phase throughout."),
    ("unit_phase_by_breath.png", "Each unit's spike phase, breath by breath",
     "Circular mean phase of each unit's spikes on every breath through the two transitions."),
]


SECTION_B = {"title": "Anoxia #2: the same transitions",
             "body": "The 27 anoxia #2 partners of the anoxia #1 singles (labelled by their anoxia #1 ids), decoded with "
                     "the anoxia #1 K = 6 model and projected on the anoxia #1 PCA axes. 8 gasps (20 in anoxia #1)."}
FIGURES_B = [
    ("timeline_into.png", "Anoxia #2, into gasping: the same sequence as anoxia #1",
     "Anoxia #2 starts from a recovery-like network: 50–65 % of each baseline cycle is in the recovery late-E state. "
     "At the end of N2 (~2325 s) it switches back to the anoxia #1 baseline late-E state and breathing speeds up "
     "(160 → 188/min). Then, as in anoxia #1: firing falls from ~2334 s, ~2 s before breathing slows; expiration leaves "
     "the baseline state at 2336.75 s; the heart rate halves within one beat at 2336.9 s (small regular deflections "
     "at the old rhythm between QRS complexes suggest 2:1 AV block); a 2.83 s apnoea from 2338.6 s; the gasp state "
     "fills each cycle from 2341.4 s, 8 breaths before the Bush rule's first gasp (2348.1 s)."),
    ("timeline_out.png", "Anoxia #2, out of gasping: expiration switches 1.7 s after the last gasp",
     "Only the first post-gasp breath is partly in the gasp state (51 %); from 2358.14 s expiration is in the recovery "
     "post-I state. Breathing climbs from 76 to ~146/min over ~11 breaths. In anoxia #1 expiration stayed in the gasp "
     "state for 11 faster breaths. The heart steps up 9.4 s after the switch (8.6 s in anoxia #1)."),
    ("timeline_through.png", "Anoxia #2, from the first gasp to early recovery",
     "8 Bush-rule gasps (2348.1–2356.5 s); the late gasps speed up and firing rises from ~2353 s. Expiration switches "
     "at 2358.14 s and the heart steps up at 2367.5 s. The population inspiratory state never leads the diaphragm by "
     "> 50 ms here, so that event isn't marked."),
    ("breathing.png", "Anoxia #2, breath by breath: the pre-gasp breaths after the apnoea are gasp-shaped",
     "Dots in condition colours; grey band = anoxia #2 baseline 10th–90th percentile. After the apnoea, pre-gasp "
     "breaths have Ti 95–128 ms at IBIs of 0.6–1.0 s, like gasps but faster. Right: breath-to-breath timing."),
    ("population.png", "Anoxia #2, population: gasp mode 8 breaths before the gasp rule, none after the last gasp",
     "Population rate, gasp-likeness (anoxia #2's own baseline and gasp mean cycles), when the inspiratory state starts "
     "relative to the diaphragm (post-gasp median −13 ms, vs −125 ms in anoxia #1), and the K = 6 state mix (anoxia #1 "
     "model). Right: gasp-likeness against breathing rate."),
    ("states_into.png", "Anoxia #2, HMM states into gasping (anoxia #1 model)",
     "Each row is one breath, coloured by the anoxia #1 K = 6 Viterbi state from 0.3 s before onset to the next onset "
     "(left) and over the cycle phase (right). Baseline late-E (aqua) gives way to recovery post-I / late-E (blue, "
     "yellow) three breaths before the apnoea, as in anoxia #1."),
    ("states_out.png", "Anoxia #2, HMM states out of gasping: recovery post-I from the second breath",
     "As the previous slide, from the last gasps into early recovery. The gasp state (orange) is gone from the second "
     "post-gasp breath; early I (pink) in late expiration stays small (5 % of post-gasp time vs 20 % in anoxia #1)."),
    ("unit_phase.png", "Anoxia #2, units: the returning expiratory units again fire later in expiration",
     "Preferred phase and modulation per window (≥ 20 spikes). #312 −98° → −29°, #350 −52° → −13°, #363 −84° → −31°, "
     "#510 −71° → −8° (baseline → post-gasp); #435 and #479 keep their phase; the inspiratory core (#331, #459, #362) "
     "keeps its phase. #422 has no anoxia #2 partner. Only 11 of 27 units reach 20 spikes in the 8 gasps."),
    ("unit_phase_by_breath.png", "Anoxia #2, each unit's spike phase, breath by breath",
     "Circular mean phase of each unit's spikes on every breath through the two transitions of anoxia #2."),
]


def write_manifest(z, run, out_slides, deck_manifest):
    """Anoxia #1: transitions.json with figures and per-unit slides, linked from slides.json as "transitions".
    Another trial: its figure slides only, appended to slides.json's "transition_sections"."""
    other = z.get("trial", "anoxia #1") != "anoxia #1"
    tu = z["tuning"].set_index(["unit_id", "condition"])
    info = z["info"].set_index("unit_id")
    units = {}
    for u in ([] if other else z["order_units"]):
        head, bullets = unit_story(tu, u)
        units[str(int(u))] = {"headline": head, "bullets": bullets, "depth_um": float(info.loc[u, "depth_um"]),
                              "raster": str(z["slide_out"] / f"unit_{u}_raster.png"),
                              "peth": str(z["slide_out"] / f"unit_{u}_peth.png")}
    windows = []
    ons = z["dia_df"]["on_sec"].to_numpy()
    for name in tw.ZOOM:
        i = np.flatnonzero(z["cond"] == name)
        windows.append({"name": name, "start_raw_s": float(ons[i[0]] + z["time_offset_s"]),
                        "end_raw_s": float(ons[i[-1] + 1] + z["time_offset_s"]), "n_breaths": int(i.size),
                        "colour": z["cond_colours"][name]})
    section = SECTION_B if other else {
        "title": "Transitions into and out of gasping",
        "body": "Pre-gasp (10 breaths) and post-gasp (16 breaths) as windows of their own: breathing, heart rate, HMM "
                "states, PCA and single units around each transition"}
    m = {"section": section,
         "figures": [{"title": t, "path": str(out_slides / f), "w": W, "h": H, "notes": n}
                     for f, t, n in (FIGURES_B if other else FIGURES)],
         "windows": windows, "units": units,
         "unit_note": "Transition slides: spikes on every breath from late anoxia to early recovery (whole cycles), "
                      "then PETHs and phase curves with pre-gasp and post-gasp as their own columns."}
    path = z["slide_out"] / "transitions.json"
    path.write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    if deck_manifest.exists():
        M = json.loads(deck_manifest.read_text(encoding="utf-8"))
        if other:
            M["transition_sections"] = sorted(set(M.get("transition_sections", [])) | {str(path)})
        else:
            M["transitions"] = str(path)
        deck_manifest.write_text(json.dumps(M, indent=1), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------------ intro slides
# The larger question and hypothesis (paraphrased from the K00 Research Strategy: Innovation, Aims 1-3), and the
# questions these recordings can address. Read by make_unit_slides.js through slides.json's "intro".
INTRO = {
    "title": "How does the respiratory network collapse into gasping, and how does it rebuild eupnea?",
    "question": "How does the ventral respiratory column (VRC) reorganize when breathing collapses from eupnea into "
                "hypoxic gasping, and how does it rebuild eupnea during autoresuscitation?",
    "hypothesis": [
        "Eupnea is a rotational population trajectory, held in place by inhibitory and sensory feedback around an "
        "inspiration-off attractor.",
        "Gasping is a ballistic collapse: the network unembeds from this manifold and periodic excitatory bursts take "
        "over.",
        "Recovery requires re-embedding: medullary inhibition, lung afferent feedback and coupled cardiorespiratory "
        "output come back together."],
    "figure_caption": "e097, 28 single units in PC1–PC3 (PCA fit on baseline). Thin: 15 cycles; thick: mean; "
                      "X: baseline inspiration-off point.",
    "subquestions_title": "Four questions these recordings let us ask",
    "subquestions": [
        {"q": "Does the network leave eupnea before breathing does?",
         "how": "The order of changes in population state, firing, heart rate and breathing over the breaths before "
                "the first gasp.",
         "aim": "K00 Aim 1", "where": "Transitions: timelines, breath by breath, HMM states into gasping"},
        {"q": "Which parts of the breath cycle are rebuilt first?",
         "how": "Inspiratory and pre-inspiratory vs post-inspiratory and expiratory states after the last gasp; when "
                "the population leads the diaphragm; single-unit phase shifts; the rotation re-forming.",
         "aim": "K00 Aims 1–3", "where": "Transitions: HMM states out of gasping, units; unit transition slides"},
        {"q": "How do heart and breathing change together?",
         "how": "Beat-by-beat heart rate against the breath-by-breath and network events of both transitions.",
         "aim": "K00 Aim 3", "where": "Transitions: timelines, breath by breath"},
        {"q": "Does the same sequence play out in a second anoxia?",
         "how": "The same 27 neurons, decoded with the same population model, in anoxia #2: 8 gasps, starting from a "
                "network that had not fully recovered.",
         "aim": "Replication", "where": "Anoxia #2 section"},
    ],
}


def slide_intro_trajectories(z, path):
    """Three small PC1-PC3 panels: baseline rotation, ballistic gasps, early-recovery rotation (15 cycles + mean)."""
    P, dt = z["P"], z["pca_bin_s"]
    ons = z["dia_df"]["on_sec"].to_numpy()
    panels = [("baseline", "eupnea\nrotation"), ("gasping", "gasping\nballistic"),
              ("early recovery", "early recovery\nrotation returns")]
    base = np.concatenate([P[int(ons[i] / dt):int(ons[i + 1] / dt), :][:, [0, 2]]
                           for i in fp.cycles_of(z, "baseline", 30)])
    lim = np.percentile(base, [1, 99], axis=0)
    pad = 0.25 * (lim[1] - lim[0])
    with plt.rc_context(FONT):
        fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.9), sharex=True, sharey=True)
        for ax, (name, title) in zip(axes, panels):
            col = z["cond_colours"][name]
            for i in fp.cycles_of(z, name, 15):
                seg = P[int(ons[i] / dt):int(ons[i + 1] / dt) + 1][:, [0, 2]]
                ax.plot(seg[:, 0], seg[:, 1], color=col, lw=0.7, alpha=0.45)
            m = tw.mean_cycle(z, name, (0, 2))
            ax.plot(m[:, 0], m[:, 1], color=INK if name == "baseline" else col, lw=2.4)
            ax.plot(z["ioff_point"][0], z["ioff_point"][2], "X", color=INK, ms=8)
            ax.set_title(title, fontsize=11, color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(lim[0, 0] - pad[0], lim[1, 0] + pad[0])
            ax.set_ylim(lim[0, 1] - pad[1], lim[1, 1] + pad[1])
            ax.set_xlabel("PC1", fontsize=10)
        axes[0].set_ylabel("PC3", fontsize=10)
        fig.tight_layout(w_pad=0.6)
        save_fig(fig, path, dpi=220)
        plt.close(fig)


def write_intro(out_slides, json_dir, deck_manifest):
    path = json_dir / "intro.json"
    path.write_text(json.dumps({**INTRO, "figure": str(out_slides / "intro_trajectories.png"), "figure_w": 6.4,
                                "figure_h": 2.9}, indent=1, ensure_ascii=False), encoding="utf-8")
    if deck_manifest.exists():
        M = json.loads(deck_manifest.read_text(encoding="utf-8"))
        M["intro"] = str(path)
        deck_manifest.write_text(json.dumps(M, indent=1), encoding="utf-8")
    return path


def main():
    a = parse_args()
    run = Path(a.run)
    out = run / "transitions"
    out_slides = out / "slides"
    out_slides.mkdir(exist_ok=True)
    c = tw.load_context(run)
    cond, pre, post, g = tw.label_transitions(c["cond"])
    z = tw.zoom_context(c, cond, out)
    T = pd.read_csv(out / "transition_breaths.tsv", sep="\t")
    T["condition"] = T["condition"].fillna("")
    beats = pd.read_csv(out / "heartbeats.tsv", sep="\t")["beat_raw_s"].to_numpy() - c["time_offset_s"]
    ev = events(z, T, beats, pre, post, g)
    for side in ("into", "out", "through"):
        slide_timeline(z, T, beats, side, pre, post, g, ev, out_slides / f"timeline_{side}.png")
    slide_strip(z, T, pre, post, g, [("rate_per_min", "breaths\n(/min)"), ("ti_ms", "Ti (ms)"), ("te_ms", "Te (ms)"),
                                     ("amp", "∫dia peak\n(z)"), ("flow_insp_peak", "peak insp.\nflow (V)"),
                                     ("hr_bpm", "heart rate\n(bpm)")],
                out_slides / "breathing.png",
                right=("ibi_s", "ibi_next", "IBI n (s)", "IBI n+1 (s)", True, "upper left"))
    slide_strip(z, T, pre, post, g, [("pop_rate_hz", "population\n(Hz/unit)"), ("gasp_likeness", "gasp-\nlikeness"),
                                     ("I_state_entry_ms_from_onset", "I state −\ndia onset\n(ms)"), "mix"],
                out_slides / "population.png",
                right=("rate_per_min", "gasp_likeness", "breathing rate (/min)", "gasp-likeness", False, "upper right"))
    sides = tw.side_ranges(pre, post, g)
    for (title, idx), side in zip(sides.items(), ("into", "out")):
        slide_states(z, idx, title, out_slides / f"states_{side}.png")
    slide_unit_phase(z, out_slides / "unit_phase.png")
    slide_unit_phase_by_breath(z, sides, out_slides / "unit_phase_by_breath.png")
    print("events:", {k: [(round(t, 2), s) for t, s in v] for k, v in ev.items()})
    if not a.no_manifest:
        print("wrote", write_manifest(z, run, out_slides, Path(a.deck_manifest)))
        if z.get("trial", "anoxia #1") == "anoxia #1":
            slide_intro_trajectories(z, out_slides / "intro_trajectories.png")
            print("wrote", write_intro(out_slides, z["slide_out"], Path(a.deck_manifest)))


if __name__ == "__main__":
    main()
