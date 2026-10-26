"""Time-lapse of the K = 6 HMM states in PC1-PC3 moving from one epoch to the next, one anoxia trial at a time.

One fixed oblique view throughout. The trial opens on its first epoch; each epoch then fades into the next and holds.
An epoch is drawn as its 5 ms bins (faint, coloured by Viterbi state) and its mean breath loop (PC1-PC3 averaged in 50
phase bins, in the epoch's colour, ◆ = inspiration onset); the baseline loop stays as a grey dashed reference. A strip
along the bottom marks the epoch on screen. PCA: the runs' own (Bush & Ramirez), with equal units on all three axes.

  cibrrig python transition_movie.py [--run <population_dynamics run>] [--elev 22 --azim -62]

Writes <run>/transitions/movie/: <trial>_transitions.gif and <trial>_epochs.png (every epoch at the movie's view).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figures_population as fp  # noqa: E402
import transition_slides as tsl  # noqa: E402
import transition_windows as tw  # noqa: E402
from figures_population import INK, INK2, GRID, state_colour  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

FPS = 15
FIRST_S, FADE_S, HOLD_S = 1.5, 1.6, 1.2    # first epoch alone, then per transition: cross-fade, next epoch alone
ELEV, AZIM = 22, -62                        # degrees; PC1 runs left to right, PC2 recedes, PC3 up
N_DRAWN = 2500                              # 5 ms bins drawn per epoch (random, seed 0)
DOT_ALPHA = 0.35
CYCLE_BINS = 50
BG = "#fcfcfb"
AXES = ["PC1", "PC2", "PC3"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", default=str(tw.RUN), help="population_dynamics.py output folder (anoxia #1 by default)")
    p.add_argument("--elev", type=float, default=ELEV)
    p.add_argument("--azim", type=float, default=AZIM)
    return p.parse_args()


def epochs(c, rng):
    k = c["ref_k"]
    cond = tw.label_transitions(c["cond"])[0]
    bcond = tw.bin_conditions(c, cond)
    P, seq, off = c["P"][:, :3], c["F"][k]["seq"], c["time_offset_s"]
    ons = c["dia_df"]["on_sec"].to_numpy()
    colours = {**c["cond_colours"], **tw.TRANSITION_COLOURS}
    out = []
    for name in tw.epoch_order(c, cond):
        idx = np.flatnonzero(bcond == name)
        sub = idx if idx.size <= N_DRAWN else np.sort(rng.choice(idx, N_DRAWN, replace=False))
        _, loop = fp.phase_binned(P, c["ph"], bcond == name, nbins=CYCLE_BINS)
        b = np.flatnonzero(cond == name)
        out.append({"name": name, "colour": colours.get(name, "#cccccc"), "pts": P[sub],
                    "sc": [state_colour(c, k, s) for s in seq[sub]], "loop": loop[~np.isnan(loop).any(axis=1)],
                    "onset": loop[CYCLE_BINS // 2],         # phase bins run -pi..pi; 0 = inspiration onset
                    "n_breaths": int(b.size), "t0": ons[b[0]] + off, "t1": ons[min(b[-1] + 1, ons.size - 1)] + off})
    return out


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def render(c, E, path, sheet_path, trial, elev, azim):
    lims = [tw.PC_LIMITS[d] for d in range(3)]
    fig = plt.figure(figsize=(9.6, 7.4), dpi=90, facecolor=BG)
    ax = fig.add_axes((0.0, 0.08, 1.0, 0.8), projection="3d")
    ax.set_facecolor(BG)
    ax.set_box_aspect([l[1] - l[0] for l in lims], zoom=1.2)
    for d, axis in enumerate((ax.xaxis, ax.yaxis, ax.zaxis)):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"].update(color=GRID, linewidth=0.5)
        axis.set_label_text(AXES[d], color=INK2)
        axis.set_tick_params(labelsize=7, colors=INK2)
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.set_zlim(*lims[2])
    ax.view_init(elev=elev, azim=azim)

    base = next(e for e in E if e["name"] == "baseline")
    ref, = ax.plot(*np.vstack([base["loop"], base["loop"][:1]]).T, color="#9a9994", lw=1.2, ls=(0, (4, 3)))
    arts = []
    for e in E:
        lp = np.vstack([e["loop"], e["loop"][:1]])
        arts.append([ax.scatter(*e["pts"].T, c=e["sc"], s=2.5, depthshade=False, edgecolors="none", alpha=0),
                     ax.plot(*lp.T, color=INK, lw=4.6, alpha=0, solid_capstyle="round")[0],
                     ax.plot(*lp.T, color=e["colour"], lw=2.8, alpha=0, solid_capstyle="round")[0],
                     ax.plot(*e["onset"], marker="D", ms=6, color=e["colour"], mec=INK, mew=1, alpha=0, ls="")[0]])

    title = fig.text(0.02, 0.965, "", fontsize=13, fontweight="bold", color=INK, va="top")
    sub = fig.text(0.02, 0.928, "", fontsize=9.5, color=INK2, va="top")
    fig.text(0.02, 0.105, "Dots: 5 ms bins coloured by HMM state · bold line: the epoch's mean breath loop · "
             "◆ inspiration onset", fontsize=8.5, color=INK2, va="top")
    strip = fig.add_axes((0.02, 0.015, 0.96, 0.035))
    strip.set_axis_off()
    strip.set_xlim(0, len(E))
    strip.set_ylim(0, 1)
    blocks = []
    for j, e in enumerate(E):
        blocks.append(strip.add_patch(Rectangle((j + 0.04, 0.05), 0.92, 0.9, color=e["colour"], alpha=0.25, lw=0)))
        strip.text(j + 0.5, 1.25, e["name"], ha="center", va="bottom", fontsize=7.5, color=INK2)
    k = c["ref_k"]
    f = c["F"][k]["feats"]
    order = np.argsort(c["colours"][k])
    handles = [plt.Line2D([], [], marker="o", ls="", color=state_colour(c, k, s), ms=6) for s in order]
    handles.append(plt.Line2D([], [], color="#9a9994", lw=1.2, ls=(0, (4, 3))))
    fig.legend(handles, [tsl.state_name(f.loc[s, "label"]) for s in order] + ["baseline mean loop"],
               loc="upper right", bbox_to_anchor=(0.995, 0.975), fontsize=7.5, frameon=False, ncol=2)

    def show(weights):
        """weights: how visible each epoch is (0..1)."""
        for (sc, outline, line, onset), w in zip(arts, weights):
            sc.set_alpha(w * DOT_ALPHA)
            for art in (outline, line, onset):
                art.set_alpha(w)
        for blk, w in zip(blocks, weights):
            blk.set_alpha(0.25 + 0.75 * w)
        on = [e for e, w in zip(E, weights) if w > 0.5] or [E[int(np.argmax(weights))]]
        e = on[0]
        ref.set_alpha(0 if e["name"] == "baseline" else 0.9)
        j = E.index(e)
        title.set_text(f"{trial} · {j + 1}. {e['name']}")
        sub.set_text(f"{e['t0']:.0f}–{e['t1']:.0f} s raw time · {e['n_breaths']} breaths")

    n_first, n_fade, n_hold = int(FIRST_S * FPS), int(FADE_S * FPS), int(HOLD_S * FPS)
    stills = []
    writer = PillowWriter(fps=FPS)
    with writer.saving(fig, str(path), dpi=90):
        w = np.zeros(len(E))
        w[0] = 1
        show(w)
        for _ in range(n_first):
            writer.grab_frame(facecolor=BG)
        stills.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        for i in range(1, len(E)):
            for t in range(n_fade + n_hold):
                w = np.zeros(len(E))
                w[i] = smooth(t / n_fade)
                w[i - 1] = 1 - w[i]
                show(w)
                writer.grab_frame(facecolor=BG)
            stills.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    h, wd = stills[0].shape[:2]
    ncol = 3
    nrow = int(np.ceil(len(stills) / ncol))
    sheet = np.full((nrow * h, ncol * wd, 3), 252, dtype=np.uint8)
    for j, im in enumerate(stills):
        sheet[(j // ncol) * h:(j // ncol + 1) * h, (j % ncol) * wd:(j % ncol + 1) * wd] = im
    Image.fromarray(sheet).resize((ncol * wd // 2, nrow * h // 2), Image.LANCZOS).save(sheet_path)


def main():
    a = parse_args()
    run = Path(a.run)
    c = tw.load_context(run)
    trial = c.get("trial", "anoxia #1")
    E = epochs(c, np.random.default_rng(0))
    out = run / "transitions" / "movie"
    out.mkdir(parents=True, exist_ok=True)
    tag = trial.replace(" #", "").replace(" ", "_")
    render(c, E, out / f"{tag}_transitions.gif", out / f"{tag}_epochs.png", trial, a.elev, a.azim)
    print(out / f"{tag}_transitions.gif")


if __name__ == "__main__":
    main()
