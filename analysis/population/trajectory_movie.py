"""Movie of one anoxia trial in the style of Bush & Ramirez 2024's supplementary video (41593_2023_1520_MOESM8_ESM):
the integrated diaphragm, the population raster and the population state in PC1-PC3 running together, from eupnea
through anoxia and gasping to recovery.

Layout (1080 x 1920, 30 fps, black with a white centre column): a title card, then three panels around a dotted "now"
line. Top: the integrated diaphragm EMG over ±1 s. Middle: every unit over ±1 s (spike counts in 10 ms bins, scaled
as raster() says; magma; rows in population_dynamics.py's order). Bottom: the run's
own PCA (Bush & Ramirez) in 3D, where every 5 ms bin played so far stays as a dot coloured by its K = 6 HMM state.
When an epoch ends its dots fade to light grey, as the reference greys out eupnea once gasping starts, and once the
next epoch ends too, only every GREY_KEEP-th of them is kept. The last 200 ms are joined by a line ending in a larger
dot. The legend lists the states seen so far: in colour, paler the rarer, with their share of the current epoch's 5 ms
bins so far, if they occur in the current epoch; grey if they don't. Each epoch's name fades in under the 3D panel
when it starts; a speed note shows while playback isn't real time.

Playback: the last BASE_S of baseline in real time, anoxia to the end of early recovery at 2x, the rest at 8x.

  cibrrig python trajectory_movie.py [--run <population_dynamics run>] [--preview] [--pcs 1,3,2] [--elev --azim]

--pcs sets which PC goes on x, y and z (vertical). The default 1,2,3 stands the breathing loop up in the PC1-PC3 wall;
1,3,2 lays it flat in PC1-PC3 with PC2 as its height, so the loop's plane rising and sinking between epochs shows.

Writes <run>/transitions/movie/<trial>_trajectory.mp4 (with --pcs 1,3,2: <trial>_trajectory_PC2-up.mp4) as H.264
through ffmpeg, which environment/cibrrig.yml installs (without ffmpeg, OpenCV's MPEG-4 Part 2, about 10x larger and
not always playable in PowerPoint); or with --preview ..._preview.png: six frames at key moments, to check the look
before the long render.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import population_dynamics as pdm  # noqa: E402
import transition_slides as tsl  # noqa: E402
import transition_windows as tw  # noqa: E402
from figures_population import INK, INK2, state_colour  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

W, H, FPS = 1080, 1920, 30
BASE_S = 8.0                    # baseline before anoxia, real time
MID_SPEED, LATE_SPEED = 2, 8    # anoxia to the end of early recovery; after that
WIN_S = 1.0                     # diaphragm and raster: ±WIN_S around now
RASTER_BIN_S = 0.010
TRAIL_S = 0.2
GREY = np.array(to_rgb("#cfceca"))
GREY_FADE_S = 1.0               # video seconds for a finished epoch's dots to turn grey
GREY_KEEP = 3
DOT_SIZE, DOT_ALPHA, GREY_ALPHA = 7, 0.6, 0.35
TITLE_S, END_S = 4.0, 1.5
LABEL_IN_S, LABEL_HOLD_S, LABEL_OUT_S = 0.4, 2.2, 0.6
CRF = 20                        # x264 quality (lower = better, larger); 18-23 is the usual range
# camera for each axis order (the PCs on x, y, z): elevation, azimuth, 3D zoom, left edge of the 3D panel
VIEWS = {(1, 2, 3): (22, -62, 0.97, 0.14),      # PC3 up: the breathing loop stands in the PC1-PC3 wall
         (1, 3, 2): (35, -45, 0.97, 0.14)}      # PC2 up: the loop lies flat in PC1-PC3, PC2 is its height


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", default=str(tw.RUN), help="population_dynamics.py output folder (anoxia #1 by default)")
    p.add_argument("--preview", action="store_true", help="six still frames instead of the movie")
    p.add_argument("--pcs", default="1,2,3", help="the PCs on the x, y and z (vertical) axes; 1,3,2 puts PC2 up")
    p.add_argument("--elev", type=float, help="camera elevation (default: VIEWS for the axis order)")
    p.add_argument("--azim", type=float, help="camera azimuth (default: VIEWS for the axis order)")
    a = p.parse_args()
    a.pcs = tuple(int(x) for x in a.pcs.split(","))
    assert sorted(a.pcs) == [1, 2, 3], "--pcs takes an order of 1, 2 and 3"
    return a


# ------------------------------------------------------------------------------------------------ data

def epoch_spans(c):
    """[(name, start, end)] in file seconds, in time order (pre-gasp and post-gasp as their own epochs)."""
    cond = tw.label_transitions(c["cond"])[0]
    ons = c["dia_df"]["on_sec"].to_numpy()
    spans = []
    for name in tw.epoch_order(c, cond):
        b = np.flatnonzero(cond == name)
        spans.append((name, ons[b[0]], ons[min(b[-1] + 1, ons.size - 1)]))
    return spans


def schedule(spans):
    """Data time (file s) and playback speed of every frame."""
    t0 = {n: a for n, a, _ in spans}
    t1 = {n: b for n, _, b in spans}
    segs = [(t0["anoxia"] - BASE_S, t0["anoxia"], 1), (t0["anoxia"], t1["early recovery"], MID_SPEED),
            (t1["early recovery"], spans[-1][2], LATE_SPEED)]
    times, speeds = [], []
    for a, b, v in segs:
        n = int(round((b - a) / v * FPS))
        times.append(a + np.arange(n) * v / FPS)
        speeds.append(np.full(n, v))
    return np.concatenate(times), np.concatenate(speeds)


def raster(c, units, n_bins):
    """Spike counts per 10 ms bin, each unit scaled to its 99.9th percentile but at least 2 spikes: a lone spike
    shows purple, bursts bright."""
    counts = pdm.bin_counts(c["spikes"], units, RASTER_BIN_S, n_bins)
    scale = np.maximum(np.percentile(counts, 99.9, axis=0), 2)
    return np.clip(counts / scale, 0, 1).T          # units x bins


# ------------------------------------------------------------------------------------------------ figure

class Movie:
    def __init__(self, c, pcs=(1, 2, 3), elev=None, azim=None):
        self.c = c
        self.pcs = pcs
        view = VIEWS.get(pcs, VIEWS[(1, 2, 3)])
        elev, azim = view[0] if elev is None else elev, view[1] if azim is None else azim
        self.zoom, self.x0 = view[2], view[3]
        k = c["ref_k"]
        self.trial = c.get("trial", "anoxia #1")
        self.off = c["time_offset_s"]
        self.spans = epoch_spans(c)
        self.times, self.speeds = schedule(self.spans)
        starts = np.array([a for _, a, _ in self.spans])
        self.frame_epoch = np.clip(np.searchsorted(starts, self.times, side="right") - 1, 0, len(starts) - 1)
        self.epoch_first = {j: int(np.flatnonzero(self.frame_epoch == j)[0]) for j in np.unique(self.frame_epoch)}
        self.P = c["P"][:, [p - 1 for p in pcs]]      # x, y, z
        self.cen = np.asarray(c["centres"])
        seq = np.asarray(c["F"][k]["seq"])
        assert seq.size == len(self.P), "HMM states and PCs are not on the same 5 ms bins"
        self.state_rgb = np.array([to_rgb(state_colour(c, k, s)) for s in range(k)])
        self.rgb = self.state_rgb[seq]
        f = c["F"][k]["feats"]
        self.state_names = [tsl.state_name(f.loc[s, "label"]) for s in range(k)]
        self.state_order = [int(s) for s in np.argsort(c["colours"][k])]
        self.cum = np.vstack([np.zeros(k, dtype=np.int64), np.cumsum(np.eye(k, dtype=np.int64)[seq], axis=0)])
        self.bin_epoch = np.clip(np.searchsorted(starts, self.cen, side="right") - 1, 0, len(starts) - 1)
        self.epoch_bin0 = np.searchsorted(self.cen, starts)
        self.units = list(c["order_units"])
        self.R = raster(c, self.units, int(np.ceil(self.cen[-1] / RASTER_BIN_S)) + 1)
        self.dia, self.sr = np.asarray(c["dia_sub"]), c["sr_sub"]
        i0, i1 = int((self.times[0] - WIN_S) * self.sr), int((self.times[-1] + WIN_S) * self.sr)
        lo, hi = np.percentile(self.dia[max(i0, 0):i1], [0.5, 99.9])
        self.dia_lim = (lo - 0.05 * (hi - lo), hi + 0.1 * (hi - lo))
        self.first_bin = int(np.searchsorted(self.cen, self.times[0]))
        self._build(k, elev, azim)

    def _build(self, k, elev, azim):
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="black")
        fig.patches.append(Rectangle((1 / 6, 0), 2 / 3, 1, transform=fig.transFigure, color="white", zorder=-10))
        self.header = fig.text(0.19, 0.982, "", fontsize=15, color=INK2, va="top")

        ad = fig.add_axes((0.215, 0.895, 0.57, 0.06))
        ad.set_axis_off()
        self.dia_line, = ad.plot([], [], color=INK, lw=1.6)
        ad.axvline(0, color=INK, ls=(0, (1.5, 2.5)), lw=2.5)
        ad.set_xlim(-WIN_S, WIN_S)
        ad.set_ylim(*self.dia_lim)

        ar = fig.add_axes((0.215, 0.535, 0.57, 0.35))
        n = len(self.units)
        self.img = ar.imshow(np.zeros((n, int(2 * WIN_S / RASTER_BIN_S))), cmap="magma", vmin=0, vmax=1,
                             aspect="auto", interpolation="nearest", extent=(-WIN_S, WIN_S, n, 0))
        ar.axvline(0, color="white", ls=(0, (1.5, 2.5)), lw=2.5)
        ar.set_yticks([])
        ar.set_xticks(np.arange(-1, 1.01, 0.5))
        ar.tick_params(labelsize=13, colors=INK)
        ar.set_xlabel("Time (s)", fontsize=14, color=INK)
        for s in ar.spines.values():
            s.set_visible(False)
        ar.plot([-WIN_S * 1.05] * 2, [n, n - 10], color=INK, lw=6, clip_on=False, solid_capstyle="butt")
        ar.text(-WIN_S * 1.1, n - 5, "10 neurons", rotation=90, ha="right", va="center", fontsize=13, color=INK)
        ar.set_xlim(-WIN_S, WIN_S)              # the scale bar outside mustn't widen it: "now" lines up with the trace

        ax = fig.add_axes((self.x0, 0.10, 0.66, 0.40), projection="3d")   # x0 < 0.17 when the box leans right
        ax.patch.set_alpha(0)
        lims = [tw.PC_LIMITS[p - 1] for p in self.pcs]
        ax.set_box_aspect([l[1] - l[0] for l in lims], zoom=self.zoom)
        for d, axis in enumerate((ax.xaxis, ax.yaxis, ax.zaxis)):
            axis.set_pane_color((1, 1, 1, 0))
            axis.set_ticks([])
            axis.set_label_text(f"PC{self.pcs[d]}", fontsize=14, color=INK2)
            axis.labelpad = -8
        ax.grid(False)
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        ax.set_zlim(*lims[2])
        ax.view_init(elev=elev, azim=azim)
        self.dots = ax.scatter([], [], [], s=DOT_SIZE, depthshade=False, edgecolors="none")
        self.trail, = ax.plot([], [], [], color=INK, lw=1.8)
        self.head = ax.scatter([], [], [], s=110, depthshade=False, edgecolors=INK, linewidths=1.2)
        # legend slots, filled per frame (see draw): two columns of three
        fig.text(0.19, 0.499, "HMM states: share of this epoch's 5 ms bins so far", fontsize=12, color=INK2, va="top")
        self.leg = []
        for j in range(k):
            x, y = 0.195 + 0.31 * (j % 2), 0.477 - 0.017 * (j // 2)
            mark = fig.add_artist(plt.Line2D([x], [y], marker="o", ms=12, ls="", transform=fig.transFigure))
            self.leg.append((mark, fig.text(x + 0.014, y, "", fontsize=12, va="center")))

        self.label = fig.text(0.5, 0.085, "", ha="center", va="center", fontsize=44, fontweight="bold", color=INK)
        self.speed = fig.text(0.5, 0.04, "", ha="center", va="center", fontsize=26, color=INK2)
        self.fig = fig

    def draw(self, i):
        """Frame i of the playback (not counting the title card); returns it as an RGB array."""
        t, J = self.times[i], self.frame_epoch[i]
        name = self.spans[J][0]
        self.header.set_text(f"e097 · {self.trial} · {t + self.off:.1f} s · {name}")

        a = int(round((t - WIN_S) * self.sr))
        x = np.arange(a, a + int(2 * WIN_S * self.sr))
        ok = (x >= 0) & (x < self.dia.size)
        self.dia_line.set_data((x[ok] / self.sr) - t, self.dia[x[ok]])
        a = int(round((t - WIN_S) / RASTER_BIN_S))
        cols = np.arange(a, a + int(2 * WIN_S / RASTER_BIN_S))
        win = np.zeros((self.R.shape[0], cols.size))
        ok = (cols >= 0) & (cols < self.R.shape[1])
        win[:, ok] = self.R[:, cols[ok]]
        self.img.set_data(win)

        i1 = int(np.searchsorted(self.cen, t, side="right"))
        idx = np.arange(self.first_bin, i1)
        ep = self.bin_epoch[idx]
        keep = (ep >= J - 1) | (idx % GREY_KEEP == 0)
        idx, ep = idx[keep], ep[keep]
        g = np.where(ep < J - 1, 1.0, np.where(ep == J - 1, min(1.0, (i - self.epoch_first[J]) / (GREY_FADE_S * FPS)),
                                               0.0))[:, None]
        rgba = np.column_stack([self.rgb[idx] * (1 - g) + GREY * g, DOT_ALPHA * (1 - g[:, 0]) + GREY_ALPHA * g[:, 0]])
        self.dots._offsets3d = tuple(self.P[idx].T)
        self.dots.set_facecolors(rgba)
        tr = np.arange(max(self.first_bin, i1 - int(TRAIL_S / pdm.PCA_BIN_S)), i1)
        self.trail.set_data_3d(*self.P[tr].T)
        if i1 > self.first_bin:
            self.head._offsets3d = tuple(self.P[[i1 - 1]].T)
            self.head.set_facecolors([self.rgb[i1 - 1]])

        # legend: states seen so far in the movie; coloured if in this epoch so far, paler the rarer, else grey
        seen = self.cum[i1] - self.cum[self.first_bin] > 0
        cnt = self.cum[i1] - self.cum[max(self.epoch_bin0[J], self.first_bin)]
        share = cnt / max(cnt.sum(), 1)
        shown = [s for s in self.state_order if seen[s]]
        for j, (mark, text) in enumerate(self.leg):
            if j >= len(shown):
                mark.set_visible(False)
                text.set_text("")
                continue
            s = shown[j]
            mark.set_visible(True)
            if share[s] > 0:
                w = 0.25 + 0.75 * share[s] / share.max()
                mark.set_color(1 - w * (1 - self.state_rgb[s]))
                pct = "<1" if share[s] < 0.005 else f"{share[s] * 100:.0f}"
                text.set_text(f"{self.state_names[s]}  {pct}%")
                text.set_color(INK)
            else:
                mark.set_color(GREY)
                text.set_text(self.state_names[s])
                text.set_color("#a3a29d")

        dt = (i - self.epoch_first[J]) / FPS
        alpha = np.interp(dt, [0, LABEL_IN_S, LABEL_IN_S + LABEL_HOLD_S, LABEL_IN_S + LABEL_HOLD_S + LABEL_OUT_S],
                          [0, 1, 1, 0])
        self.label.set_text(name[0].upper() + name[1:])
        self.label.set_alpha(alpha)
        v = self.speeds[i]
        self.speed.set_text("" if v == 1 else f"{v:g}× speed")
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def title(self):
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="black")
        fig.text(0.5, 0.53, f"e097 · {self.trial}", ha="center", fontsize=30, color="#bdbdbd")
        fig.text(0.5, 0.47, "Eupnea, anoxia and gasping,\nfollowed by recovery", ha="center", va="top", fontsize=30,
                 color="#bdbdbd", linespacing=1.4)
        fig.add_artist(plt.Line2D([0.3, 0.7], [0.39, 0.39], color="#5a5a5a", lw=1.5))
        fig.canvas.draw()
        im = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return im


def ffmpeg_exe():
    """ffmpeg on PATH, or the one conda installs in this env (on PATH only when the env is activated)."""
    env = Path(sys.prefix)
    for f in (shutil.which("ffmpeg"), env / "Library" / "bin" / "ffmpeg.exe", env / "bin" / "ffmpeg"):
        if f and Path(f).exists():
            return str(f)
    return None


class VideoOut:
    """RGB frames to H.264 through ffmpeg (yuv420p, faststart: plays in PowerPoint and browsers); without ffmpeg,
    OpenCV's MPEG-4 Part 2, about 10x larger and not always playable in PowerPoint."""

    def __init__(self, path):
        ff = ffmpeg_exe()
        self.codec = "H.264" if ff else "MPEG-4 Part 2"
        if ff:
            self.p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", str(CRF),
                                       "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                                      stdin=subprocess.PIPE)
        else:
            print("ffmpeg not found: writing MPEG-4 Part 2 with OpenCV")
            self.p = None
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    def write(self, rgb):
        if self.p:
            self.p.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        else:
            self.cv.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.p:
            self.p.stdin.close()
            if self.p.wait():
                raise RuntimeError("ffmpeg failed")
        else:
            self.cv.release()


def main():
    a = parse_args()
    run = Path(a.run)
    c = tw.load_context(run)
    m = Movie(c, a.pcs, a.elev, a.azim)
    out = run / "transitions" / "movie"
    out.mkdir(parents=True, exist_ok=True)
    tag = m.trial.replace(" #", "").replace(" ", "_") + "_trajectory" + ("" if a.pcs == (1, 2, 3) else f"_PC{a.pcs[2]}-up")
    names = [n for n, _, _ in m.spans]
    if a.preview:
        def at(name, frac):
            _, s, e = m.spans[names.index(name)]
            return min(int(np.searchsorted(m.times, s + frac * (e - s))), len(m.times) - 1)
        frames = [at("anoxia", 0) - FPS // 2, at("anoxia", 0) + FPS, at("pre-gasp", 0.7), at("gasping", 0) + FPS,
                  at("early recovery", 0.5), len(m.times) - 1]          # 2nd and 4th: 1 s in, label showing
        ims = [cv2.resize(m.draw(f), (W // 3, H // 3), interpolation=cv2.INTER_AREA) for f in frames]
        path = out / f"{tag}_preview.png"
        cv2.imwrite(str(path), cv2.cvtColor(np.hstack(ims), cv2.COLOR_RGB2BGR))
        print(path)
        return
    path = out / f"{tag}.mp4"
    vw = VideoOut(path)
    title = m.title()
    n_t = int(TITLE_S * FPS)
    for j in range(n_t):
        f = np.interp(j, [0, 0.2 * n_t, 0.8 * n_t, n_t - 1], [0, 1, 1, 0])
        vw.write((title * f).astype(np.uint8))
    last = None
    for i in range(len(m.times)):
        last = m.draw(i)
        vw.write(last)
        if i % 300 == 0:
            print(f"frame {i} / {len(m.times)}", flush=True)
    for j in range(int(END_S * FPS)):
        vw.write((last * (1 - j / (END_S * FPS))).astype(np.uint8))
    vw.close()
    print(path, f"{(n_t + len(m.times) + END_S * FPS) / FPS:.0f} s, {vw.codec}")


if __name__ == "__main__":
    main()
