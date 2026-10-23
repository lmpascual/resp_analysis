"""Anoxia #1 vs anoxia #2: figures and numbers made the same way for both trials, for side-by-side slides.

For each trial (anoxia #1: population_dynamics_v2; anoxia #2: population_dynamics_anoxia2, decoded with the anoxia #1
model) writes into <run>/figures/:
  airflow_by_condition.png      breath-averaged nasal airflow (XA5) and diaphragm EMG around diaphragm onset, baseline
                                and each recovery epoch
  epoch_metrics.tsv             per recovery epoch: median firing rate vs baseline, median modulation L, inspiratory
                                units, pairwise-correlation structure vs baseline, time in the recovery post-I and late-E
                                states
and for anoxia #2 also the figures anoxia #1 already has:
  signatures/population_peth_ion.png, breath_raster_K6_pm400ms.png, units_slides/unit_<id>_raster.png / _peth.png

  cibrrig python trial_compare.py [--units 331,362,...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import population_dynamics as pdm  # noqa: E402
import transition_windows as tw  # noqa: E402
import figures_population as fp  # noqa: E402
import figures_signatures as fsig  # noqa: E402
from figures_population import INK2, save_fig  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

RUN_A = pdm.OUT / "population_dynamics_v2"
RUN_B = pdm.OUT / "population_dynamics_anoxia2"
CACHE_A = RUN_A / "cache" / "nidq_cache.npz"
CACHE_B = RUN_B / "cache" / "nidq_cache.npz"
PRE_S, POST_S = 0.15, 0.55
FS = pdm.NIDQ_FS
POST_I_STATE, LATE_E_STATE = 2, 1          # K = 6 states of the anoxia #1 model (labels in hmm_states.tsv)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--units", default="331,362,459,312,363,374,341,350,353,391,435,510,422,425,434",
                   help="units (anoxia #1 ids) to make anoxia #2 whole-window unit slides for")
    return p.parse_args()


def recovery_conditions(c):
    gi = c["conditions"].index("gasping")
    return c["conditions"][gi + 1:]


def airflow_emg(cache, sr):
    """Airflow low-passed at 30 Hz and diaphragm EMG (300 Hz high-pass, rectified, 20 ms mean), both at sr."""
    d = np.load(cache)
    step = int(FS / sr)
    flow = sosfiltfilt(butter(4, 30 / (FS / 2), output="sos"), d["flow_v"].astype(np.float64))[::step]
    emg = sosfiltfilt(butter(4, 300 / (FS / 2), btype="high", output="sos"), d["emg_v"].astype(np.float64))
    emg = uniform_filter1d(np.abs(emg), int(0.02 * FS))[::step]
    return flow, emg


def fig_airflow_by_condition(c, cache, trial, path):
    sr = c["sr_sub"]
    flow, emg = airflow_emg(cache, sr)
    ons = c["dia_df"]["on_sec"].to_numpy()
    tt = np.arange(-int(PRE_S * sr), int(POST_S * sr)) / sr * 1e3
    names = ["baseline"] + recovery_conditions(c)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.6), sharex=True)
    rows = []
    for name in names:
        idx = np.flatnonzero(c["cond"] == name)
        i0 = (ons[idx] * sr).astype(int) - int(PRE_S * sr)
        ok = (i0 >= 0) & (i0 + tt.size < flow.size)
        F = np.stack([flow[a:a + tt.size] for a in i0[ok]])
        F -= F[:, (tt >= -150) & (tt < -100)].mean(axis=1, keepdims=True)
        E = np.stack([emg[a:a + tt.size] for a in i0[ok]])
        pre_exp = float(np.mean(F[:, (tt >= -50) & (tt < 0)].mean(axis=1) > 0.2))
        rows.append({"condition": name, "n": int(ok.sum()), "pre_onset_expiratory_flow": pre_exp})
        lw = 2.4 if name == "baseline" else 1.8
        lab = f"{name} ({ok.sum()} breaths; {pre_exp:.0%})"
        axes[0].plot(tt, F.mean(axis=0), color=c["cond_colours"][name], lw=lw, label=lab)
        axes[1].plot(tt, E.mean(axis=0), color=c["cond_colours"][name], lw=lw)
    for ax in axes:
        ax.axvline(0, color=INK2, lw=0.6, ls=":")
    axes[0].set_ylabel("nasal airflow XA5 (V, 30 Hz low-pass)\nnegative = inspiration; zero at −150 to −100 ms")
    axes[1].set_ylabel("diaphragm EMG\n(> 300 Hz, rectified, 20 ms mean, V)")
    axes[1].set_xlabel("ms from diaphragm onset")
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(f"{trial}: breath-averaged airflow and diaphragm EMG, aligned to diaphragm onset\n(% = breaths "
                 "with expiratory flow before onset: mean flow from −50 to 0 ms > 0.2 V)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, path, dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows)


def epoch_metrics(c):
    """The anoxia #1 table (population_dynamics_v2 README, 'Unit tuning' and 'HMM states') for any trial."""
    tu = c["tuning"].set_index(["unit_id", "condition"])
    names = recovery_conditions(c)
    base = tu.xs("baseline", level="condition")
    firing = base.index[base["rate_hz"] >= 0.2]
    rows = {}
    Z, bcond = c["Z"], c["bcond"]
    iu = np.triu_indices(Z.shape[1], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cb = np.corrcoef(Z[bcond == "baseline"].T)[iu]
    post = c["F"][c["ref_k"]]["post"]             # state probabilities, as the anoxia #1 table (occ_ in hmm_states.tsv)
    ons = c["dia_df"]["on_sec"].to_numpy()
    nxt = np.r_[ons[1:], np.inf]

    def n_spikes(name):                            # spikes per unit inside the condition's breath cycles
        i = np.flatnonzero(c["cond"] == name)
        return pd.Series({u: int((np.searchsorted(c["spikes"][u], nxt[i]) - np.searchsorted(c["spikes"][u],
                                                                                          ons[i])).sum())
                          for u in c["ids"]})

    def l_min(name, t):
        ok = n_spikes(name).reindex(t.index) >= 20
        return t.loc[ok, "modulation_L"].median(), int((t.loc[ok, "class"] == "inspiratory").sum())
    for name in names:
        t = tu.xs(name, level="condition")
        ratio = (t.loc[firing, "rate_hz"] / base.loc[firing, "rate_hz"]).median()
        with np.errstate(invalid="ignore", divide="ignore"):
            ce = np.corrcoef(Z[bcond == name].T)[iu]
        ok = np.isfinite(cb) & np.isfinite(ce)
        sel = bcond == name
        lm, im = l_min(name, t)
        rows[name] = {"median firing rate vs baseline": ratio,
                      "breathing modulation (median L)": t["modulation_L"].median(),
                      "inspiratory-locked units": int((t["class"] == "inspiratory").sum()),
                      "correlation structure vs baseline (r)": np.corrcoef(cb[ok], ce[ok])[0, 1],
                      "time in the recovery post-I state": post[sel, POST_I_STATE].mean(),
                      "time in the recovery late-E state": post[sel, LATE_E_STATE].mean(),
                      "median L, units with >= 20 spikes": lm, "inspiratory units, >= 20 spikes": im}
    df = pd.DataFrame(rows)
    sel = bcond == "baseline"
    lm, im = l_min("baseline", base)
    df.insert(0, "baseline", [1.0, base["modulation_L"].median(), int((base["class"] == "inspiratory").sum()), 1.0,
                              post[sel, POST_I_STATE].mean(), post[sel, LATE_E_STATE].mean(), lm, im])
    return df


def fig_peth_compare(cs, path):
    """Both trials, one row pair each: unit PETHs around inspiration onset (each unit scaled to its maximum over that
    trial's conditions; units in anoxia #1 order, a missing unit blank) over the K = 6 state occupancy."""
    from matplotlib.patches import Rectangle
    order = cs[0][1]["order_units"]
    fig = plt.figure(figsize=(12.4, 6.2))
    gs = fig.add_gridspec(5, 7, height_ratios=[2.2, 0.75, 0.55, 2.2, 0.75], hspace=0.3, wspace=0.08, left=0.075,
                          right=0.995, top=0.89, bottom=0.16)
    rows_of = {0: (0, 1), 1: (3, 4)}
    k = cs[0][1]["ref_k"]
    for r, (trial, c) in enumerate(cs):
        conds = c["conditions"]
        data = {u: fsig.peths(c, u, "on") for u in order if u in c["ids"]}
        t = next(iter(data.values()))[0]
        umax = {u: max((v[0].max() for v in d[1].values()), default=1.0) for u, d in data.items()}
        ev = fsig.events(c, "on")
        ok = fsig.valid_events(c, ev)
        dt = c["hmm_bin_s"]
        nb = int(round((fsig.PRE_S + fsig.POST_S) / dt))
        tt = (np.arange(nb) + 0.5) * dt - fsig.PRE_S
        for j, name in enumerate(conds):
            ax = fig.add_subplot(gs[rows_of[r][0], j])
            img = np.full((len(order), t.size), np.nan)
            for y, u in enumerate(order):
                if u in data and name in data[u][1]:
                    img[y] = data[u][1][name][0] / max(umax[u], 1e-9)
            ax.imshow(img, aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest",
                      extent=[t[0] * 1e3, t[-1] * 1e3, len(order), 0])
            ax.axvline(0, color="white", lw=0.6, ls=":")
            ax.set_xticks([])
            ax.set_yticks([])
            n_br = int((ok & (c["cond"] == name)).sum())
            ax.set_title(f"{name}\n({n_br})", fontsize=9.5, pad=8)
            ax.add_patch(Rectangle((0, 1.0), 1, 0.035, transform=ax.transAxes, color=c["cond_colours"][name],
                                   clip_on=False))
            if j == 0:
                ax.set_ylabel(f"{trial}\nunits", fontsize=11)
            ax = fig.add_subplot(gs[rows_of[r][1], j])
            sel = np.flatnonzero(ok & (c["cond"] == name))
            if sel.size:
                S = np.stack([c["F"][k]["seq"][int(e / dt) - int(fsig.PRE_S / dt):int(e / dt) - int(fsig.PRE_S / dt)
                                                + nb] for e in ev[sel]])
                st = np.argsort(c["colours"][k])
                ax.stackplot(tt * 1e3, np.stack([(S == s).mean(axis=0) for s in st]),
                             colors=[fp.state_colour(c, k, s) for s in st], lw=0)
            ax.set_ylim(0, 1)
            ax.set_xlim(-fsig.PRE_S * 1e3, fsig.POST_S * 1e3)
            ax.set_yticks([])
            ax.set_xticks([-400, 0, 400])
            ax.tick_params(labelsize=8, labelbottom=(r == 1))
            if j == 0:
                ax.set_ylabel("HMM\nstate", fontsize=9)
    fig.text(0.53, 0.095, "ms from inspiration onset", ha="center", fontsize=10)
    c = cs[0][1]
    st = np.argsort(c["colours"][k])
    f = c["F"][k]["feats"]
    handles = [Rectangle((0, 0), 1, 1, color=fp.state_colour(c, k, s)) for s in st]
    fig.legend(handles, [ts_state_name(f.loc[s, "label"]) for s in st], loc="lower center", ncol=6, frameon=False,
               fontsize=9, handlelength=1.1, columnspacing=1.2, bbox_to_anchor=(0.53, 0.0))
    save_fig(fig, path, dpi=200)
    plt.close(fig)


def ts_state_name(label):
    import transition_slides as tsl
    return tsl.state_name(label)


def fig_timeline_compare(cs, path, before_s=45.0, after_s=22.0):
    """Both trials side by side on a time axis of seconds from the last gasp: integrated diaphragm, heart rate,
    breathing rate (condition colours), K = 6 state, gasp-likeness."""
    import matplotlib
    fig = plt.figure(figsize=(12.4, 5.6))
    gs = fig.add_gridspec(6, 2, height_ratios=[0.16, 0.8, 0.65, 0.65, 0.2, 0.55], hspace=0.14, wspace=0.07,
                          left=0.085, right=0.995, top=0.93, bottom=0.17)
    for j, (trial, c, T, beats) in enumerate(cs):
        off, sr, dt, k = c["time_offset_s"], c["sr_sub"], c["hmm_bin_s"], c["ref_k"]
        col = {**c["cond_colours"], **tw.TRANSITION_COLOURS}
        on_raw = T["on_raw_s"].to_numpy()
        cond = T["condition"].fillna("").to_numpy()
        last = on_raw[np.flatnonzero(cond == "gasping")[-1]]
        t0, t1 = last - before_s, last + after_s
        rel = lambda x: x - last  # noqa: E731
        axes = [fig.add_subplot(gs[i, j]) for i in range(6)]
        for a in axes[1:]:
            a.sharex(axes[1])
        ax = axes[0]
        idx = np.flatnonzero((on_raw >= t0 - 5) & (on_raw <= t1 + 5))
        for a_, b_ in zip(idx[:-1], idx[1:]):
            name = cond[a_]
            x0, x1 = max(rel(on_raw[a_]), -before_s), min(rel(on_raw[b_]), after_s)
            if x1 > x0:
                ax.axvspan(x0, x1, color=col.get(name, "#ffffff"), lw=0)
        names = cond[idx]
        for nm in dict.fromkeys(n for n in names if n):
            w = idx[names == nm]
            x0 = max(rel(on_raw[w[0]]), -before_s)
            x1 = min(rel(on_raw[min(w[-1] + 1, on_raw.size - 1)]), after_s)
            if x1 - x0 > 2.5:
                ax.text((x0 + x1) / 2, 0.5, nm, ha="center", va="center", fontsize=9,
                        color="white" if nm in ("gasping", "mid recovery", "recovery 3", "later recovery",
                                                "recovery 5", "post-gasp", "pre-gasp") else "#1f1f1f")
        ax.set_ylim(0, 1)
        ax.set_xlim(-before_s, after_s)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(trial, loc="left", fontsize=13, pad=4)
        i0, i1 = int((t0 - off) * sr), int((t1 - off) * sr)
        axes[1].plot(np.arange(i0, i1) / sr + off - last, c["dia_sub"][i0:i1], color="#1f1f1f", lw=0.6)
        axes[1].set_ylim(-0.5, 1.05 * np.nanmax(c["dia_sub"][i0:i1]))
        tb, hr = beats[1:], 60 / np.diff(beats)
        m = (tb >= t0) & (tb <= t1)
        axes[2].plot(tb[m] - last, hr[m], ".", color="#1f1f1f", ms=2)
        axes[2].set_ylim(0, 750)
        sel = idx[(on_raw[idx] >= t0) & (on_raw[idx] <= t1)]
        axes[3].plot(rel(on_raw[sel]), T["rate_per_min"].to_numpy()[sel], color="#d9d8d4", lw=0.8)
        axes[3].scatter(rel(on_raw[sel]), T["rate_per_min"].to_numpy()[sel], s=14, lw=0.3, edgecolor="#52514e",
                        c=[col.get(n, "#ffffff") for n in cond[sel]], zorder=2)
        axes[3].set_ylim(0, 260)
        seq = c["F"][k]["seq"]
        scol = np.array([matplotlib.colors.to_rgb(fp.state_colour(c, k, s)) for s in range(k)])
        b0, b1 = int((t0 - off) / dt), int((t1 - off) / dt)
        axes[4].imshow(scol[seq[b0:b1]][None], aspect="auto", extent=[-before_s, after_s, 0, 1],
                       interpolation="nearest")
        gl = T["gasp_likeness"].to_numpy()
        for i in sel:
            e = on_raw[i + 1] if i + 1 < on_raw.size else on_raw[i]
            axes[5].plot([rel(on_raw[i]), rel(e)], [gl[i]] * 2, color=col.get(cond[i], "#52514e"),
                         lw=2.2)
        axes[5].set_ylim(0.2, 0.85)
        axes[5].axhline(0.5, color="#d9d8d4", lw=0.6)
        axes[5].set_xlim(-before_s, after_s)
        axes[5].set_xlabel("s from the last gasp")
        labels = ["∫ dia", "heart\n(bpm)", "breaths\n(/min)", "HMM", "gasp-\nlike"]
        for a, lab in zip(axes[1:], labels):
            a.axvline(0, color="#52514e", lw=0.7, ls=(0, (4, 3)))
            if a is not axes[5]:
                a.tick_params(labelbottom=False)
            if lab == "HMM":
                a.set_yticks([])
            if j == 0:
                a.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=10)
            else:
                a.tick_params(labelleft=lab not in ("HMM",))
    c = cs[0][1]
    k = c["ref_k"]
    st = np.argsort(c["colours"][k])
    from matplotlib.patches import Rectangle
    handles = [Rectangle((0, 0), 1, 1, color=fp.state_colour(c, k, s)) for s in st]
    fig.legend(handles, [ts_state_name(c["F"][k]["feats"].loc[s, "label"]) for s in st], loc="lower center", ncol=6,
               frameon=False, fontsize=9, handlelength=1.1, columnspacing=1.2)
    save_fig(fig, path, dpi=200)
    plt.close(fig)


def main():
    a = parse_args()
    units = [int(u) for u in a.units.split(",") if u.strip()]
    for run, cache, trial in ((RUN_A, CACHE_A, "anoxia #1"), (RUN_B, CACHE_B, "anoxia #2")):
        c = tw.load_context(run)
        c["out"] = run / "figures"
        c["out"].mkdir(exist_ok=True)
        flows = fig_airflow_by_condition(c, cache, trial, c["out"] / "airflow_by_condition.png")
        m = epoch_metrics(c)
        m.to_csv(c["out"] / "epoch_metrics.tsv", sep="\t", float_format="%.5f")
        print(trial)
        print(m.round(2).to_string())
        print(flows.to_string(index=False))
        if trial == "anoxia #2":
            c["sig_out"] = c["out"] / "signatures"
            c["slide_out"] = c["out"] / "units_slides"
            c["sig_out"].mkdir(exist_ok=True)
            c["slide_out"].mkdir(exist_ok=True)
            fsig.fig_population_peth(c, "on")
            fp.fig_breath_raster_wide(c)
            for u in [u for u in units if u in c["ids"]]:
                fsig.unit_slide_raster(c, u)
                fsig.unit_slide_peth(c, u)
    # comparison figures (both trials in one figure), into the anoxia #2 run
    ctx = [(t, tw.load_context(r)) for r, t in ((RUN_A, "anoxia #1"), (RUN_B, "anoxia #2"))]
    out = RUN_B / "figures" / "compare"
    out.mkdir(exist_ok=True)
    fig_peth_compare(ctx, out / "population_peth_ion_compare.png")
    tl = []
    for (trial, c), run in zip(ctx, (RUN_A, RUN_B)):
        T = pd.read_csv(run / "transitions" / "transition_breaths.tsv", sep="\t")
        beats = pd.read_csv(run / "transitions" / "heartbeats.tsv", sep="\t")["beat_raw_s"].to_numpy()
        tl.append((trial, c, T, beats))
    fig_timeline_compare(tl, out / "timeline_gasp_to_recovery_compare.png")


if __name__ == "__main__":
    main()
