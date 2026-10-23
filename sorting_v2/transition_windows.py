"""Zoom on the two transitions around gasping in e097 anoxia #1: eupnea -> gasping and gasping -> recovery.

population_dynamics.py leaves two runs of breaths without a condition: those after the anoxia window and before the
first Bush-rule gasp, and those after the last gasp and before early recovery. This script labels them pre-gasp and
post-gasp, keeps every other condition (and the HMM fits and PCA) as they are, and writes zoomed figures and tables to
<run>/transitions/. Nothing in <run>/figures is touched.

  transition_timeline.png         both transitions on a time axis: integrated diaphragm, airflow, heart rate, breathing
                                  rate, K = 6 HMM state, spikes of every unit, population rate, gasp-likeness
  breath_metrics.png              per breath from late anoxia to early recovery: rate, Ti, Te, duty cycle, amplitude,
                                  rise time, airflow, heart rate, population rate, gasp-likeness, I-state timing, HMM
                                  state mix; scatter of breath-to-breath timing and of gasp-likeness against rate
  breath_shapes.png               integrated diaphragm and airflow of every breath, aligned to inspiration onset
  pca_filmstrip_eupnea_to_gasp.png, pca_filmstrip_gasp_to_recovery.png
                                  each breath's population trajectory (PC1-PC2) over the baseline and gasp mean cycles
  pca_summary.png                 mean trajectories, speed and distances by phase for each window
  state_sequences.png             K = 6 Viterbi state over each breath's whole cycle, in time and in cycle phase
  state_timing.png                per window: occupancy, dwell times, when each state starts and ends in the cycle
  unit_phase_windows.png          each unit's preferred phase and modulation per window, class counts, phase shifts
  unit_phase_by_breath.png        each unit's mean spike phase on every breath through both transitions
  population_peth_ion.png, population_peth_ioff.png, unit_phase_polar.png
                                  as figures/signatures, with pre-gasp and post-gasp as their own windows
  units_slides/unit_<id>_peth.png    PETH / phase-curve progression over the six windows (as figures/units_slides)
  units_slides/unit_<id>_raster.png  spikes on every breath from late anoxia to early recovery, whole cycles
  qc_heartbeats.png               ECG snippets with the detected R peaks
  transition_breaths.tsv, transition_unit_tuning.tsv, state_runs.tsv, state_timing_summary.tsv, heartbeats.tsv

  cibrrig python transition_windows.py [--run <population_dynamics output>]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import butter, find_peaks, sosfiltfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import population_dynamics as pdm  # noqa: E402
import figures_population as fp  # noqa: E402
import figures_signatures as fsig  # noqa: E402
from figures_population import INK, INK2, GRID, save_fig, state_colour  # noqa: E402

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

RUN = pdm.OUT / "population_dynamics_v2"
NIDQ_BIN = Path(r"Z:/ramirezlab/budde/npx/animal_files/e097_g2/e097_g2_t0.nidq.bin")
NIDQ_CACHE = pdm.OUT / "intact_window" / "intact_nidq_cache.npz"
N_CH, CH_ECG, VOLTS = 9, 2, 10 / 32768
FS = pdm.NIDQ_FS            # the cache (and every breath time) counts NIDQ samples from int(1740 s * 10 kHz) at 10 kHz

ZOOM = ["baseline", "anoxia", "pre-gasp", "gasping", "post-gasp", "early recovery"]
TRANSITION_COLOURS = {"pre-gasp": "#9b3b78", "post-gasp": "#2c8a96"}   # distinct from the conditions and the states
N_CONTEXT = 15              # breaths of anoxia / early recovery either side of the transitions in the breath strips
TRAJ_BINS = 20              # cycle-phase bins per breath (10 inspiration, 10 expiration) for trajectory distances
CYCLE_PRE_S, CYCLE_POST_S = 0.3, 2.8    # whole-cycle windows around inspiration onset (gasp cycles last up to 2.7 s)
MIN_SPIKES = 20             # fewer spikes in a window: no preferred phase / class (a single spike gives L = 1)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
# cyclic phase colours without white (white means no spikes): post-I purple, mid-E blue, onset green, mid-I orange
PHASE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "phase", list(zip([0, 0.25, 0.5, 0.75, 0.875, 1.0],
                      ["#5b3a9b", "#1f8fd0", "#1fa37a", "#e8962e", "#c8384b", "#5b3a9b"])))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", default=str(RUN), help="population_dynamics.py output folder (with cache/ctx.pkl)")
    p.add_argument("--nidq-cache", default=str(NIDQ_CACHE))
    p.add_argument("--nidq-bin", default=str(NIDQ_BIN))
    return p.parse_args()


def load_context(run):
    import __main__
    __main__.RatePoissonHMM = pdm.RatePoissonHMM          # the pickle was written by population_dynamics as __main__
    with open(run / "cache" / "ctx.pkl", "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------------------------------------ windows

def label_transitions(cond):
    """pre-gasp: breaths after the anoxia window and before the first Bush-rule gasp; post-gasp: after the last gasp
    and before early recovery (both without a condition in population_dynamics.py)."""
    cond = np.asarray(cond, dtype=object).copy()
    g = np.flatnonzero(cond == "gasping")
    a = np.flatnonzero(cond == "anoxia")
    r = np.flatnonzero(cond == "early recovery")
    pre = np.arange(a[a < g[0]].max() + 1, g[0])
    post = np.arange(g[-1] + 1, r[r > g[-1]].min())
    assert not any(cond[pre]) and not any(cond[post]), "transition breaths already have a condition"
    cond[pre], cond[post] = "pre-gasp", "post-gasp"
    return cond, pre, post, g


def zoom_context(c, cond, out):
    """A copy of the population context with the six zoom windows as its conditions (phase curves recomputed)."""
    z = dict(c)
    ons = c["dia_df"]["on_sec"].to_numpy()
    cyc = np.searchsorted(ons, c["centres"], side="right") - 1
    valid = (cyc >= 0) & (cyc < ons.size - 1)
    z["cond"] = cond
    z["cyc"] = np.where(valid, cyc, -1)
    z["bcond"] = np.where(valid, cond[np.clip(cyc, 0, ons.size - 1)], "").astype(str)
    z["state_label_conditions"] = c["conditions"]          # the conditions named in the HMM state labels
    z["conditions"] = ZOOM
    z["cond_colours"] = {n: TRANSITION_COLOURS.get(n) or c["cond_colours"][n] for n in ZOOM}
    pdm.CONDITIONS = [(n, None) for n in ZOOM]            # phase_tuning only reads the names
    z["curve_bins"], z["curves"], z["tuning"] = pdm.phase_tuning(c["spikes"], c["ids"], c["dia_df"], cond,
                                                                 c["phi_t"], c["phi"])
    nxt = np.r_[ons[1:], np.inf]
    n_sp = {}
    for name in ZOOM:
        i = np.flatnonzero(cond == name)
        for u in c["ids"]:
            ts = c["spikes"][u]
            n_sp[(u, name)] = int((np.searchsorted(ts, nxt[i]) - np.searchsorted(ts, ons[i])).sum())
    tu = z["tuning"]
    tu["n_spikes"] = [n_sp[(u, n)] for u, n in zip(tu["unit_id"], tu["condition"])]
    tu["class_min_spikes"] = np.where(tu["n_spikes"] >= MIN_SPIKES, tu["class"], f"< {MIN_SPIKES} spikes")
    z["out"] = z["sig_out"] = out
    z["slide_out"] = out / "units_slides"
    z["slide_out"].mkdir(parents=True, exist_ok=True)
    k = c["ref_k"]
    f = c["F"][k]["feats"]
    z["gasp_state"] = int(f["occ_gasping"].idxmax())
    z["i_states"] = np.flatnonzero(f["phase"].to_numpy() > 0)          # early I and late I
    return z


# ------------------------------------------------------------------------------------------- physiology

def heartbeats(n_samples, t0_raw, nidq_bin, cache_dir):
    """R peaks on XA2 (ECG) in file time, and the >5 Hz ECG they were found on.
    - a peak is kept if it reaches 45 % of the 90th-percentile peak within +-2 s (the R wave shrinks in anoxia);
    - an interval < half the local median (T wave, EMG) drops the smaller of its two peaks;
    - an interval 1.6-2.4 x the local median gets the largest deflection near its middle (a missed small R wave).
    A lasting halving of the rate (from ~1824.6 s) is left alone: the local median follows it."""
    f = cache_dir / "ecg_hp.npy"
    if f.exists():
        hp = np.load(f).astype(np.float64)
    else:
        s0 = int(t0_raw * FS)
        raw = np.memmap(nidq_bin, dtype="int16", mode="r")
        ecg = np.asarray(raw[s0 * N_CH:(s0 + n_samples) * N_CH]).reshape(-1, N_CH)[:, CH_ECG].astype(np.float32)
        hp = sosfiltfilt(butter(2, 5 / (FS / 2), btype="high", output="sos"), ecg * VOLTS)
        np.save(f, hp.astype(np.float32))
    cand, _ = find_peaks(hp, distance=int(0.06 * FS), prominence=0.5)
    h, t = hp[cand], cand / FS
    lo, hi = np.searchsorted(t, t - 2.0), np.searchsorted(t, t + 2.0)
    ref = np.array([np.percentile(h[a:b], 90) for a, b in zip(lo, hi)])
    keep = list(cand[h > 0.45 * ref])
    # an extra peak (T wave, EMG) makes an interval < half the local median: drop the smaller of the pair
    changed = True
    while changed:
        changed = False
        k = np.asarray(keep)
        rr = np.diff(k) / FS
        med = pd.Series(rr).rolling(15, center=True, min_periods=3).median().to_numpy()
        bad = np.flatnonzero(rr < 0.5 * med)
        if bad.size:
            i = bad[0]
            keep.pop(i if hp[k[i]] < hp[k[i + 1]] else i + 1)
            changed = True
    k = np.asarray(keep)
    for _ in range(3):
        rr = np.diff(k)
        med = pd.Series(rr).rolling(15, center=True, min_periods=3).median().to_numpy()
        add = []
        for i in np.flatnonzero((rr > 1.6 * med) & (rr < 2.4 * med)):
            a, b = k[i] + int(0.3 * rr[i]), k[i] + int(0.7 * rr[i])
            seg = np.abs(hp[a:b])
            pk, _ = find_peaks(seg, height=0.3 * np.median(np.abs(hp[k[max(i - 5, 0):i + 6]])))
            if pk.size:
                add.append(a + pk[np.argmax(seg[pk])])
        if not add:
            break
        k = np.sort(np.r_[k, add])
    return k / FS, hp


def airflow(cache_path, sr_sub):
    """Airflow (XA5) low-passed at 50 Hz on the 1 kHz grid of the integrated diaphragm, minus a 3 s running median."""
    flow = np.load(cache_path)["flow_v"].astype(np.float64)
    lp = sosfiltfilt(butter(4, 50 / (FS / 2), output="sos"), flow)[:: int(FS / sr_sub)]
    base = median_filter(lp[::10], size=301, mode="nearest")
    return lp - np.interp(np.arange(lp.size), np.arange(base.size) * 10, base)


def inspiration_up(flow, z):
    """Airflow signed so inspiration is positive: at baseline the flow during the diaphragm burst is negative."""
    d, sr = z["dia_df"], z["sr_sub"]
    on, off = (d["on_sec"].to_numpy() * sr).astype(int), (d["off_sec"].to_numpy() * sr).astype(int)
    base = np.flatnonzero(z["cond"] == "baseline")
    return -flow if np.median([flow[on[i]:off[i]].mean() for i in base]) < 0 else flow


# ------------------------------------------------------------------------------------------- per breath

def cycle_trajectories(z, V):
    """Mean of V (bins x dims) per breath and cycle-phase bin (TRAJ_BINS over onset -> next onset): breaths x bins x
    dims, NaN where a bin is empty."""
    cyc, ph = z["cyc"], z["ph"]
    ok = (cyc >= 0) & np.isfinite(ph)
    u = np.mod(ph[ok], 2 * np.pi)                         # 0 -> pi inspiration, pi -> 2 pi expiration
    pb = np.clip((u / (2 * np.pi) * TRAJ_BINS).astype(int), 0, TRAJ_BINS - 1)
    n = len(z["cond"])
    S = np.zeros((n, TRAJ_BINS, V.shape[1]))
    N = np.zeros((n, TRAJ_BINS))
    np.add.at(S, (cyc[ok], pb), V[ok])
    np.add.at(N, (cyc[ok], pb), 1)
    with np.errstate(invalid="ignore"):
        return S / N[..., None]


def template_distance(traj, sel):
    T = np.nanmean(traj[sel], axis=0)
    return np.nanmean(np.linalg.norm(traj - T[None], axis=2), axis=1), T


def breath_table(z, flow, beats):
    d, off_s, dt, k = z["dia_df"], z["time_offset_s"], z["hmm_bin_s"], z["ref_k"]
    cond = z["cond"]
    ons, offs = d["on_sec"].to_numpy(), d["off_sec"].to_numpy()
    nxt = np.r_[ons[1:], np.nan]
    n, sr = ons.size, z["sr_sub"]
    T = pd.DataFrame({"breath": np.arange(n), "condition": cond, "on_raw_s": ons + off_s, "off_raw_s": offs + off_s})
    T["ibi_s"] = nxt - ons
    T["rate_per_min"] = 60 / T["ibi_s"]
    T["ti_ms"] = (offs - ons) * 1e3
    T["te_ms"] = (nxt - offs) * 1e3
    T["duty"] = T["ti_ms"] / 1e3 / T["ibi_s"]
    T["d_ibi_ms"] = T["ibi_s"].diff() * 1e3
    T["amp"], T["auc"] = d["amp"].to_numpy(), d["auc"].to_numpy()
    T["rise_ms"] = (d["pk_time"].to_numpy() - ons) * 1e3
    T["fall_ms"] = (offs - d["pk_time"].to_numpy()) * 1e3

    # airflow (inspiration positive)
    i_on, i_off = (ons * sr).astype(int), (offs * sr).astype(int)
    i_nx = np.r_[i_on[1:], i_on[-1]]
    fl = flow
    cols = {x: np.full(n, np.nan) for x in ("flow_insp_peak", "flow_insp_volume", "flow_exp_peak",
                                            "flow_preI_exp_mean")}
    for i in range(n - 1):
        seg = fl[i_on[i]:i_nx[i]]
        cols["flow_insp_peak"][i] = fl[i_on[i]:i_off[i] + 30].max()
        cols["flow_insp_volume"][i] = np.clip(seg, 0, None).sum() / sr
        cols["flow_exp_peak"][i] = (-seg).max()
        cols["flow_preI_exp_mean"][i] = (-fl[max(i_nx[i] - 100, 0):i_nx[i]]).mean()
    for x, v in cols.items():
        T[x] = v

    # heart rate: mean over the beats that end inside the cycle
    tb, hr = beats[1:], 60 / np.diff(beats)
    lo, hi = np.searchsorted(tb, ons), np.searchsorted(tb, np.nan_to_num(nxt, nan=ons[-1]))
    T["hr_bpm"] = [hr[a:b].mean() if b > a else np.interp(t, tb, hr) for a, b, t in zip(lo, hi, ons)]
    T["n_beats"] = hi - lo

    # population
    counts = sum(np.searchsorted(z["spikes"][u], np.nan_to_num(nxt, nan=ons[-1])) - np.searchsorted(z["spikes"][u], ons)
                 for u in z["ids"])
    T["pop_rate_hz"] = counts / T["ibi_s"] / len(z["ids"])
    seq = z["F"][k]["seq"]
    b0, b1 = (ons / dt).astype(int), (np.nan_to_num(nxt, nan=ons[-1]) / dt).astype(int)
    for s in range(k):
        T[f"state{s}_frac"] = [np.mean(seq[a:b] == s) if b > a else np.nan for a, b in zip(b0, b1)]
    # population I state (early or late I) around the diaphragm burst, and the gasp state after it
    # the block of I-state bins around the diaphragm peak (or nearest to it within the burst), extended back no
    # further than the previous offset
    isI = np.isin(seq, z["i_states"])
    pk = d["pk_time"].to_numpy()
    prev_off = np.r_[0.0, offs[:-1]]
    entry, exit_, gasp_in = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n - 1):
        lo_b, a, b = int(prev_off[i] / dt), int(ons[i] / dt), int(offs[i] / dt) + 1
        c0 = int(pk[i] / dt)
        if not isI[c0]:
            w = a + np.flatnonzero(isI[a:b])
            if not w.size:
                continue
            c0 = w[np.argmin(np.abs(w - c0))]
        s0, e = c0, c0
        while s0 > lo_b and isI[s0 - 1]:
            s0 -= 1
        while e < b1[i] and isI[e]:
            e += 1
        entry[i], exit_[i] = s0 * dt - ons[i], e * dt - offs[i]
        g = np.flatnonzero(seq[e:b1[i]] == z["gasp_state"])
        if g.size:
            gasp_in[i] = (e + g[0]) * dt - offs[i]
    T["I_state_entry_ms_from_onset"] = entry * 1e3
    T["I_state_exit_ms_from_offset"] = exit_ * 1e3
    T["gasp_state_entry_ms_from_offset"] = gasp_in * 1e3

    # distance of each breath's population pattern (all units, sqrt rates) from the baseline and gasp mean cycles
    traj = cycle_trajectories(z, z["Z"])
    for name, col in (("baseline", "d_baseline"), ("gasping", "d_gasping"), ("early recovery", "d_early_recovery")):
        T[col], _ = template_distance(traj, cond == name)
    T["gasp_likeness"] = T["d_baseline"] / (T["d_baseline"] + T["d_gasping"])
    # Bush & Ramirez PCA: path length and distance to the baseline inspiration-off point
    P3, dt_p = z["P"][:, :3], z["pca_bin_s"]
    step = np.r_[0, np.linalg.norm(np.diff(P3, axis=0), axis=1)]
    pb0, pb1 = (ons / dt_p).astype(int), (np.nan_to_num(nxt, nan=ons[-1]) / dt_p).astype(int)
    poff = np.clip((offs / dt_p).astype(int) + 1, 0, P3.shape[0] - 1)
    T["pca_path_length"] = [step[a + 1:b].sum() for a, b in zip(pb0, pb1)]
    T["pca_d_ioff_at_offset"] = np.linalg.norm(P3[poff] - z["ioff_point"], axis=1)
    T["pc2_cycle_mean"] = [z["P"][a:b, 1].mean() if b > a else np.nan for a, b in zip(pb0, pb1)]
    T.loc[n - 1, T.columns[5:]] = np.nan
    return T


def state_runs(z):
    """Every run of the reference model's Viterbi path, clipped to the breath cycles it overlaps: start / end relative
    to that breath's inspiration onset and offset, and the full run duration (dwell)."""
    seq, dt = z["F"][z["ref_k"]]["seq"], z["hmm_bin_s"]
    d = z["dia_df"]
    ons, offs = d["on_sec"].to_numpy(), d["off_sec"].to_numpy()
    change = np.flatnonzero(np.diff(seq)) + 1
    starts, ends = np.r_[0, change], np.r_[change, seq.size]
    rows = []
    for i in range(ons.size - 1):
        a, b = int(ons[i] / dt), int(ons[i + 1] / dt)
        r0, r1 = np.searchsorted(ends, a, side="right"), np.searchsorted(starts, b)
        for r in range(r0, r1):
            s, e = max(starts[r], a), min(ends[r], b)
            rows.append({"breath": i, "condition": z["cond"][i], "state": int(seq[starts[r]]),
                         "start_ms_from_onset": (s * dt - ons[i]) * 1e3, "end_ms_from_onset": (e * dt - ons[i]) * 1e3,
                         "start_ms_from_offset": (s * dt - offs[i]) * 1e3, "in_cycle_ms": (e - s) * dt * 1e3,
                         "dwell_ms": (ends[r] - starts[r]) * dt * 1e3, "starts_in_cycle": starts[r] >= a})
    return pd.DataFrame(rows)


def timing_summary(z, runs):
    rows = []
    k = z["ref_k"]
    labels = z["F"][k]["feats"]["label"]
    for name in ZOOM:
        R = runs[runs["condition"] == name]
        nb = R["breath"].nunique()
        tot = R["in_cycle_ms"].sum()
        for s in range(k):
            Rs = R[R["state"] == s]
            longest = Rs.loc[Rs.groupby("breath")["in_cycle_ms"].idxmax()] if len(Rs) else Rs
            dw = Rs.loc[Rs["starts_in_cycle"], "dwell_ms"]
            rows.append({"condition": name, "state": s, "label": labels[s], "n_breaths": nb,
                         "occupancy": Rs["in_cycle_ms"].sum() / tot if tot else np.nan,
                         "frac_breaths_with_state": Rs["breath"].nunique() / nb if nb else np.nan,
                         "median_dwell_ms": dw.median(), "n_runs": int(len(dw)),
                         "median_start_ms_from_onset": longest["start_ms_from_onset"].median(),
                         "median_end_ms_from_onset": longest["end_ms_from_onset"].median(),
                         "q25_start_ms": longest["start_ms_from_onset"].quantile(0.25),
                         "q75_start_ms": longest["start_ms_from_onset"].quantile(0.75),
                         "q25_end_ms": longest["end_ms_from_onset"].quantile(0.25),
                         "q75_end_ms": longest["end_ms_from_onset"].quantile(0.75),
                         "median_start_ms_from_offset": longest["start_ms_from_offset"].median()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- helpers

def trial_prefix(z):
    """'anoxia #2: ' for a trial other than anoxia #1 (whose figures carry no prefix)."""
    trial = z.get("trial", "anoxia #1")
    return "" if trial == "anoxia #1" else f"{trial}: "


def blocks(cond, idx):
    """Runs of one condition among the breaths idx: (name, first position, end position)."""
    c = cond[idx]
    change = np.flatnonzero(c[1:] != c[:-1]) + 1
    return [(c[a], a, b) for a, b in zip(np.r_[0, change], np.r_[change, c.size])]


def condition_strip(ax, z, idx, x=None, label=True, fontsize=8.5, lim=None, shift=0, counts=True):
    """Coloured blocks for the conditions of breaths idx, on breath positions + shift (x None) or on onset times x;
    labels centred on the part of each block inside lim."""
    for name, a, b in blocks(z["cond"], idx):
        x0 = a - 0.5 + shift if x is None else x[a]
        x1 = b - 0.5 + shift if x is None else x[b] if b < len(x) else x[-1]
        ax.add_patch(Rectangle((x0, 0), x1 - x0, 1, color=z["cond_colours"].get(name, "white"), lw=0))
        v0, v1 = (x0, x1) if lim is None else (max(x0, lim[0]), min(x1, lim[1]))
        if label and v1 > v0:
            ax.text((v0 + v1) / 2, 1.15, f"{name} ({b - a})" if counts else name, ha="center", va="bottom", fontsize=fontsize, color=INK,
                    clip_on=False)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0, labelbottom=False)


def side_ranges(pre, post, g):
    """Breaths shown for each transition: anoxia / pre-gasp / first gasps and last gasps / post-gasp / early
    recovery."""
    return {"eupnea → gasping": np.arange(pre[0] - 8, g[0] + 6),
            "gasping → recovery": np.arange(g[-1] - 5, post[-1] + 9)}


def cond_handles(z, names=ZOOM):
    return [Rectangle((0, 0), 1, 1, color=z["cond_colours"][n]) for n in names]


# --------------------------------------------------------------------------------------------- figures

def fig_timeline(z, T, flow, beats, pre, post, g, out):
    off = z["time_offset_s"]
    on_raw = T["on_raw_s"].to_numpy()
    wins = [("eupnea → gasping", on_raw[pre[0]] - 10, on_raw[g[0]] + 10, [on_raw[pre[0]], on_raw[g[0]]]),
            ("gasping → recovery", on_raw[g[-1]] - 10, on_raw[post[-1] + 1] + 10,
             [on_raw[post[0]], on_raw[post[-1] + 1]])]
    rows = ["cond", "dia", "flow", "hr", "rate", "state", "raster", "pop", "gl"]
    hr_ = [0.22, 0.9, 0.9, 0.75, 0.75, 0.3, 2.8, 0.7, 0.7]
    fig = plt.figure(figsize=(17, 14.5))
    gs = fig.add_gridspec(len(rows), 2, height_ratios=hr_, hspace=0.18, wspace=0.08, left=0.065, right=0.99,
                          top=0.9, bottom=0.085)
    sr, dt, k = z["sr_sub"], z["hmm_bin_s"], z["ref_k"]
    seq = z["F"][k]["seq"]
    scol = np.array([matplotlib.colors.to_rgb(state_colour(z, k, s)) for s in range(k)])
    units = z["order_units"]
    tb, hr = beats[1:] + off, 60 / np.diff(beats)
    for j, (title, t0, t1, marks) in enumerate(wins):
        axes = {}
        for r, name in enumerate(rows):
            axes[name] = fig.add_subplot(gs[r, j], sharex=axes.get("dia"))
        idx = np.flatnonzero((on_raw >= t0 - 3) & (on_raw <= t1 + 3))
        condition_strip(axes["cond"], z, idx, x=on_raw[idx], lim=(t0, t1), counts=False)
        axes["cond"].set_xlim(t0, t1)
        axes["cond"].set_title(title, fontsize=12, loc="left", pad=16)
        i0, i1 = int((t0 - off) * sr), int((t1 - off) * sr)
        tt = np.arange(i0, i1) / sr + off
        axes["dia"].plot(tt, z["dia_sub"][i0:i1], color=INK, lw=0.6)
        axes["flow"].plot(tt, flow[i0:i1], color=INK, lw=0.6)
        m = (tb >= t0) & (tb <= t1)
        axes["hr"].plot(tb[m], hr[m], ".", color=INK, ms=2)
        sel = idx[(on_raw[idx] >= t0) & (on_raw[idx] <= t1)]
        cols = [z["cond_colours"].get(c_, "white") for c_ in z["cond"][sel]]
        axes["rate"].plot(on_raw[sel], T["rate_per_min"].to_numpy()[sel], color=GRID, lw=0.8, zorder=1)
        axes["rate"].scatter(on_raw[sel], T["rate_per_min"].to_numpy()[sel], c=cols, s=18, zorder=2, lw=0.3,
                             edgecolor=INK2)
        b0, b1 = int((t0 - off) / dt), int((t1 - off) / dt)
        axes["state"].imshow(scol[seq[b0:b1]][None], aspect="auto", extent=[t0, t1, 0, 1], interpolation="nearest")
        ev = [z["spikes"][u][(z["spikes"][u] >= t0 - off) & (z["spikes"][u] < t1 - off)] + off for u in units]
        axes["raster"].eventplot(ev, lineoffsets=np.arange(len(units)), linelengths=0.8, linewidths=0.6, colors=INK)
        axes["raster"].set_ylim(len(units) - 0.5, -0.5)
        cnt = sum(np.histogram(e, bins=np.arange(t0, t1 + 0.01, 0.01))[0] for e in ev)
        axes["pop"].plot(np.arange(t0, t1, 0.01)[:cnt.size] + 0.005,
                         gaussian_filter1d(cnt / 0.01 / len(units), 3), color=INK, lw=0.8)
        axes["pop"].set_ylim(0, None)
        gl = T["gasp_likeness"].to_numpy()
        for i in sel:
            e = on_raw[i + 1] if i + 1 < on_raw.size else on_raw[i]
            axes["gl"].plot([on_raw[i], e], [gl[i], gl[i]], color=z["cond_colours"].get(z["cond"][i], INK2), lw=2.2)
        axes["gl"].set_ylim(0.2, 0.8)
        axes["gl"].axhline(0.5, color=GRID, lw=0.6)
        for name in rows[1:]:
            ax = axes[name]
            for x in marks:
                ax.axvline(x, color=INK2, lw=0.7, ls=(0, (4, 3)))
            if name != "gl":
                ax.tick_params(labelbottom=False)
            if name in ("state",):
                ax.set_yticks([])
        axes["gl"].set_xlim(t0, t1)
        axes["gl"].set_xlabel("raw-run time (s)")
        if j == 0:
            labels = {"dia": "∫ diaphragm\n(z)", "flow": "airflow\n(insp. up, V)", "hr": "heart rate\n(bpm)",
                      "rate": "breathing\nrate (/min)", "state": f"HMM\nK = {k}",
                      "raster": "units (baseline φ order,\ntonic last)", "pop": "population\nrate (Hz/unit)",
                      "gl": "gasp-likeness\n(per breath)"}
            for name, lab in labels.items():
                axes[name].set_ylabel(lab, fontsize=9)
            axes["raster"].set_yticks(np.arange(len(units)))
            axes["raster"].set_yticklabels([f"#{u}" for u in units], fontsize=5.5)
        else:
            axes["raster"].set_yticks([])
    fp.state_legend(fig, z, k, y=0.0, ncol=3)
    fig.suptitle(f"The two transitions around gasping ({z.get('trial', 'anoxia #1')}). Dashed lines: first pre-gasp breath, first gasp "
                 "(left); first post-gasp breath, first early-recovery breath (right).\nBreathing-rate dots and "
                 "gasp-likeness bars in condition colours; the HMM strip in state colours (legend). Heart rate: "
                 "every beat (60 / R-R).\nGasp-likeness: distance of the breath's population pattern (all units, by "
                 "cycle phase) to the baseline mean cycle / (that + distance to the gasp mean cycle); 0.5 = halfway, "
                 "higher = more gasp-like.", x=0.01, ha="left", fontsize=10)
    save_fig(fig, out / "transition_timeline.png", dpi=130)
    plt.close(fig)


STRIP_ROWS = [("rate_per_min", "rate (/min)"), ("ti_ms", "Ti (ms)"), ("te_ms", "Te (ms)"),
              ("duty", "Ti / Ttot"), ("amp", "∫dia peak (z)"), ("rise_ms", "onset → peak\n(ms)"),
              ("flow_insp_peak", "peak insp.\nflow (V)"), ("flow_insp_volume", "insp. volume\n(V·s)"),
              ("hr_bpm", "heart rate\n(bpm)"), ("pop_rate_hz", "population\nrate (Hz/unit)"),
              ("gasp_likeness", "gasp-likeness"), ("pc2_cycle_mean", "PC2 level\n(cycle mean)"),
              ("I_state_entry_ms_from_onset", "I-state entry −\ndia onset (ms)"),
              ("I_state_exit_ms_from_offset", "I-state exit −\ndia offset (ms)")]


def fig_breath_metrics(z, T, pre, post, g, out):
    idx = np.arange(pre[0] - N_CONTEXT, post[-1] + 1 + N_CONTEXT)
    x = idx - pre[0]
    k = z["ref_k"]
    nrow = len(STRIP_ROWS) + 2
    fig = plt.figure(figsize=(17, 2.6 + 1.05 * nrow))
    gs = fig.add_gridspec(nrow, 1, height_ratios=[0.25] + [1] * len(STRIP_ROWS) + [1.1], hspace=0.22, left=0.07,
                          right=0.72, top=0.95, bottom=0.07)
    cols = np.array([z["cond_colours"].get(c_, "white") for c_ in z["cond"][idx]])
    ax0 = fig.add_subplot(gs[0, 0])
    condition_strip(ax0, z, idx, shift=-N_CONTEXT)
    ax0.set_xlim(-N_CONTEXT - 0.5, x[-1] + 0.5)
    base = T[T["condition"] == "baseline"]
    axes = []
    for r, (col, lab) in enumerate(STRIP_ROWS):
        ax = fig.add_subplot(gs[r + 1, 0], sharex=ax0)
        y = T[col].to_numpy()[idx]
        lo, hi = np.nanpercentile(base[col], [10, 90])
        ax.axhspan(lo, hi, color=GRID, alpha=0.5, lw=0)
        ax.plot(x, y, color=INK2, lw=0.6, zorder=1)
        ax.scatter(x, y, c=cols, s=20, lw=0.3, edgecolor=INK2, zorder=2)
        if col == "gasp_likeness":
            ax.axhline(0.5, color=INK2, lw=0.6, ls=":")
        if col in ("te_ms", "ti_ms"):
            ax.set_yscale("log")
            ax.set_ylim((60, 250) if col == "ti_ms" else (80, 4000))
        if col.startswith("I_state"):
            ax.axhline(0, color=INK2, lw=0.6, ls=":")
        ax.set_ylabel(lab, fontsize=8.5)
        ax.tick_params(labelbottom=False)
        axes.append(ax)
    ax = fig.add_subplot(gs[-1, 0], sharex=ax0)
    order = np.argsort(z["colours"][k])
    bottom = np.zeros(idx.size)
    for s in order:
        v = np.nan_to_num(T[f"state{s}_frac"].to_numpy()[idx])
        ax.bar(x, v, bottom=bottom, width=0.9, color=state_colour(z, k, s), lw=0)
        bottom += v
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"HMM state mix\n(K = {k}, per cycle)", fontsize=8.5)
    ax.set_xlabel(f"breath (0 = first pre-gasp breath, {T['on_raw_s'][pre[0]]:.1f} s raw); grey band = baseline "
                  "10th–90th percentile")
    for a in axes + [ax]:
        for b_ in (0, g[0] - pre[0], post[0] - pre[0], post[-1] + 1 - pre[0]):
            a.axvline(b_ - 0.5, color=INK2, lw=0.6, ls=(0, (4, 3)))
    # right column: breath-to-breath timing, and population against breathing
    right = fig.add_gridspec(4, 1, left=0.785, right=0.99, top=0.93, bottom=0.06, hspace=0.45)
    gs_r = [right[i] for i in range(4)]
    panels = [("ibi_s", "ibi_next", "IBI n (s)", "IBI n+1 (s)", True),
              ("ti_ms", "te_ms", "Ti (ms)", "Te (ms)", True),
              ("rate_per_min", "gasp_likeness", "breathing rate (/min)", "gasp-likeness", False),
              ("rate_per_min", "hr_bpm", "breathing rate (/min)", "heart rate (bpm)", False)]
    T = T.assign(ibi_next=T["ibi_s"].shift(-1))
    for spec, (xc, yc, xl, yl, logs) in zip(gs_r, panels):
        a = fig.add_subplot(spec)
        for name in ZOOM:
            s_ = T[(T["condition"] == name)]
            a.scatter(s_[xc], s_[yc], s=10 if name in ("baseline", "anoxia", "early recovery") else 22,
                      color=z["cond_colours"][name], lw=0.3, edgecolor=INK2 if "gasp" in name else "none",
                      alpha=0.6 if name in ("baseline", "anoxia", "early recovery") else 1.0, label=name)
        for name in ("pre-gasp", "post-gasp"):
            s_ = T[T["condition"] == name]
            a.plot(s_[xc], s_[yc], color=z["cond_colours"][name], lw=0.7)
        if logs:
            a.set_xscale("log")
            a.set_yscale("log")
        a.set_xlabel(xl)
        a.set_ylabel(yl)
    fig.axes[-4].legend(frameon=False, fontsize=7.5, loc="upper left", markerscale=1.3)
    fig.suptitle(trial_prefix(z) + "Breath by breath from late anoxia to early recovery (dashed: first pre-gasp breath, first gasp, "
                 "first post-gasp breath, first early-recovery breath). Dots in condition colours; the bottom row "
                 "in HMM state colours (legend).", x=0.01, ha="left", fontsize=10.5)
    fp.state_legend(fig, z, k, y=0.0, ncol=3)
    save_fig(fig, out / "breath_metrics.png", dpi=130)
    plt.close(fig)


def fig_breath_shapes(z, T, flow, pre, post, out):
    sr = z["sr_sub"]
    pre_s, post_s = 0.1, 0.6
    tt = np.arange(-int(pre_s * sr), int(post_s * sr)) / sr * 1e3
    ons = z["dia_df"]["on_sec"].to_numpy()
    rows = [("dia", "∫ diaphragm (z)"), ("dia_norm", "∫ diaphragm / breath peak"),
            ("flow", "airflow, XA5\n(inspiration up, V)")]
    fig, axes = plt.subplots(len(rows), len(ZOOM) + 1, figsize=(19, 9.5), sharex=True, sharey="row")
    rng = np.random.default_rng(0)
    means = {}
    for j, name in enumerate(ZOOM):
        idx = np.flatnonzero(z["cond"] == name)
        idx = idx[(ons[idx] - pre_s > 0) & (ons[idx] + post_s < z["dia_sub"].size / sr)]
        seq_cols = None
        if name in TRANSITION_COLOURS:
            base = np.array(matplotlib.colors.to_rgb(z["cond_colours"][name]))
            seq_cols = [1 - (0.25 + 0.75 * f) * (1 - base) for f in np.linspace(0, 1, idx.size)]
        shown = idx if idx.size <= 40 else np.sort(rng.choice(idx, 40, replace=False))
        segs = {}
        for key, sig in (("dia", z["dia_sub"]), ("flow", flow)):
            i0 = (ons[idx] * sr).astype(int) - int(pre_s * sr)
            segs[key] = np.stack([sig[a:a + tt.size] for a in i0])
        segs["dia_norm"] = segs["dia"] / segs["dia"].max(axis=1, keepdims=True)
        for r, (key, lab) in enumerate(rows):
            ax = axes[r, j]
            for n_, i in enumerate(idx):
                if i not in shown:
                    continue
                col = seq_cols[n_] if seq_cols else z["cond_colours"][name]
                ax.plot(tt, segs[key][n_], color=col, lw=0.9 if seq_cols else 0.5,
                        alpha=1.0 if seq_cols else 0.35)
            m = segs[key].mean(axis=0)
            means[(name, key)] = m
            ax.plot(tt, m, color=INK, lw=1.6)
            ax.axvline(0, color=INK2, lw=0.6, ls=":")
            if r == 0:
                extra = " (light → dark = first → last)" if seq_cols else ""
                ax.set_title(f"{name}\n{idx.size} breaths{extra}", fontsize=9)
            if j == 0:
                ax.set_ylabel(lab)
        mti = np.nanmedian(T.loc[idx, "ti_ms"])
        for r in range(len(rows)):
            axes[r, j].axvline(mti, color=INK2, lw=0.6, ls=(0, (4, 3)))
    for r, (key, lab) in enumerate(rows):
        ax = axes[r, -1]
        for name in ZOOM:
            ax.plot(tt, means[(name, key)], color=z["cond_colours"][name], lw=1.6)
        ax.axvline(0, color=INK2, lw=0.6, ls=":")
        if r == 0:
            ax.set_title("means", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("ms from inspiration onset")
    fig.legend(cond_handles(z), ZOOM, loc="lower center", ncol=len(ZOOM), frameon=False)
    fig.suptitle(trial_prefix(z) + "Breath shapes aligned to inspiration onset: individual breaths (up to 40 per window; the transition "
                 "windows show every breath, shaded first → last), black = mean, dashed = median Ti",
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_fig(fig, out / "breath_shapes.png", dpi=120)
    plt.close(fig)


def mean_cycle(z, name, dims=(0, 1, 2), nbins=50):
    x, m = fp.phase_binned(z["P"][:, list(dims)], z["ph"], z["bcond"] == name, nbins=nbins)
    return np.vstack([m, m[:1]])


def fig_pca_filmstrip(z, T, title, idx, fname, out):
    P, dt = z["P"], z["pca_bin_s"]
    ons, offs = z["dia_df"]["on_sec"].to_numpy(), z["dia_df"]["off_sec"].to_numpy()
    dims = [0, 2]                  # the baseline cycle rotates in PC1-PC3; PC2 mostly separates network states
    mb, mg = mean_cycle(z, "baseline", dims), mean_cycle(z, "gasping", dims)
    ncol = 6
    nrow = int(np.ceil(idx.size / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.55 * ncol, 2.5 * nrow + 0.9), sharex=True, sharey=True)
    seg_all = np.concatenate([P[int(ons[i] / dt):int(ons[i + 1] / dt) + 1][:, dims] for i in idx])
    lim = np.percentile(seg_all, [0.5, 99.5], axis=0)
    pad = 0.08 * (lim[1] - lim[0])
    for ax, i in zip(axes.ravel(), idx):
        ax.plot(mb[:, 0], mb[:, 1], color=z["cond_colours"]["baseline"], lw=1.0, ls=(0, (3, 2)))
        ax.plot(mg[:, 0], mg[:, 1], color=z["cond_colours"]["gasping"], lw=1.0, ls=(0, (3, 2)))
        seg = P[int(ons[i] / dt):int(ons[i + 1] / dt) + 1][:, dims]
        col = z["cond_colours"].get(z["cond"][i], INK2)
        ax.plot(seg[:, 0], seg[:, 1], color=col, lw=1.6)
        ax.plot(*seg[0], "o", color=col, mec=INK, ms=5, mew=0.6)
        o = int(offs[i] / dt) + 1 - int(ons[i] / dt)
        if 0 <= o < len(seg):
            ax.plot(*seg[o], "^", color=col, mec=INK, ms=5.5, mew=0.6)
        ax.set_title(f"{i - idx[0] + 1}. {z['cond'][i]}\nIBI {T['ibi_s'][i]:.2f} s · g {T['gasp_likeness'][i]:.2f}",
                     fontsize=8)
        ax.set_xlim(lim[0, 0] - pad[0], lim[1, 0] + pad[0])
        ax.set_ylim(lim[0, 1] - pad[1], lim[1, 1] + pad[1])
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[idx.size:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("PC1")
    for ax in axes[:, 0]:
        ax.set_ylabel("PC3")
    handles = [plt.Line2D([], [], color=z["cond_colours"]["baseline"], ls=(0, (3, 2))),
               plt.Line2D([], [], color=z["cond_colours"]["gasping"], ls=(0, (3, 2))),
               plt.Line2D([], [], marker="o", ls="", color=INK2, mec=INK), plt.Line2D([], [], marker="^", ls="",
                                                                                     color=INK2, mec=INK)]
    fig.legend(handles + cond_handles(z), ["baseline mean cycle", "gasp mean cycle", "inspiration onset",
                                           "inspiration offset"] + ZOOM, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(trial_prefix(z) + f"{title}: each breath's population trajectory in PC1–PC3, the plane where the baseline cycle "
                 "rotates (PCA fit on {z.get('pca_fit', 'baseline')}; onset to next onset), in recording order; g = gasp-likeness", x=0.01,
                 ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    save_fig(fig, out / fname, dpi=120)
    plt.close(fig)


def fig_pca_summary(z, out, limits=None):
    """limits: [(xlim, ylim)] for the six panels (e.g. shared across trials, see pca_summary_shared); returns the
    limits used."""
    P, ph, bcond = z["P"], z["ph"], z["bcond"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for ax, (a, b) in zip(axes[0], ((0, 1), (0, 2), (1, 2))):
        for name in ZOOM:
            m = mean_cycle(z, name, (a, b))
            ax.plot(m[:, 0], m[:, 1], color=z["cond_colours"][name], lw=2.0 if name in TRANSITION_COLOURS else 1.4)
            ax.plot(*m[0], "o", color=z["cond_colours"][name], mec=INK, ms=5, mew=0.6)
        ax.plot(z["ioff_point"][a], z["ioff_point"][b], "X", color=INK, ms=9)
        ax.set_xlabel(f"PC{a + 1}")
        ax.set_ylabel(f"PC{b + 1}")
    axes[0, 0].set_title("mean cycle per window (● onset, X baseline inspiration-off point)", loc="left")
    xb, mb = fp.phase_binned(P[:, :3], ph, bcond == "baseline")
    edges = np.linspace(-np.pi, np.pi, mb.shape[0] + 1)
    bb = np.clip(np.digitize(ph, edges) - 1, 0, mb.shape[0] - 1)
    d_mean = np.linalg.norm(P[:, :3] - mb[bb], axis=1)
    d_ioff = np.linalg.norm(P[:, :3] - z["ioff_point"], axis=1)
    for ax, (v, lab) in zip(axes[1], ((z["speed"], "speed in PC1–3 (a.u./s), median"),
                                      (d_mean, "distance to baseline mean cycle, median"),
                                      (d_ioff, "distance to baseline insp.-off point, median"))):
        for name in ZOOM:
            x, m = fp.phase_binned(v, ph, bcond == name, stat="median")
            ax.plot(x, m, color=z["cond_colours"][name], lw=2.0 if name in TRANSITION_COLOURS else 1.4)
        fp.phase_axis(ax)
        ax.set_ylabel(lab)
    if limits is not None:
        for ax, (xl, yl) in zip(axes.ravel(), limits):
            ax.set_xlim(xl)
            ax.set_ylim(yl)
    used = [(ax.get_xlim(), ax.get_ylim()) for ax in axes.ravel()]
    fig.legend(cond_handles(z), ZOOM, loc="lower center", ncol=len(ZOOM), frameon=False)
    fig.suptitle(trial_prefix(z) + f"Population trajectories (Bush & Ramirez 2024 PCA, fit on {z.get('pca_fit', 'baseline')}) in the "
                 "six windows; the two "
                 "transition windows drawn thicker", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    save_fig(fig, out / "pca_summary.png", dpi=120)
    plt.close(fig)
    return used


EXAMPLE_S = {"pre-gasp": (0.3, 2.3), "gasping": (1.0, 1.6), "post-gasp": (0.15, 1.45)}   # (before, after onset)


def epoch_order(c, cond):
    """Conditions in time order with the transition windows in place: ..., pre-gasp, gasping, post-gasp, ..."""
    gi = c["conditions"].index("gasping")
    order = c["conditions"][:gi] + ["pre-gasp", "gasping", "post-gasp"] + c["conditions"][gi + 1:]
    return [n for n in order if (cond == n).any()]


def bin_conditions(c, cond):
    ons = c["dia_df"]["on_sec"].to_numpy()
    cyc = np.searchsorted(ons, c["centres"], side="right") - 1
    valid = (cyc >= 0) & (cyc < ons.size - 1)
    return np.where(valid, cond[np.clip(cyc, 0, ons.size - 1)], "").astype(str)


def state_rates(c, cond, rows_order):
    """Per epoch: (fraction of time in each state, units x states firing rate in Hz); units missing from the trial are
    NaN."""
    k, dt = c["ref_k"], c["hmm_bin_s"]
    seq = c["F"][k]["seq"]
    bcond = bin_conditions(c, cond)
    nbin = seq.size
    counts = np.full((nbin, len(rows_order)), np.nan)
    for j, u in enumerate(rows_order):
        if u in c["spikes"]:
            counts[:, j] = np.bincount(np.clip(np.floor(c["spikes"][u] / dt).astype(int), 0, nbin - 1),
                                       minlength=nbin)
    out = {}
    for name in epoch_order(c, cond):
        m = bcond == name
        occ = np.array([(seq[m] == s).mean() for s in range(k)])
        R = np.full((len(rows_order), k), np.nan)
        for s in range(k):
            ms = m & (seq == s)
            if ms.any():
                R[:, s] = counts[ms].sum(axis=0) / (ms.sum() * dt)
        out[name] = (occ, R)
    return out


def example_breath(c, cond, name):
    """The epoch's most typical breath (state mix closest to the epoch's mean, among its middle half of breaths), as
    figures_signatures.example_window; returns its onset (file s)."""
    k, dt = c["ref_k"], c["hmm_bin_s"]
    seq = c["F"][k]["seq"]
    ons = c["dia_df"]["on_sec"].to_numpy()
    idx = np.flatnonzero(cond == name)
    mix = np.stack([np.bincount(seq[int(ons[i] / dt):int(ons[i + 1] / dt)], minlength=k) for i in idx]).astype(float)
    mix /= np.maximum(mix.sum(axis=1, keepdims=True), 1)
    dist = np.abs(mix - mix.mean(axis=0)).sum(axis=1)
    middle = np.arange(len(idx) // 4, max(len(idx) // 4 + 1, 3 * len(idx) // 4))
    return ons[idx[middle[int(np.argmin(dist[middle]))]]]


def fig_state_examples_epochs(c, cond, out, rows_order, state_cols, xmax_hz, min_occ=0.02):
    """As figures_signatures.fig_state_progression (Jones et al. 2007 Fig. 2), K = 6 only, one row per epoch in time
    order (transition windows included), each marked by its colour, number, raw span and breath count. Left: an example
    stretch, spikes of every unit over the state probabilities. Right: each unit's firing rate in each state during
    the epoch, one fixed column per state (breath-cycle order) so a state sits in the same column in every row and in
    both trials; states under min_occ of the epoch are left empty. rows_order fixes the unit rows (a unit missing from
    this trial stays as an empty row); xmax_hz fixes the rate axis."""
    import matplotlib
    from matplotlib.patches import Rectangle
    k, dt, off = c["ref_k"], c["hmm_bin_s"], c["time_offset_s"]
    F = c["F"][k]
    order = epoch_order(c, cond)
    rates = state_rates(c, cond, rows_order)
    colours = {**c["cond_colours"], **TRANSITION_COLOURS}
    ons = c["dia_df"]["on_sec"].to_numpy()
    nu, ns = len(rows_order), len(state_cols)
    labels = F["feats"]["label"]
    import transition_slides as tsl
    H = 3.55 * len(order) + 1.9
    fig = plt.figure(figsize=(7.2 + 1.55 * ns, H))
    outer = fig.add_gridspec(len(order), 2, width_ratios=[7.0, 1.55 * ns], hspace=0.55, wspace=0.06,
                             top=1 - 1.55 / H, bottom=0.025, left=0.05, right=0.99)
    for r, name in enumerate(order):
        before, after = EXAMPLE_S.get(name, (0.15, 1.15))
        on = example_breath(c, cond, name)
        t0, t1 = max(0.0, on - before), min(c["dia_sub"].size / c["sr_sub"], on + after)
        left = outer[r, 0].subgridspec(2, 1, height_ratios=[4, 0.9], hspace=0.06)
        ax = fig.add_subplot(left[0])
        axd = fig.add_subplot(left[1], sharex=ax)
        b0, b1 = int(t0 / dt), int(t1 / dt)
        tt = (np.arange(b0, b1) + 0.5) * dt - t0
        post = F["post"][b0:b1]
        for s in state_cols:
            if post[:, s].max() < 0.05:
                continue
            col = np.array(matplotlib.colors.to_rgb(state_colour(c, k, s)))
            ax.fill_between(tt, 0, post[:, s], color=col, alpha=0.35, lw=0)
            ax.plot(tt, post[:, s], color=col * 0.75, lw=1.5)
        for y, u in enumerate(rows_order):
            if u not in c["spikes"]:
                continue
            ts = c["spikes"][u]
            ts = ts[(ts >= t0) & (ts < t1)] - t0
            ax.vlines(ts, 1 - (y + 0.85) / nu, 1 - (y + 0.15) / nu, color=INK, lw=0.7)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, sum(EXAMPLE_S.get(name, (0.15, 1.15))))
        ax.set_ylabel("P(state)", fontsize=8)
        ax.tick_params(labelbottom=False, labelsize=7)
        b = np.flatnonzero(cond == name)
        span = f"{ons[b[0]] + off:.0f}–{ons[min(b[-1] + 1, ons.size - 1)] + off:.0f} s · {b.size} breaths"
        ax.add_patch(Rectangle((0, 1.0), 1, 0.08, transform=ax.transAxes, color=colours.get(name, "#cccccc"),
                               clip_on=False))
        ax.set_title(f"{r + 1}. {name}", loc="left", fontsize=13, fontweight="bold", pad=20)
        ax.text(1.0, 1.1, f"{span}   ·   example {t0 + off:.2f}–{t1 + off:.2f} s", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color=INK2)
        ay = ax.twinx()
        ay.set_ylim(0, 1)
        ay.set_yticks(1 - (np.arange(nu) + 0.5) / nu)
        ay.set_yticklabels([f"#{u}" if u in c["spikes"] else f"#{u} –" for u in rows_order], fontsize=5)
        ay.tick_params(length=0, pad=1)
        ay.spines[["top", "right", "left", "bottom"]].set_visible(False)
        i0, i1 = int(t0 * c["sr_sub"]), int(t1 * c["sr_sub"])
        axd.plot(np.arange(i0, i1) / c["sr_sub"] - t0, c["dia_sub"][i0:i1], color=INK, lw=0.7)
        axd.set_yticks([])
        axd.set_ylabel("∫dia", fontsize=7)
        axd.set_xlabel("time from window start (s)", fontsize=8)
        axd.tick_params(labelsize=7)
        occ, R = rates[name]
        right = outer[r, 1].subgridspec(2, ns, height_ratios=[4, 0.9], hspace=0.06, wspace=0.12)
        for j, s in enumerate(state_cols):
            axb = fig.add_subplot(right[0, j])
            axb.set_ylim(nu - 0.5, -0.5)
            axb.set_xlim(0, xmax_hz)
            axb.set_yticks(np.arange(nu))
            axb.set_yticklabels([])
            axb.tick_params(axis="x", labelsize=6.5)
            axb.set_xlabel("Hz", fontsize=7)
            if occ[s] >= min_occ:
                axb.barh(np.arange(nu), np.nan_to_num(R[:, s]), color=state_colour(c, k, s), height=0.75)
                axb.set_title(f"{100 * occ[s]:.0f}%", fontsize=9, color=INK)
            else:
                axb.set_title(f"{100 * occ[s]:.0f}%", fontsize=9, color="#b0afaa")
                axb.text(0.5, 0.5, "< 2 %", transform=axb.transAxes, ha="center", va="center", fontsize=8,
                         color="#b0afaa")
                for sp in axb.spines.values():
                    sp.set_color("#e3e2dd")
            if r == 0:
                axb.add_patch(Rectangle((0, 1.13), 1, 0.05, transform=axb.transAxes, color=state_colour(c, k, s),
                                        clip_on=False))
                axb.text(0.5, 1.2, tsl.state_name(labels[s]).replace(" (", "\n("), transform=axb.transAxes,
                         ha="center", va="bottom", fontsize=8.5)
    fig.suptitle(trial_prefix(c) + f"K = {k} population HMM, epoch by epoch. Left: an example stretch, spikes of every "
                 "unit (rows, anoxia #1 baseline φ order; '–' = no partner in this trial)\nover the probability of each "
                 "state (shading, same colours as the columns). Right: each unit's firing rate in each state during "
                 "the epoch; columns are states in breath-cycle order, % = share of the epoch's time", x=0.01,
                 y=1 - 0.12 / H, ha="left", va="top", fontsize=10.5)
    path = out / f"state_examples_K{k}_by_epoch.png"
    save_fig(fig, path, dpi=100)
    plt.close(fig)
    return path


def state_examples_shared(runs):
    """fig_state_examples_epochs for several trials (decoded with the same model) with the same unit rows, state
    columns and rate axis."""
    import figures_signatures as fsig
    cs = []
    for run in runs:
        c = load_context(run)
        cs.append((c, label_transitions(c["cond"])[0], run / "figures"))
    c0 = cs[0][0]
    k = c0["ref_k"]
    rows_order = c0["order_units"]
    state_cols = fsig.cycle_order(c0, k, list(range(k)))
    xmax = 0.0
    for c, cond, _ in cs:
        for occ, R in state_rates(c, cond, rows_order).values():
            ok = occ >= 0.02
            if ok.any():
                xmax = max(xmax, np.nanmax(R[:, ok]))
    return [fig_state_examples_epochs(c, cond, out, rows_order, state_cols, xmax * 1.05) for c, cond, out in cs]


def fig_states_pca_epochs(c, cond, out, xlim=(-1.0, 2.0), ylim=(-0.8, 0.8), n_max=6000):
    """K = 6 Viterbi states in PC1-PC3 (anoxia #1 baseline PCA), one panel per epoch in time order, each marked with
    its colour, number, raw span and breath count; the baseline cloud in light grey behind every other panel."""
    from matplotlib.patches import Rectangle
    k = c["ref_k"]
    P, off = c["P"], c["time_offset_s"]
    ons = c["dia_df"]["on_sec"].to_numpy()
    cyc = np.searchsorted(ons, c["centres"], side="right") - 1
    valid = (cyc >= 0) & (cyc < ons.size - 1)
    bcond = np.where(valid, cond[np.clip(cyc, 0, ons.size - 1)], "").astype(str)
    gi = c["conditions"].index("gasping")
    order = c["conditions"][:gi] + ["pre-gasp", "gasping", "post-gasp"] + c["conditions"][gi + 1:]
    order = [n for n in order if (cond == n).any()]
    colours = {**c["cond_colours"], **TRANSITION_COLOURS}
    seq = c["F"][k]["seq"]
    rng = np.random.default_rng(0)
    ncol = 3
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 4.0 * nrow + 1.0), sharex=True, sharey=True, squeeze=False)
    base = np.flatnonzero(bcond == "baseline")
    base = base if base.size <= n_max else np.sort(rng.choice(base, n_max, replace=False))
    for i, name in enumerate(order):
        ax = axes.ravel()[i]
        idx = np.flatnonzero(bcond == name)
        sub = idx if idx.size <= n_max else np.sort(rng.choice(idx, n_max, replace=False))
        if name != "baseline":
            ax.scatter(P[base, 0], P[base, 2], s=2, color="#e3e2dd", edgecolor="none", zorder=1, rasterized=True)
        ax.scatter(P[sub, 0], P[sub, 2], c=[state_colour(c, k, s) for s in seq[sub]], s=3, alpha=0.6,
                   edgecolor="none", zorder=2, rasterized=True)
        for s in range(k):
            m = seq[idx] == s
            if m.sum() > 20:
                ax.scatter(P[idx[m], 0].mean(), P[idx[m], 2].mean(), color=state_colour(c, k, s), s=110,
                           edgecolor=INK, lw=0.9, zorder=4)
        ax.scatter(*c["ioff_point"][[0, 2]], color=INK, marker="X", s=70, zorder=5)
        b = np.flatnonzero(cond == name)
        t0, t1 = ons[b[0]] + off, ons[min(b[-1] + 1, ons.size - 1)] + off
        ax.add_patch(Rectangle((0, 1.0), 1, 0.07, transform=ax.transAxes, color=colours.get(name, "#cccccc"),
                               clip_on=False))
        ax.set_title(f"{i + 1}. {name}", loc="left", fontsize=13, fontweight="bold", pad=22)
        ax.text(1.0, 1.095, f"{t0:.0f}–{t1:.0f} s · {b.size} breaths", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=10, color=INK2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if i % ncol == 0:
            ax.set_ylabel("PC3")
        if i >= len(order) - ncol:
            ax.set_xlabel("PC1")
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")
    order_s = np.argsort(c["colours"][k])
    f = c["F"][k]["feats"]
    import transition_slides as tsl
    handles = [plt.Line2D([], [], marker="o", ls="", color=state_colour(c, k, s), mec=INK, ms=9) for s in order_s]
    handles += [plt.Line2D([], [], marker="X", ls="", color=INK, ms=9),
                plt.Line2D([], [], marker="o", ls="", color="#e3e2dd", ms=6)]
    labels = [tsl.state_name(f.loc[s, "label"]) for s in order_s] + ["baseline inspiration-off point",
                                                                       "baseline points (reference)"]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10.5)
    fig.suptitle(trial_prefix(c) + f"K = {k} HMM states in PC1–PC3 (PCA fit on {c.get('pca_fit', 'baseline')}), one "
                 "panel per epoch in time order.\nSmall dots: 5 ms bins coloured by Viterbi state; big dots: each "
                 "state's mean position in that epoch", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    path = out / f"hmm_states_pca_K{k}_by_epoch.png"
    save_fig(fig, path, dpi=130)
    plt.close(fig)
    return path


def pca_summary_shared(runs):
    """pca_summary.png for several trials (population_dynamics runs projected on the same PCA axes) with the same axis
    limits in every panel: the union of each trial's own limits."""
    zs = []
    for run in runs:
        c = load_context(run)
        cond = label_transitions(c["cond"])[0]
        zs.append((zoom_context(c, cond, run / "transitions"), run / "transitions"))
    own = [fig_pca_summary(z, out) for z, out in zs]
    shared = [((min(l[i][0][0] for l in own), max(l[i][0][1] for l in own)),
               (min(l[i][1][0] for l in own), max(l[i][1][1] for l in own))) for i in range(len(own[0]))]
    for z, out in zs:
        fig_pca_summary(z, out, limits=shared)
    return shared


def state_image(z, idx, mode):
    """Rows = breaths idx. mode 'time': state every 5 ms from onset - CYCLE_PRE_S to onset + CYCLE_POST_S, blank
    after the next onset. mode 'phase': state over 100 cycle-phase steps (onset -> offset -> next onset)."""
    k, dt = z["ref_k"], z["hmm_bin_s"]
    seq, ph = z["F"][k]["seq"], z["ph"]
    ons = z["dia_df"]["on_sec"].to_numpy()
    if mode == "time":
        n0, n1 = int(CYCLE_PRE_S / dt), int(CYCLE_POST_S / dt)
        img = np.full((idx.size, n0 + n1), -1)
        for r, i in enumerate(idx):
            b = int(ons[i] / dt)
            row = seq[b - n0:b + n1].astype(int).copy()
            row[n0 + int((ons[i + 1] - ons[i]) / dt):] = -1
            img[r, :row.size] = row
        return img
    img = np.full((idx.size, 100), -1)
    for r, i in enumerate(idx):
        b = np.arange(int(ons[i] / dt), int(ons[i + 1] / dt))
        u = np.mod(ph[b], 2 * np.pi)
        ok = np.isfinite(u)
        col = np.clip((u[ok] / (2 * np.pi) * 100).astype(int), 0, 99)
        img[r, col] = seq[b[ok]]
        for c_ in range(1, 100):                       # fill phase steps no 5 ms bin landed on
            if img[r, c_] < 0:
                img[r, c_] = img[r, c_ - 1]
    return img


def fig_state_sequences(z, sides, out):
    k = z["ref_k"]
    rgb = np.array([matplotlib.colors.to_rgb(state_colour(z, k, s)) for s in range(k)] + [(1, 1, 1)])
    d = z["dia_df"]
    ons, offs = d["on_sec"].to_numpy(), d["off_sec"].to_numpy()
    n_rows = [len(v) for v in sides.values()]
    H = 3 + 0.27 * sum(n_rows)
    fig = plt.figure(figsize=(17, H))
    gs = fig.add_gridspec(2, 2, height_ratios=n_rows, width_ratios=[2.3, 1], hspace=0.12, wspace=0.28, left=0.05,
                          right=0.9, top=1 - 0.8 / H, bottom=1.3 / H)
    for r, (title, idx) in enumerate(sides.items()):
        for j, mode in enumerate(("time", "phase")):
            ax = fig.add_subplot(gs[r, j])
            img = state_image(z, idx, mode)
            if mode == "time":
                ext = [-CYCLE_PRE_S, CYCLE_POST_S, idx.size, 0]
                ax.imshow(rgb[img], aspect="auto", interpolation="nearest", extent=ext)
                ax.plot((offs[idx] - ons[idx]), np.arange(idx.size) + 0.5, "|", color=INK, ms=6, mew=1.2)
                ax.plot((ons[idx + 1] - ons[idx]), np.arange(idx.size) + 0.5, "|", color=INK, ms=9, mew=1.6)
                ax.axvline(0, color=INK, lw=0.8)
                ax.set_xlim(-CYCLE_PRE_S, 2.45)
                ax.set_xlabel("s from inspiration onset (short tick: offset; long tick: next onset)"
                              if r == 1 else "")
                ax.set_title(title, loc="left", fontsize=11)
                ax.set_yticks(np.arange(idx.size) + 0.5)
                ax.set_yticklabels([f"{T_raw:.1f}" for T_raw in ons[idx] + z["time_offset_s"]], fontsize=6.5)
                ax.set_ylabel("breath onset (raw s)")
            else:
                ax.imshow(rgb[img], aspect="auto", interpolation="nearest", extent=[0, 2 * np.pi, idx.size, 0])
                ax.axvline(np.pi, color=INK, lw=0.8, ls=(0, (4, 3)))
                ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
                ax.set_xticklabels(["onset", "", "offset", "", "next onset"])
                ax.set_yticks([])
                ax.set_title("same breaths over the cycle phase (inspiration | expiration)", loc="left", fontsize=9)
                # condition bars and labels at the right
                x0 = 2 * np.pi * 1.02
                for name, a, b in blocks(z["cond"], idx):
                    ax.add_patch(Rectangle((x0, a), 0.25, b - a, color=z["cond_colours"].get(name, "white"),
                                           clip_on=False))
                    ax.text(x0 + 0.35, (a + b) / 2, f"{name}\n({b - a})", va="center", fontsize=8.5,
                            clip_on=False)
                ax.set_xlim(0, 2 * np.pi)
    fp.state_legend(fig, z, k, y=0.0, ncol=3)
    fig.suptitle(trial_prefix(z) + f"K = {k} HMM state (Viterbi, 5 ms) over each breath's whole cycle through the two transitions; "
                 "colours are states. White: after the next onset.", x=0.01, ha="left", fontsize=10.5)
    save_fig(fig, out / "state_sequences.png", dpi=130)
    plt.close(fig)


def fig_state_timing(z, runs, summ, out):
    k = z["ref_k"]
    order = list(np.argsort(z["colours"][k]))
    labels = z["F"][k]["feats"]["label"]
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 6, height_ratios=[0.9, 1.0, 1.2], hspace=0.85, wspace=0.3, left=0.09, right=0.99,
                          top=0.93, bottom=0.08)
    ax = fig.add_subplot(gs[0, :3])
    for y, name in enumerate(ZOOM):
        left = 0.0
        S = summ[summ["condition"] == name].set_index("state")
        for s in order:
            v = S.loc[s, "occupancy"]
            ax.barh(y, v, left=left, color=state_colour(z, k, s), height=0.75, lw=0)
            if v > 0.06:
                ax.text(left + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white")
            left += v
    ax.set_yticks(range(len(ZOOM)))
    ax.set_yticklabels([f"{n} ({int(summ.loc[summ.condition == n, 'n_breaths'].iloc[0])})" for n in ZOOM])
    ax.set_ylim(len(ZOOM) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of time in each state (breaths)")
    ax.set_title("A. State occupancy per window", loc="left")
    ax = fig.add_subplot(gs[0, 4:])
    for s in order:
        v = summ[summ["state"] == s].set_index("condition").loc[ZOOM, "frac_breaths_with_state"]
        ax.plot(range(len(ZOOM)), v, "o-", color=state_colour(z, k, s), lw=1.6, ms=5)
    ax.set_xticks(range(len(ZOOM)))
    ax.set_xticklabels(ZOOM, rotation=20, ha="right")
    ax.set_ylabel("fraction of breaths that visit the state")
    ax.set_title("B. How many breaths visit each state", loc="left")
    # C: dwell times per state
    rng = np.random.default_rng(1)
    for j, s in enumerate(order):
        ax = fig.add_subplot(gs[1, j])
        R = runs[(runs["state"] == s) & runs["starts_in_cycle"]]
        for x, name in enumerate(ZOOM):
            v = R.loc[R["condition"] == name, "dwell_ms"].to_numpy()
            if not v.size:
                continue
            ax.scatter(x + rng.uniform(-0.25, 0.25, v.size), v, s=5, color=z["cond_colours"][name], lw=0, alpha=0.7)
            ax.plot([x - 0.32, x + 0.32], [np.median(v)] * 2, color=INK, lw=1.6)
        ax.set_yscale("log")
        ax.set_xticks(range(len(ZOOM)))
        ax.set_xticklabels([n.replace(" recovery", "\nrecovery") for n in ZOOM], rotation=90, fontsize=7.5)
        ax.add_patch(Rectangle((0, 1.02), 1, 0.06, color=state_colour(z, k, s), transform=ax.transAxes,
                               clip_on=False))
        ax.set_title(labels[s].replace(" | ", "\n"), fontsize=8, pad=14)
        if j == 0:
            ax.set_ylabel("dwell time (ms)\n(black: median)")
            yC = ax.get_position().y1
    # D: where in the cycle each state sits (longest run per breath)
    tmax = 2.6
    for j, name in enumerate(ZOOM):
        ax = fig.add_subplot(gs[2, j])
        S = summ[summ["condition"] == name].set_index("state")
        Tz = z["dia_df"]
        sel = z["cond"] == name
        ti = np.median((Tz["off_sec"] - Tz["on_sec"]).to_numpy()[sel])
        ibi = np.median(np.diff(Tz["on_sec"].to_numpy())[sel[:-1]])
        for y, s in enumerate(order):
            f = S.loc[s, "frac_breaths_with_state"]
            if not f >= 0.1:
                continue
            a, b = S.loc[s, "median_start_ms_from_onset"] / 1e3, S.loc[s, "median_end_ms_from_onset"] / 1e3
            ax.barh(y, b - a, left=a, color=state_colour(z, k, s), height=0.6, alpha=0.35 + 0.65 * f, lw=0)
            ax.plot([S.loc[s, "q25_start_ms"] / 1e3, S.loc[s, "q75_start_ms"] / 1e3], [y - 0.38] * 2, color=INK,
                    lw=0.8)
            ax.plot([S.loc[s, "q25_end_ms"] / 1e3, S.loc[s, "q75_end_ms"] / 1e3], [y + 0.38] * 2, color=INK, lw=0.8)
            ax.text(tmax * 0.99, y, f"{f:.0%}", ha="right", va="center", fontsize=7, color=INK2)
        ax.axvline(0, color=INK, lw=0.8)
        ax.axvline(ti, color=INK, lw=0.8, ls=(0, (4, 3)))
        ax.axvline(ibi, color=INK2, lw=1.2)
        ax.set_xlim(-0.02, tmax)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([fsig.short_label(labels[s]) for s in order] if j == 0 else [], fontsize=7.5)
        if j == 0:
            yD = ax.get_position().y1
        ax.set_xlabel("s from inspiration onset")
        ax.set_title(name, fontsize=9.5, color=INK)
        ax.add_patch(Rectangle((0, 1.0), 1, 0.035, color=z["cond_colours"][name], transform=ax.transAxes,
                               clip_on=False))
    fig.text(0.01, yD + 0.045, "D. Where each state sits in the cycle: bar from median start to median end of its "
             "longest run per breath (whiskers: interquartile start above, end below; opacity and % = share of breaths "
             "that visit it);\n    black line = onset, dashed = median offset, grey = median next onset. State labels: "
             + ", ".join(f"{fsig.cond_abbr(n)} {n}" for n in z["state_label_conditions"]) + ", ABG all but gasping",
             fontsize=9.5, va="bottom")
    fig.text(0.01, yC + 0.06, "C. Dwell times: every run of the state that starts inside a breath of the window",
             fontsize=9.5, va="bottom")
    fp.state_legend(fig, z, k, y=0.0, ncol=3)
    fig.suptitle(trial_prefix(z) + f"HMM states (K = {k}) in the six windows; colours are states", x=0.01, ha="left", fontsize=11)
    save_fig(fig, out / "state_timing.png", dpi=120)
    plt.close(fig)


def fig_unit_phase_windows(z, out):
    tu = z["tuning"].set_index(["unit_id", "condition"])
    units = z["order_units"]
    fig = plt.figure(figsize=(17, 13.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.15, 1.15], height_ratios=[1, 1], hspace=0.35, wspace=0.3,
                          left=0.06, right=0.98, top=0.86, bottom=0.06)
    ax = fig.add_subplot(gs[:, 0])
    R = 0.42
    for y, u in enumerate(units):
        for x, name in enumerate(ZOOM):
            ax.add_patch(plt.Circle((x, y), R, fill=False, color=GRID, lw=0.6))
            if (u, name) not in tu.index:
                continue
            th, L = tu.loc[(u, name), "preferred_phase"], tu.loc[(u, name), "modulation_L"]
            if not np.isfinite(L * th) or tu.loc[(u, name), "n_spikes"] < MIN_SPIKES:
                ax.text(x, y, "–", ha="center", va="center", color=INK2, fontsize=8)
                continue
            tonic = not L >= pdm.TONIC_MAX_L
            ax.plot([x, x + R * L * np.cos(th)], [y, y - R * L * np.sin(th)], color=z["cond_colours"][name],
                    lw=1.0 if tonic else 2.4, solid_capstyle="round")
            ax.plot(x + R * L * np.cos(th), y - R * L * np.sin(th), "o", color=z["cond_colours"][name],
                    ms=2.5 if tonic else 4)
    ax.set_xlim(-0.6, len(ZOOM) - 0.4)
    ax.set_ylim(len(units) - 0.5, -0.9)
    ax.set_aspect("equal")
    ax.set_xticks(range(len(ZOOM)))
    ax.set_xticklabels(ZOOM, fontsize=8, rotation=50, ha="left", rotation_mode="anchor")
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(units)))
    short = {"inspiratory": "ins", "expiratory": "exp", "tonic": "ton"}
    ax.set_yticklabels([f"#{u} ({short.get(tu.loc[(u, 'baseline'), 'class_min_spikes'], 'silent')})" for u in units],
                       fontsize=7.5)
    for s in ("left", "bottom", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_title("A. Preferred phase (direction; onset at right, offset at\nleft, inspiration above) and modulation "
                 f"L (length;\ncircle = 1); thin = tonic (L < 0.2); – = < {MIN_SPIKES} spikes", loc="left", fontsize=9,
                 pad=62)
    # B: class counts and median L
    ax = fig.add_subplot(gs[0, 1])
    classes = [("inspiratory", "#3f3e3a"), ("expiratory", "#8f8d86"), ("tonic", "#d9d8d4"),
               (f"< {MIN_SPIKES} spikes", "#ffffff")]
    bottom = np.zeros(len(ZOOM))
    for cls, col in classes:
        v = np.array([(z["tuning"][(z["tuning"]["condition"] == n)]["class_min_spikes"] == cls).sum() for n in ZOOM])
        ax.bar(range(len(ZOOM)), v, bottom=bottom, color=col, label=cls, width=0.7, edgecolor=GRID, lw=0.6)
        for x, (b_, vv) in enumerate(zip(bottom, v)):
            if vv:
                ax.text(x, b_ + vv / 2, str(vv), ha="center", va="center", fontsize=8,
                        color="white" if col in ("#3f3e3a", "#8f8d86") else INK)
        bottom += v
    ax.set_xticks(range(len(ZOOM)))
    ax.set_xticklabels(ZOOM, rotation=20, ha="right")
    ax.set_ylabel("units")
    ax.set_ylim(0, 33)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=4, columnspacing=1.0, handlelength=1.2)
    ax.set_title("B. Units per class (cibrrig 2025 rule)", loc="left")
    ax = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(2)
    for x, name in enumerate(ZOOM):
        sel = (z["tuning"]["condition"] == name) & (z["tuning"]["n_spikes"] >= MIN_SPIKES)
        v = z["tuning"].loc[sel, "modulation_L"].dropna().to_numpy()
        ax.scatter(x + rng.uniform(-0.2, 0.2, v.size), v, s=12, color=z["cond_colours"][name], lw=0)
        ax.plot([x - 0.3, x + 0.3], [np.median(v)] * 2, color=INK, lw=1.8)
    ax.axhline(pdm.TONIC_MAX_L, color=INK2, lw=0.6, ls=":")
    ax.set_xticks(range(len(ZOOM)))
    ax.set_xticklabels(ZOOM, rotation=20, ha="right")
    ax.set_ylabel("modulation L (black: median)")
    ax.set_title(f"C. Breathing modulation per unit (≥ {MIN_SPIKES} spikes)", loc="left")
    # D: phase shifts between consecutive windows
    pairs = [("anoxia", "pre-gasp"), ("pre-gasp", "gasping"), ("gasping", "post-gasp"),
             ("post-gasp", "early recovery")]
    sub = gs[1, 1:].subgridspec(1, 4, wspace=0.35)
    for j, (a, b) in enumerate(pairs):
        ax = fig.add_subplot(sub[j])
        for u in units:
            if (u, a) not in tu.index or (u, b) not in tu.index:
                continue
            La, Lb = tu.loc[(u, a), "modulation_L"], tu.loc[(u, b), "modulation_L"]
            pa, pb = tu.loc[(u, a), "preferred_phase"], tu.loc[(u, b), "preferred_phase"]
            if not np.isfinite(La * Lb * pa * pb) or min(tu.loc[(u, a), "n_spikes"],
                                                         tu.loc[(u, b), "n_spikes"]) < MIN_SPIKES:
                continue
            both = min(La, Lb) >= pdm.TONIC_MAX_L
            ax.scatter(pa, pb, s=10 + 60 * min(La, Lb), color=INK if both else GRID, lw=0, zorder=2)
            if both and abs(np.angle(np.exp(1j * (pb - pa)))) > np.pi / 2:
                ax.text(pa, pb, f" #{u}", fontsize=7, color=INK, va="center")
        ax.plot([-np.pi, np.pi], [-np.pi, np.pi], color=INK2, lw=0.6, ls=":")
        for v in (0,):
            ax.axhline(v, color=GRID, lw=0.6)
            ax.axvline(v, color=GRID, lw=0.6)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_aspect("equal")
        ticks = [-np.pi, 0, np.pi]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(["-π", "0", "π"])
        ax.set_yticklabels(["-π", "0", "π"])
        ax.set_xlabel(f"φ {a}")
        ax.set_ylabel(f"φ {b}")
        ax.add_patch(Rectangle((0, 1.02), 0.48, 0.05, color=z["cond_colours"][a], transform=ax.transAxes,
                               clip_on=False))
        ax.add_patch(Rectangle((0.52, 1.02), 0.48, 0.05, color=z["cond_colours"][b], transform=ax.transAxes,
                               clip_on=False))
    fig.text(0.40, 0.475, "D. Preferred phase in consecutive windows (units with ≥ 20 spikes in both; black: "
             "modulated in both, size ∝ L; grey: tonic in either; labelled: shift > 90°)", fontsize=9.5)
    fig.legend(cond_handles(z), ZOOM, loc="lower center", ncol=len(ZOOM), frameon=False)
    fig.suptitle(trial_prefix(z) + "Single-unit breath-phase alignment in the six windows (φ: 0 = inspiration onset, ±π = offset; "
                 "cibrrig phase curves and vector mean)", x=0.01, ha="left", fontsize=11)
    save_fig(fig, out / "unit_phase_windows.png", dpi=125)
    plt.close(fig)


def breath_phase_image(z, idx):
    """Units (baseline φ order) x breaths idx, RGBA: circular mean φ of the unit's spikes in the cycle (PHASE_CMAP),
    blended towards white for 1-5 spikes, white for none."""
    units = z["order_units"]
    ons = z["dia_df"]["on_sec"].to_numpy()
    phi_t, phi = z["phi_t"], z["phi"]
    img = np.ones((len(units), idx.size, 4))
    for r, u in enumerate(units):
        ts = z["spikes"][u]
        for c_, i in enumerate(idx):
            s = ts[(ts >= ons[i]) & (ts < ons[i + 1])]
            if not s.size:
                continue
            p = phi[np.clip(np.searchsorted(phi_t, s), 0, phi.size - 1)]
            mu = np.angle(np.exp(1j * p).mean())
            col = np.array(PHASE_CMAP((mu + np.pi) / (2 * np.pi)))
            a = min(1.0, 0.25 + 0.75 * s.size / 6)        # 6+ spikes = full opacity
            img[r, c_, :3] = 1 - a * (1 - col[:3])
    return img


def fig_unit_phase_by_breath(z, sides, out):
    """Per unit and breath: circular mean spike phase (colour, cyclic) and spike count (opacity)."""
    units = z["order_units"]
    ons = z["dia_df"]["on_sec"].to_numpy()
    cmap = PHASE_CMAP
    widths = [len(v) for v in sides.values()]
    fig = plt.figure(figsize=(18, 10.5))
    gs = fig.add_gridspec(2, 2, width_ratios=widths, height_ratios=[0.07, 1], hspace=0.04, wspace=0.06, left=0.07,
                          right=0.93, top=0.88, bottom=0.12)
    for j, (title, idx) in enumerate(sides.items()):
        img = breath_phase_image(z, idx)
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(img, aspect="auto", interpolation="nearest", extent=[-0.5, idx.size - 0.5, len(units), 0])
        ax.set_yticks(np.arange(len(units)) + 0.5)
        ax.set_yticklabels([f"#{u}" for u in units] if j == 0 else [], fontsize=7)
        ax.set_xticks(np.arange(idx.size)[::3])
        ax.set_xticklabels([f"{t:.1f}" for t in ons[idx][::3] + z["time_offset_s"]], rotation=90, fontsize=7.5)
        ax.set_xlabel("breath onset (raw s)")
        top = fig.add_subplot(gs[0, j], sharex=ax)
        condition_strip(top, z, idx, fontsize=8)
        top.set_xlim(-0.5, idx.size - 0.5)
        top.set_title(title, loc="left", fontsize=11, pad=18)
    cax = fig.add_axes([0.94, 0.35, 0.012, 0.3])
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(-np.pi, np.pi))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cb.set_ticklabels(["-π (offset)", "-π/2 (mid E)", "0 (onset)", "π/2 (mid I)", "π (offset)"])
    cax.set_title("mean spike φ", fontsize=9, loc="left")
    fig.suptitle(trial_prefix(z) + "Each unit's mean spike phase on every breath through the two transitions (units in baseline φ "
                 "order, tonic last).\nColour: circular mean φ of the unit's spikes in that cycle (warm = "
                 "inspiration, cool = expiration); paler = 1–5 spikes, white = none", x=0.01, ha="left", fontsize=10.5)
    save_fig(fig, out / "unit_phase_by_breath.png", dpi=125)
    plt.close(fig)


def unit_transition_raster(z, u, idx, out):
    """Slide-sized: spikes on every breath from late anoxia to early recovery, whole cycles from inspiration onset."""
    ons, offs = z["dia_df"]["on_sec"].to_numpy(), z["dia_df"]["off_sec"].to_numpy()
    ts = z["spikes"][u]
    with plt.rc_context(fsig.SLIDE_FONT):
        fig = plt.figure(figsize=(8.7, 6.6))
        ax = fig.add_axes([0.1, 0.11, 0.66, 0.79])
        lo, hi = np.searchsorted(ts, ons[idx] - CYCLE_PRE_S), np.searchsorted(ts, ons[idx + 1])   # one cycle
        xs = np.concatenate([ts[a:b] - ons[i] for a, b, i in zip(lo, hi, idx)])
        ys = np.repeat(np.arange(idx.size) + 0.5, hi - lo)
        ax.scatter(offs[idx] - ons[idx], np.arange(idx.size) + 0.5, s=14, marker="|", color="#8f8d86", lw=1.2)
        ax.scatter(ons[idx + 1] - ons[idx], np.arange(idx.size) + 0.5, s=30, marker="|", color="#8f8d86", lw=1.8)
        ax.scatter(xs, ys, s=4.0, marker="s", color=INK, lw=0, zorder=3)
        ax.axvline(0, color=INK2, lw=0.9, ls=(0, (4, 3)))
        ax.set_ylim(idx.size, 0)
        ax.set_xlabel("s from inspiration onset")
        ax.set_ylabel("breath (recording order)")
        x0 = 2.45 + 0.04
        for name, a, b in blocks(z["cond"], idx):
            ax.add_patch(Rectangle((x0, a), 0.1, b - a, color=z["cond_colours"].get(name, "white"), clip_on=False))
            ax.text(x0 + 0.16, (a + b) / 2, f"{name}\n{ons[idx[a]] + z['time_offset_s']:.0f}–"
                    f"{ons[idx[b - 1]] + z['time_offset_s']:.0f} s ({b - a})", va="center", fontsize=10,
                    clip_on=False)
        ax.set_xlim(-CYCLE_PRE_S, 2.45)
        fig.suptitle(trial_prefix(z) + f"#{u}: spikes on each breath through both transitions\nfrom 0.3 s before "
                     "onset to the next onset (grey ticks: inspiration offset, next onset)", x=0.01, ha="left",
                     fontsize=12)
        save_fig(fig, z["slide_out"] / f"unit_{u}_raster.png", dpi=220)
        plt.close(fig)


def fig_heartbeat_qc(z, beats, hp, out, wins):
    off = z["time_offset_s"]
    fig, axes = plt.subplots(len(wins), 1, figsize=(14, 1.7 * len(wins)))
    for ax, w in zip(axes, wins):
        a, b = int((w - off) * FS), int((w + 2 - off) * FS)
        tt = np.arange(a, b) / FS + off
        ax.plot(tt, hp[a:b], color=INK, lw=0.5)
        m = (beats + off >= w) & (beats + off < w + 2)
        ax.plot(beats[m] + off, hp[(beats[m] * FS).astype(int)], "v", color="#d93636", ms=5)
        ax.set_title(f"{w}–{w + 2} s raw", loc="left", fontsize=8)
        ax.set_ylabel("ECG (V, >5 Hz)")
    axes[-1].set_xlabel("raw-run time (s)")
    fig.suptitle(trial_prefix(z) + "Heartbeat detection check: ECG (XA2) with the detected R peaks", x=0.01, ha="left")
    fig.tight_layout()
    save_fig(fig, out / "qc_heartbeats.png", dpi=100)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------- main

def summarise(T, z, summ):
    cols = ["rate_per_min", "ibi_s", "ti_ms", "te_ms", "duty", "amp", "rise_ms", "flow_insp_peak", "flow_insp_volume",
            "hr_bpm", "pop_rate_hz", "gasp_likeness", "I_state_entry_ms_from_onset", "I_state_exit_ms_from_offset",
            f"state{z['gasp_state']}_frac"]
    tab = T[T["condition"].isin(ZOOM)].groupby("condition")[cols].median().loc[ZOOM]
    tab.insert(0, "n", T[T["condition"].isin(ZOOM)].groupby("condition").size().loc[ZOOM])
    return tab


def main():
    a = parse_args()
    run = Path(a.run)
    out = run / "transitions"
    (out / "cache").mkdir(parents=True, exist_ok=True)
    c = load_context(run)
    cond, pre, post, g = label_transitions(c["cond"])
    z = zoom_context(c, cond, out)
    n_samples = int(round(c["dia_sub"].size / c["sr_sub"] * FS))
    beats, ecg_hp = heartbeats(n_samples, c["time_offset_s"], a.nidq_bin, out / "cache")
    flow = inspiration_up(airflow(a.nidq_cache, c["sr_sub"]), z)
    T = breath_table(z, flow, beats)
    runs = state_runs(z)
    summ = timing_summary(z, runs)
    T.to_csv(out / "transition_breaths.tsv", sep="\t", index=False, float_format="%.4f")
    z["tuning"].to_csv(out / "transition_unit_tuning.tsv", sep="\t", index=False, float_format="%.4f")
    runs.to_csv(out / "state_runs.tsv", sep="\t", index=False, float_format="%.1f")
    summ.to_csv(out / "state_timing_summary.tsv", sep="\t", index=False, float_format="%.3f")
    pd.DataFrame({"beat_raw_s": beats + c["time_offset_s"],
                  "rr_ms": np.r_[np.nan, np.diff(beats) * 1e3]}).to_csv(out / "heartbeats.tsv", sep="\t", index=False,
                                                                        float_format="%.4f")
    sides = side_ranges(pre, post, g)
    on_raw = T["on_raw_s"].to_numpy()
    rr_jump = beats[2:][np.abs(np.log(np.diff(beats[1:]) / np.diff(beats[:-1]))) > np.log(1.6)] + c["time_offset_s"]
    wins = sorted({round(float(t), 1) for t in [on_raw[0] + 30, on_raw[pre[0]] - 1, on_raw[pre[0]] + 1, on_raw[g[0]] + 5,
                                                 on_raw[post[0]] + 3, on_raw[post[-1] + 1] + 5]}
                  | {round(float(t) - 1, 1) for t in rr_jump[(rr_jump > on_raw[pre[0]] - 10)][:2]})
    fig_heartbeat_qc(z, beats, ecg_hp, out, wins)
    fig_timeline(z, T, flow, beats, pre, post, g, out)
    fig_breath_metrics(z, T, pre, post, g, out)
    fig_breath_shapes(z, T, flow, pre, post, out)
    names = {"eupnea → gasping": "pca_filmstrip_eupnea_to_gasp.png",
             "gasping → recovery": "pca_filmstrip_gasp_to_recovery.png"}
    for title, idx in sides.items():
        fig_pca_filmstrip(z, T, title, idx, names[title], out)
    fig_pca_summary(z, out)
    fig_state_sequences(z, sides, out)
    fig_state_timing(z, runs, summ, out)
    fig_unit_phase_windows(z, out)
    fig_unit_phase_by_breath(z, sides, out)
    for align in ("on", "off"):
        fsig.fig_population_peth(z, align)
    fsig.fig_polar(z)
    strip = np.arange(pre[0] - N_CONTEXT, post[-1] + 1 + N_CONTEXT)
    for u in z["order_units"]:
        fsig.unit_slide_peth(z, u)
        unit_transition_raster(z, u, strip, out)
    pd.set_option("display.width", 250)
    print(summarise(T, z, summ).round(3).T.to_string())
    print(f"pre-gasp breaths {pre[0]}-{pre[-1]} ({pre.size}), gasps {g[0]}-{g[-1]} ({g.size}), "
          f"post-gasp {post[0]}-{post[-1]} ({post.size})")


if __name__ == "__main__":
    main()
