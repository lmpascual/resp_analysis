"""Write ``run_summary.md``: the numbers of one run, generated from the files in its output folder.

Everything here is read from files the pipeline wrote (run_manifest.json, config_used.yaml,
<sorter>/units.tsv, postprocessing_report.json, epoch_stability.tsv, comparison/, review/), so the
summary can't drift from the results. Reasons for the settings, interpretation and open questions
belong in the hand-written analysis record (docs/), which quotes these numbers.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from .config import DEFAULTS_PATH

LABELS = ("good", "mua", "non_soma", "noise")
VERDICT_ORDER = ("footprint differs", "unclear", "same footprint; a neighbour looks alike", "same footprint")


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _fmt(v) -> str:
    return f"`{v}`" if isinstance(v, str) else "`" + json.dumps(v) + "`"


def config_changes(cfg: dict) -> list[tuple[str, str, str]]:
    """(setting, default, value used) for every setting that differs from defaults.yaml."""
    defaults = _flatten(yaml.safe_load(DEFAULTS_PATH.read_text()))
    used = _flatten(cfg)
    rows = []
    keys = set(defaults) | set(used)
    for k in sorted(keys):
        if k.startswith(("paths.", "_")) or k == "experiment_name":
            continue
        if any(o.startswith(k + ".") for o in keys):  # an empty section filled in by the config: its entries are listed
            continue
        if defaults.get(k, "(not set)") != used.get(k, "(not set)"):
            rows.append((k, _fmt(defaults.get(k, "(not set)")), _fmt(used.get(k, "(not set)"))))
    return rows


def _read(path: Path, **kw) -> pd.DataFrame | None:
    return pd.read_csv(path, sep="\t", **kw) if path.exists() else None


def _worst_verdicts(stab: pd.DataFrame) -> pd.Series:
    rank = {v: i for i, v in enumerate(VERDICT_ORDER)}
    worst = stab.assign(r=stab.verdict.map(rank)).groupby("unit_id").r.min()
    return worst.map(dict(enumerate(VERDICT_ORDER)))


def _table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def write_run_summary(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    man = json.loads((out / "run_manifest.json").read_text())
    cfg = yaml.safe_load((out / "config_used.yaml").read_text())
    sorters = man.get("sorters", [])
    L = [f"# Run summary: {man.get('experiment', cfg.get('experiment_name'))}", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M} by `summarize_run.py` from the files in this folder. "
         "Don't edit it by hand: rerun the script. Reasons and interpretation go in the analysis record.", ""]

    # provenance
    env = man.get("environment", {})
    L += ["## Provenance", "",
          f"- Config: `{man.get('config_file')}` (as used: `config_used.yaml`)",
          f"- Log of the last run: `{man.get('log_file')}`",
          f"- Pipeline commit: `{env.get('git_commit')}`",
          f"- Input: `{cfg['input']['source']}` from `{cfg['input']['folder']}` ({cfg['input']['stream_id']})"]
    pre = out / "preprocessing" / "preprocessing_info.json"
    if pre.exists():
        p = json.loads(pre.read_text())
        L.append(f"- Recording: {p['duration_s']:.1f} s at {p['sampling_frequency']:.2f} Hz, "
                 f"{p['n_channels_in']} channels in, {p['n_channels_out']} out; bad channels {p.get('bad_channel_counts')}")
    L.append(f"- Drift correction: {'on (' + cfg['motion']['preset'] + ')' if cfg['motion']['enabled'] else 'off'}")
    L += ["", "| package | version |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in env.items() if k != "git_commit"]
    L.append("")

    # settings
    rows = config_changes(cfg)
    L += ["## Settings that differ from the defaults", "",
          "Each of these needs a reason in the analysis record's decision log.", ""]
    L += _table(["setting", "default", "used"], [list(r) for r in rows]) if rows else ["(none)"]
    L.append("")

    # timings
    t = man.get("timings_s", {})
    if t:
        L += ["## Stage timings", "", "Stages skipped by `--redo` keep the time from the run that did them.", ""]
        L += _table(["stage", "minutes"], [[k, f"{v / 60:.1f}"] for k, v in t.items()])
        L.append("")

    # per sorter
    data = {}
    for s in sorters:
        units = _read(out / s / "units.tsv", index_col="unit_id")
        rep_p = out / s / "postprocessing_report.json"
        data[s] = {"units": units, "rep": json.loads(rep_p.read_text()) if rep_p.exists() else {},
                   "stab": _read(out / s / "epoch_stability.tsv"),
                   "matches": _read(out / "comparison" / f"{s}_matches.tsv", index_col="unit_id")}
    rows = []

    def row(name, f):
        rows.append([name] + [f(data[s]) for s in sorters])

    row("units before cleaning (incl. extracted)", lambda d: d["rep"].get("units_from_sorter", ""))
    row("spikes before cleaning", lambda d: f"{d['rep'].get('spikes_from_sorter', 0):,}")
    row("double detections removed", lambda d: f"{d['rep'].get('spikes_from_sorter', 0) - d['rep'].get('spikes_after_duplicate_removal', 0):,}")
    row("redundant pairs (one unit dropped)", lambda d: len(d["rep"].get("redundant_pairs", [])))
    row("units after cleaning", lambda d: d["rep"].get("units_after_cleaning", ""))
    for lab in LABELS:
        row(f"label {lab}", lambda d, lab=lab: int((d["units"].label == lab).sum()))
    row("positive-going units (good / mua / non_soma / noise)",
        lambda d: " / ".join(str(int(((d["units"].amplitude_median > 0) & (d["units"].label == lab)).sum())) for lab in LABELS))
    row("found by both sorters (all / good)",
        lambda d: "" if d["matches"] is None else
        f"{int(d['matches'].found_by_both.sum())} / {int(d['matches'].found_by_both.reindex(d['units'].index[d['units'].label == 'good']).fillna(False).astype(bool).sum())}")
    row("merge suggestion groups", lambda d: d["rep"].get("merge_suggestion_groups", ""))
    L += ["## Units", ""] + _table([""] + sorters, rows)
    L += ["", "Polarity is the sign of `amplitude_median`. Cross-sorter matching excludes "
          f"{cfg['comparison']['exclude_labels']} units and counts agreement ≥ {cfg['comparison']['match_score']}.", ""]

    # extracted units
    ex_rows = []
    for s in sorters:
        for name, r in data[s]["rep"].get("extracted_units", {}).items():
            u = r.get("unit_id")
            lab = data[s]["units"].label.get(u, "") if u is not None else ""
            ex_rows.append([s, name, u, r.get("n_spikes"), f"{r['first_last_s'][0]:.1f}–{r['first_last_s'][1]:.1f} s" if r.get("first_last_s") else "",
                            ", ".join(f"#{k}: {v}" for k, v in r.get("removed_from", {}).items()) or "none", lab])
    if ex_rows:
        L += ["## Units extracted by config rules", ""]
        L += _table(["sorter", "rule", "unit", "spikes", "span", "spikes taken from", "label"], ex_rows)
        L.append("")

    # epoch stability
    if any(d["stab"] is not None for d in data.values()):
        L += ["## Epoch stability", "",
              "Per unit, the worst verdict over all epoch pairs with ≥ 30 spikes in each epoch.", ""]
        if cfg.get("epochs"):
            L.append("Epochs (file time): " + ", ".join(f"{e['name']} {e['start_s']}–{e['end_s']} s" for e in cfg["epochs"]))
            L.append("")
        rows = []
        worst = {s: _worst_verdicts(d["stab"]) if d["stab"] is not None else pd.Series(dtype=object) for s, d in data.items()}
        for v in VERDICT_ORDER:
            rows.append([v] + [int((worst[s] == v).sum()) for s in sorters])
        rows.append(["not compared (too few spikes)"] + [len(data[s]["units"]) - len(worst[s]) for s in sorters])
        L += _table(["verdict"] + sorters, rows)
        for s in sorters:
            w = worst[s]
            diff = w.index[w == "footprint differs"]
            if len(diff):
                labs = data[s]["units"].label.reindex(diff)
                good = [str(u) for u in diff if labs.get(u) == "good"]
                L.append(f"\n{s}: footprint differs for {len(diff)} units"
                         + (f", of which good: {', '.join('#' + g for g in good)}" if good else ", none of them good"))
        L.append("")

    # manual review
    rev_rows = []
    for s in sorters:
        rev = _read(out / "review" / f"{s}_unit_review.tsv", dtype=str, keep_default_na=False)
        if rev is None or not len(rev):
            continue
        ids = rev.unit_id.astype(int)
        units = data[s]["units"]
        missing = [i for i in ids if i not in units.index]
        changed = [f"#{i} {l}→{units.at[i, 'label']}" for i, l in zip(ids, rev.label_at_review)
                   if i in units.index and units.at[i, "label"] != l]
        verdicts = rev.verdict.replace("", "(none)").value_counts()
        rev_rows.append([s, len(rev), ", ".join(f"{k} {v}" for k, v in verdicts.items()),
                         ", ".join(f"{k} {v}" for k, v in rev.label_at_review.value_counts().items()),
                         ", ".join(map(str, missing)) or "none", ", ".join(changed) or "none",
                         rev.reviewed_at.max()[:10] if "reviewed_at" in rev else ""])
    L += ["## Manual review (`review/`)", ""]
    L += _table(["sorter", "units reviewed", "verdicts", "label when reviewed", "no longer in the sorting",
                 "label changed since review", "last review"], rev_rows) if rev_rows else ["No reviews saved yet."]
    L.append("")

    path = out / "run_summary.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
