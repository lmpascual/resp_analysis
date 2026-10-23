"""Zip the parts of finished sorts that the analyses need, so a collaborator can run them on the same units.

    python pack_results.py <outputs>/e097/intact_window_v2 <outputs>/e097/anoxia2_window_v2 --out e097_sorted_results.zip

A rerun of the sorting won't give the same unit ids (Lupin's detection is unseeded; Kilosort 4 on a GPU isn't
guaranteed to repeat), and the review verdicts are keyed by unit id. So, to reproduce the analyses on the reviewed
units, share this zip along with the raw animal folder. Per sort it holds the run's provenance (config_used.yaml,
run_summary.md, run_manifest.json, logs/), review/, and for each sorter units.tsv, alf/, the curation and stability
tables, sorter_params_used.json and epoch_templates.npz, and the parts of the SortingAnalyzer that the review browser,
analysis/units/match_sorts.py and optotag.py read (the sorting, the average templates, unit locations, sparsity).
About 90 MB per window, compressed. It leaves out the rest of the analyzer (waveforms, PCA), the Phy export, the raw
sorter output and the report figures (GBs).

Unzip it into the collaborator's outputs folder: the paths inside start with the sort's folder relative to
<outputs> (e.g. e097/intact_window_v2/...).
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from npxsort.paths import path  # noqa: E402

RUN_FILES = ["config_used.yaml", "run_summary.md", "run_manifest.json"]
RUN_DIRS = ["review", "logs"]
SORTER_FILES = ["units.tsv", "removed_spikes.tsv", "epoch_stability.tsv", "epoch_activity.tsv",
                "postprocessing_report.json", "sorter_params_used.json", "merge_suggestions.csv"]
SORTER_DIRS = ["alf", "bombcell"]
ANALYZER_PARTS = ["recording.json", "recording_info", "settings.json", "sorting", "sparsity_mask.npy",
                  "spikeinterface_info.json", "extensions/unit_locations", "extensions/templates/average.npy",
                  "extensions/templates/info.json", "extensions/templates/params.json",
                  "extensions/templates/run_info.json"]


def files_of(run: Path):
    for name in RUN_FILES:
        if (run / name).is_file():
            yield run / name
    for d in RUN_DIRS + ["comparison"]:
        if (run / d).is_dir():
            yield from (f for f in sorted((run / d).rglob("*")) if f.is_file())
    for sorter in ("kilosort4", "lupin"):
        s = run / sorter
        if not s.is_dir():
            continue
        for name in SORTER_FILES:
            if (s / name).is_file():
                yield s / name
        for d in SORTER_DIRS:
            if (s / d).is_dir():
                yield from (f for f in sorted((s / d).rglob("*")) if f.is_file())
        if (s / "epoch_templates.npz").is_file():
            yield s / "epoch_templates.npz"
        for part in ANALYZER_PARTS:
            f = s / "analyzer" / part
            if f.is_file():
                yield f
            elif f.is_dir():
                yield from (g for g in sorted(f.rglob("*")) if g.is_file())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="sort output folders (inside the outputs folder)")
    p.add_argument("--out", required=True, help="zip file to write")
    p.add_argument("--extra", nargs="*", default=[],
                   help="other files under the outputs folder to include (e.g. e097/e097/e097.docx)")
    a = p.parse_args()
    root = path("outputs")
    files = []
    for run in map(Path, a.runs):
        if not (run / "config_used.yaml").is_file():
            sys.exit(f"{run} is not a finished sort (no config_used.yaml)")
        files += list(files_of(run))
    files += [root / e for e in a.extra]
    total = 0
    with zipfile.ZipFile(a.out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            arc = f.resolve().relative_to(root.resolve()).as_posix()
            z.write(f, arc)
            total += f.stat().st_size
    print(f"{len(files)} files, {total / 1e6:.0f} MB before compression -> {a.out} ({Path(a.out).stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
