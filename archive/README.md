# Archive: the pre-sorting_v2 pipeline and analyses

Moved here on 2026-10-22. Nothing in this folder is used by the current pipeline (`sorting/`) or analyses
(`analysis/`). It is kept so old figures can be traced back to the code that made them.

**To run any of it, check out the tag `legacy-pipeline`** (`git checkout legacy-pipeline`). That is the last
commit with this code in its original place; its imports (`src.pipeline...`, `scripts...`) assume that layout
and won't work from inside `archive/`.

| Folder | What it held | Replaced by |
|---|---|---|
| `pipeline_v1/src/pipeline` | The first sorting pipeline: SpikeGLX/CatGT/NIDQ readers, sliding-window Lupin / MountainSort 5 / Kilosort 4 runs with cross-window consensus, its own QC and HTML diagnostics. Several modules exist twice (`io/` vs `processing/neural_ap/`, `processors/` vs `processing/analog_nidq/`) | `sorting/` (npxsort). Its "Lupin" was SpyKING-CIRCUS 2 with clustering skipped above 500 k peaks, so its units are not comparable; see `sorting/README.md` |
| `pipeline_v1/src/analysis` | Epoch spiking statistics, a non-stationarity / split detector | `sorting/npxsort/stability.py`, `analysis/units/epoch_outliers.py` |
| `pipeline_v1/tests` | Tests for the old pipeline | `sorting/tests/simulate_and_test.py` |
| `pipeline_v1/*.json`, `*.md`, `pyproject.toml`, `requirements.txt`, `README.md`, `docs/` | Old pipeline configs (e097 pre-laser 60 s, laser 226 s, diving reflex 240 s), guide, collaborator report, packaging | `sorting/configs/`, `README.md` at the repository root |
| `scripts/` | Old analysis and figure scripts. The ~100 files directly in `scripts/` only forward to the copies in `scripts/pipeline` and `scripts/visualization` | — |
| `scripts/pipeline` | Old sorting runs, Phy export fixes, laser-train extraction, QC recomputation | `sorting/` |
| `scripts/visualization` | Old QC dashboards, rasters, breath-onset PSTHs, phase rasters, publication-figure replication, presentation decks | `analysis/population/` (PETHs, phase curves, rasters), `analysis/units/` |
| `scripts/experiments/e097` | e097 analyses on old-pipeline sorts: sub-window sorting, 5–7-state HMMs, laser PETHs, diving reflex 240 s, pre-laser 60 s, session dashboards. Never on GitHub before this commit (`.gitignore`'s `e097/` pattern) | `analysis/population/`, `analysis/units/optotag.py`. The session summary (`generate_e097_session_dashboard.py`) moved to `analysis/session/e097_session_summary.py` |
| `scripts/experiments/{e086,lp21,lp22,tp_tacr1}` | Other animals, all on old-pipeline sorts: e086 opto window; LP21 Bush et al. 2024 manifold replication and Lebedev rotation checks; LP22 stationarity and dropouts; TP_Tacr1 pacemaker / kindling and rotational trajectories | Not redone yet. Re-sort these recordings with `sorting/` first |
| `figures/` | Diagnostic PNGs from the old pipeline (ECG, flow, artefacts, sorter benchmark) | — |

Not redone with the current pipeline: the e097 diving-reflex (240 s) and pre-laser (60 s) analyses, and every
animal other than e097. Their old outputs are on the analysis PC under `outputs/_archive/` (not in git).
