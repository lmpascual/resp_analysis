# Analysis record template

Copy this file to `docs/<recording>_analysis_record.md` at the start of each recording's sorting, and fill it in
as you go: [e097_analysis_record.md](e097_analysis_record.md) is a filled-in example. Two documents work
together:

- **`run_summary.md`** (written by the pipeline into the output folder, or by `summarize_run.py`): versions,
  settings that differ from the defaults, unit counts, extracted units, epoch verdicts, review progress. It is
  generated, so the numbers can't be mistyped or go stale.
- **The analysis record** (this file, kept in git): what only a person can write. Why each setting was chosen,
  what was concluded from what evidence, which units go into analysis, and what is still open.

Rules of thumb:
- **Date every entry** and say what evidence it rests on (a file, a figure, a unit id with a time).
- **Never delete or rewrite an entry.** Mark it superseded and add the new one, so the history stays.
- **Write the unit-inclusion rule before the downstream analysis,** and apply it to every recording alike.
- **Commit the record with the code** so each version points at the pipeline commit it describes.

---

## 1. Question and scope
- Question:
- Window(s) sorted and why:
- What was deliberately not sorted:

## 2. Data provenance
| | |
|---|---|
| Animal (id, strain/genotype, sex, weight, age) | |
| Date, anaesthesia, procedures | |
| Target and probe (type, serial, shank, depth) | |
| Acquisition (software version, sampling rate, gain) | |
| Raw data location | |
| Preprocessing command (e.g. CatGT, from the output `.meta`) | |
| Config file | |
| Output folder | |

- **Time base:** how output times map to raw-run time and to the session's event log.
- **Retention:** which files must be kept for a rerun or for Phy.

## 3. Processing
Pipeline stages and a pointer to `run_summary.md`. A table of runs: date, log file, what changed, why.

## 4. Decision log
| # | Date | Decision | Why / evidence | Alternatives; how to undo |
|---|---|---|---|---|
| D1 | | | | |

Every row of "Settings that differ from the defaults" in `run_summary.md` needs a decision here. So does
every threshold you set by looking at the data, every rule-extracted unit, and every epoch boundary.

## 5. Findings
| # | Finding | Evidence | Status (established / provisional) |
|---|---|---|---|
| F1 | | | |

## 6. Which units go into analysis
The rule, written down before the downstream analysis:
- which labels count;
- whether and how manual review overrides labels;
- what happens to flagged units (epoch changes, duplicates, contamination);
- recording-level criteria (yield, noise, drift).

## 7. Curation status
| What | Where | State (date) |
|---|---|---|

## 8. Open questions and known issues
| # | Question | What would settle it |
|---|---|---|

## 9. How to reproduce
Exact commands, the pipeline commit, the environment, which results are not bit-identical and why.

## 10. Change history
| Date | Change | Commit |
|---|---|---|

---

## Where this structure comes from

No journal requires a single standard form for spike-sorting records. These sources say what such a record
should contain, and each section above answers one of them.

| Source | What it asks you to record | Section |
|---|---|---|
| Sandve et al. 2013, *Ten simple rules for reproducible computational research*, PLoS Comput Biol 9(10):e1003285, [doi:10.1371/journal.pcbi.1003285](https://doi.org/10.1371/journal.pcbi.1003285) | How every result was produced; no manual edits to data; exact program versions; scripts under version control; random seeds; data behind each figure; each statement in the text linked to the result behind it | 3, 5, 9; `run_summary.md` |
| Wilson et al. 2017, *Good enough practices in scientific computing*, PLoS Comput Biol 13(6):e1005510, [doi:10.1371/journal.pcbi.1005510](https://doi.org/10.1371/journal.pcbi.1005510) | Keep raw data unchanged; record every processing step; keep a changelog and a plain-text record of the project; put code and docs under version control | 2, 9, 10 |
| Schnell 2015, *Ten simple rules for a computational biologist's laboratory notebook*, PLoS Comput Biol 11(9):e1004385, [doi:10.1371/journal.pcbi.1004385](https://doi.org/10.1371/journal.pcbi.1004385) | A dated notebook is the primary record of computational work, as for bench work: what was done, why, and what came of it | 4, 5 |
| Hill, Mehta & Kleinfeld 2011, *Quality metrics to accompany spike sorting of extracellular signals*, J Neurosci 31(24):8699–8705, [doi:10.1523/JNEUROSCI.0971-11.2011](https://doi.org/10.1523/JNEUROSCI.0971-11.2011) | Report quality metrics (estimated false positives and false negatives) with every sorted unit, not just the unit count | 5; `units.tsv` |
| Harris et al. 2016, *Improving data quality in neuronal population recordings*, Nat Neurosci 19(9):1165–1174, [doi:10.1038/nn.4365](https://doi.org/10.1038/nn.4365) | State how sorting quality was checked, and the confounds that could produce wrong conclusions | 5, 8 |
| International Brain Laboratory et al. 2025, *Reproducibility of in vivo electrophysiological measurements in mice*, eLife 13:RP100840, [doi:10.7554/eLife.100840](https://doi.org/10.7554/eLife.100840) | Explicit, pre-set inclusion criteria at recording level (yield, noise, drift, artefacts) and unit level (refractory contamination, missed spikes, amplitude): the "RIGOR" criteria. Differences in preprocessing and inclusion change results between labs | 6 |
| IBL 2022, *Spike sorting pipeline for the International Brain Laboratory* (white paper), [figshare 19705522](https://figshare.com/articles/online_resource/Spike_sorting_pipeline_for_the_International_Brain_Laboratory/19705522) | Document the pipeline's changes from stock software, its QC tools, and its known issues | 3, 8 |
| Buccino et al., *Efficient and reproducible pipelines for spike sorting large-scale electrophysiology data*, eLife RP110170 (preprint [doi:10.1101/2025.11.12.687966](https://doi.org/10.1101/2025.11.12.687966)) | Small differences in software versions or parameters change sorting results: pin versions and parameters, ideally in a container | 9; `run_summary.md` |
| Percie du Sert et al. 2020, *The ARRIVE guidelines 2.0*, PLoS Biol 18(7):e3000410, [doi:10.1371/journal.pbio.3000410](https://doi.org/10.1371/journal.pbio.3000410) | The animal study around the recording: study design, sample size, inclusion/exclusion criteria, randomisation, blinding, outcome measures, statistics, animals, procedures, results ("Essential 10") | 2, 6 |
| Rübel et al. 2022, *The Neurodata Without Borders ecosystem for neurophysiological data science*, eLife 11:e78362, [doi:10.7554/eLife.78362](https://doi.org/10.7554/eLife.78362) | Share data with standard metadata (NWB, archived on DANDI) so others can reuse it | for publication |

Also cite the software itself in methods: SpikeInterface, Kilosort 4, Lupin, CatGT and bombcell (Fabre et al.
2023, [doi:10.5281/zenodo.8172821](https://doi.org/10.5281/zenodo.8172821)); use each package's own
citation page for the others.
