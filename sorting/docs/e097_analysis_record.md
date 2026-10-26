# e097 intact window: analysis record

This is the record of what was done to the e097 spike sorting and why: decisions with reasons, findings with
the evidence behind them, curation status and open questions. Keep it up to date as you go: add a row whenever
you change a setting, reject a unit set or draw a conclusion. Don't rewrite old entries: mark them superseded.

- **Numbers come from the run.** `run_summary.md` in the output folder is generated from the run's files
  (`summarize_run.py`, also written at the end of every pipeline run). Quote it here; don't retype counts by hand.
- **Methods live in the docs.** [unit_labels_explained.md](unit_labels_explained.md) covers what each stage and
  label means; [waveform_shape_criteria.md](waveform_shape_criteria.md) covers the shape rules;
  [phy_tutorial.md](phy_tutorial.md) covers Phy. This record links to them rather than repeating them.
- **Template.** [analysis_record_template.md](analysis_record_template.md) is the same skeleton for the next
  recording, with the published standards it draws on.

Status of this record: started 2026-10-19; last updated 2026-10-26. Outputs described: the 2026-10-19
`--redo postprocess` run at pipeline commit `d003f64`.

---

## 1. Question and scope

- **Question.** Which ventral respiratory column neurons can be isolated in this window, and how do they behave
  across eupnea, anoxia, gasping, recovery and the active-expiration shift? In particular: when a unit switches
  from expiratory to inspiratory firing during gasping, is it the same neuron changing phase, or a different
  neuron being recruited?
- **Sorted.** One 420 s window, raw-run time 1740–2160 s. It contains baseline, 15 s anoxia #1 (N2,
  1800–1815 s raw) with gasping, recovery, and the active-expiration shift (2078–2150 s raw).
- **Not sorted here.** The rest of the 110 min session: laser trains 1–8 (840–1440 s raw), anoxia #2, diving
  reflex, kainic acid.
- **TODO (you):** why this window was chosen and what downstream analysis will use the units for.

## 2. Data provenance

| | |
|---|---|
| Animal | e097 (listed as "e097 (e096)" in the session file), Vglut2:ai32, male, 35.8 g |
| Recording date | 2026-09-04 |
| Anaesthesia | isoflurane induction; urethane 1.2 g/kg + 0.15 g/kg supplement |
| Target | ventral respiratory column (preBötC / rVRG) |
| Probe | Neuropixels 2.0 four-shank (type 2013, SN 22420022704), shank 1 active, 4500 µm depth; headstage/base SN 524 |
| Acquisition | SpikeGLX (appVersion 20260826), 29999.82 Hz, AP gain 100 |
| Raw data | `Z:/ramirezlab/budde/npx/animal_files`, run `e097_g2`, probe 0 |
| Session metadata | `outputs/e097/e097_whole_session_data.json` (milestones, breathing and heart rate) |
| CatGT command | `CatGT -dir=Z:/ramirezlab/budde/npx/animal_files -run=e097 -g=2 -t=0 -prb_fld -ap -prb=0 -startsecs=1740 -maxsecs=420 -apfilter=butter,12,300,9000 -loccar_um=75,300 -dest=C:/SCRATCH_neuropixels/catgt_e097_intact_420s` (from the output's `.meta`, `catGTCmdline0`; made 2026-10-13) |
| Sorting input | `C:/SCRATCH_neuropixels/catgt_e097_intact_420s/catgt_e097_g2/e097_g2_tcat.imec0.ap.bin` |
| Config | [configs/e097_intact_window.yaml](../configs/e097_intact_window.yaml) |
| Outputs | `C:/Users/lpascu/resp_analysis/outputs/e097/intact_window_v2` |

**Time base.** Every time in the outputs (units, ALF, Phy, epochs, review notes) is seconds from the start of
the CatGT file. Add **1740 s** for raw-run time and the session file's milestones.

**Keep.** The sorting input and `scratch_dir/preprocessed` are both on `C:/SCRATCH_neuropixels`. Phy needs
the preprocessed binary, and a rerun needs the CatGT file. If scratch is cleared, regenerate the CatGT file
with the command above before rerunning.

## 3. Processing

SpikeInterface 0.105.2, Kilosort 4.1.7 (GPU) and Lupin, labelled with bombcell's rules. Stages:
preprocess → sort → postprocess (cleaning, metrics, labels, rule-extracted units) → epoch stability → compare →
export. Versions, settings that differ from the defaults, timings and unit counts are in **`run_summary.md`**.
`run_manifest.json`, `config_used.yaml` and `logs/` hold the raw provenance.

| Run (log) | What | Why |
|---|---|---|
| 2026-10-15 `run_20261015_094012`, `_135919` | preprocessing; DREDGE drift estimate | — |
| 2026-10-16 `run_20261016_090550`, `_113819` | sorting to export, without drift correction; Lupin first ran out of memory | D2, D3 |
| 2026-10-19 `run_20261019_160643` | `--redo postprocess` at `d003f64` | adds the extraction rule (D9) and the epoch stability stage (D10) |

## 4. Decision log

Each decision: what, why, the evidence, what else was considered, and how to undo it. Dates are when the
decision was made.

| # | Date | Decision | Why / evidence | Alternatives; how to undo |
|---|---|---|---|---|
| D1 | 2026-10-13 | Sort CatGT output: 12th-order Butterworth 300–9000 Hz, local CAR (exclude 75 µm, include 300 µm), phase-shifted. The sorters don't re-reference: Kilosort runs with `do_CAR: false`; Lupin keeps its whitening, but its own filter and CMR are replaced by a 2nd-order Bessel 300–7000 Hz band-pass (`npxsort/sorters.py`) | Standard CatGT preprocessing; a second common reference would only remove signal | `input.source: spikeglx_raw`; filtering may affect shape metrics (O5) |
| D2 | 2026-10-16 | No drift correction (`motion.enabled: false`, Kilosort `do_correction: false`, Lupin `apply_motion_correction: false`) | DREDGE found median 0.26 µm displacement. Its only large "motion" (+70/−45 µm) fell in the ~85–125 s activity dropout during gasping, where it is unreliable, and correcting it bent the spike bands. Evidence: `preprocessing/motion.png`, `motion_summary.json` (both left from the 2026-10-15 run) | Turn correction back on if a later window drifts |
| D3 | 2026-10-16 | Lupin runs with `n_jobs: 4` (everything else 12) | With 12 spawned workers each got a ~1.2 GB copy of 1.85 M peak features and memory ran out (Errno 22 / truncated pickle). Only speed changes | More RAM, or `pool_engine: thread` |
| D4 | 2026-10-16 | `presence_ratio` reported but not used for labels | Respiratory neurons can fall silent during apnea and anoxia; bombcell would demote them to mua for it (#422 has presence 0.43) | Remove the override from `curation.threshold_overrides` |
| D5 | 2026-10-16 | Kilosort `torch_device: cuda` | Fail loudly if the GPU isn't visible instead of silently running on the CPU | — |
| D6 | 2026-10-16 | Bad channels by noise level (flat < 0.2×, noisy > 5× the probe median) | CatGT output is already referenced, so the IBL coherence method can't be used. Result: no channels removed | `preprocessing.bad_channels` |
| D7 | 2026-10-19 | Review amplitudes on each unit's largest-µV channel, not Kilosort's main channel | Kilosort picks its main channel on whitened templates, which differs from the largest raw channel for 59 of 553 KS4 units | Review browser: click a lane |
| D8 | 2026-10-19 | Per-spike review flags (they don't change labels or spike trains): **footprint mismatch** if relative deviation > 0.5 and > 6 noise SD, or polarity reversal (opposite/same ratio > 2, > 2× the template's, > 10 SD); **duplicate** if a spike of another unit lies within 0.1 ms and 100 µm, for unit pairs with ≥ 5 coincidences and ≥ 4× the 0.5–2.5 ms flank rate | Calibrated on e097 so that the flags mark the cases seen by eye (#435's 242.5–242.8 s spikes, the gasping spikes in #361). Simpler rules flagged 16–20% of all spikes as duplicates | Thresholds in `npxsort/spikefeatures.py`, `review.py` |
| D13 | 2026-10-26 | **Add a soft-normalised PCA next to the Bush & Ramirez PCA** (it doesn't replace it): each unit's rate (5 ms bins, 10 ms Gaussian, Hz) divided by its range + 5 sp/s (Churchland et al. 2012), with the range and the PCA axes taken from anoxia #1's baseline breath-averaged cycle (50 phase bins); every 5 ms bin of both anoxia trials is projected on those axes | F18: the current PC2 is one unit. Soft normalisation is defined on trial-averaged responses. Applied to single 5 ms bins it made PC1, PC2 and PC3 each 99% one unit (#122, #89, #121: fast, weakly modulated units whose spike noise it scales up). Fitting on the mean cycle keeps spike noise out of the axes | Also tried on the 5 ms bins (2026-10-26): z-scoring each unit (PC1's largest unit 13%, #89 moves to PC6), and the current square-root input with axes fit on the mean cycle (PC1 unchanged, #89 moves to PC8). Undo: delete `analysis/population/pca_softnorm.py` and the `*_softnorm_*` figures |
| D9 | 2026-10-19 | Extract the positive-going gasping cell on ch 148/150 as its own unit (rule `gasp_pos_ch148`: positive peaks > 5 SD on ch 148, footprint from 95–125 s, cosine ≥ 0.95) | ~27 large positive spikes at 103.7–109.4 s that neither sorter gave a unit: Kilosort put 26 into #361, Lupin into #1079. Seen by eye on the trace | Delete the rule; Phy split by hand ([phy_tutorial.md](phy_tutorial.md) §4) |
| D10 | 2026-10-19 | Epochs for the stability check (file time): baseline 0–60, anoxia 60–85, gasping 85–130, recovery 140–338, active_expiration 338–410 s; 130–140 s left out as transition | From the session milestones and breathing rate (gasping: < ~100 bpm from ~1825 to ~1870 s raw). Set after looking at the data, which is fine for a waveform check but not for testing firing differences between epochs | `epochs` in the config |
| D11 | 2026-10-19 | Epoch verdicts from effect size only: **same footprint** = footprint cosine ≥ 0.99 and waveform r ≥ 0.9; **differs** = cosine < 0.97 (about how different the nearest other neuron looks); **same footprint, a neighbour looks alike** if a nearby unit also has cosine ≥ 0.99. The discriminant AUC is reported, not used | During gasping the whole tissue shifts ~0.3–0.7 µm. With thousands of spikes the AUC detects that for nearly every unit, so a significance test would call every unit "changed". Thresholds calibrated on e097 | Thresholds in `npxsort/stability.py` |
| D12 | 2026-10-19 | Seed the noise-level estimate, spike locations and quality metrics from `postprocessing.seed` (0), as random spike sampling already was | F14: an unseeded noise estimate flipped a unit's label between runs. Takes effect from the next postprocessing run; the current outputs used the random seed stored in `analyzer/extensions/noise_levels/params.json`, so SNRs will shift slightly once more | `npxsort/postprocess.py` |

## 5. Findings

Status: **established** = checked against the code or data and unlikely to change; **provisional** = depends
on thresholds or on review not yet done.

| # | Finding | Evidence | Status |
|---|---|---|---|
| F1 | **Labels** (2026-10-19 rerun). Kilosort 4: 554 units; good 52, mua 299, non_soma 73, noise 130. Lupin: 686; good 51, mua 494, non_soma 87, noise 54. 48 of 52 KS4 good units and 47 of 51 Lupin good units are found by both sorters (agreement ≥ 0.5) | `run_summary.md` | established for this run |
| F2 | **Positive-going units can never be good or mua** under bombcell's rules in SpikeInterface 0.105.2: the shape rules assume a negative trough. KS4 130, Lupin 27 (each including the extracted unit); median SNR ~5 | [unit_labels_explained.md §5](unit_labels_explained.md) | established |
| F3 | **MUA is decided by two metrics.** `sliding_rp_violation` is NaN below ~410 spikes and can't pass below ~720 spikes in 420 s (~1.7 Hz), so every low-rate unit is mua. `l_ratio` medians are 1.09 (KS4) and 3.03 (Lupin) against a 0.3 threshold | same | established |
| F4 | **The shape-width gate does nothing here.** 88–93% of negative units count as "narrow"; non_soma comes from a small repolarisation peak (A/C > 3) | [waveform_shape_criteria.md](waveform_shape_criteria.md) | established |
| F5 | **Kilosort 4.1.7 bug:** the refractory-period split veto never fires (`swarmsplitter.py:57`, `len(st2 == 0)` is always the array length) | source | established |
| F6 | **Lupin is not reproducible run to run:** its detection-threshold sampling is unseeded despite `seed: 0` (1.85 M vs 1.60 M peaks in two runs on the same data) | run logs | established |
| F7 | Some KS4 unit pairs are the same spikes counted twice, e.g. #151/#161: 1615 coincident spikes vs 13 expected | review duplicate flags | provisional |
| F8 | **A positive-going cell fires only during the gasps** (ch 148/150, 27 spikes, 103.7–109.4 s). Now unit KS4 #566 and Lupin #1229; bombcell labels both noise (F2) | `postprocessing_report.json` → `extracted_units` | established |
| F9 | **KS4 #355** (expiratory → inspiratory during gasping) **keeps the same footprint in every epoch**: cosine ≥ 0.998, waveform r ≥ 0.993, amplitude within 4%; the nearest other unit is clearly different (0.955). This supports one neuron changing its phase, not another neuron taking over | `kilosort4/epoch_stability.tsv` (2026-10-19 rerun); review browser epoch panel | verdict established; interpretation provisional |
| F10 | **KS4 #434:** same footprint across epochs; its amplitude depends on firing rate. The low-amplitude spikes are close in time to the large ones, so this looks like one neuron, not two | review note; epoch check | provisional |
| F11 | **KS4 #435:** well isolated except about 6 spikes at 242.5–242.8 s that belong to a smaller unit on ch 240 (clearest on ch 243) | review note; footprint-mismatch flags | established |
| F12 | Gasping shifts every unit's footprint along the probe by ~0.3–0.7 µm, consistent with tissue movement rather than unit changes | `epoch_stability.tsv` `shift_vs_probe_um` | provisional |
| F13 | On simulated drifting Neuropixels data, 17 of 18 (60 s) and 17 of 17 (300 s) bombcell-good units were single true neurons; many true neurons were labelled mua in short windows | [README](../README.md#testing), `tests/simulate_and_test.py` | established (simulation only) |
| F14 | **SNR changed between runs of the same sorting.** SpikeInterface drew a random seed for the noise-level estimate, so SNR moved by a median 2.3% (up to 6–8%) between the 2026-10-16 run and the rerun. Lupin #790 dropped from good to mua (SNR 5.16 → 4.94; the threshold is 5). `l_ratio` also moved slightly for 70 KS4 / 102 Lupin units (median < 0.5%); cause not established. Fixed for later runs (D12) | `analyzer/extensions/noise_levels/params.json` (random seed); comparison with `unit_inspection.xlsx` made from the earlier run | established |
| F15 | **KS4 #361 after the gasping spikes were taken out:** footprint cosine ≥ 0.999 against gasping in every pair, but waveform r 0.89–0.92 (0.86 before), so two pairs are still "unclear". Something besides the 26 positive spikes still differs during gasping | `kilosort4/epoch_stability.tsv` | provisional |
| F16 | **KS4 #187 (good) looks like a different neuron during gasping.** Only 41 spikes fire then, with a different footprint (cosine 0.96, waveform r 0.75, 26% smaller, 2 µm shift); every other epoch pair is identical (cosine 1.000). Likely the neuron stops during gasping and another neuron's spikes were assigned to it | same | provisional; review |
| F17 | **Footprint differs in some epoch pair** for 19 KS4 units (good: #187) and 9 Lupin units (none good). "Unclear" for 169 KS4 and 235 Lupin units, mostly from small spike counts in short epochs | `run_summary.md` → Epoch stability | established for this run |
| F18 | **PC2 of the Bush & Ramirez PCA is one unit, KS4 #89, and its discrete steps are that unit's single spikes.** #89 is tonic (~30 Hz, L 0.04); its loading is +0.997 and it explains 99.8% of PC2's variance. The PCA has no per-unit scaling, so #89's spike noise, which the breathing rhythm doesn't share, gets an axis of its own. A lone spike can only fall at one of the 9 offsets the 10 ms Gaussian reaches (0, ±5, … ±40 ms), so it gives only 9 PC2 values, from −0.27 (40 ms away) to +0.17 (same bin); these match the PC2 histogram's peaks in both anoxia trials, and PC2 above +0.17 needs ≥ 2 spikes. PC2's floor (≈ −0.28) means #89 is silent: pre-gasp and gasping sit on it because #89 nearly stops then (0.1 Hz during gasping in anoxia #1, 1.1 Hz in anoxia #2). PC3 is 54% #353; PC1 is spread (largest #459, 26%). The HMM is fit on rates, not on the PCA, so the states are unaffected | `analysis/population/pca_softnorm.py` prints the loadings and makes `population_dynamics_v2/figures/pca_pc2_single_unit_steps.png` | established |
| F19 | **Soft-normalised PCA (D13).** PC1 is spread across units (largest #510, 13%) and still runs through the breathing cycle (late I against late E / pre-I). PC2 (51% #353, then #350, #435, #312) separates eupnea (≈ 0) from pre-gasp and gasping (≈ −3) in both trials, with recovery in between and moving back towards 0. No discrete steps; #89 sits mostly on PC5. PC1–PC3 carry 74, 9 and 6% of the baseline mean cycle's variance but only 3.6, 1.4 and 2.5% of single 5 ms bins' variance (the rest is spike noise), so read the state means, not the clouds | `hmm_states_pca_softnorm_K6_by_epoch*.png` in both runs' `figures/`; `population_dynamics_v2/cache/pca_softnorm.npz` | provisional |

## 6. Which units go into analysis

**Not decided yet.** Write the rule down here *before* looking at the downstream results, and apply it the same
way to every recording. For a well-defined rule, decide:

1. **Which bombcell labels count.** Good only? Good plus mua that pass review? F2 and F3 mean bombcell alone
   would exclude every positive-going unit and every unit under ~1.7 Hz, including real respiratory neurons.
2. **Whether manual review overrides labels,** and with which verdicts (`single`; `unsure`?).
3. **What happens to units flagged in review:** footprint differs between epochs, duplicates, contaminating spikes.
4. **Rule-extracted units** (F8): include them by hand, since bombcell labels them noise.

For comparison, the IBL's RIGOR single-unit criteria are: refractory-period contamination below 10% with 90%
confidence, an amplitude-cutoff (missed spikes) test, and median amplitude > 50 µV. Their recording-level
criteria include yield ≥ 0.1 passing units per channel and AP-band RMS < 40 µV (IBL et al. 2025, eLife).

## 7. Curation status

| What | Where | State (2026-10-19) |
|---|---|---|
| Trace review, KS4 | `outputs/.../review/kilosort4_unit_review.tsv` (backup `kilosort4_unit_review.backup_2026-10-19.tsv`) | 24 units: single 12, unsure 5, multi 4, noise 1, merged 1, no verdict 1. Of the 14 good units reviewed: single 9, unsure 3, multi 1 (#11), no verdict yet 1 (#435) |
| Trace review, Lupin | `review/lupin_unit_review.tsv` | not started |
| Inspection workbook | `outputs/.../unit_inspection.xlsx` (`make_inspection_workbook.py`) | in use. Made before the 2026-10-19 rerun: it lacks KS4 #566 and Lupin #1229, and still shows Lupin #790 as good (now mua, F14) |
| Phy | `<sorter>/phy` | not used. A rerun's export overwrites it: copy it first |

`run_summary.md` lists, after each rerun, whether any reviewed unit is gone or has a new label.

## 8. Open questions and known issues

| # | Question | What would settle it |
|---|---|---|
| O1 | Inclusion rule for analysis (§6) | your decision |
| O2 | Positive-going units: somatic or axonal? Bombcell can't judge them (F2) | flip the template sign for shape rules (code change); look for propagation along the probe |
| O3 | Are `l_ratio` (0.3) and the sliding refractory test calibrated for dense brainstem recordings? | use `rp_contamination` for low-count units; compare with review verdicts |
| O4 | Lupin's unseeded detection (F6) | find or add a seed; until then, rerunning Lupin changes unit ids, and Lupin reviews must be redone |
| O5 | Does CatGT's 12th-order filter change the shape metrics bombcell's defaults were set on? | sort the same window from raw SpikeGLX with a gentler filter |
| O6 | Ground truth for the labels | e097 is Vglut2:ai32 (ChR2 in Vglut2 neurons), and laser trains 1–8 (840–1440 s raw) fall outside this window. **Don't widen the sorted window to include them** (concern raised 2026-10-19): in your experience sorting gets worse over longer windows, drift has more time to build up, and brief units like the gasping-recruited ones become an even smaller fraction of the spikes and get merged or lost (O9). Instead: (a) project this window's templates into the laser window by template matching (no re-clustering, so unit ids are kept) and test each unit's light response for latency, jitter and an evoked waveform that matches its spontaneous one; or (b) sort the laser window on its own and match units to this sort by footprint. Either way, first measure drift between the two windows (motion estimate only, no correction). Short-latency, low-jitter responses are needed to call a unit Vglut2+, since the light also drives synaptic followers and caused apneas |
| O9 | Brief units are lost at two levels. **Sorter:** a 27-spike gasping cell was merged into #361 (F8). **Labels:** a unit can't be good below ~720 spikes (F3); the gasping units #2 (156 spikes) and #422 (662) are mua only because `sliding_rp_violation` is NaN, though both have zero refractory violations (`rp_contamination` 0), `l_ratio` ≤ 0.003 and SNR 10–15. Presence ratio isn't the cause (D4) | Labels: judge low-count units on `rp_contamination` instead of the sliding test (O3). Sorter: extraction rules (D9), sorting the gasping epoch on its own, or a residual search for gasping-epoch spikes no existing template explains |
| O7 | Units whose footprint differs between epochs (19 KS4, 9 Lupin; F16, F17) and #361's remaining gasping difference (F15) | review them, starting with #187 (epoch filter in the review browser) |
| O8 | Does Kilosort 4 on the GPU give the same units on a rerun? | rerun `--redo sort` into a copy and compare |
| O10 | Which PCA should the population figures and slides use: Bush & Ramirez (comparable with their papers, but PC2 is unit #89, F18) or the soft-normalised one (F19)? | your decision |

## 9. How to reproduce

The whole chain, from the raw animal folder to the population figures, is in the
[repository README](../../README.md). The sorting step alone:

```bat
conda activate npxsort
cd sorting
git checkout <pipeline commit from run_summary.md>
python run_sorting.py configs\e097_intact_window.yaml
python summarize_run.py <outputs>\e097\intact_window_v2
python review_units.py <outputs>\e097\intact_window_v2 --time-offset-s 1740
```

Commits before 2026-10-22 have the pipeline in `sorting_v2/` with machine paths in the configs; from then on it is in
`sorting/`, and the configs' `${outputs}` etc. come from `configs/local_paths.yaml`.

- **Same software.** Package versions are in `run_summary.md`. Install from `environment/npxsort.yml` and
  `environment/npxsort-requirements.txt`.
- **Not bit-identical.** Lupin changes run to run (F6). Kilosort 4 on the GPU is untested (O8). Postprocessing
  is seeded from `postprocessing.seed` (0): random spike sampling always; noise levels, spike locations and
  quality metrics since D12. The current outputs predate D12.
- **Review files survive reruns.** `review/` is never overwritten, but unit ids are only stable if the sorter
  output is reused (`--redo postprocess` keeps it; `--redo sort` doesn't).

## 10. Change history

| Date | Change | Commit |
|---|---|---|
| 2026-10-19 | Record started; numbers from the `--redo postprocess` rerun at `d003f64`; D12 | `98e0c90` |
| 2026-10-22 | Repository cleanup: `sorting_v2/` renamed `sorting/`, analysis scripts moved to `analysis/`, machine paths moved to `configs/local_paths.yaml`; §9 updated. No setting or result changed | branch `repo-cleanup` |
| 2026-10-26 | D13, F18, F19, O10: the population PCA's PC2 is unit #89; a soft-normalised PCA added beside it | `main` |
