# sorting_v2 — Neuropixels spike sorting, rebuilt

A clean replacement for the old `src/pipeline/sorting` code. It imports nothing from the old
pipeline. It runs **Kilosort 4** and **Lupin** on the same drift-corrected data. It computes
**real** quality metrics from the waveforms, labels units with **bombcell**, and reports which
units both sorters found.

## What changed from the old pipeline

| Old pipeline | sorting_v2 |
|---|---|
| "Lupin" was SpyKING-CIRCUS 2 with clustering silently skipped when >500k peaks → one unit per channel | Real Lupin sorter (SpikeInterface 0.105.2) and Kilosort 4, run unmodified |
| No drift correction (comment claimed Kriging) | DREDGE drift estimate once, shared by both sorters; `motion.png` to check it |
| SNR and amplitude computed from spike count, amplitude cutoff hardcoded, depth = list position | All metrics from SpikeInterface on the preprocessed voltage; depth from unit localisation |
| "Good" = hardcoded ISI ≤ 0.5; config thresholds ignored | Bombcell labels (noise / mua / good / non_soma), thresholds saved with the results |
| NumPy/sklearn functions replaced program-wide with PyTorch to dodge crashes | No patches. `check_env.py` tests the math libraries so a broken install fails up front |
| Errors swallowed with `except: pass` | Errors stop the run; figure failures are logged with a traceback |

## Pipeline

1. **preprocess**: load the CatGT output → detect bad channels (dead / noisy / outside brain) and
   remove them → estimate drift with DREDGE and interpolate → save one int16 binary to scratch.
   For raw SpikeGLX input it also does phase shift, 300 Hz high-pass and median CMR.
2. **sort**: Kilosort 4 (GPU) and Lupin on that binary. The data was already referenced and
   drift-corrected, so Kilosort 4 runs with `do_CAR=False` and `do_correction=False`. Lupin gets
   a 300–7000 Hz band-pass plus its own whitening, and its internal motion correction is off.
   Every parameter actually used is written to `<sorter>/sorter_params_used.json`.
3. **postprocess**: drop excess spikes and duplicate spikes (< 0.3 ms), and remove redundant
   units (≥ 80% shared spikes). Then build a SortingAnalyzer: waveforms, templates, amplitudes,
   locations, correlograms, PCA. Quality metrics: SNR, amplitude median and cutoff, refractory
   violations (ISI, RP, sliding RP), presence ratio, drift, sd ratio, isolation distance, L-ratio
   and d′. Template metrics are computed too. Bombcell then labels each unit. Likely over-split
   pairs go to `merge_suggestions.csv` for review; they are never merged automatically.
4. **compare**: matches non-noise units between the two sorters by spike-time agreement (0.4 ms).
   Adds `match_<other>`, `agreement` and `found_by_both` columns to each unit table.
5. **export**: `units.tsv`, ALF arrays, a Phy folder with bombcell labels pre-filled as Phy groups,
   and per-unit report figures.

## Setup (once, on the sorting PC)

```bat
cd C:\Users\lpascu\resp_analysis\sorting_v2
conda env create -f environment.yml
conda activate npxsort
:: Install the CUDA build of torch FIRST. Get the exact command for Windows + pip + your CUDA
:: version from pytorch.org ("Get Started"); `nvidia-smi` shows the newest CUDA your driver supports.
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python check_env.py
```

`check_env.py` must print `Environment OK.` If the linear-algebra check fails, the NumPy BLAS
build is the problem (the source of the old OpenBLAS crash). Switch to MKL with
`conda install -c conda-forge numpy "libblas=*=*mkl"`, then re-run the check. If only the
process-pool check fails, set `job_kwargs.pool_engine: thread` in the config.

Phy works best in its own environment (you already have one). Point it at
`<output>/<sorter>/phy/params.py`.

## Running

```bat
conda activate npxsort
cd C:\Users\lpascu\resp_analysis\sorting_v2
:: Optional: check drift first
python run_sorting.py configs\e097_intact_window.yaml --stop-after preprocess
:: Look at outputs\e097\intact_window_v2\preprocessing\motion.png, then run the rest:
python run_sorting.py configs\e097_intact_window.yaml
```

Stages resume: re-running skips finished stages. `--redo postprocess` re-runs postprocessing and
everything after it, e.g. after changing curation thresholds. Each run writes a log to
`<output>/logs/`, the fully resolved config to `config_used.yaml`, and package versions, GPU,
timings and label counts to `run_manifest.json`.

For a new recording, copy `configs/e097_intact_window.yaml` and edit the paths. Any setting from
`npxsort/defaults.yaml` can be overridden there. Unknown keys are an error, so a typo can't
silently do nothing.

## Outputs

```
<output_dir>/
  preprocessing/   bad_channels.csv, motion.png, motion_summary.json, motion/, preprocessing_info.json
  kilosort4/ and lupin/
    units.tsv                 one row per unit: label, peak channel, x/depth (um), every metric, cross-sorter match
    alf/                      spike_times.npy (s), spike_clusters.npy, units.tsv
    phy/                      open in Phy; cluster groups = bombcell labels (non_soma shown as mua)
    report/                   per-unit summary figures
    bombcell/                 per-metric pass/fail tables, thresholds_used.json
    merge_suggestions.csv     candidate over-splits to review in Phy
    analyzer/                 SpikeInterface SortingAnalyzer (si.load_sorting_analyzer)
    sorting_raw/              sorter output before cleaning
    sorter_params_used.json, postprocessing_report.json
  comparison/      <sorter>_matches.tsv, agreement_scores.csv, agreement_matrix.png
  run_manifest.json, config_used.yaml, logs/
```

**Which units to analyse:** start with `label == "good"`. Units with `found_by_both == True` are
the most trustworthy. Units labelled `mua` are real spikes but not well isolated. `noise` units
are mostly not spikes, **except positive-going units**: bombcell's waveform rules can never label
a unit whose largest deflection is upward `good` or `mua` (see `docs/unit_labels_explained.md`).
Check a sample of each label with `review_units.py` or Phy before relying on the thresholds.

## Reviewing units

```bat
conda activate npxsort
python review_units.py C:\Users\lpascu\resp_analysis\outputs\e097\intact_window_v2 --time-offset-s 1740
```

This opens a browser page (served from 127.0.0.1 only; keep the console window open) that shows,
for one unit at a time, the preprocessed voltage of its highest-amplitude channels with its spike
times marked, ISI violations, the best-matching unit from the other sorter, nearby units' spikes,
amplitude over the whole recording, the template, the ISI histogram, and every bombcell criterion
with its value. Windows from 50 ms to the whole recording; it reads the scratch binary on demand.
Filter by label, failed criterion, polarity, cross-sorter match or review status; press `?` for
keys. Spike dots and the amplitude overview use each unit's largest channel in µV. Click a
channel's label to plot that channel instead. The sorter's main channel, on which all metrics
were computed, is tagged; for Kilosort it can differ (see `docs/unit_labels_explained.md`).

Two per-spike flags help find contaminated units without scrolling every unit. On first launch they
take about 25 s to compute; after that they're cached in `scratch_dir/review_cache`.
- **Footprint mismatch** (purple diamond, key `F`): the spike's amplitude pattern across the unit's
  top channels differs from the template's by more than 50% after scaling, and by more than 6 noise
  SDs. Typically a neighbouring neuron's spike assigned to this unit.
- **Duplicate** (dark ring, key `X`): a nearby unit (largest channels within 100 µm) has a spike within
  0.1 ms. This is only counted for unit pairs that coincide at 0.1 ms at least 4× more often than at
  0.5–2.5 ms, so synchronous firing isn't counted; what remains is the same event assigned to two units.

The unit list shows the counts (F, D). You can filter and sort by them. Verdicts and notes are saved to `<output>/review/<sorter>_unit_review.tsv`.
`docs/unit_labels_explained.md` explains every processing step and labelling rule.
`docs/waveform_shape_criteria.md` illustrates the waveform-shape rules (noise / non_soma), where
their cutoffs come from, and why they may not suit brainstem units.

**Times:** ALF spike times are seconds from the start of the sorted file. For e097 that's the
CatGT extract, so add 1740 s for raw-run time.

**Phy needs the scratch binary:** Phy's `params.py` points at `scratch_dir/preprocessed`. Keep
that folder while you curate. The analyzer and unit tables don't need it.

## Settings worth knowing

- `curation.threshold_overrides`: changes bombcell thresholds, e.g.
  `{mua: {snr: {greater: 4}}}`. `null` drops a metric from labelling. The e097 config drops
  `presence_ratio`, because respiratory units can fall silent during apnea and shouldn't be
  demoted for it.
- `sorters.<name>.params`: any Kilosort 4 or Lupin parameter (e.g. Kilosort `Th_universal`,
  `Th_learned`; Lupin `detect_threshold`). Asking a sorter to correct drift while
  `motion.enabled` is true is rejected, so the data can't be corrected twice.
- `motion.enabled: false`: skips SpikeInterface drift correction, and each sorter then corrects
  drift itself.
- `input.source: spikeglx_raw`: sorts raw SpikeGLX instead of CatGT output, with all
  preprocessing done here.

## Testing

`tests/simulate_and_test.py` builds a simulated drifting Neuropixels recording with known spike
trains. It runs the whole pipeline on it and scores each sorter against the truth. It also
checks that bombcell's `good` units really are single true neurons. Run it once on the sorting PC:

```bat
python tests\simulate_and_test.py --workdir C:\SCRATCH_neuropixels\npxsort_test
```

Results from development (Linux, CPU, simulated 128-channel Neuropixels with a ±15 µm zigzag drift):

- Drift: DREDGE recovered the simulated drift (estimated range 30 µm vs. 30 µm true).
- 60 s, both sorters: 17 of 18 units labelled `good` were single true neurons (agreement ≥ 0.8).
  `noise` units had a median agreement of 0.001 with any true neuron. Many true neurons were
  labelled `mua`, mainly because the sliding refractory-period test needs more spikes than 60 s
  provides.
- 300 s, Lupin only: 17 of 17 `good` units were true neurons. They covered 17 of the 21 neurons
  Lupin detected well. `mua` units had a median agreement of 0.05 and `noise` 0.001.

Recordings as long as e097's (420 s) behave like the 300 s case. Very short windows will label
real neurons `mua` more often.

## Notes

- SpikeInterface 0.105.2's `dredge` motion preset passes an invalid `batching_mode='offline'`.
  The pipeline swaps in DREDGE's default (`'full'`) and logs a warning saying so
  (`preprocess._estimate_motion_kwargs`).
- `input.source: catgt` and `si_folder` assume the data is already common-referenced. If you
  run CatGT without `-loccar`/`-gblcar`, use `spikeglx_raw` instead.
