# From voltage to "good unit": every step, and where units drop out

This explains what happens to the e097 intact-window recording
(`outputs/e097/intact_window_v2`, run 2026-10-16) from the probe to the final labels. Every
number comes from the installed code (SpikeInterface 0.105.2, Kilosort 4.1.7, Lupin 2026.01) or
from that run's output files. File references are to the installed packages.

Spikes can be lost, split or demoted at six points:

1. preprocessing (filters that change waveform shape),
2. detection (thresholds and polarity),
3. clustering (minimum sizes, splits, merges),
4. the pipeline's own cleaning (duplicate spikes, redundant units),
5. bombcell labelling (noise / non_soma / mua / good),
6. cross-sorter matching (`found_by_both`).

---

## 1. Preprocessing

**Recording.**
- Probe: Neuropixels 2.0 (type 2013), one shank, 384 channels.
- Layout: two columns at x = 500 and 532 µm; rows every 15 µm from y = 0 to 2865 µm.
- Sampling rate: 29,999.8 Hz.
- Data: int16, 3.027 µV per bit (±0.62 V range / 2048 / gain 100).

**CatGT** (command line recorded in the `.ap.meta`):

| Option | What it does |
|---|---|
| `-startsecs=1740 -maxsecs=420` | Extracts 1740–2160 s of run g2. Add 1740 s to file time to get run time. |
| (default) | Corrects the ADC multiplexing time shift between channels. |
| `-apfilter=butter,12,300,9000` | 12th-order Butterworth band-pass, 300–9000 Hz. |
| `-loccar_um=75,300` | Local common-average reference: from each channel, subtracts the mean of the channels 75–300 µm away on the same shank. Channels closer than 75 µm are excluded, so the spike's own footprint isn't subtracted. |

**npxsort** (`npxsort/preprocess.py`):
- **Bad channels.** A channel is "dead" if its MAD noise is below 0.2× the probe median, and "noisy" above 5×. All 384 were good. The IBL coherence method isn't used, because CatGT's local reference makes it label active channels dead.
- **No drift correction.** `motion.enabled: false` in the config.
- **Saved unchanged.** The CatGT int16 values are written to `scratch/preprocessed`. Every quality metric, bombcell, Phy and `review_units.py` use exactly this binary.

**Inside the sorters.** Both sorters transform the data again before detecting spikes. The amplitudes they threshold are therefore in "whitened noise SD" units, not µV.

| | Kilosort 4 | Lupin |
|---|---|---|
| Extra filter | 3rd-order Butterworth high-pass at 300 Hz, applied zero-phase, on top of CatGT's 300 Hz (`preprocessing.py:126-154`) | 300–7000 Hz 2nd-order Bessel band-pass, zero-phase (set by the pipeline instead of Lupin's own filter + CMR) |
| Reference | Per-batch mean removed; no median CAR (`do_CAR: false`) | None |
| Whitening | Local ZCA whitening over the 32 nearest channels (~120 µm median radius). Covariance from 9 batches of 2 s. | Local ZCA whitening over channels within 100 µm. Covariance from 20 random 0.5 s chunks. |
| Drift correction | Off (`nblocks=0`) | Off |

Whitening makes noise roughly unit variance and decorrelates neighbouring channels. That is why both sorters' thresholds are in SD units and don't depend on µV gain.

---

## 2. Spike detection

### Kilosort 4 (`spikedetect.py`, `template_matching.py`)

1. **Universal templates learned from the data.**
   - On every 25th batch, Kilosort takes samples where the whitened signal exceeds |6| SD and is the largest |value| within ±4 channels and ±0.17 ms.
   - Clips are aligned on the absolute peak. **Either polarity counts.**
   - It extracts 6 principal components and 6 k-means temporal shapes. In this run 5 shapes have a negative trough and 1 has a positive peak.
2. **First detection pass.**
   - The data are matched against those shapes at 1149 virtual positions (383 depths × 3 columns), each with Gaussian spatial footprints 10–50 µm wide.
   - Kilosort takes the |score| (`spikedetect.py:155-157`), keeps local maxima within ±20 samples and ~127 µm, and accepts scores above **Th_universal = 9** SD.
   - Result: 1,204,051 spikes.
3. **Second detection pass with learned templates.**
   - After the first clustering, the 593 cluster templates are matched against the data and subtracted iteratively ("peeling").
   - A spike is accepted when ⟨template, data⟩/‖template‖ > **Th_learned = 8**.
   - Result: 2,554,264 spikes.

**Polarity:** Kilosort detects upward spikes. It uses |score| at detection, and learned templates carry the unit's own polarity. In the final output, 136 of 566 templates peak positive.

### Lupin (`peak_detection/matched_filtering.py`, `sorters/internal/lupin.py`)

1. **Noise.** MAD/0.6745 per whitened channel.
2. **Prototype waveform.**
   - A simple detector finds the first 10,000 negative threshold crossings (5× MAD, 50 µm, 1 ms exclusion).
   - Their median waveform, scaled so the trough is −1, is the prototype.
3. **Matched filter.**
   - The prototype is cross-correlated with every channel and pooled spatially at 5 virtual depths (0–120 µm).
   - A detection is a local maximum above **5 × the MAD of the filter output** (`detect_threshold: 5`). This is not 5× the voltage noise.
   - Within ±1.47 ms and 50 µm only the strongest detection survives. A smaller neuron firing close in time and space to a bigger one is therefore not detected.
4. All 1,601,479 peaks went to clustering. The cap is 5000 per channel × 384 = 1.92 M.

**Polarity:**
- With `peak_sign: neg`, only spikes that correlate *positively* with the negative prototype are detected (`matched_filtering.py:68-70`).
- An upward spike gives a negative correlation and is missed, unless its negative phase alone clears the threshold. It is then timed at that trough.
- Only 26 of Lupin's 729 templates are positive-dominant (Kilosort 4: 136 of 566). **Lupin under-detects upward units.**

**Reproducibility:** Lupin's detection thresholds are estimated from random chunks that are not seeded (`get_random_data_chunks` without a seed). Peak counts therefore change from run to run: the failed run found 1,847,456 peaks and the successful run 1,601,479. Lupin results will differ somewhat if you re-run it.

---

## 3. Clustering

### Kilosort 4 (`clustering_qr.py`, `swarmsplitter.py`, `template_matching.py`)

**Clustering within each region.**
- Spikes are grouped into 96 regions every 30 µm along the probe.
- In each region, PC features from nearby channels (60–84 dimensions) form a nearest-neighbour graph (10 neighbours, built on every 20th spike).
- Graph clustering starts from 200 k-means++ seeds and optimises a modularity-like score.
- The result is a tree, pruned top-down. A branch is split if its modularity ratio is < 0.2 or its 1-D bimodality score is ≥ 0.6.
- Result: 652 clusters (first pass), then 615 (final pass, after the learned-template detection).

**Merging.** 615 → 566 units.
- Clusters are visited from largest to smallest.
- A cluster can absorb a neighbour when:
  - the absorbing cluster passes Kilosort's autocorrelogram (ACG) refractory test;
  - their templates correlate ≥ 0.5;
  - their cross-correlogram (CCG) shows a refractory gap (R12 < 0.25, Q12 < 0.05).
- Two fragments that both fail the ACG test can never be merged.

**Kilosort's own "good" (KSLabel):** ACG refractory ratio R12 < 0.2 and Q12 < 0.2. This gave 275 units. It is not used downstream; the pipeline uses bombcell.

**Bug in Kilosort 4.1.7.** The refractory veto that should stop a refractory-clean neuron from being split never fires. `swarmsplitter.py:57` reads `len(st2 == 0)` where `len(st2) == 0` was meant. So amplitude changes, such as bursts or slow drift, can split one neuron into fragments with matching templates. Look for these as same-shape neighbours with interleaved spike times.

### Lupin (`clustering/iterative_isosplit.py`, `itersplit_tools.py`, `tools.py`)

**Seeding and splitting.**
- Each peak starts in the group of its detection channel (384 seed groups). Features are 5 SVD components on channels within 120 µm.
- Groups are split recursively, up to 3 levels, with isosplit on 4 local PCA features (channels within 60 µm).
- Groups with fewer than 25 peaks are never split, so they can hold two neurons.
- Sub-clusters with fewer than 10 peaks are discarded.

**Template cleaning.** A template is kept only if all of these hold:
- at least one channel has peak-to-peak / noise ≥ 4;
- its largest deflection is within ±0.2 ms of the alignment point;
- it has channels left after sparsifying.

**Merging.** Templates with L1 similarity > 0.8 (allowing ±0.3 ms lag) are merged. Merging is transitive, so A~B and B~C merges A, B and C.

**Low-firing cut.** Clusters with fewer than **41 peaks** (0.1 Hz × 420 s) are dropped *before* template matching. Their spikes are never recovered.

**Template matching (Wobble).** This found 3,289,853 spikes for 729 templates.
- A spike is accepted only if 2⟨T,x⟩ − ‖T‖² > **50** (whitened units²).
- For low-energy templates (‖T‖² ≈ 40) that means only spikes at least 1.1× the template size pass. 280 templates have ‖T‖² < 100.
- So small units lose their smaller spikes, which also distorts their amplitude-cutoff estimate.

**Final auto-merge.** 729 → 691 units.
- It merges neighbours (≤ 50 µm apart, template difference ≤ 0.35) whose merged spike train stays refractory.
- Only units with ≥ 100 spikes and contamination ≤ 0.2 are eligible.
- It removes spikes closer than 3 ms within a merged unit. **Genuine bursts with intervals under 3 ms lose spikes here.**

---

## 4. The pipeline's own cleaning (`npxsort/postprocess.py: clean_sorting`)

- **Duplicate spikes.** Spikes of the same unit within 0.3 ms are removed (Kilosort 4: 3,345 spikes; Lupin: none).
- **Redundant units.** When two units share ≥ 80% of their spikes, the one with more spikes is kept.
- Result: Kilosort 4 566 → 553 units, Lupin 691 → 685.

---

## 5. Labels: noise, non_soma, mua, good

**Which channel the metrics use.** Every per-unit metric and shape rule uses the unit's *main channel*.
- For Kilosort units, SpikeInterface takes it from Kilosort's own `templates.npy`: the channel with the largest peak-to-peak value. Those templates are in whitened space (`phykilosortextractors.py:527-537`).
- Whitening reshapes amplitudes across neighbouring channels, so this isn't always the largest channel in µV.
- In e097 it differs for 59 of 553 Kilosort 4 units (11%). Usually the main channel is a neighbour with about 95% of the amplitude, but for 8 units it has under 80%. Example: unit 434's metrics use ch 191 (203 µV peak-to-peak) rather than ch 240 (267 µV).
- Lupin's main channels come from the mean template and differ for 35 of 685 units, all by less than 20%.

**How the inputs are computed.**
- Labels come from bombcell as implemented in SpikeInterface (`curation/bombcell_curation.py`). The thresholds are in `<sorter>/bombcell/thresholds_used.json`.
- The template is the **mean of up to 500 random spikes** on the unit's main channel, from −1.0 to +2.0 ms, upsampled 10×.
- On that template, the "trough" is always the most prominent **negative** dip. The waveform is never flipped for upward units.

**Order of decisions:**

1. **noise.** Fail *any* noise rule. A NaN value counts as a fail.
2. Among the rest: **mua** if *any* MUA rule fails, otherwise **good**. NaN is ignored here, except for `sliding_rp_violation`, where NaN counts as a fail.
3. Among the rest: **non_soma** overrides both good and mua if the non-somatic rule holds.

Bounds are inclusive for passing: a value exactly on the threshold passes.

### Noise rules (pass conditions)

| Metric | Meaning | Passes if |
|---|---|---|
| num_positive_peaks | prominent (≥ 30% of max) positive peaks | ≤ 2 |
| num_negative_peaks | prominent troughs | ≤ 1 |
| peak_to_trough_duration | trough → next positive peak | 0.1–1.15 ms |
| waveform_baseline_flatness | baseline (−1 to −0.5 ms) wobble / spike size | ≤ 0.5 |
| peak_after_to_trough_ratio | repolarisation peak / trough | ≤ 0.8 |
| exp_decay | spatial decay of amplitude across channels; NaN if the fit fails | 0.01–0.1 per µm |

### Non-somatic rule

A unit is non_soma when:

> [ peak_before_width < 0.15 ms **and** trough_width < 0.2 ms ] **and**
> [ peak_before / trough > 3 **or** peak_before / peak_after > 3 ]
> **— or —** largest positive peak / trough > 0.8

A NaN width counts as "narrow", which pushes the unit toward non_soma.

`waveform_shape_criteria.md` illustrates these shape rules on real e097 templates. It traces the cutoffs to their sources (the authors' empirical choices, plus a cortical-culture study for the axonal rules) and checks them against brainstem literature.

### MUA rules (pass conditions)

| Metric | Meaning | Passes if |
|---|---|---|
| amplitude_median | median spike amplitude, peak channel | \|x\| ≥ 30 µV |
| snr | template peak / channel noise (MAD) | ≥ 5 |
| amplitude_cutoff | estimated fraction of spikes missing below threshold; NaN under 500 spikes | ≤ 0.2 |
| num_spikes | | ≥ 300 |
| sliding_rp_violation | contamination at 90% confidence, refractory periods 0.5–10 ms | ≤ 0.1; **NaN fails** |
| drift_ptp | depth drift, 60 s bins; NaN for low firing | ≤ 100 µm |
| isolation_distance | Mahalanobis isolation in PC space | ≥ 20 |
| l_ratio | contamination from other units in PC space | ≤ 0.3 |

`presence_ratio` was removed by your config.

### What actually decided the labels in this run

| | Kilosort 4 | Lupin |
|---|---|---|
| good | 52 | 52 |
| mua | 299 | 493 |
| non_soma | 73 | 87 |
| noise | 129 | 53 |

**MUA.** Two rules do almost all the work:
- **`sliding_rp_violation`** fails for 244 Kilosort 4 and 408 Lupin MUA units. Most of those values are NaN: 173 and 336. NaN means the test never reached 90% confidence. That happens with heavy contamination, but also simply with **too few spikes**. In 420 s, a unit with *zero* violations is NaN below about 410 spikes and cannot pass below about 720 spikes (about 1.7 Hz). Every unit under 410 spikes is MUA for this reason alone.
- **`l_ratio`** fails for 237 Kilosort 4 and 457 Lupin MUA units. The median l_ratio is 1.09 (Kilosort 4) and 3.03 (Lupin), against a threshold of 0.3. It is computed in a ~130-dimensional space (5 PCs × ~26 channels) from ≤ 500 spikes. In this dense recording the threshold looks poorly calibrated. `isolation_distance` never fails.
- **Single rule only.** 39 / 17 MUA units fail only the sliding refractory test, and 41 / 54 fail only l_ratio. These are the best candidates for reclassification.
- **Never fail here:** `amplitude_cutoff`, `drift_ptp` and `isolation_distance` don't fail for any unit in either sorter.

**Noise.**
- `peak_to_trough_duration` is the main trigger: 100 Kilosort 4 units (89 failing it alone) and 30 Lupin units.
- Then `exp_decay`, `peak_after_to_trough_ratio` and `num_negative_peaks`.
- 4 Kilosort 4 units are noise only because `exp_decay` is NaN (the spatial fit failed).

**Non_soma.** The usual route is `peak_before / peak_after > 3`. In most such units the "peak after" was a fallback value (no clear repolarisation peak was found), which inflates the ratio.

**Upward (positive-going) units can never be good or mua.**
- For a unit whose largest deflection is positive, the "trough" is its small undershoot. So `main_peak_to_trough_ratio` is > 1, which exceeds 0.8, and the unit is non_soma.
- More often it is labelled noise first, because:
  - `peak_to_trough_duration` is measured to some late maximum and exceeds 1.15 ms, or
  - `peak_after_to_trough_ratio` > 0.8.

| Polarity | Kilosort 4 good / mua / noise / non_soma | Lupin good / mua / noise / non_soma |
|---|---|---|
| negative-going | 52 / 299 / 33 / 40 | 52 / 493 / 34 / 80 |
| **positive-going** | **0 / 0 / 96 / 33** | **0 / 0 / 19 / 7** |

Positive units are not small: their median SNR is 4.8 (Kilosort 4) and 5.5 (Lupin). Some may be axonal or dendritic signals, which is what "non_soma" is meant to catch. But somatic spikes recorded from the far side of a cell can also be upward. **Review them by eye** with the viewer's polarity filter.

`bombcell/labeling_results_narrow.csv` has a `passed` column that is not the labelling logic. It marks every NaN as failed, and it ignores `abs` on amplitude_median, so every negative-going unit shows "failed". Use the values from that file, not its pass flags. The review viewer shows the real logic.

---

## 6. Cross-sorter matching (`npxsort/compare.py`, SpikeInterface `compare_two_sorters`)

- **Noise units are excluded on both sides** before comparing. A unit whose counterpart was labelled noise therefore has no match.
- **Spike matching.** Two spikes match if they are within **0.4 ms** of each other.
- **Agreement** between unit A and unit B = matched / (n_A + n_B − matched). This is 1 for identical trains and penalises both missed and extra spikes.
- **Pairing.** Units are paired one-to-one with the Hungarian algorithm on the agreement matrix. A pair counts as **found_by_both** when its agreement is ≥ 0.5.
- **Split units.** If one sorter split a neuron in two, only one fragment can be paired, and its agreement is lowered by the other fragment's spikes. The viewer shows each unit's *best* match even below 0.5, and how many spikes the two units share.

---

## 7. Suggested review categories

Each of these is one filter combination in `review_units.py`:

| Question | Filter |
|---|---|
| Are upward units real somatic neurons? | polarity = positive-going |
| Is the sliding refractory-period test demoting clean low-rate units? | label = mua, criterion = sliding_rp_violation (check the ISI histogram) |
| Is l_ratio demoting well-isolated units? | label = mua, criterion = l_ratio (look at the traces and nearby units) |
| Is `peak_to_trough_duration` throwing away real units? | label = noise, criterion = peak_to_trough_duration |
| Do units found by only one sorter look real? | label = good, match = not found by both |
| Are good units actually split or merged? | label = good; look for nearby units with the same template, or a matched unit with many unshared spikes (`U` key) |

Record a verdict (keys 1–6) and a note per unit. The verdicts in
`review/<sorter>_unit_review.tsv` can then be tabulated against the failed criteria to decide
which thresholds to change.

---

## 8. Phy

Phy 2.1.0rc1 is installed in the `cibrrig` environment:

```bat
conda activate cibrrig
phy template-gui C:\Users\lpascu\resp_analysis\outputs\e097\intact_window_v2\kilosort4\phy\params.py
```

**Unit numbers differ.** Phy renumbers clusters, so Phy `cluster_id` is not the `unit_id` in `units.tsv` or the viewer. For example, Phy cluster 12 is unit 13. Use the **`si_unit_id`** column in Phy's cluster table to cross-reference. The `group` column is pre-filled from bombcell (non_soma shown as mua), and every metric is a column. The filter box takes expressions such as `bombcell == "noise"` or `snr > 5`.

**Views:**
- **WaveformView:** spike waveforms across nearby channels.
- **FeatureView:** PC space. Separated clouds mean a split is justified; one continuous cloud argues for merging.
- **AmplitudeView:** amplitude over time. Look for drift, sudden jumps (a sign of a split or merge), and a cut-off lower edge (missed spikes).
- **CorrelogramView:** ACG and CCG for refractoriness.
- **SimilarityView:** the most similar clusters, which are merge candidates.

**TraceView** shows the voltage on all channels, with the selected clusters' spikes coloured. It is useful but awkward for this purpose: it shows the whole probe and is navigated by keyboard.

**Default shortcuts** (Phy 2.1):

| Keys | Action |
|---|---|
| `space` / `shift+space` | next / previous similar cluster |
| `↓` / `↑` | next / previous cluster in the list |
| `alt+g` / `alt+m` / `alt+n` | label as good / mua / noise |
| `g` / `k` | merge / split (draw a lasso in FeatureView first) |
| `alt+pgdown` / `alt+pgup` | TraceView: next / previous spike of the selection |
| `alt+←` / `alt+→` | TraceView: scroll |
| `alt+s` | TraceView: highlight the selected clusters' spikes |
| `ctrl+wheel` | TraceView: trace height |
| `o` | WaveformView: overlap waveforms |
| `ctrl+z` | undo |

Phy saves its changes to `cluster_group.tsv` and `spike_clusters.npy` in the `phy` folder. Those changes are not read back by the pipeline.

For flipping quickly through units on just their peak channels, use `review_units.py` instead (see the README).
