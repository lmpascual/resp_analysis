# Using Phy with this pipeline

Phy 2.1.0rc1 is installed in the `cibrrig` environment. Every shortcut below was checked against that
version's source (`...\envs\cibrrig\Lib\site-packages\phy`).

## Before you start: what Phy changes, and what it doesn't

- **Phy edits stay in the `phy` folder.** Merges, splits and labels are saved there as
  `spike_clusters.npy`, `cluster_group.tsv` and `cluster_info.tsv`.
- **Nothing else reads them back.** The pipeline (`run_sorting.py`, `units.tsv`, ALF), the review
  browser (`review_units.py`) and the inspection workbook don't see Phy edits.
- **Re-exporting overwrites them.** Re-running the pipeline's `export` stage (for example
  `--redo postprocess`) rewrites the `phy` folder. Copy the folder first if you've curated in it.
- **To remove spikes for the analyses, use the config instead.** `curation.remove_spikes` (see the
  README and `npxsort/remove.py`) removes a unit's spikes in a time window inside the pipeline, so
  `units.tsv`, ALF, the browser and the next Phy export all have the change, and it survives reruns.
- **Unit numbers differ.** Phy renumbers clusters at export, so Phy's `cluster_id` isn't the
  `unit_id` in `units.tsv` or the review browser. The **`si_unit_id`** column gives the mapping.
  After you split or merge, the resulting clusters get new ids and have no `si_unit_id`.
- **It needs the scratch binary.** `params.py` points at `scratch_dir/preprocessed`, so keep that
  folder while you use Phy.

## 1. Launch

```bat
conda activate cibrrig
phy template-gui C:\Users\lpascu\resp_analysis\outputs\e097\intact_window_v2\kilosort4\phy\params.py
```

Use `...\lupin\phy\params.py` for Lupin.

The first launch builds a cache (a `.phy` folder next to `params.py`), which takes a few minutes for
550–700 units. Later launches are fast.

## 2. The window

| View | What it shows | Use it for |
|---|---|---|
| **ClusterView** (table, top left) | One row per cluster: `id`, `group`, `n_spikes`, `fr`, `ch`, `depth`, `si_unit_id`, `bombcell` and every metric column. Click a header to sort. | Picking clusters; filtering |
| **SimilarityView** | Clusters most similar to the selected one | Merge candidates |
| **WaveformView** | Individual spikes (grey) and the mean on nearby channels | Shape, consistency, contamination |
| **FeatureView** | PCA features of the selected clusters on two channels, plus feature vs time | Splitting; checking separation |
| **AmplitudeView** | Amplitude of every spike against time | Drift, dropouts, two amplitude bands, a cut-off lower edge |
| **CorrelogramView / ISIView** | Auto- and cross-correlograms, ISI histogram | Refractoriness; merge decisions (a shared refractory dip) |
| **FiringRateView** | Rate over time | Epoch-dependent firing |
| **TraceView** | Voltage on all channels, with the selected clusters' spikes coloured | Checking spikes on the trace |
| **ProbeView** | Where the selected clusters sit on the probe | Location |

- **`group`** is pre-filled from bombcell: good, mua, noise. `non_soma` is shown as mua; the real
  label is in the `bombcell` column.
- **Filter box** (top of ClusterView): it takes expressions over the column names. Examples:
  - `bombcell == "mua" && sliding_rp_violation > 0.1`
  - `si_unit_id == 355`
  - `snr > 5 && n_spikes > 1000`

  Press `esc` to clear the filter.

## 3. Reviewing a unit

1. **Find it.** Click its row in ClusterView, or filter `si_unit_id == <id>` to find a unit you saw
   in the review browser.
2. **WaveformView.** The waveforms should be consistent: a single shape, no second population.
   `o` overlaps the channels, `m` toggles the mean, `w` switches between waveforms and templates,
   and `ctrl+wheel` changes the box size.
3. **AmplitudeView.** Look for one continuous band. Two bands, a jump, or a band that disappears and
   reappears suggest two neurons or drift. `a` cycles amplitude types. `alt+click` at a time moves
   TraceView there.
4. **CorrelogramView.** A real single unit has an empty centre (refractory period).
5. **FeatureView.** One compact cloud means one unit. To compare channels, `ctrl+click` a channel's
   box in WaveformView to use it for FeatureView's first axis, or `ctrl+right click` for the second.
6. **TraceView.** `alt+pgdown` / `alt+pgup` jump to the next / previous spike of the selection,
   `alt+left` / `alt+right` scroll, `ctrl+wheel` sets trace height, and `alt+s` highlights the
   selected spikes.
7. **Label it** with `alt+g` (good), `alt+m` (mua) or `alt+n` (noise).
8. **Move on.** `down` / `up` go to the next / previous cluster in the table. `space` /
   `shift+space` step through similar clusters.

## 4. Splitting

1. Select the cluster.
2. In **FeatureView** (or AmplitudeView), `ctrl+click` around the spikes to separate, building a
   polygon. `ctrl+right click` closes the lasso.
3. Press **`k`**. Phy creates new clusters: the lassoed spikes, and the rest.
4. Check both new clusters in WaveformView and CorrelogramView. `ctrl+z` undoes the split.

**Worked example: the positive-going gasping unit in KS4 #361.** About 27 large positive spikes on
ch 148/150 at 103–110 s were absorbed into #361.
- Filter `si_unit_id == 361`.
- In AmplitudeView, look at 100–112 s; in FeatureView use ch 148 and ch 150 (`ctrl+click` /
  `ctrl+right click` their boxes in WaveformView). The gasping spikes form a separate group.
- Lasso them and press `k`.

You don't need to do this for e097: the `gasp_pos_ch148` rule in the config already pulls these spikes
into their own unit (`extracted_as` in `units.tsv`). The steps above are how you'd handle the next case
by hand.

## 5. Merging

1. Select one cluster, then a candidate in SimilarityView (or `ctrl+click` a second row in ClusterView).
2. Merge only if the waveforms match **and** the cross-correlogram shows the same refractory dip as the
   autocorrelograms.
3. Press **`g`**. `ctrl+z` undoes it.

## 6. Saving

Press **`ctrl+s`**. Phy also asks on close. Saving writes `spike_clusters.npy`, `cluster_group.tsv` and
`cluster_info.tsv` in the `phy` folder.

## 7. Checking whether a unit changes between epochs (e.g. KS4 #355)

Phy has no notion of epochs. Its WaveformView mixes spikes from the whole recording, so it can't show
whether the gasping-epoch spikes come from the same neuron. You can:
- watch AmplitudeView and FiringRateView for a change at the epoch boundary;
- use TraceView at a baseline time and a gasping time (`alt+click` in AmplitudeView) and compare the
  spikes by eye.

For the comparison itself use the review browser, which overlays the unit's mean waveform per epoch and
lists `epoch_stability.tsv` (footprint cosine, waveform correlation, verdict). For #355 every epoch has
the same footprint (cosine ≥ 0.998, waveform r ≥ 0.993), and its most similar other unit is clearly
different (0.955). That supports one neuron changing its firing rather than another neuron taking over.

## 8. Using Phy's result in analysis

Phy keeps spike times in samples. Cluster labels are in `cluster_info.tsv` (written on save) or
`cluster_group.tsv`.

```python
import numpy as np, pandas as pd
phy = r"C:\Users\lpascu\resp_analysis\outputs\e097\intact_window_v2\kilosort4\phy"
fs = 29999.821974390747
t = np.load(phy + r"\spike_times.npy").ravel() / fs + 1740.0       # raw-run time, s
clu = np.load(phy + r"\spike_clusters.npy").ravel()
groups = pd.read_csv(phy + r"\cluster_group.tsv", sep="\t", index_col="cluster_id")["group"]
good_trains = {c: t[clu == c] for c in groups.index[groups == "good"]}
```

## Shortcut sheet (Phy 2.1.0rc1)

| Keys | Where | Action |
|---|---|---|
| `down` / `up` | any | next / previous cluster in the table |
| `space` / `shift+space` | any | next / previous similar cluster |
| `alt+g` / `alt+m` / `alt+n` | any | label good / mua / noise |
| `g` / `k` | any | merge selected / split lassoed |
| `ctrl+z` / `ctrl+y` | any | undo / redo |
| `ctrl+s` | any | save |
| `esc` | ClusterView | clear the filter |
| `ctrl+click`, `ctrl+right click` | FeatureView, AmplitudeView | add lasso point, close lasso |
| `ctrl+click` / `ctrl+right click` a channel | WaveformView | use it for FeatureView's first / second axis |
| `o`, `m`, `w` | WaveformView | overlap channels, mean waveforms, waveforms/templates |
| `a`, `alt+click` | AmplitudeView | amplitude type; move TraceView to that time |
| `alt+pgdown` / `alt+pgup` | TraceView | next / previous spike of the selection |
| `alt+left` / `alt+right`, `shift+alt+left/right` | TraceView | scroll, jump |
| `ctrl+wheel`, `alt+s` | TraceView | trace height; highlight selected spikes |
| `ctrl+wheel`, `alt+wheel` | CorrelogramView | window size, bin size |
