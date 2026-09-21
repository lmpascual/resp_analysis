# Iterative Sliding-Window Spike Sorting & Cross-Window Consensus Pipeline Guide

This guide provides step-by-step instructions for running the **Iterative Sliding-Window Spike Sorting Pipeline** with **Cross-Window Unit Agreement Consensus**, **Customizable Allen Single-Unit QC Criteria**, **Automated Run Journaling**, and **Multi-Recording Support**.

---

## 📖 1. Why Sliding-Window Spike Sorting?

When sorting high-density Neuropixels probes (384 channels) globally:
- SVD clustering often over-splits dense multi-unit background into dozens of fragile sub-templates.
- During synchronous network bursts, overlapping templates can trigger spatial collision cancellations in deconvolution matching pursuit (such as Lupin's OMP-SVD), producing artificial blank dropout gaps in rasters.

**The Sliding-Window Solution**:
1. Slices the probe into overlapping channel sub-bands (defaults to all **384 channels**, sorted 50 channels at a time with 20-channel overlaps: `0–50`, `30–80`, `60–110`, ..., `330–384`).
2. Sorts each 50-channel slice independently, eliminating multi-template collisions.
3. Compares spike coincidence in the 20-channel overlap regions across adjacent windows.
4. Units with **Agreement $\ge 50\%$** ($90\%\text{–}100\%$ observed) are validated as **Cross-Window Confirmed Single Units**.
5. Preserves non-overlapping units (e.g. channels 0–29 and 350–384) and evaluates Allen Single-Unit Quality Metrics across all clusters.
6. Automatically creates a **Run Journal** (`run_journal.md`) documenting all settings, window statistics, and QC results for the session.

---

## 🚀 2. Quick-Start Execution

### Option A: Run Interactively with the Configuration Wizard (Recommended)

To configure parameters for any recording interactively:

```powershell
.\.venv\Scripts\python.exe scripts/configure_pipeline_wizard.py
```

- **Interactive Prompts**:
  1. **Recording & Output Paths**: Enter raw/preprocessed ephys directory and output root folder.
  2. **CatGT Preprocessing**: AP filter (default `butter,12,300,9000`), local CAR `loccar_um` (default `[75.0, 300.0]`), `start_secs`, and `max_secs`.
  3. **Motion Correction**: Prompts whether to enable motion drift estimation, displays all default motion parameters for verification.
  4. **Sliding-Window Settings**:
     - Channel Selection Range: Defaults to **all 384 channels (`0-384`)**, or enter custom range (e.g. `0-330`).
     - Window Size: Defaults to `50` channels (or custom).
     - Overlap: Defaults to `20` channels (or custom).
     - Agreement Threshold: Defaults to `0.50` (50% mutual spike coincidence, or custom e.g. `0.90`).
  5. **Allen Single-Unit QC Criteria**:
     - Displays all default thresholds:
       - Max ISI Refractory Violation Rate (`qc_isi_cutoff`): `0.10` (< 10% violations)
       - Min Presence Ratio (`qc_presence_ratio`): `0.90` (Active in $\ge 90\%$ of recording)
       - Max Amplitude Cutoff (`qc_amp_cutoff`): `0.10` (< 10% truncated distribution)
       - Min Firing Rate (`qc_min_fr_hz`): `0.10` Hz
       - Refractory Period Window (`isi_threshold_s`): `1.5` ms
     - Prompts: `Keep these recommended Allen Single-Unit QC criteria? [Y/n]`. If no, allows customizing each threshold.
  6. **Sorter Selection**: `1 = Lupin (Recommended)`, `2 = MountainSort 5`, `3 = Kilosort 4`, `4 = All 3`.

---

### Option B: Execute the Pipeline Directly via CLI

Run the master sliding-window pipeline on any recording:

```powershell
# 1. Run full sliding-window pipeline on your recording
.\.venv\Scripts\python.exe scripts/run_sliding_window_pipeline.py --rec-dir "Z:\path\to\your_recording" --output-dir "./output_my_experiment" --sorter lupin

# 2. Or launch with the interactive config wizard first
.\.venv\Scripts\python.exe scripts/run_sliding_window_pipeline.py --interactive
```

#### CLI Command-Line Arguments:
| Argument | Description | Default |
| :--- | :--- | :--- |
| `--rec-dir <path>` | Path to SpikeGLX recording (`.bin`/`.meta`) | From config |
| `--output-dir <path>` | Root output folder for this recording session | From config |
| `--config <path>` | Path to configuration JSON | `pipeline_config.json` |
| `--sorter <name>` | Sorter algorithm (`lupin`, `mountainsort5`, `kilosort4`) | `lupin` |
| `--interactive` | Launch interactive wizard before sorting | `False` |
| `--max-windows <N>` | Limit execution to first $N$ windows (for quick tests) | `None` (All) |

---

## 📊 3. Output Directory Structure & Run Journaling

When executed, the pipeline creates an isolated output folder with an **Automated Run Journal**:

```
<output_dir>/sliding_window_pipeline/<sorter_name>/
├── consensus_master/                          # Master Unified Sorting Output
│   ├── run_journal.md                         # 📝 Automated Run Journal & Execution Summary
│   ├── run_journal.json                       # Machine-readable parameters & metrics
│   ├── master_consensus_qc_summary.csv        # Master cluster QC table (metrics, SUA status)
│   ├── cross_window_pairwise_matches.csv      # Pairwise window agreement % and spike counts
│   ├── cluster_info.tsv                       # TSV formatted table for Phy GUI
│   ├── lupinslidingconsensusmaster_qc_interactive_raster.html  # 5-Panel Synchronized Interactive Raster
│   ├── lupinslidingconsensusmaster_qc_raster_preview.png       # Publication-ready static preview
│   ├── alf/                                   # Unified ALF numpy arrays
│   │   ├── spike_times.npy                    # De-duplicated global spike times (seconds)
│   │   ├── spike_clusters.npy                 # Unified cluster IDs
│   │   └── cluster_info.tsv                   # Metadata
│   └── phy/                                   # Complete Phy GUI Bundle
│       ├── params.py                          # Phy configuration
│       ├── templates.npy                      # Mean unit waveform templates
│       ├── amplitudes.npy                     # Spike peak amplitudes
│       └── similar_templates.npy              # Template similarity matrix
│
└── windows/                                   # Individual Sub-Window Sortings
    ├── win_00_ch0_50/                         # Window 0 (Channels 0–49, Depths 0–375 µm)
    ├── win_01_ch30_80/                        # Window 1 (Channels 30–79, Depths 225–600 µm)
    ├── win_02_ch60_110/                       # Window 2 (Channels 60–109, Depths 450–825 µm)
    └── ...
```

---

## 🔍 4. Inspecting Visualizations & Phy GUI

### 1. Synchronized 5-Panel Interactive HTML Raster

Open the master consensus interactive dashboard in your default browser:

```powershell
Start-Process "<output_dir>\sliding_window_pipeline\lupin\consensus_master\lupinslidingconsensusmaster_qc_interactive_raster.html"
```

- **5 Synchronized Panels**:
  1. **Respiratory Flow Rate Trace** (Plethysmography airflow)
  2. **Diaphragm EMG Trace** (100–2000 Hz bandpass motor burst drive)
  3. **Electrocardiogram / ECG Trace** (Cardiac QRS complexes)
  4. **Population Instantaneous Firing Rate (PSTH)** (50 ms binned rate)
  5. **Depth-Resolved Single-Unit Spike Raster** (Passing single units across depth)

---

### 2. Launching the Interactive Phy GUI

To inspect individual spike waveforms, cross-channel templates, autocorrelograms, and amplitudes:

```powershell
cd '<output_dir>\sliding_window_pipeline\lupin\consensus_master\phy'
phy template-gui params.py
```

---

## ⚡ 5. Fast Dynamic QC Re-Evaluation & Epoch Presence Tracking

If you want to adjust QC thresholds (such as lowering `presence_ratio` for burst-active/transient neurons) without re-running the heavy spike sorting algorithms:

```powershell
.\.venv\Scripts\python.exe scripts/recompute_qc_and_consensus.py --isi-cutoff 0.10 --presence-ratio 0.05 --min-fr 0.05
```

### Features:
1. **Instant Execution (~1-2 seconds)**: Operates directly on the saved `.npy` and `.tsv` files.
2. **Epoch Presence Profiling**: Divides the session into 30s epochs to analyze temporal activation vs quiescence.
3. **Biological State vs Probe Drift Diagnostics**: Generates `master_temporal_epoch_and_presence_diagnostics.png` (Rasters, Temporal Activation Heatmap, and ISI vs Presence quality landscape).
4. **Auto-Regenerates Dashboard**: Updates `consensus_master/` and the synchronized 5-panel interactive HTML raster.

---

## ⚙️ 6. Configuration Reference (`pipeline_config.json`)

```json
{
    "raw_data_dir": "./raw_data",
    "output_dir": "./output",
    "catgt": {
        "enabled": true,
        "apfilter": "butter,12,300,9000",
        "loccar_um": [75.0, 300.0],
        "start_secs": 0.0,
        "max_secs": 60.0,
        "t_shift": true
    },
    "motion": {
        "enabled": false,
        "preset": "kilosort_like",
        "interpolate_method": "kriging"
    },
    "sliding_window": {
        "enabled": true,
        "channel_range": [0, 384],
        "window_size": 50,
        "overlap": 20,
        "agreement_threshold": 0.50
    },
    "sorters": {
        "enabled_sorters": ["lupin"],
        "qc_isi_cutoff": 0.10,
        "qc_presence_ratio": 0.90,
        "qc_amp_cutoff": 0.10,
        "qc_min_fr_hz": 0.10,
        "isi_threshold_s": 0.0015
    }
}
```
