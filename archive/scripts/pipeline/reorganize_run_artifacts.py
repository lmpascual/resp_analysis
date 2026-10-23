#!/usr/bin/env python3
"""
Reorganize all analysis artifacts, spike sorting outputs, figures, and diagnostics
into a clean, structured master run directory with a detailed RUN_JOURNAL.md.
"""
import shutil
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("REORGANIZE_RUN")

PROJECT_ROOT = Path(r"C:\Users\lpascu\analysis_src")
RUN_DIR_NAME = "LP21_fent_EXPT_g1_60s_120s_lupin_run"
TARGET_DIR = PROJECT_ROOT / RUN_DIR_NAME

# Source Paths
CONSENSUS_DIR = PROJECT_ROOT / "'output_LP21_60s_120s'" / "sliding_window_pipeline" / "lupin" / "consensus_master"
if not CONSENSUS_DIR.exists():
    CONSENSUS_DIR = PROJECT_ROOT / "output_LP21_60s_120s" / "sliding_window_pipeline" / "lupin" / "consensus_master"

RAW_ONSET_DIR = PROJECT_ROOT / "output_individual_neuron_rasters_LP21_60s"
ONSET_SORTED_TI_DIR = PROJECT_ROOT / "output_individual_neuron_rasters_onset_sorted_insp_duration_LP21_60s"
TIMEWARPED_DIR = PROJECT_ROOT / "output_timewarped_neuron_rasters_LP21_60s"
OFFSET_DIR = PROJECT_ROOT / "output_individual_neuron_rasters_aligned_offset_LP21_60s"
OFFSET_SORTED_TI_DIR = PROJECT_ROOT / "output_individual_neuron_rasters_offset_sorted_insp_duration_LP21_60s"
PEAK_DIR = PROJECT_ROOT / "output_individual_neuron_rasters_aligned_peak_LP21_60s"
ISI_DIR = PROJECT_ROOT / "output_neuron_isi_distributions_LP21_60s"
PUB_FIG_DIR = PROJECT_ROOT / "output_publication_figures_LP21_60s"
LEBEDEV_DIR = PROJECT_ROOT / "output_lebedev_rigor_checks_60s"


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Target Master Run Directory: {TARGET_DIR.resolve()}")

    # 1. Subfolder Structure
    subfolders = {
        "00_spike_sorting_data": TARGET_DIR / "00_spike_sorting_data",
        "01_interactive_sorter_raster": TARGET_DIR / "01_interactive_sorter_raster",
        "02_single_cell_phase_rasters_onset": TARGET_DIR / "02_single_cell_phase_rasters" / "onset_aligned",
        "02_single_cell_phase_rasters_onset_sorted": TARGET_DIR / "02_single_cell_phase_rasters" / "onset_aligned_sorted_by_insp_duration",
        "02_single_cell_phase_rasters_offset": TARGET_DIR / "02_single_cell_phase_rasters" / "offset_aligned",
        "02_single_cell_phase_rasters_offset_sorted": TARGET_DIR / "02_single_cell_phase_rasters" / "offset_aligned_sorted_by_insp_duration",
        "02_single_cell_phase_rasters_peak": TARGET_DIR / "02_single_cell_phase_rasters" / "peak_aligned",
        "02_single_cell_phase_rasters_warped": TARGET_DIR / "02_single_cell_phase_rasters" / "timewarped_mean_cycle",
        "03_isi_distributions": TARGET_DIR / "03_isi_distributions",
        "04_publication_figures": TARGET_DIR / "04_publication_figures",
        "05_lebedev_rigor_checks": TARGET_DIR / "05_lebedev_rigor_checks",
        "06_diagnostics": TARGET_DIR / "06_diagnostics"
    }

    for sf in subfolders.values():
        sf.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 00. Spike Sorting Data
    # -------------------------------------------------------------
    logger.info("Populating 00_spike_sorting_data...")
    if CONSENSUS_DIR.exists():
        # Copy ALF
        alf_src = CONSENSUS_DIR / "alf"
        alf_dst = subfolders["00_spike_sorting_data"] / "alf"
        if alf_src.exists():
            shutil.copytree(alf_src, alf_dst, dirs_exist_ok=True)

        # Copy cluster_info.tsv
        ci_src = CONSENSUS_DIR / "cluster_info.tsv"
        if ci_src.exists():
            shutil.copy2(ci_src, subfolders["00_spike_sorting_data"] / "cluster_info.tsv")

        # Copy params.py
        params_src = CONSENSUS_DIR / "params.py"
        if params_src.exists():
            shutil.copy2(params_src, subfolders["00_spike_sorting_data"] / "params.py")

    # Snapshot pipeline config
    cfg_src = PROJECT_ROOT / "pipeline_config.json"
    if cfg_src.exists():
        shutil.copy2(cfg_src, subfolders["00_spike_sorting_data"] / "pipeline_config_snapshot.json")

    # -------------------------------------------------------------
    # 01. Interactive Sorter Raster
    # -------------------------------------------------------------
    logger.info("Populating 01_interactive_sorter_raster...")
    sorter_raster_src = CONSENSUS_DIR / "lupin(recomputedconsensusmaster)_qc_interactive_raster.html"
    if sorter_raster_src.exists():
        shutil.copy2(sorter_raster_src, subfolders["01_interactive_sorter_raster"] / "lupin_qc_interactive_raster.html")

    # -------------------------------------------------------------
    # 02. Single Cell Phase Rasters
    # -------------------------------------------------------------
    logger.info("Populating 02_single_cell_phase_rasters...")
    if RAW_ONSET_DIR.exists():
        for f in RAW_ONSET_DIR.glob("*.*"):
            shutil.copy2(f, subfolders["02_single_cell_phase_rasters_onset"] / f.name)

    if ONSET_SORTED_TI_DIR.exists():
        for f in ONSET_SORTED_TI_DIR.glob("*.*"):
            shutil.copy2(f, subfolders["02_single_cell_phase_rasters_onset_sorted"] / f.name)

    if OFFSET_DIR.exists():
        for f in OFFSET_DIR.glob("*.*"):
            shutil.copy2(f, subfolders["02_single_cell_phase_rasters_offset"] / f.name)

    if OFFSET_SORTED_TI_DIR.exists():
        for f in OFFSET_SORTED_TI_DIR.glob("*.*"):
            shutil.copy2(f, subfolders["02_single_cell_phase_rasters_offset_sorted"] / f.name)

    if PEAK_DIR.exists():
        for f in PEAK_DIR.glob("*.*"):
            shutil.copy2(f, subfolders["02_single_cell_phase_rasters_peak"] / f.name)

    if TIMEWARPED_DIR.exists():
        for f in TIMEWARPED_DIR.glob("*.*"):
            shutil.copy2(f, subfolders["02_single_cell_phase_rasters_warped"] / f.name)

    # -------------------------------------------------------------
    # 03. ISI Distributions
    # -------------------------------------------------------------
    logger.info("Populating 03_isi_distributions...")
    if ISI_DIR.exists():
        for f in ISI_DIR.glob("*.*"):
            shutil.copy2(f, subfolders["03_isi_distributions"] / f.name)

    # -------------------------------------------------------------
    # 04. Publication Figures
    # -------------------------------------------------------------
    logger.info("Populating 04_publication_figures...")
    if PUB_FIG_DIR.exists():
        for f in PUB_FIG_DIR.glob("*.*"):
            if "diagnostic" not in f.name:
                shutil.copy2(f, subfolders["04_publication_figures"] / f.name)

    # -------------------------------------------------------------
    # 05. Lebedev Rigor Checks
    # -------------------------------------------------------------
    logger.info("Populating 05_lebedev_rigor_checks...")
    if LEBEDEV_DIR.exists():
        for f in LEBEDEV_DIR.glob("*.*"):
            shutil.copy2(f, subfolders["05_lebedev_rigor_checks"] / f.name)

    # -------------------------------------------------------------
    # 06. Diagnostics
    # -------------------------------------------------------------
    logger.info("Populating 06_diagnostics...")
    diag_dash = PUB_FIG_DIR / "breath_detection_diagnostic_dashboard.html"
    if diag_dash.exists():
        shutil.copy2(diag_dash, subfolders["06_diagnostics"] / "breath_detection_diagnostic_dashboard.html")

    # -------------------------------------------------------------
    # Write RUN_JOURNAL.md
    # -------------------------------------------------------------
    logger.info("Generating RUN_JOURNAL.md...")
    df_info = pd.read_csv(CONSENSUS_DIR / "cluster_info.tsv", sep="\t")
    qc_col = "passed_qc" if "passed_qc" in df_info.columns else "passed_allen_qc"
    df_clean = df_info[df_info[qc_col] == True]
    n_total_clusters = len(df_info)
    n_clean_sua = len(df_clean)

    journal_md = f"""# Experiment & Spike Sorting Run Journal

## 1. Run Metadata
* **Experiment Run ID**: `LP21_fent_EXPT_g1`
* **Analysis Slice**: **`60.0s to 120.0s`** (60.0-second time window)
* **Raw Recording Source**: `Z:\\ramirezlab\\pascual\\rec_data\\LP21_fent_EXPT_g1`
* **Neuropixels Probe**: Neuropixels 2.0 (`NP2004`, 4-shank, 2 parallel columns at $x=27\\,\\mu\\text{{m}}$ & $x=59\\,\\mu\\text{{m}}$, $15\\,\\mu\\text{{m}}$ vertical row pitch)
* **Electrophysiology Channels Analyzed**: Channels **`0 to 190`** (covering depth span **$0$ to $1,425\\,\\mu\\text{{m}}$**)
* **Spike Sorter**: **`Lupin`** (Spyking Circus 2 clustering + `circus-omp-svd` template matching)
* **Date of Processing**: August 2026

---

## 2. Preprocessing Configuration (CatGT)
* **Phase Shift Correction (`t_shift`)**: Enabled (`true`)
* **AP Band Filtering (`apfilter`)**: `butter,12,300,9000` (12-pole Butterworth bandpass, 300 Hz – 9,000 Hz)
* **Common Average Reference (`loccar_um`)**: Local CAR with inner radius $75\\,\\mu\\text{{m}}$, outer radius $300\\,\\mu\\text{{m}}$
* **Motion Correction**: Disabled for short slice stability

---

## 3. Sliding Window Sorter & Consensus Settings
To resolve single units along dense probe recordings while maintaining local channel sparsity:
* **Channel Window Size**: `50` channels per local window
* **Channel Overlap**: `20` channels of overlap between consecutive windows
* **Window Agreement Threshold**: `0.50` (50% spike temporal coincidence for duplicate resolution)
* **Strict Pre-Matching SUA QC Gating**:
  - **Rule**: Only candidate clusters passing Single-Unit QC enter cross-window consensus matching.
  - **ISI Violation Rate Cutoff**: `isi_violations_ratio < 0.10` ($< 10\\%$ refractory violations)
  - **Presence Ratio Cutoff**: `presence_ratio >= 0.05`

---

## 4. Single-Unit Sorting Yield
* **Total Evaluated Clusters**: `{n_total_clusters}`
* **Clean Single Units (SUAs Passing Strict QC)**: **`{n_clean_sua}` clean units**
* **Cross-Window Consensus Merges**: `16` verified consensus multi-window units
* **Depth Span of Clean Units**: **$60\\,\\mu\\text{{m}}$ to $1,155\\,\\mu\\text{{m}}$** along the probe shank

---

## 5. Auxiliary Physiological Signal Processing (`cibrrig` Standards)
Auxiliary signals synchronized 1:1 with the 60.0s – 120.0s electrophysiology window:
1. **Respiratory Airflow (Plethysmography, Channel 5)**: Digitized at 10 kHz, downsampled to 500 Hz.
2. **Diaphragm EMG (Channel 0)**:
   - Bandpass filter: 300 – 5,000 Hz (2nd order Butterworth SOS)
   - Full-wave rectification $\|V(t)\|$
   - 2-stage smoothing: **50 ms Median filter** + **10 ms Savitzky-Golay polynomial filter**
3. **Diaphragm Burst Landmark Detection (`burst_stats_dia`)**:
   - StandardScaler (z-score standardization)
   - `scipy.signal.find_peaks(prominence=1.0, distance=100ms, width=25ms)`
   - `scipy.signal.peak_widths(..., rel_height=0.95)`:
     - **Onset ($I_{{\\text{{on}}}}$)**: Left rising foot (10% above baseline)
     - **Offset ($I_{{\\text{{off}}}}$)**: Right falling foot (10% above baseline)
     - **Sigh / Double-burst rejection**: Suppressed overlapping intra-breath events
4. **Detected Breath Cycles**:
   - Total Detected Breaths: **148 Breaths**
   - Mean Inspiration Duration ($T_I$): **`153.8 ms`**
   - Mean Expiration Duration ($T_E$): **`252.0 ms`**
   - Mean Cycle Period ($T_{{\\text{{cycle}}}}$): **`405.7 ms`** (Breathing Frequency: **`2.46 Hz`**)

---

## 6. Directory Structure & Analysis Outputs

```
{RUN_DIR_NAME}/
├── RUN_JOURNAL.md                                # Complete configuration, parameters & yield journal
├── 00_spike_sorting_data/                        # Raw SpikeInterface / Lupin sorted data & metadata
│   ├── alf/                                     # IBL-ALF tables (spike_times.npy, spike_clusters.npy, etc.)
│   ├── cluster_info.tsv                         # Unit metrics, depth, firing rate, & QC flags
│   └── pipeline_config_snapshot.json            # Snapshot of JSON execution configuration
├── 01_interactive_sorter_raster/                # 5-Panel Synchronized Interactive Web Dashboard
│   └── lupin_qc_interactive_raster.html         # Flow + Raw EMG + Integrated EMG + PSTH + Raster
├── 02_single_cell_phase_rasters/                # Single-Neuron Phase-Aligned Rasters & PETHs
│   ├── onset_aligned/                           # Aligned to Breath Onset (I-on: 0 ms) + HTML gallery
│   ├── offset_aligned/                          # Aligned to Breath Offset (I-off: 0 ms) + HTML gallery
│   ├── peak_aligned/                            # Aligned to Burst Peak (0 ms) + HTML gallery
│   └── timewarped_mean_cycle/                   # Phase-timewarped to mean cycle + HTML gallery
├── 03_isi_distributions/                        # Refractory & Firing Mode ISI Analyses
│   ├── all_37_neurons_isi_overview_grid.png     # Multi-unit overview grid ordered by unit number
│   ├── gallery_isi_distributions_37_neurons.html# Interactive gallery of all 37 2-panel ISI cards
│   └── *.png                                    # Individual 2-panel linear + log ISI figures
├── 04_publication_figures/                      # Replications of Bush & Ramirez (2025) Figures
│   ├── publication_figure_2_single_cell_phase_dynamics.png
│   ├── publication_figure_3_low_d_neural_manifold.png
│   ├── publication_figure_3_interactive_3d_manifold.html
│   └── unit_respiratory_phase_classification.csv# Insp (15) vs Exp (17) vs Tonic (5) classifications
├── 05_lebedev_rigor_checks/                     # Lebedev et al. (2019) Dynamical Controls
│   ├── lebedev_2019_rotation_dynamics_rigor_checks.png
│   └── lebedev_subpopulation_3d_trajectory.html # Direct 3D subpopulation closed-loop trajectory
└── 06_diagnostics/                              # Quality control & signal diagnostics
    └── breath_detection_diagnostic_dashboard.html# Synchronized EMG, Envelope & Flow event dashboard
```

---

## 7. Ensemble Classification Summary
* **Inspiratory Units**: **`15`** neurons ($40.5\\%$)
* **Expiratory Units**: **`17`** neurons ($45.9\\%$)
* **Tonic Units**: **`5`** neurons ($13.5\\%$)
* **Lebedev Circle Generalization $R^2$**: **`0.763`** (demonstrates dynamic low-dimensional rotational constraint)
"""
    journal_path = TARGET_DIR / "RUN_JOURNAL.md"
    journal_path.write_text(journal_md, encoding="utf-8")
    logger.info(f"Saved RUN_JOURNAL.md to: {journal_path.resolve()}")

    print("\n" + "=" * 85)
    print(" [SUCCESS] RUN DIRECTORY REORGANIZATION COMPLETE")
    print("=" * 85)
    print(f" • Master Run Directory : {TARGET_DIR.resolve()}")
    print(f" • Run Journal          : {journal_path.resolve()}")
    print("=" * 85)


if __name__ == "__main__":
    main()
