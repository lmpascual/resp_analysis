# resp_analysis: Modular Neurophysiology & Cardiorespiratory Analysis Pipeline

A modular Python analysis and spike sorting framework built on Nicholas Bush's `cibrrig` foundation for processing **Neuropixels electrophysiology** (`neural_ap`) and **analog NIDQ signals** (`analog_nidq`: diaphragm EMG, plethysmography airflow, ECG, time sync TTL pulses, optic fiber laser pulses).

---

## 📖 Key Documentation

- **[Collaborator Spike Sorting Technical Report](file:///c:/Users/lpascu/analysis_src/COLLABORATOR_SPIKE_SORTING_PIPELINE_REPORT.md)**: Full parameter decision manifest, intermediate diagnostic output inventory, and empirical benchmark evaluation (Lupin SVD-OMP vs MountainSort 5 vs Kilosort 4).
- **[Sliding Window Pipeline Guide](file:///c:/Users/lpascu/analysis_src/SLIDING_WINDOW_PIPELINE_GUIDE.md)**: 50-channel sub-band sliding window execution, cross-window consensus assembly, and Phy GUI curation.

---

## 🌟 Key Features

1. **Allen Institute for Neural Dynamics (AIND) Stream Architecture**:
   - `src/pipeline/processing/neural_ap`: Memory-mapped SpikeGLX `.ap.bin` reader, CatGT bandpass & local CAR (`loccar [32, 2]`), non-rigid Kriging motion correction, single-channel thresholding, and unit modulation.
   - `src/pipeline/processing/analog_nidq`: Multi-channel NIDQ demux, diaphragm EMG filtering & TKEO envelope, airflow plethysmography, ECG R-peaks & HRV, opto laser TTL edge detection, and Phase Response Curves (PRC).
2. **Interactive Parameter Checkpoint Wizard**: Interactive CLI wizard allows you to review, approve, or override parameters on-the-fly, generating a reproducible `pipeline_config.json`.
3. **Multi-Sorter Sliding Window Consensus**: Runs Lupin (SVD-OMP), MountainSort 5, or Kilosort 4 over 50-channel sub-bands with 10-channel overlap, followed by temporal ($\pm 0.4\,\text{ms}$) and Jaccard ($\ge 0.50$) cross-window consensus merging.
4. **Automated Quality Control & AIND Sidecars**: Evaluates Allen Institute and IBL SUA criteria (ISI violations $< 0.5\%$, amplitude cutoff $< 10\%$, presence ratio $> 90\%$, SNR $> 2.5$) and automatically writes standard AIND sidecar files (`subject.json`, `acquisition.json`, `quality_control.json`, `processing.json`).
5. **Bush et al. 2024 Literature Replication**: Reproduces single-cell respiratory phase dynamics (Fig 2) and 3D rotational state-space neural manifolds (Fig 3) across respiratory cycles.

---

## 📁 Repository Structure

```
resp_analysis/
├── docs/                                      # Collaborator manuals & technical reports
│   └── collaborator_spike_sorting_pipeline_report.md
├── src/
│   └── pipeline/
│       ├── processing/
│       │   ├── neural_ap/                     # Neuropixels AP stream (SpikeGLX, CatGT, Motion)
│       │   └── analog_nidq/                   # Analog stream (EMG, Airflow, ECG, Opto)
│       ├── sorting/                           # Lupin, MS5, KS4, Sliding Window Consensus
│       ├── diagnostics/                       # Diagnostic plotters & stationarity checkers
│       ├── journaler.py                       # Checkpoint approvals, run journal, AIND JSON sidecars
│       ├── visualizer.py                      # Multi-panel dashboards & interactive HTML
│       ├── config.py & models.py              # Parameter schemas & typed data models
│       ├── io/ & processors/ & motion/        # Backward-compatibility shims
│       └── pipeline_runner.py                 # Master pipeline orchestrator
├── scripts/
│   ├── pipeline/                              # Core pipeline runners & wizards
│   ├── experiments/                           # Animal-specific scripts (LP21, LP22, e097, TP_Tacr1_1, e086)
│   ├── visualization/                         # Diagnostic plotting & interactive dashboards
│   └── presentations/                         # Presentation decks & slide generators
├── figures/
│   ├── benchmarks/                            # Sorter comparison drop window & benchmark figures
│   └── diagnostics/                           # Raw waveform zooms & signal inspection plots
├── outputs/                                   # Experiment outputs organized by subject (Gitignored)
├── exploratory/                               # Tier 1 private sandbox for prototyping (Gitignored)
├── external/                                  # External clones (inspiration-off-attractor, pytau)
└── tests/                                     # Automated test suite (Pytest)
```

---

## 🚀 Quickstart Guide

### 1. Launch Parameter Configuration Wizard
```bash
python scripts/pipeline/configure_pipeline_wizard.py
```

### 2. Run Modular Pipeline
```bash
python scripts/pipeline/run_modular_pipeline.py --config pipeline_config.json --non-interactive
```

### 3. Replicate Bush et al. 2024 Manifolds
```bash
python scripts/experiments/lp21/replicate_bushetal2024_manifold_figures.py
```

### 4. Run Test Suite
```bash
pytest tests/
```

---

## 🐙 Linking to GitHub (Private Repository)

```bash
git remote add origin https://github.com/<your-username>/resp_analysis.git
git branch -M main
git push -u origin main
```

