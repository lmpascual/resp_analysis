# Modular Spike Sorting & Multimodal Cardiorespiratory Analysis Pipeline
## Technical Report & Collaborator Reference Manual

**Author**: Ramirez Lab (Pascual Lab / Collaborators)  
**Repository**: `resp_analysis`  
**Date**: September 2026  
**Primary Application**: Neuropixels High-Density Electrophysiology Coupled to Diaphragm EMG, Plethysmography Airflow, ECG, and Optogenetic Perturbation  
**Target Brain Regions**: Ventral Respiratory Column (VRC: Bötzinger Complex, pre-Bötzinger Complex, Rostral/Caudal VRG)

---

## Executive Summary

Recording neural activity in respiratory brainstem networks using Neuropixels probes presents unique computational challenges that conventional cortical spike-sorting pipelines fail to address:
1. **Mechanical & Non-Stationary Brainstem Motion**: Diaphragmatic contractions and vascular pulsations produce localized, rhythmic axial displacement (5–25 µm drift) synchronized to the breath cycle.
2. **High-Frequency Burst Dynamics & Intracellular Waveform Attenuation**: Pacemaker and bursting interneurons in the pre-Bötzinger complex undergo rapid spike-frequency adaptation and substantial intracellular amplitude decrement during inspiratory bursts, causing traditional template-matching sorters (e.g. Kilosort) to drop spikes selectively during the critical inspiratory phase.
3. **Multimodal Synchronization Demands**: Disentangling neural activity requires sample-accurate microsecond alignment between 30 kHz Neuropixels AP electrophysiology, 10 kHz NIDQ analog DAQ channels (diaphragm EMG, plethysmography airflow, ECG), and optogenetic laser pulse TTLs.

To overcome these failure modes, we developed the **Modular Spike Sorting & Multimodal Cardiorespiratory Pipeline (`resp_analysis`)**. This report details:
- The **end-to-end processing architecture** from raw SpikeGLX binary streams to curated single units and cardiorespiratory manifolds.
- A **comprehensive parameter decision manifest** documenting every algorithm, threshold, and parameter across all 6 pipeline stages.
- An **inventory of all intermediate diagnostic outputs** enabling step-by-step visual inspection of decisions and data stationarity.
- An **empirical multi-sorter benchmark** comparing **Lupin (SVD-OMP)**, **MountainSort 5 (branchless ISO-SPLIT)**, and **Kilosort 4**, featuring a forensic analysis of burst-phase spike dropouts.
- A **reproducibility guide** enabling collaborators to execute the entire pipeline end-to-end on their own machines.

---

## 1. Pipeline Architecture & Data Flow

```
[ SpikeGLX Acquisition: 30 kHz AP Stream + 10 kHz NIDQ Analog Stream ]
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
   [ neural_ap Stream ]              [ analog_nidq Stream ]
             │                                 │
             ▼                                 ▼
     Stage 1: CatGT Filter              Stage 1: Demux & Conditioning
   (300-9000 Hz, loccar 32/2)          (EMG: 300-3000 Hz bandpass, TKEO)
             │                         (Flow: 0.1-50 Hz tidal volume)
             ▼                         (ECG: 5-100 Hz, R-peaks, instant HR)
   Stage 2: Motion Correction          (Sync/Opto: TTL thresholding)
  (Locally exclusive peaks,                     │
   Kriging interpolation)                       ▼
             │                         Stage 2: Respiratory Phase
             ▼                         (Continuous phase ϕ, I/E transitions)
   Stage 3: 50-Ch Sliding Window                │
     Multi-Sorter Execution                     │
  (Lupin OMP / MS5 / Kilosort4)                 │
             │                                  │
             ▼                                  │
   Stage 4: Consensus Merging                   │
  (Temporal ±0.4ms, Jaccard ≥0.50)              │
             │                                  │
             ▼                                  │
   Stage 5: Quality Control                     │
  (Allen Institute / IBL SUA QC)                │
             │                                  │
             └────────────────┬─────────────────┘
                              ▼
           [ Stage 6: Cross-Modal Integration ]
           - Bush et al. 2024 Replicated Manifolds (PCA / 3D)
           - Cycle-by-Cycle Single-Neuron Phase Tuning Curves
           - Interactive Plotly HTML Dashboards & Phy GUI Export
           - Allen Institute (AIND) JSON Sidecar Provenance
```

---

## 2. Complete Step-by-Step Parameter Decision Manifest

Every parameter in the pipeline is explicit, inspectable, and overrideable via interactive checkpoint wizard or config JSON.

### Stage 1: Hardware Acquisition & Binary Stream Ingestion
| Parameter | Default Value | Unit / Format | Engineering Rationale & Failure Mode Avoidance |
| :--- | :--- | :--- | :--- |
| `fs_ap` | `30000.0` | Hz | Standard SpikeGLX Neuropixels 1.0/2.0 AP sampling rate. |
| `fs_nidq` | `10000.0` | Hz | Sufficient to resolve high-frequency diaphragm EMG motor units and sub-millisecond opto TTL edges. |
| `reader_backend` | `raw_mmap` | Memory Map | Directly targets and memory-maps `.ap.bin` and `.nidq.bin` streams independently. Bypasses Neo multi-stream crashes when 30 kHz probe and 10 kHz analog files co-occur in the same folder. |
| `data_dtype` | `int16` | Dtype | Memory-mapped access prevents RAM saturation during multi-channel processing. |

> [!NOTE]
> **Why `raw_mmap` instead of default Neo stream reading?**
> SpikeGLX stores both 30 kHz Neuropixels AP probe streams and 10 kHz NIDAQ analog streams (diaphragm EMG, plethysmography, ECG, opto TTLs) in the same recording folder. Standard SpikeInterface / Neo loaders often crash with stream enumeration/ambiguity errors when two different sampling rates co-occur in one folder. The `raw_mmap` backend directly parses the `.meta` text headers and memory-maps each binary stream independently from disk via `np.memmap`. This eliminates stream ambiguity crashes and enables instant random access to any time window with near-zero RAM overhead, even on >20 GB recordings.

### Stage 2: Signal Conditioning & Preprocessing (`CatGT`)
| Parameter | Default Value | Unit / Format | Engineering Rationale & Failure Mode Avoidance |
| :--- | :--- | :--- | :--- |
| `highpass_cutoff` | `300.0` | Hz | Attenuates low-frequency local field potentials (LFP) and mechanical drift. |
| `lowpass_cutoff` | `9000.0` | Hz | Anti-aliasing margin below Nyquist frequency ($15\,\text{kHz}$). |
| `loccar_inner` | `32` | Channels | Local Common Average Referencing radius. Global CAR introduces false negative artifact dips across the probe when large bursts occur; local CAR isolates referencing to nearby shank neighbors. |
| `loccar_outer` | `2` | Channels | Spatial guard band excluding immediate adjacent channels from the reference pool to avoid waveform subtraction. |
| `apfilter` | `butterworth_4` | Order 4 | Clean -24 dB/octave rolloff without ringing artifacts. |

### Stage 3: Motion Estimation & Non-Rigid Drift Correction
| Parameter | Default Value | Unit / Format | Engineering Rationale & Failure Mode Avoidance |
| :--- | :--- | :--- | :--- |
| `peak_detector` | `locally_exclusive`| Method | Detects local minima in spatial neighborhoods; prevents double-counting large spikes across adjacent sites. |
| `peak_threshold` | `5.0` | MAD (SD) | Median Absolute Deviation threshold; discriminates action potentials from thermal noise floor. |
| `localizer` | `monopolar_triangulation` | Algorithm | Triangulates 3D spike origin $(x, y, z)$ based on relative amplitude decay across neighboring electrodes. |
| `motion_preset` | `nonrigid_fast_and_accurate` | Preset | Spatial Kriging with localized spatial bins ($50\,\mu\text{m}$). Matches non-uniform respiratory brainstem displacement where dorsal channels experience different displacement than ventral channels. |
| `rigid_test` | `disabled` | Flag | Brainstem drift is non-uniform along the probe; rigid assumption fails. |

### Stage 4: Spike Sorting Engines & 50-Channel Sliding Window
| Parameter | Default Value | Unit / Format | Engineering Rationale & Failure Mode Avoidance |
| :--- | :--- | :--- | :--- |
| `window_size` | `50` | Channels | 50-channel sub-bands ($1000\,\mu\text{m}$ spatial coverage). Reduces memory footprint, limits clustering complexity, and isolates local anatomies (e.g. pre-BötC). |
| `window_overlap` | `10` | Channels | $200\,\mu\text{m}$ spatial overlap ensures units at window boundaries are sampled redundantly in both windows and merged during consensus. |
| `sorter_engine` | `lupin` / `mountainsort5` / `kilosort4` | Engine | Multi-sorter comparative framework. Lupin uses SVD-OMP (Matching Pursuit) to resolve overlapping spikes without rigid templates; MountainSort 5 uses branchless ISO-SPLIT; Kilosort 4 uses generative drift-tracking templates. |
| `lupin_svd_rank` | `8` | Components | Number of SVD singular vectors per spatial neighborhood capturing waveform variation during burst amplitude attenuation. |
| `lupin_omp_thresh` | `4.5` | SD | Residual energy stopping criterion for Orthogonal Matching Pursuit. |

### Stage 5: Cross-Window Consensus & De-duplication
| Parameter | Default Value | Unit / Format | Engineering Rationale & Failure Mode Avoidance |
| :--- | :--- | :--- | :--- |
| `coincidence_window` | `0.4` | ms (12 samples) | Spike matching tolerance across overlapping sub-bands. Accounts for slight sample jitter in peak assignment. |
| `jaccard_threshold` | `0.50` | Index ($0\dots1$) | Two units from adjacent windows sharing $\ge 50\%$ identical spikes within $\pm 0.4\,\text{ms}$ are merged into a single master unit. |
| `shank_assignment` | `centroid_depth` | µm | Master unit depth is calculated from the energy-weighted amplitude profile across all probe sites. |

### Stage 6: Quality Control (Allen Institute for Neural Dynamics & IBL Standards)
| Quality Metric | SUA Passing Threshold | Unit | Failure Mode Detected / Rationale |
| :--- | :--- | :--- | :--- |
| `isi_violation_rate` | `< 0.005` ($< 0.5\%$) | Ratio | Refractory period violations ($< 1.5\,\text{ms}$). Detects multi-unit contamination or false merged clusters. |
| `amplitude_cutoff` | `< 0.10` ($< 10\%$) | Ratio | Quantifies missing spikes due to detection threshold using a Gaussian tail fit. $> 10\%$ indicates severe spike dropout. |
| `presence_ratio` | `> 0.90` ($> 90\%$) | Ratio | Detects severe physical unit drift out of electrode recording sphere or intermittent recording silencing. |
| `snr` | `> 2.5` | Ratio | Peak-to-peak waveform amplitude relative to background noise MAD. |
| `unit_type` | `SUA` vs `MUA` vs `Noise` | Classification | Single units meeting all 4 criteria are flagged as `passed_allen_qc = True`. |

---

## 3. Empirical Multi-Sorter Benchmark & Forensic Analysis

We evaluated **Lupin (Garcia et al. 2026)**, **MountainSort 5 (MS5)**, and **Kilosort 4 (KS4)** on synchronized Neuropixels recordings from the mouse ventral respiratory column (`LP21`, `LP22`, `TP_Tacr1_1`, `e097`).

### 3.1 Benchmark Dataset & Precise Snippet Provenance

To provide a controlled, reproducible head-to-head evaluation under identical biophysical conditions, all three sorters were executed on the following synchronized recording snippet:

- **Animal Subject**: Mouse `LP21` (Adult C57BL/6J, acute preparation).
- **Session / Experiment Run**: `LP21_fent_EXPT_g1` (Pre-fentanyl baseline state).
- **Raw Data Provenance**:
  - **Neuropixels AP Stream**: `LP21_fent_EXPT_g1_t0.imec0.ap.bin` (30 kHz, Neuropixels 2.0 probe).
  - **NIDAQ Analog Stream**: `LP21_fent_EXPT_g1_t0.nidq.bin` (10 kHz; Channel 0: Diaphragm EMG, Channel 5: Plethysmography Airflow, Channel 1: ECG, Channel 7: Sync TTL).
  - **Raw Server Path**: `Z:\ramirezlab\pascual\rec_data\LP21_fent_EXPT_g1`
- **Probe Channels Analyzed**: Rostral probe tip channels (Channels 0 to 50, covering the ventral-most 1000 µm of the probe targeting the rostral Ventral Respiratory Column / pre-Bötzinger and Bötzinger complexes).
- **Precise Temporal Windows**:
  1. **60-Second Quantitative Yield Benchmark Window**: `t = 0.0 s` to `t = 60.0 s` (Pre-fentanyl stable eupneic baseline breathing; ~150 breaths/min, diaphragm EMG SNR ≥ 12 dB). Also validated across the continuous `t = 60.0 s` to `t = 120.0 s` slice (`outputs/LP21/output_LP21_60s_120s`).
  2. **High-Resolution Forensic Burst Window (`sorter_comparison_drop_window.png`)**: A 1.0-second zoom (`t = 40.0 s` to `t = 41.0 s`) centered directly across a peak inspiratory diaphragmatic burst transition, accompanied by single-breath verification windows (`t = 7.0 s` to `t = 9.0 s` in `figures/diagnostics/four_breath_raw_ap_comparison.png`).
- **Standardized Preprocessing**: All three sorters received identical CatGT conditioning (300-9000 Hz bandpass, local CAR `loccar [32, 2]`) and non-rigid Kriging motion correction.

### 3.2 Quantitative Yield Comparison (60s Benchmark Window)

| Sorter Engine | Algorithm Category | Total Units Extracted | Clean SUA (Passed Allen QC) | Mean ISI Violation Rate | Burst Phase Retention (%) | Compute Time (60s recording) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Lupin (SVD-OMP)** | Sparse Matching Pursuit | **142** | **37** | **0.18%** | **94.2%** | ~4.2 min |
| **MountainSort 5** | ISO-SPLIT Clustering | 98 | 24 | 0.31% | 76.8% | **~1.8 min** |
| **Kilosort 4** | Generative Template Matching | 114 | 19 | 0.84% | 58.4% | ~3.5 min |

### 3.3 The "Burst-Induced Amplitude Drop" Phenomenon (`sorter_comparison_drop_window.png`)

A critical finding from our forensic investigation of inspiratory burst windows is documented in `figures/benchmarks/sorter_comparison_drop_window.png`:

```
                       INSPIRATION ONSET
                              │
Diaphragm EMG:                ▼  ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ (High-amplitude burst)
                              ┌────────────────────────────────────────┐
Raw AP Channel:               │  | | |||||||||||||| | |                │ (Spike amplitude drops 40%)
                              └────────────────────────────────────────┘
Lupin (SVD-OMP):              │  | | |||||||||||||| | |                │ (Spikes captured continuously: 100%)
MountainSort 5:               │  | | |||    ||||    | |                │ (Partial dropout in burst core: 76%)
Kilosort 4:                   │  | |                  | |              │ (Severe dropout during peak burst: 42%)
```

#### Forensic Findings:
1. **Intracellular Amplitude Attenuation**: During intense inspiratory bursts, respiratory neurons (e.g. pre-BötC pacemakers) fire at instantaneous rates exceeding $60\text{–}120\,\text{Hz}$. Sodium channel inactivation causes spike amplitudes to drop by up to $40\text{–}50\%$ relative to isolated baseline spikes.
2. **Template Matching Failure (Kilosort 4)**: KS4 relies on rigid spatiotemporal templates. When spike waveforms rapidly shrink during bursts, their dot product with the baseline template falls below the template matching threshold, causing KS4 to drop the densest portion of the burst.
3. **Clustering Splitting Failure (MountainSort 5)**: MS5's branchless clustering often splits the attenuated burst spikes from the isolated spikes into two separate clusters, creating an artificial low-amplitude "noise" cluster that fails Allen QC.
4. **Why Lupin Succeeds (SVD-OMP)**: Lupin projects candidate spikes onto an SVD basis representing waveform variance across amplitudes. By iteratively subtracting waveforms using Orthogonal Matching Pursuit (OMP), it tracks spikes even when amplitudes shrink or when overlapping waveforms co-occur during the peak of inspiration.

---

## 4. Inventory of Intermediate Diagnostic Outputs

To enable collaborators to evaluate every processing stage, the pipeline generates a standardized set of diagnostic figures and reports.

| Diagnostic Output Filename | Generating Script / Stage | Key Features Inspected | Evaluation Criteria & Red Flags |
| :--- | :--- | :--- | :--- |
| `*_emg_stationarity.png` | `src.pipeline.visualizer` / Stage 1 | Diaphragm EMG raw, bandpass, envelope, breath onsets, instant RR tachogram, running SNR (dB). | **Pass**: SNR $\ge 10\,\text{dB}$; steady RR; clean onset triangles.<br>**Fail**: Baseline drifting; SNR $< 8\,\text{dB}$; breath onset false positives during apnea. |
| `*_ecg_rpeaks.png` | `src.pipeline.visualizer` / Stage 1 | ECG raw, filtered (5-100 Hz), detected R-peaks, instantaneous HR tachogram (bpm), HRV SDNN. | **Pass**: R-peaks hit sharp QRS peaks; physiological mouse HR ($400\text{–}650\,\text{bpm}$).<br>**Fail**: T-wave double-triggering; missing beats during movement. |
| `*_opto_sync.png` | `src.pipeline.visualizer` / Stage 1 | Digital sync TTL square waves (1 Hz) and laser optogenetic TTL pulses (20 Hz). | **Pass**: Square edges with zero jitter; pulse counts match acquisition journal.<br>**Fail**: Rounded TTL edges; missing pulses. |
| `*_opto_prc.png` | `src.pipeline.visualizer` / Stage 1 | Phase Response Curve: Laser phase ($\phi \in [0, 1]$) vs Phase shift ($\Delta \phi$). | **Pass**: Clear inspiratory advance ($\Delta\phi < 0$) or expiratory delay ($\Delta\phi > 0$) curve.<br>**Fail**: Completely uniform scatter indicating non-functional stimulation. |
| `*_diaphragm_interactive.html` | `src.pipeline.visualizer` / Stage 1 | Interactive Plotly 3-panel dashboard with range slider, zoom buttons, and shaded laser pulse overlays. | High-resolution inspection of breath transitions, gasping, and sigh events. |
| `motion_drift_map.png` | `src.pipeline.motion` / Stage 2 | Spatial peak scatter $(t, z)$ overlaid with Kriging drift estimate trajectories. | **Pass**: Smooth drift lines tracking respiratory motion bands ($< 30\,\mu\text{m}$).<br>**Fail**: Erratic discontinuous jumps indicating poor peak localization. |
| `sorter_comparison_drop_window.png`| `scripts.visualization` / Stage 3 | Multi-panel comparison of Lupin vs MS5 vs KS4 over a 500 ms inspiratory burst window. | Direct verification of whether sorters retain spikes during peak diaphragmatic activity. |
| `bushetal2024_replicated_single_cell_phase_dynamics.png` | `scripts.experiments.lp21` / Stage 6 | Single-unit respiratory phase tuning curves ($0\text{–}360^\circ$) replicating Bush et al. 2024 Fig 2. | **Pass**: Clear segregation of Inspiratory (I-phase peak) vs Expiratory (E-phase peak) neurons. |
| `bushetal2024_replicated_neural_manifold_pca.png` | `scripts.experiments.lp21` / Stage 6 | Low-dimensional neural state-space manifold (PC1 vs PC2 vs PC3) color-coded by respiratory cycle phase. | **Pass**: Smooth rotational attractor loop tracing inspiration $\rightarrow$ expiration $\rightarrow$ transition. |
| `bushetal2024_replicated_interactive_3d_manifold.html` | `scripts.experiments.lp21` / Stage 6 | Interactive Plotly 3D manifold viewer with continuous trajectory animation. | Enables 3D camera rotation to inspect neural state dynamics and cycle separation. |
| `*_run_journal.json` & `*_provenance.md` | `src.pipeline.journaler` / All | Machine-readable parameters, Git commit hash, Python version, summary statistics. | 100% exact reproduction record for publications. |
| `aind_metadata/*.json` | `src.pipeline.journaler` / All | AIND-standard `subject.json`, `acquisition.json`, `quality_control.json`, `processing.json`. | Direct compliance with Allen Institute for Neural Dynamics data portals. |

---

## 5. Repository Structure & Directory Organization

The repository is structured into modular tiers to support both core production pipelines and exploratory experiments:

```
resp_analysis/
├── docs/                                      # Collaborator manuals, guides, and reports
│   └── collaborator_spike_sorting_pipeline_report.md
│
├── src/                                       # Core Production Package
│   └── pipeline/
│       ├── config.py                          # Unified parameter dataclasses
│       ├── models.py                          # Typed data structures (SessionResults, AnalogData, EphysData)
│       ├── journaler.py                       # Checkpoint approvals, run journal, AIND JSON sidecars
│       ├── pipeline_runner.py                 # Master orchestration engine
│       ├── visualizer.py                      # Multi-panel diagnostic dashboards & interactive HTML
│       │
│       ├── processing/                        # Consolidated Stream Processing (AIND-compliant)
│       │   ├── neural_ap/                     # Neuropixels Electrophysiology Stream
│       │   │   ├── io_spikeglx.py             # Memory-mapped binary reader
│       │   │   ├── io_catgt.py                # CatGT bandpass & loccar filter
│       │   │   ├── io_ephys.py                # ALF spike train reader & synthetic generator
│       │   │   ├── proc_ephys.py              # Unit modulation & PSTH computation
│       │   │   ├── proc_blech_single_channel.py # Single-channel AP filtering
│       │   │   ├── motion_corrector.py        # Peak detection, localization, Kriging interpolation
│       │   │   └── motion_presets.py          # Custom drift presets
│       │   │
│       │   └── analog_nidq/                   # Multimodal Analog Stream
│       │       ├── io_nidq.py                 # NIDQ demux & synthetic generator
│       │       ├── proc_emg.py                # Diaphragm EMG filtering, TKEO envelope, breath onsets
│       │       ├── proc_flow.py               # Airflow plethysmography & tidal volume
│       │       ├── proc_ecg.py                # ECG bandpass, R-peak detection, instant HR, HRV
│       │       ├── proc_sync_opto.py          # TTL edge detection & pulse timing
│       │       ├── proc_opto_phase.py         # Phase Response Curve (PRC) analysis
│       │       └── proc_diaphragm_transition.py # Diaphragm state transition detector
│       │
│       ├── sorting/                           # Spike Sorting & Cross-Window Consensus
│       │   ├── sorter_runner.py               # Multi-sorter runner
│       │   ├── lupin_pipeline.py              # SVD-OMP matching pursuit sorter
│       │   ├── sliding_window_runner.py       # 50-channel sub-band sliding window runner
│       │   ├── sliding_window_consensus.py    # Cross-window de-duplication & consensus assembly
│       │   ├── postprocessing.py              # Allen / IBL QC metrics & Phy export
│       │   └── comparator.py                  # Pairwise bipartite coincidence matrix
│       │
│       ├── diagnostics/                       # Diagnostic Inspection Engines
│       │   ├── preproc_inspector.py           # CatGT PSD, raw vs filtered, noise floor
│       │   ├── motion_inspector.py            # Drift trajectories & peak scatter
│       │   └── sorting_inspector.py           # Multi-sorter benchmarks & agreement heatmaps
│       │
│       ├── io/                                # Backward-compatibility shims
│       ├── processors/                        # Backward-compatibility shims
│       └── motion/                            # Backward-compatibility shims
│
├── scripts/                                   # Categorized Command-Line Scripts & Tools
│   ├── pipeline/                              # Core pipeline runners & configuration wizards
│   │   ├── configure_pipeline_wizard.py       # Interactive parameter configuration wizard
│   │   ├── run_modular_pipeline.py            # End-to-end modular runner
│   │   ├── run_sliding_window_pipeline.py     # 50-channel sliding window runner
│   │   ├── run_catgt_preprocessing.py         # Batch CatGT runner
│   │   ├── recompute_qc_and_consensus.py      # Rapid consensus re-assembly
│   │   └── export_to_phy_gui.py               # Phy GUI dataset exporter
│   │
│   ├── experiments/                           # Animal Subject Specific Analyses
│   │   ├── lp21/                              # Subject LP21 (Fentanyl & baseline)
│   │   │   ├── replicate_bushetal2024_manifold_figures.py # Bush et al. 2024 Fig 2 & Fig 3 replication
│   │   │   ├── run_lp21_ms5_and_kilosort_sorting.py
│   │   │   └── run_lebedev_rotation_checks.py
│   │   ├── lp22/                              # Subject LP22 (Multi-epoch & dropout analysis)
│   │   ├── e097/                              # Subject e097 (Opto laser trains, gasping transitions)
│   │   ├── tp_tacr1/                          # Subject TP_Tacr1_1 (Pacemaker isolation & rotational dynamics)
│   │   └── e086/                              # Subject e086 (Opto window analysis)
│   │
│   ├── visualization/                         # Plotting & Diagnostic Generation
│   │   ├── generate_split_bushetal2024_panels.py # Multi-panel publication composite figures
│   │   ├── plot_interactive_sorter_raster.py  # Interactive browser-based unit raster
│   │   ├── plot_top_phasic_units_psth.py      # Phasic PSTH polar & cartesian plots
│   │   └── plot_all_37_neurons_peth_with_sd.py # Population PETH with standard deviation bands
│   │
│   └── presentations/                         # Presentation Decks & Slide Builders
│       ├── build_presentation_deck.py
│       └── build_45min_comprehensive_deck.py
│
├── figures/                                   # Curated Diagnostic Figures & Sorter Benchmarks
│   ├── benchmarks/                            # Multi-sorter comparison figures
│   │   └── sorter_comparison_drop_window.png
│   └── diagnostics/                           # Raw waveform zooms, breath transitions, ECG previews
│
├── outputs/                                   # All Experiment Outputs Organized by Subject (Gitignored)
│   ├── LP21/
│   │   ├── bushetal2024_replicated_manifold_20min/
│   │   ├── bushetal2024_replicated_manifold_60s/
│   │   └── ...
│   ├── LP22/
│   ├── TP_Tacr1_1/
│   ├── e097/
│   ├── e086/
│   └── misc/
│
├── exploratory/                               # Tier 1 Private Sandbox for Prototyping (Gitignored)
├── external/                                  # External Git Repositories (inspiration-off-attractor, pytau)
└── tests/                                     # Automated Unit & Integration Test Suite
```

---

## 6. End-to-End Walkthrough & Replication Commands

Collaborators can replicate our entire analysis pipeline using the following step-by-step commands:

### Step 1: Environment Setup
```bash
# Clone the repository
git clone <private-repo-url>/resp_analysis.git
cd resp_analysis

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies in editable mode
pip install -e .
```

### Step 2: Interactive Parameter Configuration Wizard
Launch the interactive configuration wizard to specify channel maps, filter cutoffs, and sorting parameters:
```bash
python scripts/pipeline/configure_pipeline_wizard.py
```
This writes an approved `pipeline_config.json` file.

### Step 3: Run the Modular Pipeline (Demo or Live Data)
Run the pipeline with automated non-interactive execution:
```bash
python scripts/pipeline/run_modular_pipeline.py --config pipeline_config.json --non-interactive
```
If raw `.ap.bin` or `.nidq.bin` files are not present in the specified directory, the pipeline automatically generates a verified synthetic test dataset, exercises all filter banks, detects breaths and R-peaks, and generates the diagnostic figures and AIND sidecar metadata in the output folder.

### Step 4: Replicate Bush et al. 2024 Manifold & Phase Dynamics Figures
To reproduce the single-cell phase dynamics (Fig 2) and 3D rotational neural manifold (Fig 3):
```bash
python scripts/experiments/lp21/replicate_bushetal2024_manifold_figures.py \
    --consensus-dir "outputs/LP21/output_LP21_60s_120s/sliding_window_pipeline/lupin/consensus_master" \
    --raw-dir "Z:/ramirezlab/pascual/rec_data/LP21_fent_EXPT_g1" \
    --out-dir "outputs/LP21/bushetal2024_replicated_manifold_60s"
```

### Step 5: Export to Phy GUI for Manual Verification
```bash
python scripts/pipeline/export_to_phy_gui.py --input-dir "outputs/LP21/output_LP21_60s_120s/sliding_window_pipeline/lupin/consensus_master"
phy template-gui outputs/LP21/output_LP21_60s_120s/sliding_window_pipeline/lupin/consensus_master/phy/params.py
```

---

## 7. Quality Assurance & Reproducibility Verification

All components are protected by automated tests in `tests/`:
```bash
pytest tests/
```
Current Status: **6 passed in 3.71s**
- `test_config_serialization`: Verifies round-trip JSON parameter persistence.
- `test_synthetic_nidq_generation`: Verifies multimodal signal generation.
- `test_emg_processing`: Tests TKEO envelope and breath onset thresholding.
- `test_ecg_processing`: Tests bandpass filter, R-peak detector, and HRV SDNN.
- `test_pulse_extraction`: Tests sub-millisecond opto pulse edge detection.
- `test_pipeline_non_interactive`: Verifies end-to-end execution, diagnostic dashboard rendering, provenance records, and AIND sidecar generation.


---

## 8. Relevant Sources, Methodological Lineage & Open-Source Repositories

This pipeline is directly grounded in and extends key methodological frameworks, algorithms, and open-source tools from the electrophysiology and respiratory neurobiology communities:

### 8.1 Modular Spike Sorting & The Lupin Engine
- **Garcia et al. (2026)**: *Opening the black box: a modular approach to spike sorting*. (bioRxiv / Nature Methods).
  - *Local reference*: C:\Users\lpascu\OneDrive - SCH\PAPERS\Tools\2026 Garcia, Opening the black box - a modular approach to spike sorting.pdf
  - *Contribution & Implementation*: Formulates the core algorithmic foundation of our spike sorting pipeline. Specifically:
    - **Lupin Sorter (spikeinterface.sorters.lupin)**: Deploys Sparse SVD-OMP (Orthogonal Matching Pursuit) matching pursuit to resolve dense overlapping waveforms and intracellular amplitude decrement during high-frequency inspiratory bursts without dropping spikes.
    - **Locally Exclusive Peak Detection (spikeinterface.sortingcomponents.peak_detection)**: Discovers spatio-temporal extrema across localized probe neighborhoods, eliminating duplicate threshold triggers.
    - **Non-Rigid Kriging Motion Correction (spikeinterface.sortingcomponents.motion)**: Computes continuous non-rigid displacement maps along the probe and reconstructs spatial voltage profiles via Kriging interpolation, compensating for diaphragmatic respiratory motion.
    - **50-Channel Sliding Window & Consensus Assembly**: Partitions high-density probes into overlapping spatial sub-bands to limit memory overhead and isolate localized brainstem nuclei, followed by bipartite Jaccard graph merging.
- **Buccino et al. (2020)**: *SpikeInterface, a unified framework for spike sorting*. **eLife**, 9, e61834. DOI: 10.7554/eLife.61834.
  - *Local reference*: C:\Users\lpascu\OneDrive - SCH\PAPERS\Tools\2020 Buccino, SpikeInterface.pdf
  - *Repository*: [SpikeInterface GitHub](https://github.com/SpikeInterface/spikeinterface)
  - *Contribution*: The foundational interoperability API, memory-mapped recording extractors, sorting containers, and automated quality metric calculation routines.

### 8.2 Respiratory Rhythmogenesis, Attractor Dynamics & CIBRRIG Lineage
- **Bush & Ramirez (2025/2024)**: *An inspiration-off attractor supports the robust and flexible control of breathing*. Nature / bioRxiv.
  - *Local reference*: C:\Users\lpascu\OneDrive - SCH\PAPERS\2025 Bush Ramirez, An inspiration-off attractor supports the robust and flexible control of breathing.pdf
  - *Repository*: [inspiration-off-attractor (VLAD) GitHub](https://github.com/cibr-uw/inspiration-off-attractor) (mirrored locally in xternal/inspiration-off-attractor/).
  - *Contribution*: Establishes the mathematical and empirical basis for low-dimensional respiratory manifold reconstruction (PCA attractor loops across respiratory cycles, replicated in our Stage 6 pipeline as ushetal2024_replicated_neural_manifold_pca.png and ushetal2024_replicated_single_cell_phase_dynamics.png).
- **Bush & Ramirez (2024)**: *Latent neural population dynamics in the respiratory rhythm generator*.
  - *Local reference*: C:\Users\lpascu\OneDrive - SCH\PAPERS\2024 Bush Ramirez, Latent neural population dynamics.pdf
  - *Repository*: [CIBRRIG GitHub](https://github.com/cibr-uw/cibrrig) / [nbush257/cibrrig](https://github.com/nbush257/cibrrig).
  - *Contribution*: Provided the initial analog DAQ processing patterns, SpikeGLX metadata parsing conventions, and cardiorespiratory alignment routines which were refactored and modularized into src/pipeline/processing/analog_nidq/.
- **Foundational Ramirez Lab Cardiorespiratory & Opioid Studies**:
  - *Baertsch & Ramirez (2018, 2019, 2021)*: Dual mechanisms of opioid-induced respiratory depression (OIRD) in the pre-Bötzinger complex; inspiratory rhythmogenic state shifts.
  - *Moreira, Ramirez, et al. (2025)*: Retrotrapezoid nucleus (RTN) neuronal modulation and fentanyl-induced respiratory depression.
  - *Arthurs, Palmiter, & Baertsch (2023)*: Parabrachial tachykinin1 (Tacr1) circuit contributions to state-dependent breathing control.
  - *Huff & Ramirez (2022, 2023, 2024)*: Optogenetic stimulation of pre-Bötzinger and Postinspiratory (PiCo) complexes during swallowing-breathing coordination.

### 8.3 Comparative Spike Sorting Engines
- **MountainSort 5**:
  - *Magland et al. (2023)*: Fast, parameter-light, branchless ISO-SPLIT spike sorting for high-channel recordings.
  - *Repository*: [MountainSort 5 GitHub](https://github.com/flatironinstitute/mountainsort5)
  - *Role*: Evaluated in our multi-sorter benchmark as a high-precision, drift-resistant spatial clustering baseline.
- **Kilosort 4**:
  - *Pachitariu, Stringer, et al. (2024)*: Next-generation spike sorting with generative drift correction and template matching.
  - *Repository*: [Kilosort GitHub](https://github.com/MouseLand/Kilosort)
  - *Role*: Evaluated in our multi-sorter benchmark; identified the template-matching burst dropout failure mode during inspiration (sorter_comparison_drop_window.png).

### 8.4 Signal Preprocessing, Hardware & Manual Curation
- **CatGT**:
  - *Karsh, Harris, et al. (Janelia Research Campus)*: High-performance C++ tool for common average referencing and demultiplexing SpikeGLX binary streams.
  - *Repository*: [CatGT GitHub](https://github.com/billkarsh/CatGT)
  - *Role*: Implemented in src/pipeline/processing/neural_ap/io_catgt.py for local CAR (loccar [32, 2]) and bandpass filtering (300-9000 Hz).
- **Phy GUI**:
  - *Rossant et al. (CortexLab / UCL)*: Interactive visual inspection and manual curation graphical interface for large-scale electrophysiology datasets.
  - *Repository*: [Phy GitHub](https://github.com/cortex-lab/phy)
  - *Role*: Integrated via scripts/pipeline/export_to_phy_gui.py for collaborator single-unit inspection and waveform curation.
- **PyTau**:
  - *Mahmood, A. (2024)*: Bayesian changepoint detection for neural time series and behavioral state transitions.
  - *Repository*: [PyTau GitHub](https://github.com/abuzarmahmood/pytau) (mirrored locally in xternal/pytau/).
  - *Role*: Available for detecting discrete state transitions (e.g. baseline vs apnea vs gasping vs opioid depression).

### 8.5 Metadata Governance & Standards
- **Allen Institute for Neural Dynamics (AIND) Data Schema**:
  - *Repository*: [AIND Data Schema GitHub](https://github.com/AllenNeuralDynamics/aind-data-schema)
  - *Role*: Direct source for our automated sidecar metadata generator (subject.json, cquisition.json, quality_control.json, processing.json) in src/pipeline/journaler.py.
- **International Brain Laboratory (IBL)**:
  - *Repository*: [IBL Brainbox Quality Metrics](https://github.com/int-brain-lab/ibllib)
  - *Role*: Formulates our automated single-unit QC thresholds (refractory ISI violation <0.5%, amplitude cutoff <10%, presence ratio >90%, SNR >2.5).

