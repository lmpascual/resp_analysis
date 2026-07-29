# Modular Neurophysiology & Cardiorespiratory Analysis Pipeline

A modular Python framework built on Nicholas Bush's `cibrrig` foundation for processing **Neuropixel electrophysiology** and **analog NIDQ signals** (diaphragm EMG, ECG, time sync TTL pulses, optic fiber laser pulses).

---

## 🌟 Key Features

1. **Modular Modality Architecture**: Designed so additional modalities (such as video tracking or olfactometer data) can be plugged in easily as subclassed processors/loaders without modifying core pipeline logic.
2. **Interactive Parameter Checkpoints**: Before processing raw binary data, an interactive CLI checkpoint displays all parameters (filter frequencies, notch filters, smoothing window sizes, threshold multipliers, and channel mappings). You can review, approve as-is, or edit parameters on-the-fly.
3. **Visual Quality & Stationarity Inspection**: Generates multi-panel diagnostic plots for:
   - **EMG Transformation & SNR Stationarity**: Raw vs. bandpass filtered vs. envelope waveforms, detected breath peaks, and time-resolved Signal-to-Noise Ratio (SNR) in dB over time to check for electrode drift or noise artifacts.
   - **ECG & Heart Rate**: Filtered ECG waveforms, detected R-peaks, and instantaneous heart rate (HR) tachograms with HRV (SDNN).
   - **Opto Laser & Sync Edge Alignment**: Pulse onset/offset timestamps aligned against TTL waveforms.
   - **Neuropixel Depth Rasters & Stationarity**: Probe depth vs. time unit spike rasters to inspect neural stationarity and drift.
4. **100% Journaled Reproducibility**: Automatically generates a `run_journal.json` and human-readable `provenance.md` document recording the exact raw data paths, Git commit hash, software versions, parameter settings, and step-by-step execution log.

---

## 📁 Repository Structure

```
analysis_src/
├── pyproject.toml              # Editable package configuration & dependencies
├── README.md                   # Project documentation & GitHub instructions
├── .gitignore                  # Git ignore rules for neurophys data & binary files
├── scripts/
│   └── run_local_preproc.py    # Main CLI runner (interactive checkpoint or batch mode)
├── src/
│   └── pipeline/
│       ├── config.py           # Parameter dataclasses & JSON schema helpers
│       ├── models.py           # Data structures for signals, events, & ephys
│       ├── journaler.py        # Interactive parameter checkpoints & provenance logger
│       ├── visualizer.py       # Diagnostic plotting engine (EMG SNR, ECG, Opto, Npx)
│       ├── io/
│       │   ├── io_nidq.py      # SpikeGLX NIDQ binary loader & synthetic data generator
│       │   └── io_ephys.py     # ALF Neuropixel spike loader & synthetic unit generator
│       ├── processors/
│       │   ├── proc_emg.py     # Diaphragm EMG filtering, envelope, SNR, & breath peaks
│       │   ├── proc_ecg.py     # ECG bandpass filter, R-peaks, HR, & HRV
│       │   ├── proc_sync_opto.py # Edge detection for time sync & optic laser TTLs
│       │   └── proc_ephys.py   # Unit spike alignment to breath onsets & depth profiles
│       └── pipeline_runner.py # Main orchestration engine
└── tests/
    └── test_pipeline.py        # Pytest test suite
```

---

## 🚀 Quickstart Guide

### 1. Run Interactive Checkpoint Mode
To run with interactive parameter confirmation and visual inspection:
```bash
python scripts/run_local_preproc.py --animal-id p05 --session-id session_01 --raw-dir ./raw_data --out-dir ./output
```

### 2. Run Non-Interactive Batch Mode
For automated script execution or cluster runs:
```bash
python scripts/run_local_preproc.py --animal-id p05 --session-id session_01 --non-interactive
```

### 3. Replicate Analysis from Saved Journal
To re-run an exact analysis using a saved provenance journal:
```bash
python scripts/run_local_preproc.py --config ./output/p05_session_01_run_journal.json --non-interactive
```

---

## 🐙 Linking to GitHub

To save your progress to GitHub:

1. **Create a new empty repository on GitHub** (e.g. `https://github.com/your-username/analysis_src.git`).
2. **Add remote and push initial commit**:
```bash
git remote add origin https://github.com/your-username/analysis_src.git
git branch -M main
git push -u origin main
```
