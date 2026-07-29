# Analysis Provenance & Reproduction Journal

- **Timestamp**: `2026-07-29T15:58:48.967025`
- **Subject**: `TP_Tacr1_1`
- **Session**: `npx2_g4`
- **Git Commit**: `b0bf8ee74a495ed265f5fce7f165d43644e55fcc (dirty)`
- **Python Version**: `3.12.13`
- **Raw Data Directory**: `Z:\ramirezlab\pascual\rec_data\TP_Tacr1_1_npx2_g4`

---

## 1. Approved Execution Parameters

### Diaphragm EMG Processing
- **Channel Index**: `0`
- **Sampling Frequency**: `10000.0 Hz`
- **Bandpass Filter**: `300.0 Hz` to `3000.0 Hz` (Notch `60.0 Hz`)
- **Rectification**: `True`
- **Smoothing Window**: `50.0 ms`
- **Peak Threshold**: `2.5 SD`
- **Apnea Threshold**: `2.0 s`

### ECG Processing
- **Channel Index**: `3`
- **Sampling Frequency**: `10000.0 Hz`
- **Bandpass Filter**: `5.0 Hz` to `100.0 Hz` (Notch `60.0 Hz`)
- **R-Peak Threshold**: `3.0 SD`

### Opto Laser & Time Sync
- **Sync Channel Index**: `7`
- **Opto Laser Channel Index**: `2`
- **Pulse Threshold**: `2.5 V`

---

## 2. Summary Results & Data Quality Metrics

```json
{
  "duration_seconds": 703.4100437505085,
  "breaths_detected": 856,
  "mean_respiratory_rate_bpm": 80.00294380953558,
  "emg_mean_snr_db": 40.14756618420838,
  "apnea_events_count": 10,
  "r_peaks_detected": 174,
  "mean_heart_rate_bpm": 43.884800648157686,
  "hrv_sdnn_ms": 3821.180567017202,
  "sync_pulses_count": 0,
  "opto_pulses_count": 1080,
  "neuropixel_units_count": 8,
  "diagnostic_plots": {
    "raw_all_channels": "output_TP_Tacr1_1\\TP_Tacr1_1_npx2_g4_raw_all_channels.png",
    "emg_stationarity": "output_TP_Tacr1_1\\TP_Tacr1_1_npx2_g4_emg_stationarity.png",
    "ecg_rpeaks": "output_TP_Tacr1_1\\TP_Tacr1_1_npx2_g4_ecg_rpeaks.png",
    "opto_sync": "output_TP_Tacr1_1\\TP_Tacr1_1_npx2_g4_opto_sync.png",
    "npx_drift": "output_TP_Tacr1_1\\TP_Tacr1_1_npx2_g4_npx_drift.png"
  }
}
```

---

## 3. Reproduction Instructions

To replicate this exact analysis from raw data, execute:
```bash
python scripts/run_local_preproc.py --config "output_TP_Tacr1_1\TP_Tacr1_1_npx2_g4_run_journal.json" --non-interactive
```
