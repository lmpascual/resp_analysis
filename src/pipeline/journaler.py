"""
Provenance Journaler & Interactive Parameter Checkpoint Module.
Records exact execution metadata, git commit hash, raw data paths, and parameter overrides.
"""
import sys
import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
from .config import PipelineConfig

def get_git_commit_hash() -> str:
    """Retrieve the current Git commit hash if in a repository."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
        if dirty:
            commit += " (dirty)"
        return commit
    except Exception:
        return "git_commit_unknown"

def checkpoint_approve_parameters(config: PipelineConfig, interactive: bool = True) -> Tuple[PipelineConfig, bool]:
    """
    Present an interactive parameter approval checkpoint.
    Allows user to review, edit parameters, or approve as-is.
    """
    print("\n" + "="*80)
    print(" [CHECKPOINT] PIPELINE PARAMETER CHECKPOINT & REPRODUCIBILITY JOURNAL")
    print("="*80)
    print(f" Subject ID   : {config.animal_id}")
    print(f" Session ID   : {config.session_id}")
    print(f" Raw Directory: {config.raw_data_dir}")
    print(f" Output Dir   : {config.output_dir}")
    print("\n[CONFIGURED PROCESSING PARAMETERS]")
    print(f" - EMG Channel Index: {config.emg.channel_index} | Fs: {config.emg.fs} Hz")
    print(f"    Passband: {config.emg.highpass_cutoff} Hz - {config.emg.lowpass_cutoff} Hz | Notch: {config.emg.notch_freq} Hz")
    print(f"    Smoothing Window: {config.emg.smoothing_window_ms} ms | Peak Threshold: {config.emg.peak_threshold_sd} SD")
    print(f"    Apnea Threshold: {config.emg.apnea_threshold_s} s")
    print(f" - ECG Channel Index: {config.ecg.channel_index} | Fs: {config.ecg.fs} Hz")
    print(f"    Passband: {config.ecg.highpass_cutoff} Hz - {config.ecg.lowpass_cutoff} Hz | R-Peak Threshold: {config.ecg.r_peak_threshold_sd} SD")
    print(f" - Sync Channel Index: {config.sync_opto.sync_channel_index} | Opto Channel Index: {config.sync_opto.opto_channel_index}")
    print(f"    Pulse TTL Threshold: {config.sync_opto.pulse_threshold_v} V")
    print(f" - Neuropixel AP Fs: {config.ephys.fs_ap} Hz | PSTH Bin: {config.ephys.bin_size_ms} ms")
    print("="*80)

    if not interactive:
        print(" -> Non-interactive mode enabled. Parameters approved automatically.")
        return config, True

    while True:
        choice = input("\nDo you approve these parameters to proceed? [y = Yes / e = Edit / q = Quit]: ").strip().lower()
        if choice in ['y', 'yes', '']:
            print(" -> Parameters approved by user.")
            return config, True
        elif choice in ['q', 'quit']:
            print(" -> Processing cancelled by user.")
            return config, False
        elif choice in ['e', 'edit']:
            print("\n--- PARAMETER OVERRIDE MENU ---")
            print("1. EMG Bandpass Highpass (current: {} Hz)".format(config.emg.highpass_cutoff))
            print("2. EMG Smoothing Window (current: {} ms)".format(config.emg.smoothing_window_ms))
            print("3. EMG Peak Threshold SD (current: {} SD)".format(config.emg.peak_threshold_sd))
            print("4. ECG R-Peak Threshold SD (current: {} SD)".format(config.ecg.r_peak_threshold_sd))
            print("5. Opto Pulse TTL Threshold (current: {} V)".format(config.sync_opto.pulse_threshold_v))
            print("6. Done Editing")
            
            sub = input("Select parameter to edit (1-6): ").strip()
            if sub == '1':
                val = input("Enter new EMG highpass Hz: ").strip()
                if val: config.emg.highpass_cutoff = float(val)
            elif sub == '2':
                val = input("Enter new EMG smoothing window ms: ").strip()
                if val: config.emg.smoothing_window_ms = float(val)
            elif sub == '3':
                val = input("Enter new EMG peak threshold SD: ").strip()
                if val: config.emg.peak_threshold_sd = float(val)
            elif sub == '4':
                val = input("Enter new ECG R-peak threshold SD: ").strip()
                if val: config.ecg.r_peak_threshold_sd = float(val)
            elif sub == '5':
                val = input("Enter new Opto pulse TTL threshold V: ").strip()
                if val: config.sync_opto.pulse_threshold_v = float(val)
            elif sub == '6':
                print(" -> Edits updated. Re-displaying parameters.")

def write_run_journal(config: PipelineConfig, execution_summary: Dict[str, Any]) -> Path:
    """
    Write detailed run_journal.json and provenance.md for 100% exact reproduction.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    git_commit = get_git_commit_hash()
    
    journal_data = {
        "timestamp": timestamp,
        "git_commit": git_commit,
        "python_version": sys.version,
        "platform": sys.platform,
        "animal_id": config.animal_id,
        "session_id": config.session_id,
        "raw_data_dir": str(config.raw_data_dir),
        "config": config.to_dict(),
        "execution_summary": execution_summary
    }

    # 1. JSON journal
    journal_json_path = output_dir / f"{config.animal_id}_{config.session_id}_run_journal.json"
    with open(journal_json_path, 'w') as f:
        json.dump(journal_data, f, indent=4)

    # 2. Markdown provenance report
    provenance_md_path = output_dir / f"{config.animal_id}_{config.session_id}_provenance.md"
    md_content = f"""# Analysis Provenance & Reproduction Journal

- **Timestamp**: `{timestamp}`
- **Subject**: `{config.animal_id}`
- **Session**: `{config.session_id}`
- **Git Commit**: `{git_commit}`
- **Python Version**: `{sys.version.split()[0]}`
- **Raw Data Directory**: `{config.raw_data_dir}`

---

## 1. Approved Execution Parameters

### Diaphragm EMG Processing
- **Channel Index**: `{config.emg.channel_index}`
- **Sampling Frequency**: `{config.emg.fs} Hz`
- **Bandpass Filter**: `{config.emg.highpass_cutoff} Hz` to `{config.emg.lowpass_cutoff} Hz` (Notch `{config.emg.notch_freq} Hz`)
- **Rectification**: `{config.emg.rectify}`
- **Smoothing Window**: `{config.emg.smoothing_window_ms} ms`
- **Peak Threshold**: `{config.emg.peak_threshold_sd} SD`
- **Apnea Threshold**: `{config.emg.apnea_threshold_s} s`

### ECG Processing
- **Channel Index**: `{config.ecg.channel_index}`
- **Sampling Frequency**: `{config.ecg.fs} Hz`
- **Bandpass Filter**: `{config.ecg.highpass_cutoff} Hz` to `{config.ecg.lowpass_cutoff} Hz` (Notch `{config.ecg.notch_freq} Hz`)
- **R-Peak Threshold**: `{config.ecg.r_peak_threshold_sd} SD`

### Opto Laser & Time Sync
- **Sync Channel Index**: `{config.sync_opto.sync_channel_index}`
- **Opto Laser Channel Index**: `{config.sync_opto.opto_channel_index}`
- **Pulse Threshold**: `{config.sync_opto.pulse_threshold_v} V`

---

## 2. Summary Results & Data Quality Metrics

```json
{json.dumps(execution_summary, indent=2)}
```

---

## 3. Reproduction Instructions

To replicate this exact analysis from raw data, execute:
```bash
python scripts/run_local_preproc.py --config "{journal_json_path}" --non-interactive
```
"""
    with open(provenance_md_path, 'w') as f:
        f.write(md_content)

    print(f" -> Journal & provenance records created:")
    print(f"    - {journal_json_path}")
    print(f"    - {provenance_md_path}")
    return journal_json_path
