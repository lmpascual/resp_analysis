"""
Programmatic Multi-Sorter Runner:
Executes Kilosort 4, MountainSort 5, SpyKING-CIRCUS 2, Tridesclous 2, and Lupin
without manual GUI interaction, ensuring complete parameter reproducibility.
"""
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

import spikeinterface.full as si
import spikeinterface.sorters as ss

from ..config import SorterConfig, LupinConfig
from .lupin_pipeline import run_lupin_sorting
from .postprocessing import export_sorting_alf_and_qc

logger = logging.getLogger("SORTER_RUNNER")

def run_single_sorter(
    sorter_name: str,
    recording: si.BaseRecording,
    output_dir: Path,
    scratch_dir: Path,
    config: Optional[SorterConfig] = None,
    lupin_config: Optional[LupinConfig] = None,
    verbose: bool = True
) -> Tuple[si.BaseSorting, pd.DataFrame, float]:
    """
    Runs a single spike sorter and returns (sorting_obj, df_qc_metrics, runtime_s).
    """
    config = config or SorterConfig()
    sorter_name = sorter_name.lower().strip()
    out_sorter_dir = output_dir / sorter_name
    scratch_sorter_dir = scratch_dir / f"{sorter_name}_working"
    
    out_sorter_dir.mkdir(parents=True, exist_ok=True)
    if scratch_sorter_dir.exists():
        shutil.rmtree(scratch_sorter_dir, ignore_errors=True)

    fs = recording.get_sampling_frequency()
    dur_s = recording.get_total_duration()

    logger.info(f"[{sorter_name.upper()}] Starting spike sorting...")
    t0 = time.time()

    if sorter_name == "kilosort4":
        ks4_params = dict(
            do_CAR=config.do_CAR,
            do_correction=True,
            save_extra_vars=True
        )
        sorting = ss.run_sorter(
            sorter_name='kilosort4',
            recording=recording,
            folder=str(scratch_sorter_dir),
            remove_existing_folder=True,
            verbose=verbose,
            **ks4_params
        )

    elif sorter_name == "mountainsort5":
        ms5_params = ss.get_default_sorter_params('mountainsort5')
        ms5_params['scheme'] = '2'
        ms5_params['detect_threshold'] = 5.5
        ms5_params['filter'] = False
        ms5_params['whiten'] = False
        sorting = ss.run_sorter(
            sorter_name='mountainsort5',
            recording=recording,
            folder=str(scratch_sorter_dir),
            remove_existing_folder=True,
            verbose=verbose,
            **ms5_params
        )

    elif sorter_name == "lupin":
        sorting = run_lupin_sorting(
            recording=recording,
            output_folder=out_sorter_dir,
            config=lupin_config,
            n_jobs=config.n_jobs,
            verbose=verbose
        )

    elif sorter_name == "spykingcircus2":
        sc2_params = ss.get_default_sorter_params('spykingcircus2')
        sc2_params['apply_preprocessing'] = False
        sc2_params['apply_motion_correction'] = False
        sc2_params['job_kwargs'] = {'n_jobs': config.n_jobs}
        sorting = ss.run_sorter(
            sorter_name='spykingcircus2',
            recording=recording,
            folder=str(scratch_sorter_dir),
            remove_existing_folder=True,
            verbose=verbose,
            **sc2_params
        )

    elif sorter_name == "tridesclous2":
        tdc2_params = ss.get_default_sorter_params('tridesclous2')
        tdc2_params['apply_preprocessing'] = False
        tdc2_params['apply_motion_correction'] = False
        sorting = ss.run_sorter(
            sorter_name='tridesclous2',
            recording=recording,
            folder=str(scratch_sorter_dir),
            remove_existing_folder=True,
            verbose=verbose,
            **tdc2_params
        )

    else:
        raise ValueError(f"Unsupported sorter: {sorter_name}")

    t_elapsed = time.time() - t0
    logger.info(f"[{sorter_name.upper()}] Sorting finished in {t_elapsed:.1f}s. Found {len(sorting.get_unit_ids())} units.")

    # Export ALF and QC Metrics
    df_qc = export_sorting_alf_and_qc(
        sorting=sorting,
        fs=fs,
        duration_s=dur_s,
        out_dir=out_sorter_dir,
        sorter_name=sorter_name.upper(),
        recording=recording
    )

    return sorting, df_qc, t_elapsed

def run_multi_sorter_suite(
    sorter_names: List[str],
    recording: si.BaseRecording,
    base_out_dir: Path,
    scratch_dir: Path,
    config: Optional[SorterConfig] = None,
    lupin_config: Optional[LupinConfig] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Executes a list of spike sorters sequentially and compiles comparative outputs.
    """
    results = {}
    for name in sorter_names:
        name_clean = name.strip()
        if not name_clean:
            continue
        try:
            sorting_obj, df_qc, runtime_s = run_single_sorter(
                sorter_name=name_clean,
                recording=recording,
                output_dir=base_out_dir,
                scratch_dir=scratch_dir,
                config=config,
                lupin_config=lupin_config
            )
            results[name_clean] = {
                "sorting": sorting_obj,
                "df_qc": df_qc,
                "runtime_s": runtime_s,
                "n_units": len(df_qc),
                "n_qc_passed": df_qc["passed_allen_qc"].sum() if len(df_qc) > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error executing sorter '{name_clean}': {e}", exc_info=True)

    return results
