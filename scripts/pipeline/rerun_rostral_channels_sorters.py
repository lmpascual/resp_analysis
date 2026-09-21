#!/usr/bin/env python3
"""
Rerun Multi-Sorter Benchmark on Rostral Probe Tip Channels (Channels 0 - 23, Depths 0 - 165 µm)
Compares MountainSort 5, Kilosort 4, and Lupin specifically on rostral multi-unit channels.
Extracts waveform templates, SNR, ISI violations, and Allen Institute Single-Unit QC.
"""
import sys
import os
import shutil
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import spikeinterface.full as si

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.sorting.lupin_pipeline import run_lupin_sorting
from src.pipeline.sorting.postprocessing import export_sorting_alf_and_qc, export_to_phy_bundle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ROSTRAL_SORTER_BENCHMARK")

def main():
    catgt_meta = Path(r"C:\SCRATCH_neuropixels_test\catgt_output\catgt_LP21_fent_EXPT_g1\LP21_fent_EXPT_g1_tcat.imec0.ap.meta")
    out_base = Path(r"c:\Users\lpascu\analysis_src\output_LP21_fent_EXPT_g1\rostral_tip_benchmark")
    out_base.mkdir(parents=True, exist_ok=True)

    # 1. Load Preprocessed Recording
    logger.info(f"Loading CatGT recording from: {catgt_meta}")
    rec = si.read_spikeglx(catgt_meta.parent, stream_id="imec0.ap")
    
    # Select Rostral Channels 0 to 23 (First 24 channels, y = 0 to 165 µm)
    rostral_chan_ids = rec.channel_ids[:24]
    logger.info(f"Slicing recording to first 24 rostral channels: {rostral_chan_ids}")
    rec_rostral = rec.channel_slice(channel_ids=rostral_chan_ids)
    
    # Save a temporary preprocessed slice for ultra-fast sorting
    scratch_rec_dir = Path(r"C:\SCRATCH_neuropixels_test\rostral_slice_rec")
    if scratch_rec_dir.exists():
        shutil.rmtree(scratch_rec_dir)
    rec_rostral = rec_rostral.save(folder=scratch_rec_dir, n_jobs=4, chunk_duration="1s")
    logger.info(f"Saved rostral slice recording: {rec_rostral.get_num_channels()} channels, {rec_rostral.get_total_duration():.1f}s")

    sorters = ["mountainsort5", "kilosort4", "lupin"]
    results_summary = []

    for sorter_name in sorters:
        logger.info(f"\n{'='*70}\n RUNNING {sorter_name.upper()} ON ROSTRAL CHANNELS (0-23)\n{'='*70}")
        sorter_out_dir = out_base / sorter_name
        if sorter_out_dir.exists():
            shutil.rmtree(sorter_out_dir)

        try:
            if sorter_name == "mountainsort5":
                sorting = si.run_sorter(
                    sorter_name="mountainsort5",
                    recording=rec_rostral,
                    output_folder=sorter_out_dir / "si_output",
                    scheme="2",  # scheme 2 for dense multielectrode
                    detect_threshold=5.0,
                    verbose=True
                )
            elif sorter_name == "kilosort4":
                sorting = si.run_sorter(
                    sorter_name="kilosort4",
                    recording=rec_rostral,
                    output_folder=sorter_out_dir / "si_output",
                    Th_universal=8,
                    Th_learned=7,
                    verbose=True
                )
            elif sorter_name == "lupin":
                sorting = run_lupin_sorting(
                    recording=rec_rostral,
                    output_folder=sorter_out_dir / "si_output",
                    n_jobs=4,
                    verbose=True
                )

            # Compute Quality Metrics & Export ALF / QC summary
            qc_df = export_sorting_alf_and_qc(
                sorting=sorting,
                fs=rec_rostral.get_sampling_frequency(),
                duration_s=rec_rostral.get_total_duration(),
                out_dir=sorter_out_dir,
                sorter_name=sorter_name,
                recording=rec_rostral
            )

            # Save summary CSV
            csv_path = sorter_out_dir / f"{sorter_name}_rostral_qc_summary.csv"
            qc_df.to_csv(csv_path, index=False)

            n_total = len(qc_df)
            n_sua = int(np.sum(qc_df["passed_allen_qc"]))
            n_mua = n_total - n_sua

            logger.info(f"[{sorter_name.upper()}] Done: {n_total} Total Clusters -> {n_sua} SUA (Single Units), {n_mua} MUA/Noise")
            results_summary.append({
                "sorter": sorter_name,
                "total_clusters": n_total,
                "sua_single_units": n_sua,
                "mua_noise_clusters": n_mua,
                "mean_firing_rate_hz": qc_df["firing_rate_hz"].mean(),
                "mean_snr": qc_df["snr"].mean(),
                "mean_isi_violations": qc_df["isi_violations"].mean() if "isi_violations" in qc_df.columns else np.nan
            })

        except Exception as e:
            logger.error(f"Failed running {sorter_name}: {e}", exc_info=True)

    df_res = pd.DataFrame(results_summary)
    df_res.to_csv(out_base / "rostral_sorters_comparison_summary.csv", index=False)
    print("\n" + "="*80)
    print(" ROSTRAL CHANNELS (0-23) MULTI-SORTER BENCHMARK COMPLETE")
    print("="*80)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
