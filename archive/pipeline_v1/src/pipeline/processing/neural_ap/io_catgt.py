"""
CatGT wrapper for Neuropixels AP preprocessing:
- SpikeGLX ADC Phase Shift Correction (-t_shift)
- High-Order Butterworth Bandpass Filtering (-apfilter)
- Local Common Average Referencing (-loccar_um)
"""
import subprocess
import re
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from src.pipeline.config import CatGTConfig

logger = logging.getLogger("IO_CATGT")

def parse_spikeglx_run_name(data_path: Path) -> Tuple[Path, str, int, int, int]:
    """
    Infers dir, run_name, gate (g), trigger (t), and probe index from a SpikeGLX folder structure.
    Example: Z:\\...\\LP21_fent_EXPT_g1\\LP21_fent_EXPT_g1_imec0 ->
             parent=Z:\\...\\, run='LP21_fent_EXPT', g=1, t=0, probe=0
    """
    data_path = Path(str(data_path).replace("/", "\\"))
    
    # Check if folder ends in _imec<N>
    probe_idx = 0
    folder_name = data_path.name
    match_probe = re.search(r"_imec(\d+)$", folder_name)
    if match_probe:
        probe_idx = int(match_probe.group(1))
        parent_dir = data_path.parent
        run_folder = data_path.parent.name
        data_root = data_path.parent.parent
    else:
        run_folder = folder_name
        data_root = data_path.parent

    # Parse gate from run_folder (e.g., LP21_fent_EXPT_g1)
    match_gate = re.search(r"^(.*)_g(\d+)$", run_folder)
    if match_gate:
        run_name = match_gate.group(1)
        gate_idx = int(match_gate.group(2))
    else:
        run_name = run_folder
        gate_idx = 1

    trigger_idx = 0
    return data_root, run_name, gate_idx, trigger_idx, probe_idx

def build_catgt_command(
    config: CatGTConfig,
    raw_dir: Path,
    dest_dir: Optional[Path] = None,
    start_secs: Optional[float] = None,
    max_secs: Optional[float] = None
) -> Tuple[list, Path]:
    """
    Builds the CLI argument list for CatGT.exe.
    """
    catgt_exe = Path(config.catgt_path)
    if not catgt_exe.exists():
        raise FileNotFoundError(f"CatGT executable not found at: {catgt_exe}")

    data_root, run_name, gate_idx, trigger_idx, probe_idx = parse_spikeglx_run_name(raw_dir)
    if config.run_name:
        run_name = config.run_name
    gate_idx = config.gate if config.gate is not None else gate_idx
    trigger_idx = config.trigger if config.trigger is not None else trigger_idx
    probe_idx = config.probe if config.probe is not None else probe_idx

    target_dest = dest_dir or (Path(config.dest_dir) if config.dest_dir else (raw_dir.parent / "catgt_processed"))
    target_dest.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(catgt_exe),
        f"-dir={str(data_root).replace('/', chr(92))}",
        f"-run={run_name}",
        f"-g={gate_idx}",
        f"-t={trigger_idx}",
        "-prb_fld",
        "-ap",
        f"-prb={probe_idx}",
    ]

    # In CatGT v2.2+, tshift is on by default. Use -no_tshift to disable
    if not config.t_shift:
        cmd.append("-no_tshift")

    if config.apfilter:
        cmd.append(f"-apfilter={config.apfilter}")

    if config.loccar_um:
        inner, outer = config.loccar_um
        cmd.append(f"-loccar_um={int(inner)},{int(outer)}")

    s_secs = start_secs if start_secs is not None else config.start_secs
    if s_secs is not None:
        cmd.append(f"-startsecs={s_secs}")

    m_secs = max_secs if max_secs is not None else config.max_secs
    if m_secs is not None:
        cmd.append(f"-maxsecs={m_secs}")

    cmd.append(f"-dest={target_dest}")

    return cmd, target_dest

def run_catgt(
    config: CatGTConfig,
    raw_dir: Path,
    dest_dir: Optional[Path] = None,
    start_secs: Optional[float] = None,
    max_secs: Optional[float] = None
) -> Tuple[Path, Path]:
    """
    Executes CatGT and returns (catgt_ap_bin_path, catgt_ap_meta_path).
    """
    cmd, target_dest = build_catgt_command(
        config=config,
        raw_dir=raw_dir,
        dest_dir=dest_dir,
        start_secs=start_secs,
        max_secs=max_secs
    )

    logger.info("Executing CatGT with command:")
    logger.info(" ".join(cmd))

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if proc.returncode != 0:
        logger.error(f"CatGT failed with returncode {proc.returncode}")
        logger.error(f"Stdout:\n{proc.stdout}")
        logger.error(f"Stderr:\n{proc.stderr}")
        raise RuntimeError(f"CatGT processing failed: {proc.stderr}")

    logger.info("CatGT execution completed successfully.")
    
    # Locate output .ap.bin in target_dest
    out_bin_files = list(target_dest.rglob("*.ap.bin"))
    if not out_bin_files:
        raise FileNotFoundError(f"CatGT completed but no .ap.bin found in {target_dest}")

    out_bin = out_bin_files[0]
    out_meta = out_bin.with_suffix(".meta")
    logger.info(f"CatGT AP Output: {out_bin} ({out_bin.stat().st_size / (1024*1024):.2f} MB)")
    return out_bin, out_meta
