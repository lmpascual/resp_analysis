#!/usr/bin/env python3
"""
Re-render Lupin subwindow rasters for e097 with the correct NIDQ start offset
(companion to fix_subwindows_depths_and_rasters.py, which only handled Kilosort 4).

Lupin depths come from channel geometry and are already within the probe span,
so only the NIDQ synchronization is re-applied here. Sorter names are kept
identical to the original renders so the *_sorter_comparison.html links stay valid.
"""
import re
import sys
import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.visualization.plot_interactive_sorter_raster import build_interactive_sorter_raster

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FIX_LUPIN_SUBWINDOW_RASTERS")

SUBWINDOWS = [
    ("win1_baseline", "Lupin (SVD-OMP) - Window 1 Baseline", 1740.0),
    ("win2_anox1", "Lupin (SVD-OMP) - Anoxic Challenge 1 (Gas Switch)", 1800.0),
    ("win3_anox2", "Lupin (SVD-OMP) - Anoxic Challenge 2 (Apnea / HR Nadir)", 1836.0),
    ("win4_gasp_early", "Lupin (SVD-OMP) - Early Gasping (Autoresuscitation Surge)", 1860.0),
    ("win5_gasp_late", "Lupin (SVD-OMP) - Late Gasping (Tachypneic Transition)", 1950.0),
    ("win6_flow_shift", "Lupin (SVD-OMP) - Active Expiration Shift", 2070.0),
    ("win7_recovery", "Lupin (SVD-OMP) - Post-Challenge Recovery", 2136.0),
]


def main():
    base = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting"
    for w_id, name, t_start in SUBWINDOWS:
        lupin_dir = base / w_id / "lupin"
        if not lupin_dir.exists():
            logger.warning(f"{w_id}: no lupin directory, skipping")
            continue
        logger.info(f"Re-rendering {w_id} Lupin raster with start_t_s = {t_start:.1f}s")
        build_interactive_sorter_raster(sorter_dir=lupin_dir, sorter_name=name, start_t_s=t_start)

    # Verify every raster link in the comparison pages resolves to a real file
    logger.info("Checking comparison-page links...")
    for comp in sorted(base.glob("win*/*_sorter_comparison.html")):
        html = comp.read_text(encoding="utf-8", errors="ignore")
        for link in sorted(set(re.findall(r'(?:src|href)="([^"#?]+\.html)"', html))):
            if link.startswith("http"):
                continue
            ok = (comp.parent / link).exists()
            if not ok:
                logger.warning(f"  BROKEN LINK in {comp.name}: {link}")


if __name__ == "__main__":
    main()
