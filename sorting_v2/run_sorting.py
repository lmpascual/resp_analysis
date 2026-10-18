"""Run the sorting pipeline from a config file.

    python run_sorting.py configs/e097_intact_window.yaml
    python run_sorting.py configs/e097_intact_window.yaml --redo postprocess
    python run_sorting.py configs/e097_intact_window.yaml --stop-after preprocess
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npxsort import load_config, run_pipeline, STAGES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="experiment YAML")
    parser.add_argument("--redo", choices=STAGES, help="re-run this stage and every stage after it")
    parser.add_argument("--stop-after", choices=STAGES, help="stop after this stage (e.g. to inspect motion first)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_pipeline(cfg, redo=args.redo, stop_after=args.stop_after)
    return 0


if __name__ == "__main__":  # required on Windows: worker processes re-import this file
    sys.exit(main())
