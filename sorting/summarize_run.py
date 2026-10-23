"""Write run_summary.md for a finished run (the pipeline also writes it at the end of every run).

    python summarize_run.py <outputs>/e097/intact_window_v2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npxsort.summary import write_run_summary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="the run's paths.output_dir")
    args = parser.parse_args()
    print(write_run_summary(args.output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
