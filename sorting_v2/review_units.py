"""Open the unit trace-review browser for a finished run.

    python review_units.py C:/Users/lpascu/resp_analysis/outputs/e097/intact_window_v2
    python review_units.py <output_dir> --time-offset-s 1740      # also show raw-run time (e097: CatGT -startsecs)
    python review_units.py <output_dir> --scratch-dir D:/moved_scratch --port 8800 --no-browser

Needs the run's scratch folder (the preprocessed binary). Leave this window open while reviewing;
Ctrl+C stops the server. Verdicts and notes go to <output_dir>/review/<sorter>_unit_review.tsv.
"""
import argparse
import logging
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npxsort.review import ReviewData, serve  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="the run's paths.output_dir (holds config_used.yaml, <sorter>/units.tsv)")
    parser.add_argument("--scratch-dir", help="override paths.scratch_dir from config_used.yaml")
    parser.add_argument("--time-offset-s", type=float, default=0.0, help="added to file time to display raw-run time")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    data = ReviewData(Path(args.output_dir), args.scratch_dir, args.time_offset_s)
    server = serve(data, args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"\nUnit review running at {url}  (Ctrl+C to stop)\n", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
