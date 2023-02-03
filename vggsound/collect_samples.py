"""Extract frames from videos."""
import argparse
import csv
import logging
import pathlib
import pprint
import shutil
import subprocess
import sys

import joblib
import tqdm

import utils


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument(
        "-f", "--filename", type=pathlib.Path, help="input filename"
    )
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input directory"
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    parser.add_argument(
        "-n", "--n_samples", default=500, type=int, help="number of samples"
    )
    parser.add_argument(
        "-s",
        "--skip_existing",
        default=False,
        action="store_true",
        help="whether to skip existing outputs",
    )
    parser.add_argument(
        "-e",
        "--ignore_exceptions",
        default=False,
        action="store_true",
        help="whether to ignore all exceptions",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()

    # Make sure the output directory exists
    args.out_dir.mkdir(exist_ok=True)

    # Read samples
    for i, row in enumerate(
        csv.reader(open(args.filename, "r"), delimiter=",")
    ):
        if i >= args.n_samples:
            break
        video_id = pathlib.Path(row[0]).stem
        filename = args.in_dir / video_id[0] / video_id[1] / f"{video_id}.mp4"
        out_filename = args.out_dir / f"{video_id}.mp4"
        shutil.copyfile(filename, out_filename)


if __name__ == "__main__":
    main()
