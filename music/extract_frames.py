"""Extract frames from videos."""
import argparse
import logging
import pathlib
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
        "-i", "--in_dir", type=pathlib.Path, help="input directory"
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    parser.add_argument(
        "-f", "--fps", default=1, type=int, help="frames per second"
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
        "-j", "--jobs", default=1, type=int, help="number of jobs"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)


def extract(filename, out_filename, fps):
    """Extract the audio from a video."""
    # Extract frames using the given fps
    subprocess.check_output(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            filename,
            "-r",
            str(fps),
            out_filename,
        ]
    )
    return filename


@utils.suppress_outputs
@utils.ignore_exceptions
def extract_ignore_exceptions(filename, out_filename, fps):
    """Extract the audio from a video, ignoring all exceptions."""
    return extract(filename, out_filename, fps)


def process(filename, out_dir, fps, skip_existing, ignore_exceptions, quiet):
    """Process wrapper for multiprocessing."""
    # Extract filename components
    youtube_id = filename.stem
    instrument = filename.parent.name
    out_filename = out_dir / instrument / youtube_id / "%6d.jpg"

    # Skip if exists
    if skip_existing and out_filename.parent.is_dir():
        return

    # Make sure the output directory exists
    out_filename.parent.mkdir(exist_ok=True, parents=True)

    # Extract frames using the given fps
    if ignore_exceptions:
        return extract_ignore_exceptions(filename, out_filename, fps)
    return extract(filename, out_filename, fps)


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()

    # Set up a console logger
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format="%(message)s"
    )

    # Make sure the output directory exists
    args.out_dir.mkdir(exist_ok=True)

    # Iterate over all the MP4 files
    filenames = list(args.in_dir.rglob("*.mp4"))
    if args.jobs == 1:
        pbar = tqdm.tqdm(filenames, ncols=80)
        for filename in pbar:
            pbar.set_postfix_str(filename.stem)
            process(
                filename,
                args.out_dir,
                args.fps,
                args.skip_existing,
                args.ignore_exceptions,
                args.quiet,
            )
    else:
        joblib.Parallel(n_jobs=args.jobs, verbose=5)(
            joblib.delayed(process)(
                filename,
                args.out_dir,
                args.fps,
                args.skip_existing,
                args.ignore_exceptions,
                args.quiet,
            )
            for filename in filenames
        )


if __name__ == "__main__":
    main()
