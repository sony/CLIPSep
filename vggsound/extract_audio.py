"""Extract audio from videos."""
import argparse
import logging
import pathlib
import pprint
import subprocess
import sys

import joblib
import tqdm

import utils


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract audio from videos.")
    parser.add_argument(
        "-i",
        "--in_dir",
        default=pathlib.Path("data/vggsound/vggsound/"),
        type=pathlib.Path,
        help="input directory",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default=pathlib.Path("data/vggsound/audio/"),
        type=pathlib.Path,
        help="output directory",
    )
    parser.add_argument(
        "-r", "--rate", default=16000, type=int, help="sampling rate"
    )
    parser.add_argument(
        "-s",
        "--skip_existing",
        action="store_true",
        help="whether to skip existing outputs",
    )
    parser.add_argument(
        "-e",
        "--ignore_exceptions",
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


def extract(filename, out_filename, rate):
    """Extract the audio from a video."""
    subprocess.check_output(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            filename,
            "-vn",
            "-ar",
            str(rate),
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            out_filename,
        ]
    )
    return filename


@utils.suppress_outputs
@utils.ignore_exceptions
def extract_ignore_exceptions(filename, out_filename, rate):
    """Extract the audio from a video, ignoring all exceptions."""
    return extract(filename, out_filename, rate)


def process(
    filename, in_dir, out_dir, rate, skip_existing, ignore_exceptions, quiet
):
    """Process wrapper for multiprocessing."""
    # Get output filename
    out_filename = out_dir / filename.relative_to(in_dir).with_suffix(".wav")

    # Skip if exists
    if skip_existing and out_filename.is_file():
        return

    # Make sure the output directory exists
    out_filename.parent.mkdir(exist_ok=True, parents=True)

    # Extract the audio
    if ignore_exceptions:
        return extract_ignore_exceptions(filename, out_filename, rate)
    return extract(filename, out_filename, rate)


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()

    # Set up a console logger
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
    )

    # Log command called
    logging.info(f"Running command: python {' '.join(sys.argv)}")

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

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
                args.in_dir,
                args.out_dir,
                args.rate,
                args.skip_existing,
                args.ignore_exceptions,
                args.quiet,
            )
    else:
        joblib.Parallel(n_jobs=args.jobs, verbose=5)(
            joblib.delayed(process)(
                filename,
                args.in_dir,
                args.out_dir,
                args.rate,
                args.skip_existing,
                args.ignore_exceptions,
                args.quiet,
            )
            for filename in filenames
        )


if __name__ == "__main__":
    main()
