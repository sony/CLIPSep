"""Extract frames from videos."""
import argparse
import logging
import pathlib
import sys

import ffmpeg
import joblib
import tqdm

import utils


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument(
        "-i",
        "--in_dir",
        type=pathlib.Path,
        required=True,
        help="input directory",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=pathlib.Path,
        required=True,
        help="output directory",
    )
    parser.add_argument(
        "-r", "--rate", default=11025, type=int, help="sampling rate"
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


def process(filename, out_dir, rate, skip_existing, ignore_exceptions, quiet):
    """Process wrapper for multiprocessing."""
    # Extract filename components
    youtube_id = filename.stem
    instrument = filename.parent.name

    # Skip if existing
    if (
        skip_existing
        and (out_dir / instrument / f"{youtube_id}.mp3").is_file()
    ):
        return

    # Make sure the output directory exists
    (out_dir / instrument).mkdir(exist_ok=True)

    # Extract audio
    stream = ffmpeg.input(filename, vn=None)
    stream = ffmpeg.output(
        stream,
        str(out_dir / instrument / f"{youtube_id}.mp3"),
        ar=rate,
        acodec="mp3",
    )
    try:
        ffmpeg.run(
            stream,
            capture_stdout=True,
            capture_stderr=True,
            overwrite_output=True,
        )
    except ffmpeg.Error:
        if not ignore_exceptions:
            raise RuntimeError(f"Failed to extract the audio for {filename}")
    return filename


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
        for filename in (pbar := tqdm.tqdm(filenames, ncols=80)) :
            pbar.set_postfix_str(filename.stem)
            process(
                filename,
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
