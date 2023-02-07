"""Download videos from YouTube."""
import argparse
import logging
import pathlib
import subprocess
import sys

import joblib
import pafy
import tqdm

import utils


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download videos from YouTube."
    )
    parser.add_argument(
        "-i",
        "--in_filename",
        default=pathlib.Path("data/vggsound/vggsound.csv"),
        type=pathlib.Path,
        help="input filename",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default=pathlib.Path("data/vggsound/vggsound"),
        type=pathlib.Path,
        help="output directory",
    )
    parser.add_argument(
        "-d", "--duration", type=int, default=10, help="video duration"
    )
    parser.add_argument(
        "-t",
        "--trials",
        type=int,
        default=5,
        help="number of failed trials before skipping",
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
        "-j", "--jobs", type=int, default=1, help="number of jobs"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)


def download(youtube_id, start, duration, filename):
    """Download a YouTube video."""
    # Get the URL
    youtube_url = "https://www.youtube.com/watch?v={}".format(youtube_id)
    video = pafy.new(youtube_url)
    url = video.getbest().url

    # Call ffmpeg to download the video
    subprocess.check_output(
        [
            "ffmpeg",
            "-loglevel",
            "fatal",
            "-y",
            "-ss",
            str(start),
            "-i",
            url,
            "-t",
            str(duration),
            filename,
        ]
    )
    return url


@utils.suppress_outputs
@utils.ignore_exceptions
def download_ignore_exceptions(url, start, duration, filename):
    """Download a YouTube video, ignoring all exceptions."""
    return download(url, start, duration, filename)


def process(
    youtube_id,
    out_dir,
    start,
    duration,
    subset,
    trials=1,
    skip_existing=False,
    ignore_exceptions=False,
):
    """Wrapper for multiprocessing."""
    out_filename = (
        out_dir / subset / youtube_id[0] / youtube_id[1] / f"{youtube_id}.mp4"
    )
    if skip_existing and out_filename.is_file():
        return
    out_filename.parent.mkdir(exist_ok=True, parents=True)

    count = 0
    while count < trials:
        # Download the video
        if ignore_exceptions:
            download_ignore_exceptions(
                youtube_id, start, duration, out_filename
            )
        else:
            download(youtube_id, start, duration, out_filename)

        if out_filename.is_file():
            break

        # Increment failure counter
        count += 1

    # Log the failure case
    if count >= trials:
        logging.warning(f"Failed on: {youtube_id}")

    return youtube_id


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()

    # Set up a console logger
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format="%(message)s"
    )

    # Make sure the output directory exists
    args.out_dir.mkdir(exist_ok=True, parents=True)
    (args.out_dir / "train").mkdir(exist_ok=True)
    (args.out_dir / "test").mkdir(exist_ok=True)

    # Load the input CSV file
    data = utils.load_csv_text(args.in_filename, True)

    # Iterate over the ID lists
    logging.info(f"Iterating over the videos...")
    if args.jobs == 1:
        pbar = tqdm.tqdm(data, ncols=80)
        for youtube_id, start, _, subset in pbar:
            pbar.set_postfix_str(youtube_id)
            process(
                youtube_id,
                args.out_dir,
                start,
                args.duration,
                subset,
                args.trials,
                args.skip_existing,
                args.ignore_exceptions,
            )
    else:
        joblib.Parallel(args.jobs, verbose=5)(
            joblib.delayed(process)(
                youtube_id,
                args.out_dir,
                start,
                args.duration,
                subset,
                args.trials,
                args.skip_existing,
                args.ignore_exceptions,
            )
            for youtube_id, start, _, subset in data
        )


if __name__ == "__main__":
    main()
