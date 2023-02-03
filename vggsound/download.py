"""Download videos from YouTube."""
import argparse
import logging
import os
import pathlib
import sys
import time

import joblib
import tqdm
import youtube_dl

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


def download(youtube_id, downloader=youtube_dl):
    """Download a YouTube video."""
    downloader.download([f"https://www.youtube.com/watch?v={youtube_id}"])
    return youtube_id


@utils.suppress_outputs
@utils.ignore_exceptions
def download_ignore_exceptions(youtube_id, downloader=youtube_dl):
    """Download a YouTube video, ignoring all exceptions."""
    return download(youtube_id, downloader)


def process(
    youtube_id,
    ydl_options,
    out_dir,
    start,
    subset,
    skip_existing=False,
    ignore_exceptions=False,
):
    """Wrapper for multiprocessing."""
    # Update output filename template
    filename = (
        out_dir / subset / youtube_id[0] / youtube_id[1] / f"{youtube_id}.mp4"
    )
    ydl_options["outtmpl"] = str(filename)
    start_time_str = time.strftime("%H:%M:%S", time.gmtime(int(start)))
    ydl_options["postprocessor_args"] = f"--ss {start_time_str} -t 0:0:10"

    if skip_existing and filename.is_file():
        return

    # Create a downloader
    with youtube_dl.YoutubeDL(ydl_options) as ydl:
        ydl._err_file = open(os.devnull, "w")
        # Download the video
        if ignore_exceptions:
            return download_ignore_exceptions(youtube_id, ydl)
        return download(youtube_id, ydl)


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
    (args.out_dir / "train").mkdir(exist_ok=True)
    (args.out_dir / "test").mkdir(exist_ok=True)

    # Load the input CSV file
    data = utils.load_csv_text(args.in_filename, True)

    # Set Youtube downloader options
    ydl_options = {
        "format": "mp4",
        "quiet": True,
        "cachedir": False,
        "nooverwrites": args.skip_existing,
        "ignoreerrors": args.ignore_exceptions,
    }

    # Iterate over the ID lists
    logging.info(f"Iterating over the videos...")
    if args.jobs == 1:
        pbar = tqdm.tqdm(data, ncols=80)
        for youtube_id, start, _, subset in pbar:
            pbar.set_postfix_str(youtube_id)
            process(
                youtube_id,
                ydl_options,
                args.out_dir,
                start,
                subset,
                args.skip_existing,
                args.ignore_exceptions,
            )
    else:
        joblib.Parallel(args.jobs, verbose=5)(
            joblib.delayed(process)(
                youtube_id,
                ydl_options,
                args.out_dir,
                start,
                subset,
                args.skip_existing,
                args.ignore_exceptions,
            )
            for youtube_id, start, _, subset in data
        )


if __name__ == "__main__":
    main()
