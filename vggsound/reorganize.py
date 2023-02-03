"""Reorganize the dataset."""
import argparse
import logging
import pathlib
import shutil
import sys

import tqdm

import utils


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Reorganize the dataset.")
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input directory"
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    parser.add_argument(
        "-m", "--mode", choices=("video", "audio", "frames"), help="mode"
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

    # Set up a console logger
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format="%(message)s"
    )

    # Make sure the output directory exists
    args.out_dir.mkdir(exist_ok=True)
    (args.out_dir / args.mode).mkdir(exist_ok=True)

    logging.info("Start processing the files...")
    if args.mode == "video":
        # Iterate over all the videos
        filenames = list((args.in_dir / "video").rglob("*.mp4"))
        pbar = tqdm.tqdm(filenames, ncols=80)
        for filename in pbar:
            youtube_id = filename.stem
            pbar.set_postfix_str(youtube_id)

            # Copy video file
            suffix = f"{youtube_id[0]}/{youtube_id[1]}/{youtube_id}"
            out_filename = args.out_dir / "video" / f"{suffix}.mp4"
            out_filename.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(filename, out_filename)

    elif args.mode == "audio":
        # Iterate over all the audio files
        filenames = list((args.in_dir / "audio").rglob("*.wav"))
        pbar = tqdm.tqdm(filenames, ncols=80)
        for filename in pbar:
            youtube_id = filename.stem
            pbar.set_postfix_str(youtube_id)

            # Copy audio file
            suffix = f"{youtube_id[0]}/{youtube_id[1]}/{youtube_id}"
            out_filename = args.out_dir / "audio" / f"{suffix}.wav"
            out_filename.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(filename, out_filename)

    elif args.mode == "frames":
        # Iterate over all the frame files
        filenames = list((args.in_dir / "frames").rglob("000001.jpg"))
        pbar = tqdm.tqdm(filenames, ncols=80)
        for filename in pbar:
            youtube_id = filename.parent.stem
            pbar.set_postfix_str(youtube_id)

            # Copy frames
            suffix = f"{youtube_id[0]}/{youtube_id[1]}/{youtube_id}"
            out_filename = args.out_dir / "frames" / suffix
            out_filename.parent.mkdir(exist_ok=True, parents=True)
            shutil.copytree(filename.parent, out_filename, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
