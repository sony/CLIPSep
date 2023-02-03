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
        "-i", "--in_dir", type=pathlib.Path, help="input directory"
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    parser.add_argument(
        "-n", "--n_frames", type=int, help="number of frames to extract"
    )
    parser.add_argument("-f", "--fps", type=int, help="frames per second")
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


def get_duration(filename, ignore_exceptions):
    """Return the duration of a video."""
    metadata = ffmpeg.probe(filename)
    duration = None
    for stream in metadata["streams"]:
        if stream["codec_type"] == "video":
            duration = stream.get("duration")
            if duration is None:
                continue
            break
        continue

    # If duration information not found
    if duration is None:
        try:
            duration = metadata["format"]["duration"]
        except KeyError:
            if not ignore_exceptions:
                raise RuntimeError(
                    f"Cannot find duration information for: {filename}"
                )
            logging.error(f"Cannot find duration information for: {filename}")
            return

    return float(duration)


def process(
    filename, out_dir, n_frames, fps, skip_existing, ignore_exceptions, quiet
):
    """Process wrapper for multiprocessing."""
    # Extract filename components
    youtube_id = filename.stem
    instrument = filename.parent.name

    # Skip if existing
    if skip_existing and (out_dir / instrument / youtube_id).is_dir():
        return

    # Make sure the output directory exists
    (out_dir / instrument).mkdir(exist_ok=True)
    (out_dir / instrument / youtube_id).mkdir(exist_ok=True)

    if n_frames is not None:
        # Get the duration of the video
        duration = get_duration(filename, ignore_exceptions)

        # Extract n evenly-spaced frames from the video
        stream = ffmpeg.input(filename, ss=1)
        stream = ffmpeg.filter(
            stream,
            "fps",
            fps=10 / (duration - 5) if duration > 20 else 10 / duration,
            round="up",
        )
        stream = ffmpeg.output(
            stream,
            str(out_dir / instrument / youtube_id / "%6d.png"),
            vcodec="png",
            t=duration - 10 if duration > 20 else duration,
        )
        ffmpeg.run(stream, quiet=quiet, overwrite_output=True)

    # elif fps == 1:
    #     duration = get_duration(filename, ignore_exceptions)
    #     for t in range(int(duration)):
    #         count_trials = 0
    #         is_successful = False
    #         while not is_successful and count_trials < 10:
    #             try:
    #                 stream = ffmpeg.input(filename, ss=t + 0.01 * count_trials)
    #                 stream = ffmpeg.output(
    #                     stream,
    #                     str(
    #                         out_dir
    #                         / instrument
    #                         / youtube_id
    #                         / f"{t + 1:06d}.png"
    #                     ),
    #                     vframes=1,
    #                 )
    #                 ffmpeg.run(
    #                     stream,
    #                     capture_stdout=True,
    #                     capture_stderr=True,
    #                     overwrite_output=True,
    #                 )
    #                 is_successful = True
    #             except ffmpeg.Error:
    #                 count_trials += 1
    #         if not is_successful:
    #             raise RuntimeError(
    #                 f"Failed to extract the frame at {t}s for {filename}"
    #             )

    else:
        # Extract frames using the given fps
        stream = ffmpeg.input(filename)
        stream = ffmpeg.filter(stream, "fps", fps=fps, round="up")
        stream = ffmpeg.output(
            stream,
            str(out_dir / instrument / youtube_id / "%06d.png"),
            vcodec="png",
        )
        ffmpeg.run(stream, quiet=quiet, overwrite_output=True)

    return filename


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()

    if args.n_frames is None and args.fps is None:
        raise ValueError("`n_frames` and `fps` must not be both None.")
    elif args.n_frames is not None and args.fps is not None:
        raise ValueError("Either `n_frames` or `fps` must be given.")

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
                args.n_frames,
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
                args.n_frames,
                args.fps,
                args.skip_existing,
                args.ignore_exceptions,
                args.quiet,
            )
            for filename in filenames
        )


if __name__ == "__main__":
    main()
