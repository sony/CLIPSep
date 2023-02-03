import argparse
import pathlib
import random


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--root_audio",
        type=pathlib.Path,
        help="root for extracted audio files",
    )
    parser.add_argument(
        "-f",
        "--root_frame",
        type=pathlib.Path,
        help="root for extracted video frames",
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    parser.add_argument(
        "--fps", default=1, type=int, help="fps of video frames"
    )
    parser.add_argument(
        "--ratio",
        default=0.2,
        type=float,
        help="percentage of the validation set",
    )
    parser.add_argument("--seed", default=1234, type=int, help="manual seed")
    return parser.parse_args(args=args, namespace=namespace)


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()

    # Set random seed
    random.seed(args.seed)

    # find all audio/frames pairs
    infos = []
    filenames = list(args.root_audio.rglob("*.wav"))
    for filename in filenames:
        frame_dir = args.root_frame / filename.parent.name / filename.stem
        n_frames = len(list(frame_dir.rglob("*.jpg")))
        label = filename.parent.name
        if n_frames > args.fps * 5:
            infos.append(f"{filename},{frame_dir},{n_frames},{label}")
    print(f"{len(infos)} audio/frames pairs found.")

    # split train/val
    n_train = int(len(infos) * (1 - args.ratio))
    random.shuffle(infos)
    trainset = infos[:n_train]
    valset = infos[n_train:]
    for name, subset in zip(("train", "val"), (trainset, valset)):
        filename = args.out_dir / f"{name}.csv"
        with open(filename, "w") as f:
            for item in subset:
                f.write(item + "\n")
        print(f"{len(subset)} items saved to {filename}.")

    print("Done!")


if __name__ == "__main__":
    main()
