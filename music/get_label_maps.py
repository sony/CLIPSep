import argparse
import pathlib

import utils


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--in_filename",
        default=pathlib.Path("MUSIC_solo_videos.json"),
        type=pathlib.Path,
        help="input filename",
    )
    parser.add_argument(
        "-o",
        "--out_filename",
        default=pathlib.Path("data/MUSIC/solo/labels.json"),
        type=pathlib.Path,
        help="output filename",
    )
    return parser.parse_args(args=args, namespace=namespace)


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()

    # Load the JSON file
    data = utils.load_json(args.in_filename)

    # Collect all labels
    all_labels = sorted(set(data["videos"].keys()))

    # Create the label map
    label_map = {label: i for i, label in enumerate(all_labels)}

    # Save the label map
    utils.save_json(args.out_filename, label_map)
    print(f"Saved the label map to {args.out_filename}.")


if __name__ == "__main__":
    main()
