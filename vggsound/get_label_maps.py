import argparse
import pathlib

import utils


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--in_filename",
        default=pathlib.Path("vggsound.csv"),
        type=pathlib.Path,
        help="input filename",
    )
    parser.add_argument(
        "-o",
        "--out_filename",
        default=pathlib.Path("data/vggsound/labels.json"),
        type=pathlib.Path,
        help="output filename",
    )
    return parser.parse_args(args=args, namespace=namespace)


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()

    # Load the input CSV file
    data = utils.load_csv_text(args.in_filename, True)

    # Collect all labels
    all_labels = sorted(set(row[2].replace(", ", " ") for row in data))

    # Create the label map
    label_map = {label: i for i, label in enumerate(all_labels)}

    # Save the label map
    utils.save_json(args.out_filename, label_map)
    print(f"Saved the label map to {args.out_filename}.")


if __name__ == "__main__":
    main()
