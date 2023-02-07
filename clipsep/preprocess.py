"""Preprocess the data."""
import argparse
import logging
import pathlib
import pprint
import sys

import imageio
import joblib
import numpy as np
import torchvision.transforms
import tqdm
from PIL import Image

import utils


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess the data.")
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input directory"
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    parser.add_argument("--img_size", default=224, type=int, help="image size")
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


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def transform(n_px):
    """Preprocessing transformations used in the CLIP model."""
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                n_px,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            ),
            torchvision.transforms.CenterCrop(n_px),
            _convert_image_to_rgb,
            torchvision.transforms.ToTensor(),
        ]
    )


def preprocess(filename, out_filename, preprocess_func):
    """Preprocess the image."""
    # Open the image
    img = Image.open(filename).convert("RGB")

    # Preprocess the image
    preprocessed = preprocess_func(img).numpy().transpose(1, 2, 0)

    # Write the processed image
    imageio.imwrite(out_filename, (preprocessed * 255).astype(np.uint8))

    return filename


@utils.suppress_outputs
@utils.ignore_exceptions
def preprocess_ignore_exceptions(filename, out_filename, preprocess_func):
    """Preprocess the image, ignoring all exceptions."""
    return preprocess(filename, out_filename, preprocess_func)


def process(
    filename,
    in_dir,
    out_dir,
    preprocess_func,
    skip_existing,
    ignore_exceptions,
    quiet,
):
    """Process wrapper for multiprocessing."""
    # Get output filename
    out_filename = out_dir / filename.relative_to(in_dir)

    # Skip if exists
    if skip_existing and out_filename.is_file():
        return

    # Make sure the output directory exists
    out_filename.parent.mkdir(exist_ok=True, parents=True)

    # Preprocess the image
    if ignore_exceptions:
        return preprocess_ignore_exceptions(
            filename, out_filename, preprocess_func
        )
    return preprocess(filename, out_filename, preprocess_func)


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

    # Load the clip model
    preprocess = transform(args.img_size)

    # Iterate over all the MP4 files
    filenames = list(args.in_dir.rglob("*.jpg"))
    if args.jobs == 1:
        pbar = tqdm.tqdm(filenames, ncols=80)
        for filename in pbar:
            pbar.set_postfix_str(filename.parent.stem)
            process(
                filename,
                args.in_dir,
                args.out_dir,
                preprocess,
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
                preprocess,
                args.skip_existing,
                args.ignore_exceptions,
                args.quiet,
            )
            for filename in filenames
        )


if __name__ == "__main__":
    main()
