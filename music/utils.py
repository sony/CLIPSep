"""Utility functions."""
import contextlib
import json
import os
import pathlib
import warnings

import numpy as np


def save_args(filename, args):
    """Save the command-line arguments."""
    args_dict = {}
    for key, value in vars(args).items():
        if isinstance(value, pathlib.Path):
            args_dict[key] = str(value)
        else:
            args_dict[key] = value
    save_json(filename, args_dict)


def inverse_dict(d):
    """Return the inverse dictionary."""
    return {v: k for k, v in d.items()}


def save_txt(filename, data):
    """Save a list to a TXT file."""
    with open(filename, "w", encoding="utf8") as f:
        for item in data:
            f.write(f"{item}\n")


def load_txt(filename):
    """Load a TXT file as a list."""
    with open(filename, encoding="utf8") as f:
        return [line.strip() for line in f]


def save_json(filename, data):
    """Save data as a JSON file."""
    with open(filename, "w", encoding="utf8") as f:
        json.dump(data, f)


def load_json(filename):
    """Load data from a JSON file."""
    with open(filename, encoding="utf8") as f:
        return json.load(f)


def save_csv(filename, data, header=""):
    """Save data as a CSV file."""
    np.savetxt(
        filename, data, fmt="%d", delimiter=",", header=header, comments=""
    )


def load_csv(filename, skiprows=1):
    """Load data from a CSV file."""
    return np.loadtxt(filename, dtype=int, delimiter=",", skiprows=skiprows)


def ignore_exceptions(func):
    """Decorator that ignores all errors and warnings."""

    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return func(*args, **kwargs)
            except Exception:
                return None

    return inner


def suppress_outputs(func):
    """Decorator that suppresses writing to stdout and stderr."""

    def inner(*args, **kwargs):
        devnull = open(os.devnull, "w")
        with contextlib.redirect_stdout(devnull):
            with contextlib.redirect_stderr(devnull):
                return func(*args, **kwargs)

    return inner


def resolve_paths(func):
    """Decorator that resolves all paths."""

    def inner(*args, **kwargs):
        parsed = func(*args, **kwargs)
        for key in vars(parsed).keys():
            if isinstance(getattr(parsed, key), pathlib.Path):
                setattr(
                    parsed, key, getattr(parsed, key).expanduser().resolve()
                )
        return parsed

    return inner
