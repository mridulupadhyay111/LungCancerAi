"""
utils.py

Utility functions.
"""

from pathlib import Path
import json


def ensure_dir(path):

    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True
    )

    return path


def save_json(data, filename):

    with open(filename, "w") as f:

        json.dump(
            data,
            f,
            indent=4
        )