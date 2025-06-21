from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, List

__all__ = [
    "discover_images",
    "extract_number_from_filename",
    "hash_text",
]


def discover_images(directory: Path, patterns: Iterable[str] | None = None) -> List[Path]:
    """Return a naturally sorted list of image files inside *directory*.

    Parameters
    ----------
    directory
        Folder to scan for images.
    patterns
        Glob patterns used to match image files. If *None* the default set
        ``("*.png", "*.jpg", "*.jpeg", "*.webp")`` is used.
    """

    patterns = patterns or ("*.png", "*.jpg", "*.jpeg", "*.webp")
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(directory.glob(pattern))

    paths.sort(key=extract_number_from_filename)
    return paths


def extract_number_from_filename(path: Path) -> int:
    """Extract the first integer from *path.name* for natural sorting.

    If no digits are present the function returns ``0`` so such files are
    sorted first.
    """

    match = re.search(r"(\d+)", path.name)
    return int(match.group(1)) if match else 0


def hash_text(text: str) -> str:
    """Return an MD5 hash of *text* after normalising whitespace & case."""

    normalised = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalised.encode()).hexdigest() 