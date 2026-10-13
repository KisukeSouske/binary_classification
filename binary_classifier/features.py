"""
Loading the cat photos as raw inputs for the pipeline.

A photo becomes a small grayscale thumbnail, flattened. That vector is NOT
given to the classifier directly - it has far more entries than there are
photos - but compressed by PCA (see `decomposition.py`) inside the pipeline.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from binary_classifier import images

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
THUMB_SIZE = (16, 16)  # (height, width); 256 raw pixels per photo

# EXIF tags: DateTimeOriginal lives in the Exif sub-IFD, DateTime in the main one
EXIF_IFD = 0x8769
EXIF_DATETIME_ORIGINAL = 36867
EXIF_DATETIME = 306


def thumbnail(path) -> np.ndarray:
    """Flattened grayscale thumbnail with values in [0, 1]."""
    return images.load_image(path, size=THUMB_SIZE, color_mode="grayscale").ravel()


def photo_date(path) -> str | None:
    """Day the photo was taken ("YYYY-MM-DD") from EXIF, or None if absent."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            raw = exif.get_ifd(EXIF_IFD).get(EXIF_DATETIME_ORIGINAL) or exif.get(EXIF_DATETIME)
    except OSError:
        return None
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw).strip()[:10], "%Y:%m:%d").date().isoformat()
    except ValueError:
        return None


def list_images(folder) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_folders(negative_folder, positive_folder) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """
    Loads every image in both folders.
    Returns X (one thumbnail per row), y (0.0 = negative_folder,
    1.0 = positive_folder) and the file paths in the same order.
    """
    X, y, paths = [], [], []
    for label, folder in enumerate([negative_folder, positive_folder]):
        for path in list_images(folder):
            X.append(thumbnail(path))
            y.append(float(label))
            paths.append(path)
    if not X:
        raise ValueError("No images found in the given folders.")
    return np.array(X), np.array(y), paths


def session_groups(paths) -> np.ndarray:
    """
    One group label per photo: its class folder plus the day it was taken.
    Photos without a date each get their own group, since nothing ties them
    to any other photo.
    """
    groups = []
    for path in paths:
        day = photo_date(path)
        groups.append(f"{Path(path).parent.name}/{day}" if day else f"undated/{Path(path).name}")
    return np.array(groups)
