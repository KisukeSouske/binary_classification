"""Turning image files into the fixed-shape numeric arrays the model expects.

Everything downstream - feature extraction, training, prediction - assumes
every sample has exactly the same shape and the same value range. This module
is where that guarantee is made: one path in, one float array in [0, 1] out,
whatever the file on disk happened to be (JPEG or PNG, palette or RGBA,
portrait or landscape, 300x200 or 4000x3000).
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Registry of the colour spaces we support: our name -> (Pillow mode, channels).
# The channel count is what decides the rank of the returned array, so a caller
# can work out the sample shape without loading a file first.
COLOR_MODES = {
    "grayscale": ("L", 1),
    "rgb": ("RGB", 3),
}

DEFAULT_IMAGE_SIZE = (64, 64)
PIXEL_MAX = 255.0


def _as_size(size) -> tuple[int, int]:
    """Validate a (height, width) pair.

    Note the order: it is the numpy convention, matching the shape of the array
    this module returns, NOT Pillow's (width, height). The two are swapped once,
    here, so no caller has to think about it again.
    """
    try:
        height, width = size
    except (TypeError, ValueError):
        raise ValueError(f"size must be a (height, width) pair, got {size!r}.") from None
    height, width = int(height), int(width)
    if height <= 0 or width <= 0:
        raise ValueError(f"size entries must be positive, got {(height, width)}.")
    return height, width


def load_image(
    path,
    size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    color_mode: str = "grayscale",
) -> np.ndarray:
    """Load one image file as a standardized float array.

    The pipeline, in order: honour any EXIF orientation, flatten to RGB,
    resize to the requested size, convert to the requested colour space, and
    scale the pixels to [0, 1].

    RGB comes before the resize on purpose: palette ("P") and 16-bit images
    resample badly in their native mode, and dropping alpha afterwards would
    mean resampling a channel that is about to be discarded.

    Parameters:
    path (str | Path): Path to the image file.
    size (tuple[int, int]): Target (height, width), in numpy order. The image
        is stretched to exactly this shape; the source aspect ratio is not
        preserved.
    color_mode (str): One of COLOR_MODES - "grayscale" or "rgb".

    Returns:
    np.ndarray: float array with values in [0, 1]. Shape is (height, width)
        for grayscale and (height, width, 3) for rgb.

    Raises:
    FileNotFoundError: If `path` does not point at an existing file.
    ValueError: If `size` or `color_mode` is invalid, or if the file is not a
        readable image.
    """
    height, width = _as_size(size)
    if color_mode not in COLOR_MODES:
        raise ValueError(
            f"Unknown color_mode {color_mode!r}; available: {sorted(COLOR_MODES)}."
        )
    pillow_mode, _channels = COLOR_MODES[color_mode]

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No image file at {str(path)!r}.")

    try:
        with Image.open(path) as opened:
            # Phone cameras store the rotation as metadata rather than rotating
            # the pixels; without this, the same scene loads transposed
            # depending on how the device was held.
            image = ImageOps.exif_transpose(opened)
            image = image.convert("RGB")
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            if pillow_mode != "RGB":
                image = image.convert(pillow_mode)
            pixels = np.asarray(image, dtype=float)
    except UnidentifiedImageError:
        raise ValueError(f"{str(path)!r} is not a readable image file.") from None

    # 8-bit channels after the conversions above, so the maximum is always 255
    # regardless of the source's original bit depth.
    return pixels / PIXEL_MAX
