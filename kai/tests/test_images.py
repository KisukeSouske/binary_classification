from binary_classifier import images
import numpy as np
import pytest
from PIL import Image


def write_image(tmp_path, name="sample.png", size=(30, 20), color=(255, 0, 0), mode="RGB"):
    """Write a solid-colour image. `size` is Pillow order: (width, height)."""
    path = tmp_path / name
    Image.new(mode, size, color).save(path)
    return path


# Shape tests
@pytest.mark.parametrize("source_size", [(30, 20), (400, 400), (17, 233)])
def test_output_shape_is_fixed_regardless_of_the_source(tmp_path, source_size):
    path = write_image(tmp_path, size=source_size)

    pixels = images.load_image(path, size=(64, 64))

    assert pixels.shape == (64, 64)


def test_size_is_height_by_width_in_numpy_order(tmp_path):
    # A 32x8 request must come back as 32 rows by 8 columns, not the reverse.
    path = write_image(tmp_path, size=(200, 100))

    assert images.load_image(path, size=(32, 8)).shape == (32, 8)
    assert images.load_image(path, size=(32, 8), color_mode="rgb").shape == (32, 8, 3)


def test_rgb_keeps_three_channels(tmp_path):
    path = write_image(tmp_path)

    pixels = images.load_image(path, size=(16, 16), color_mode="rgb")

    assert pixels.shape == (16, 16, 3)


# Value tests
@pytest.mark.parametrize(
    "color,expected", [((255, 255, 255), 1.0), ((0, 0, 0), 0.0), ((128, 128, 128), 128 / 255)]
)
def test_pixels_are_scaled_into_the_unit_range(tmp_path, color, expected):
    path = write_image(tmp_path, color=color)

    pixels = images.load_image(path, size=(8, 8))

    assert pixels.dtype == np.float64
    np.testing.assert_allclose(pixels, expected, atol=1e-6)
    assert pixels.min() >= 0.0 and pixels.max() <= 1.0


def test_rgb_channels_come_back_in_r_g_b_order(tmp_path):
    path = write_image(tmp_path, color=(255, 0, 0))

    pixels = images.load_image(path, size=(8, 8), color_mode="rgb")

    np.testing.assert_allclose(pixels[..., 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(pixels[..., 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(pixels[..., 2], 0.0, atol=1e-6)


def test_grayscale_uses_luma_weights(tmp_path):
    # Pillow's RGB->L is the ITU-R 601-2 luma: 0.299R + 0.587G + 0.114B, so a
    # pure red image is NOT 1.0 in grayscale. Worth pinning down: a plain
    # channel average would give 1/3 here.
    path = write_image(tmp_path, color=(255, 0, 0))

    pixels = images.load_image(path, size=(8, 8))

    np.testing.assert_allclose(pixels, 0.299, atol=2e-3)


# Input format tests
@pytest.mark.parametrize("mode,color", [("P", 3), ("RGBA", (255, 0, 0, 128)), ("L", 200)])
def test_reads_palette_alpha_and_grayscale_sources(tmp_path, mode, color):
    path = write_image(tmp_path, name=f"{mode}.png", mode=mode, color=color)

    pixels = images.load_image(path, size=(8, 8), color_mode="rgb")

    assert pixels.shape == (8, 8, 3)
    assert np.all(np.isfinite(pixels))


def test_jpeg_and_png_agree_on_a_flat_image(tmp_path):
    # A solid colour survives JPEG compression, so the two formats must give
    # the same array - any difference would mean the decode path differs.
    png = write_image(tmp_path, name="flat.png", color=(10, 200, 90))
    jpeg = write_image(tmp_path, name="flat.jpg", color=(10, 200, 90))

    np.testing.assert_allclose(
        images.load_image(png, size=(8, 8), color_mode="rgb"),
        images.load_image(jpeg, size=(8, 8), color_mode="rgb"),
        atol=2 / 255,
    )


def test_exif_orientation_is_applied(tmp_path):
    # Phone cameras leave the pixels as the sensor read them and record the
    # rotation as metadata. Two files with identical pixel data but different
    # orientation tags depict different scenes, and must not load alike.
    base = Image.new("L", (20, 10), 0)
    base.paste(255, (0, 0, 10, 10))  # left half white, right half black
    plain = tmp_path / "plain.jpg"
    base.save(plain)
    exif = Image.Exif()
    exif[274] = 6  # Orientation: rotate 270 CW when displaying
    rotated = tmp_path / "rotated.jpg"
    base.save(rotated, exif=exif)

    upright = images.load_image(plain, size=(8, 8))
    turned = images.load_image(rotated, size=(8, 8))

    assert not np.allclose(upright, turned)
    # untagged: split runs left-to-right, so every row is identical
    np.testing.assert_allclose(upright[0], upright[-1])
    # tagged: the split now runs top-to-bottom, so every column is identical
    np.testing.assert_allclose(turned[:, 0], turned[:, -1])
    assert turned[0].min() > 0.9 and turned[-1].max() < 0.1


# Failure tests
def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        images.load_image(tmp_path / "nope.png")


def test_directory_is_not_a_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        images.load_image(tmp_path)


def test_non_image_file_raises(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("this is not an image")

    with pytest.raises(ValueError, match="not a readable image"):
        images.load_image(path)


def test_invalid_color_mode_raises(tmp_path):
    path = write_image(tmp_path)

    with pytest.raises(ValueError, match="Unknown color_mode"):
        images.load_image(path, color_mode="hsv")


@pytest.mark.parametrize("size", [(0, 64), (64, -1), 64, (64, 64, 3)])
def test_invalid_size_raises(tmp_path, size):
    path = write_image(tmp_path)

    with pytest.raises(ValueError):
        images.load_image(path, size=size)
