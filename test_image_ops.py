import os

import numpy as np
import pytest
from PIL import Image

from hvpy import (
    grayscale, resize, normalize, rotate,
    adjust_brightness_contrast, threshold,
    gaussian_blur3x3, frame_diff,
    crop, flip_horizontal
)

# --- Fixtures ---
@pytest.fixture(scope="module")
def rgb_image():
    # simple 2×2 test pattern with distinct RGB values
    return np.array([
        [[255,   0,   0], [  0, 255,   0]],
        [[  0,   0, 255], [255, 255, 255]],
    ], dtype=np.uint8)

@pytest.fixture(scope="module")
def grayscale_image(rgb_image):
    # grayscale version for threshold tests
    gray = grayscale(rgb_image)
    assert gray.ndim == 3 and gray.shape[2] == 1
    return gray

# --- Core Tests ---
def test_grayscale_values_and_shape(rgb_image):
    gray = grayscale(rgb_image)
    assert gray.shape == (2, 2, 1)
    expected = np.empty((2,2), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            r, g, b = rgb_image[i,j]
            expected[i,j] = 0.299*r + 0.587*g + 0.114*b
    assert np.allclose(gray[...,0], expected, atol=1e-5)


def test_normalize_scales_to_unit(rgb_image):
    norm = normalize(rgb_image.astype(np.float32))
    assert np.issubdtype(norm.dtype, np.floating)
    assert 0.0 <= norm.min() <= norm.max() <= 1.0

@pytest.mark.parametrize("new_h,new_w,expected_pixel", [
    (1, 1, [255, 0, 0]),   # downsample picks top-left
    (4, 4, None),          # upsample: shape only
])
def test_resize_shape_and_content(rgb_image, new_h, new_w, expected_pixel):
    out = resize(rgb_image, new_h, new_w)
    assert out.shape == (new_h, new_w, 3)
    if expected_pixel is not None:
        assert np.array_equal(out[0,0], expected_pixel)


def test_rotate_identity(rgb_image):
    rotated = rotate(rgb_image, 0)
    assert rotated.shape == rgb_image.shape
    assert np.array_equal(rotated, rgb_image)


def test_rotate_90_shape(rgb_image):
    h, w = rgb_image.shape[:2]
    rotated = rotate(rgb_image, 90)
    assert rotated.shape == (w, h, 3)

# --- New Operation Tests ---

def test_adjust_brightness_contrast_midpoint():
    img = np.full((1,1,1), 100, dtype=np.uint8)
    # brightness +10, contrast 1.0: (100-127.5)*1 +127.5 +10 = 110
    out = adjust_brightness_contrast(img, brightness=10.0, contrast=1.0)
    assert out.dtype == img.dtype
    assert out[0,0,0] == 110


def test_threshold_binary(grayscale_image):
    # grayscale image values in 0-255; threshold at mean (~127)
    t_hi = threshold(grayscale_image, thresh=150)
    t_lo = threshold(grayscale_image, thresh=50)
    assert set(np.unique(t_hi.flatten())) <= {0,255}
    assert set(np.unique(t_lo.flatten())) <= {0,255}
    # values >150 should be 255, so max is 255, min at least 0
    assert t_hi.max() == 255 and t_hi.min() == 0


def test_gaussian_blur_constant():
    img = np.full((3,3,1), 100, dtype=np.uint8)
    out = gaussian_blur3x3(img)
    # constant image blurred stays constant
    assert out.shape == img.shape
    assert np.all(out == 100)


def test_frame_diff_detects_changes(rgb_image):
    prev = rgb_image
    curr = rgb_image.copy()
    curr[0,0,0] = prev[0,0,0] + 20  # diff =20 >15
    out = frame_diff(prev, curr)
    assert out.dtype == prev.dtype
    assert out[0,0,0] == 255
    # pixels without change should be 0
    assert out[1,1,1] == 0


def test_crop_region(rgb_image):
    # 2×2 image, crop 1×1 at (1,1)
    out = crop(rgb_image, top=1, left=1, height=1, width=1)
    assert out.shape == (1,1,3)
    assert np.array_equal(out[0,0], rgb_image[1,1])


def test_flip_horizontal(rgb_image):
    out = flip_horizontal(rgb_image)
    h, w = rgb_image.shape[:2]
    assert out.shape == (h, w, 3)
    # leftmost should match original rightmost
    assert np.array_equal(out[:,0], rgb_image[:,w-1])

# --- Real Image Sanity Check ---
@pytest.fixture(scope="module")
def example_array():
    fname = "example.jpg"
    for d in ("tests", "."):
        path = os.path.join(d, fname)
        if os.path.isfile(path):
            img = Image.open(path).convert("RGB")
            return np.asarray(img, dtype=np.uint8)
    pytest.skip(f"Could not find {fname}")


def test_example_operations_run(example_array):
    h, w = example_array.shape[:2]

    # grayscale
    g = grayscale(example_array)
    assert g.shape == (h, w, 1)

    # normalize
    n = normalize(example_array.astype(np.float32))
    assert np.issubdtype(n.dtype, np.floating)
    assert 0.0 <= n.min() <= n.max() <= 1.0

    # resize
    for new_h, new_w in ((100, 100), (h//2, w//2)):
        rsz = resize(example_array, new_h, new_w)
        assert rsz.shape == (new_h, new_w, 3)

    # rotate
    for angle in (0, 45, 90, 180):
        rot = rotate(example_array, angle)
        assert rot.ndim == 3 and rot.shape[2] == 3
        if angle == 0:
            assert np.array_equal(rot, example_array)
        else:
            assert rot.shape != example_array.shape or not np.array_equal(rot, example_array)
