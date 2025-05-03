# image_ops.py
import numpy as np
from numba import njit

@njit
def grayscale(image):
    h, w, _ = image.shape
    gray_image = np.empty((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            r, g, b = image[i, j]
            gray_image[i, j] = 0.299 * r + 0.587 * g + 0.114 * b
    return gray_image[..., np.newaxis]

@njit
def resize(image, new_h, new_w):
    h, w = image.shape[:2]
    scale_h, scale_w = h / new_h, w / new_w
    resized_image = np.empty((new_h, new_w, image.shape[2]), dtype=image.dtype)
    for i in range(new_h):
        for j in range(new_w):
            orig_x = min(int(i * scale_h), h - 1)
            orig_y = min(int(j * scale_w), w - 1)
            resized_image[i, j] = image[orig_x, orig_y]
    return resized_image

@njit
def normalize(image):
    return image / 255.0

@njit
def rotate(image, angle):
    radians = np.deg2rad(angle)
    cos_theta, sin_theta = np.cos(radians), np.sin(radians)
    h, w = image.shape[:2]
    new_h = int(abs(h * cos_theta) + abs(w * sin_theta))
    new_w = int(abs(w * cos_theta) + abs(h * sin_theta))

    rotated_image = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
    cx, cy = w // 2, h // 2
    new_cx, new_cy = new_w // 2, new_h // 2

    for i in range(new_h):
        for j in range(new_w):
            y = int((i - new_cy) * cos_theta + (j - new_cx) * sin_theta + cy)
            x = int(-(i - new_cy) * sin_theta + (j - new_cx) * cos_theta + cx)
            if 0 <= x < w and 0 <= y < h:
                rotated_image[i, j] = image[y, x]

    return rotated_image

@njit
def adjust_brightness_contrast(image, brightness: float, contrast: float):
    # brightness in [-255,255], contrast in [0,∞), 1.0 = unchanged
    h, w, c = image.shape
    out = np.empty_like(image)
    for i in range(h):
        for j in range(w):
            for k in range(c):
                v = image[i, j, k]
                v = (v - 127.5) * contrast + 127.5 + brightness
                out[i, j, k] = 0 if v < 0 else (255 if v > 255 else v)
    return out

@njit
def threshold(image, thresh: float):
    # assumes grayscale input
    h, w = image.shape[:2]
    out = np.empty((h, w, 1), dtype=image.dtype)
    for i in range(h):
        for j in range(w):
            out[i, j, 0] = 255 if image[i, j, 0] > thresh else 0
    return out

@njit
def gaussian_blur3x3(image):
    kernel = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32) / 16
    h, w, c = image.shape
    out = np.empty_like(image)
    for i in range(h):
        for j in range(w):
            for k in range(c):
                acc = 0.0
                for ki in range(-1,2):
                    for kj in range(-1,2):
                        ii = min(max(i+ki,0), h-1)
                        jj = min(max(j+kj,0), w-1)
                        acc += image[ii,jj,k] * kernel[ki+1,kj+1]
                out[i,j,k] = acc
    return out

@njit
def frame_diff(prev, curr):
    h, w, c = curr.shape
    out = np.empty_like(curr)
    for i in range(h):
        for j in range(w):
            for k in range(c):
                d = int(curr[i,j,k]) - int(prev[i,j,k])
                out[i,j,k] = 255 if abs(d) > 15 else 0
    return out

@njit
def crop(image, top, left, height, width):
    out = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    h, w = image.shape[:2]
    for i in range(height):
        for j in range(width):
            y = min(max(top + i, 0), h-1)
            x = min(max(left + j, 0), w-1)
            out[i,j] = image[y,x]
    return out

@njit
def flip_horizontal(image):
    h, w, c = image.shape
    out = np.empty_like(image)
    for i in range(h):
        for j in range(w):
            out[i, w-1-j] = image[i, j]
    return out
