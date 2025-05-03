import logging
import numpy as np
from numba import njit, prange
from typing import Callable, List, Dict

# Configure module-level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    "grayscale",
    "resize",
    "normalize",
    "rotate",
    "adjust_brightness_contrast",
    "threshold",
    "gaussian_blur3x3",
    "frame_diff",
    "crop",
    "flip_horizontal",
    "generate_merged_kernel",
    "example_merged_grayscale_thresh",
]

# ---------------- Pixel-wise Operations ----------------

@njit(parallel=True)
def grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to single-channel luminance float32 output.
    Uses Rec.601 coefficients.
    """
    h, w, _ = image.shape
    out = np.empty((h, w), dtype=np.float32)
    for i in prange(h):
        for j in range(w):
            r, g, b = image[i, j]
            out[i, j] = 0.299*r + 0.587*g + 0.114*b
    return out[..., np.newaxis]

@njit(parallel=True)
def resize(image: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """
    Nearest-neighbor resize to (new_h, new_w).
    """
    h, w = image.shape[:2]
    scale_h, scale_w = h / new_h, w / new_w
    out = np.empty((new_h, new_w, image.shape[2]), dtype=image.dtype)
    for i in prange(new_h):
        for j in range(new_w):
            y = min(int(i * scale_h), h - 1)
            x = min(int(j * scale_w), w - 1)
            out[i, j] = image[y, x]
    return out

@njit
def normalize(image: np.ndarray) -> np.ndarray:
    """
    Scale uint8 image to float32 in [0.0, 1.0].
    """
    return image.astype(np.float32) / 255.0

@njit(parallel=True)
def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image by `angle` degrees (nearest-neighbor).
    """
    rad = np.deg2rad(angle)
    cos_t, sin_t = np.cos(rad), np.sin(rad)
    h, w = image.shape[:2]
    new_h = int(abs(h*cos_t) + abs(w*sin_t))
    new_w = int(abs(w*cos_t) + abs(h*sin_t))
    out = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
    cx, cy = w//2, h//2
    ncx, ncy = new_w//2, new_h//2
    for i in prange(new_h):
        for j in range(new_w):
            yy = int((i-ncy)*cos_t + (j-ncx)*sin_t + cy)
            xx = int((j-ncx)*cos_t - (i-ncy)*sin_t + cx)
            if 0 <= xx < w and 0 <= yy < h:
                out[i, j] = image[yy, xx]
    return out

@njit(parallel=True)
def adjust_brightness_contrast(
    image: np.ndarray, brightness: float = 0.0, contrast: float = 1.0
) -> np.ndarray:
    """
    Apply linear brightness/contrast adjustment.
    brightness: [-255,255], contrast: [0,inf), 1.0 = no change
    """
    h, w, c = image.shape
    out = np.empty_like(image)
    for i in prange(h):
        for j in range(w):
            for k in range(c):
                v = (image[i,j,k] - 127.5)*contrast + 127.5 + brightness
                out[i,j,k] = 0 if v < 0 else (255 if v > 255 else v)
    return out

@njit(parallel=True)
def threshold(image: np.ndarray, thresh: float) -> np.ndarray:
    """
    Produce binary mask (0 or 255) from single-channel input.
    """
    h, w = image.shape[:2]
    out = np.empty((h, w, 1), dtype=image.dtype)
    for i in prange(h):
        for j in range(w):
            out[i, j, 0] = 255 if image[i,j,0] > thresh else 0
    return out

@njit(parallel=True)
def gaussian_blur3x3(image: np.ndarray) -> np.ndarray:
    """
    3x3 separable Gaussian blur: first horizontal, then vertical.
    """
    h, w, c = image.shape
    temp = np.empty_like(image)
    out = np.empty_like(image)
    # horizontal
    for i in prange(h):
        for j in range(w):
            for k in range(c):
                left = image[i, max(j-1,0), k]
                center = image[i, j, k]
                right = image[i, min(j+1,w-1), k]
                temp[i, j, k] = (left + 2*center + right) / 4.0
    # vertical
    for i in prange(h):
        for j in range(w):
            for k in range(c):
                up = temp[max(i-1,0), j, k]
                center = temp[i, j, k]
                down = temp[min(i+1,h-1), j, k]
                out[i, j, k] = (up + 2*center + down) / 4.0
    return out

@njit(parallel=True)
def frame_diff(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """
    Highlight pixels whose absolute difference > 15 between two frames.
    """
    h, w, c = curr.shape
    out = np.empty_like(curr)
    for i in prange(h):
        for j in range(w):
            for k in range(c):
                d = int(curr[i,j,k]) - int(prev[i,j,k])
                out[i,j,k] = 255 if abs(d) > 15 else 0
    return out

@njit(parallel=True)
def crop(
    image: np.ndarray, top: int, left: int, height: int, width: int
) -> np.ndarray:
    """
    Extract a rectangle, clamping out-of-bounds to nearest border.
    """
    h, w = image.shape[:2]
    out = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    for i in prange(height):
        for j in range(width):
            yy = min(max(top+i, 0), h-1)
            xx = min(max(left+j,0), w-1)
            out[i,j] = image[yy, xx]
    return out

@njit
def flip_horizontal(image: np.ndarray) -> np.ndarray:
    """
    Mirror image left-to-right.
    """
    return image[:, ::-1, :]

# ---------------- Kernel Merging Utility ----------------

def generate_merged_kernel(
    name: str, sequence: List[Dict[str, str]]
) -> Callable:
    """
    Create a fused @njit(parallel=True) kernel by inlining per-pixel steps.
    sequence: list of {"loop_body": code snippet}.
    Returns the compiled function.
    """
    # Build source lines
    lines = [
        "@njit(parallel=True)",
        f"def {name}(image, *params):",
        "    h, w, _ = image.shape",
        "    out = np.empty((h, w, 1), dtype=image.dtype)",
        "    # unpack parameters",
        "    vars = params",
        "    for i in prange(h):",
        "        for j in range(w):",
        "            r, g, b = image[i, j]",
    ]
    # inline each step
    for step in sequence:
        lines.append(f"            {step['loop_body']}")
    lines.append("    return out")

    src = "\n".join(lines)
    env = {"np": np, "prange": prange, "njit": njit}
    exec(src, globals(), env)
    func = env[name]
    logger.info("Generated merged kernel '%s'", name)
    return func


def example_merged_grayscale_thresh(thresh: float) -> Callable:
    """Return a kernel that does grayscale + threshold in one pass."""
    seq = [
        {"loop_body": "gray = 0.299*r + 0.587*g + 0.114*b"},
        {"loop_body": "out[i, j, 0] = 255 if gray > vars[0] else 0"},
    ]
    return generate_merged_kernel("grayscale_then_thresh", seq)

# ---------------- CLI for testing ----------------
if __name__ == "__main__":
    import argparse
    from PIL import Image
    parser = argparse.ArgumentParser(description="Run image_ops demo.")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output prefix")
    args = parser.parse_args()

    img = np.array(Image.open(args.input).convert('RGB'), dtype=np.uint8)
    logger.info("Loaded %s", args.input)

    # Demo pipeline
    blurred = gaussian_blur3x3(img)
    Image.fromarray(blurred).save(f"{args.output}_blur.png")
    logger.info("Saved blurred image.")

    thresh = 128.0
    merged_fn = example_merged_grayscale_thresh(thresh)
    merged = merged_fn(blurred, thresh)
    Image.fromarray(merged.squeeze()).save(f"{args.output}_merged.png")
    logger.info("Saved merged grayscale+threshold image.")
