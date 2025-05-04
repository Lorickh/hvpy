# library

Motivation:

Alongside our paper published in ToC, we provided a lightweight “snack” utility for video-analysis workloads. Traditional libraries such as OpenCV, PyTorch, and TensorFlow can be too large to package into a Docker image for serverless platforms (e.g. AWS Lambda). To address this, we implemented a custom operator and further optimized it using a NUMA-aware JIT so that it runs efficiently on both CPU and GPU.

We’ve open-sourced this operator because we recognize that others face the same challenge, and we’d rather not reinvent the wheel. Some users suggested adding JIT-parallelism, which we tried, but it didn’t yield significant gains on our local machines. For best results, use hvpy.py; if you’re targeting architectures such as AMD Genoa or Venice, switch to hvpy_parallel.py.


---

## Features

- **Numba-accelerated kernels**: All heavy loops are JIT-compiled with `@njit(parallel=True)` and use `prange` for multi-core speed.
- **Pixel-wise transforms**:  
  - `grayscale`  
  - `resize` (nearest-neighbor)  
  - `normalize` (0–255 → 0.0–1.0)  
  - `rotate` (nearest-neighbor)  
  - `adjust_brightness_contrast`  
  - `threshold`  
  - `gaussian_blur3x3` (separable 3×3)  
  - `frame_diff` (highlight pixels whose frame-to-frame diff > 15)  
  - `crop` (clamps OOB to border)  
  - `flip_horizontal`
- **Kernel-merging utility**:  
  - `generate_merged_kernel(name, sequence)`  
  - `example_merged_grayscale_thresh(thresh)`  
- **Plain NumPy API**: Inputs and outputs are `numpy.ndarray` with zero hidden state.
- **Built-in logging & demo CLI**: Module-level `logging` plus a `__main__` entrypoint using Pillow for quick tests.

---

## 📦 Installation

```bash
git clone [git@github.com:Lorickh/hvpy.git](https://github.com/Lorickh/hvpy.git)
pip install -r requirements.txt
