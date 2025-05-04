# library

Motivation:

Alongside our paper published in ToC, we provided a lightweight “snack” utility for video-analysis workloads. Traditional libraries such as OpenCV, PyTorch, and TensorFlow can be too large to package into a Docker image for serverless platforms (e.g. AWS Lambda). To address this, we implemented a custom operator and further optimized it using a NUMA-aware JIT so that it runs efficiently on both CPU and GPU.

We’ve open-sourced this operator because we recognize that others face the same challenge, and we’d rather not reinvent the wheel. Some users suggested adding JIT-parallelism, which we tried, but it didn’t yield significant gains on our local machines. For best results, use hvpy.py; if you’re targeting architectures such as AMD Genoa or Venice, switch to hvpy_parallel.py.

some Examples
<p align="center">
  <img src="https://github.com/user-attachments/assets/7ce538a3-d9be-4554-9dd6-a2fba9ff5a7c" alt="blurred" width="120" />
  <img src="https://github.com/user-attachments/assets/fde6a2ab-25b8-427f-b3bd-67c47ced76a2" alt="07_threshold" width="120" />
  <img src="https://github.com/user-attachments/assets/c7a72ad1-22ce-403c-8344-744ec0f2104e" alt="06_blurred" width="120" />
  <img src="https://github.com/user-attachments/assets/d0726342-e5a4-4b9f-b780-06154ba9b6f0" alt="05_bright_contrast" width="120" />
  <img src="https://github.com/user-attachments/assets/ced21479-841d-4242-8c85-4d1b15a6a40c" alt="04_rotated" width="120" />
  <img src="https://github.com/user-attachments/assets/103aa50d-4c5f-4f6a-8ac9-d5e5e2443694" alt="03_resized" width="120" />
  <img src="https://github.com/user-attachments/assets/03bc1d31-088b-4295-9c45-139c76d76524" alt="02_normalized" width="120" />
  <img src="https://github.com/user-attachments/assets/023b20a4-ac3f-4832-9e7b-c0545028baf1" alt="01_grayscale" width="120" />
  <img src="https://github.com/user-attachments/assets/88dae03a-c00d-4926-8444-7bf011d60cad" alt="02_normalized (again)" width="120" />
</p>

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
