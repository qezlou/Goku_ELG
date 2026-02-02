# GPU Installation Guide for gal_goku

This guide explains how to install the `gal_goku` package with proper GPU support for TensorFlow.

## Installation Options

### 1. Default Installation (GPU Support)

The default installation now includes GPU-enabled TensorFlow:

```bash
pip install -e ./src/gal_goku/
```

This will install `tensorflow[and-cuda]` which includes:
- TensorFlow with CUDA support
- CUDA libraries (cuDNN, cuBLAS, etc.)
- GPU acceleration capabilities

### 2. CPU-Only Installation

For systems without GPU or when you specifically want CPU-only:

```bash
pip install -e ./src/gal_goku/[cpu]
```

This will install `tensorflow-cpu` instead of the GPU version.

### 3. Full GPU Installation (with monitoring)

For complete GPU support including monitoring capabilities:

```bash
pip install -e ./src/gal_goku/[gpu]
```

This includes additional GPU monitoring tools.

### 4. Development Installation

For development with testing and linting tools:

```bash
pip install -e ./src/gal_goku/[dev]
```

Or combine with GPU support:

```bash
pip install -e ./src/gal_goku/[gpu,dev]
```

## Verifying GPU Installation

After installation, verify GPU support:

```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.experimental.list_physical_devices('GPU'))
print("CUDA available:", tf.test.is_built_with_cuda())

# Test GPU functionality
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float64)
        b = tf.constant([[2.0], [1.0]], dtype=tf.float64)
        result = tf.matmul(a, b)
        print("GPU test successful:", result.numpy().flatten())
else:
    print("No GPU available - using CPU")
```

## System Requirements for GPU Support

### NVIDIA GPU Requirements
- NVIDIA GPU with compute capability 3.5 or higher
- CUDA-compatible GPU driver

### CUDA Requirements
- CUDA 12.3 or later (automatically installed with tensorflow[and-cuda])
- cuDNN 8.9 or later (automatically installed)

### Checking Your System
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version (if manually installed)
nvcc --version

# Check GPU compute capability
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

## Troubleshooting GPU Installation

### Common Issues and Solutions

#### 1. "No module named 'tensorflow'" after installation
```bash
# Reinstall with explicit GPU support
pip uninstall tensorflow tensorflow-gpu
pip install tensorflow[and-cuda]==2.19.0
```

#### 2. "Could not load dynamic library 'libcudart.so'"
This usually means CUDA libraries are not properly installed. The `tensorflow[and-cuda]` installation should handle this automatically.

#### 3. GPU not detected by TensorFlow
```python
import tensorflow as tf
# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
```

#### 4. Out of GPU memory errors
- Reduce batch size or model size
- Enable memory growth (done automatically in the updated code)
- Monitor GPU usage with `nvidia-smi`

### Environment Variables
Set these environment variables for optimal GPU performance:

```bash
# Suppress TensorFlow warnings (optional)
export TF_CPP_MIN_LOG_LEVEL=2

# Enable GPU memory growth by default
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Use specific GPU (if multiple GPUs available)
export CUDA_VISIBLE_DEVICES=0
```

## Docker Installation (Alternative)

For containerized environments with GPU support:

```dockerfile
FROM tensorflow/tensorflow:2.19.0-gpu

WORKDIR /app
COPY . .

# Install the package
RUN pip install -e ./src/gal_goku/

# Verify GPU support
RUN python -c "import tensorflow as tf; print('GPUs:', len(tf.config.experimental.list_physical_devices('GPU')))"
```

Run with GPU support:
```bash
docker run --gpus all -it your_image_name
```

## Performance Comparison

You can benchmark CPU vs GPU performance:

```python
import time
import tensorflow as tf
import numpy as np

# Generate test data
X = np.random.randn(1000, 10).astype(np.float64)
Y = np.random.randn(1000, 5).astype(np.float64)

# Test CPU performance
with tf.device('/CPU:0'):
    start = time.time()
    result_cpu = tf.matmul(X, tf.random.normal([10, 5], dtype=tf.float64))
    cpu_time = time.time() - start

# Test GPU performance (if available)
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    with tf.device('/GPU:0'):
        start = time.time()
        result_gpu = tf.matmul(X, tf.random.normal([10, 5], dtype=tf.float64))
        gpu_time = time.time() - start
    print(f"CPU time: {cpu_time:.4f}s, GPU time: {gpu_time:.4f}s")
    print(f"GPU speedup: {cpu_time/gpu_time:.2f}x")
else:
    print("GPU not available for comparison")
```

## Next Steps

After successful installation:
1. Run the test script: `python test_gpu_training.py`
2. Verify your multifidelity training uses GPU
3. Monitor GPU usage during training with `nvidia-smi`
4. Optimize your model parameters for GPU memory usage