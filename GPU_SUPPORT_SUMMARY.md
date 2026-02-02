# GPU/CUDA Support for Multifidelity Training - Summary

This document summarizes the changes made to enable GPU/CUDA support for your multifidelity training pipeline.

## Changes Made

### 1. BaseMFCoregEmu Class (`emus_multifid.py`)

**Added GPU Configuration:**
- New `use_gpu=True` parameter in `__init__()` method
- Added `_configure_gpu()` method to handle GPU setup:
  - Detects available GPUs
  - Enables memory growth to avoid allocating all GPU memory
  - Sets mixed precision for better performance
  - Handles fallback to CPU if GPU is unavailable
  - Stores device name (`/GPU:0` or `/CPU:0`)

**Updated Training Method:**
- Wrapped model creation and training in `tf.device()` context
- Convert numpy arrays to TensorFlow tensors before training
- Move training data to GPU during optimization loops
- Log which device is being used for training

**Updated Prediction Method:**
- Use GPU device context for inference
- Convert input data to TensorFlow tensors on GPU
- Convert results back to numpy for post-processing

### 2. LatentMFCoregionalizationSVGP Class (`linear_svgp.py`)

**Enhanced GPU Support in Optimization:**
- Auto-convert numpy arrays to TensorFlow tensors
- Ensure tensors stay on correct device during optimization
- Convert loss/KL history to Python floats to avoid device issues
- Maintain device context throughout optimization loop

### 3. Updated Child Classes

**HmfNativeBins and XiNativeBins:**
- Added `use_gpu=True` parameter to constructors
- Pass GPU parameter to parent BaseMFCoregEmu class

## How to Use

### Basic Usage with GPU (Default)
```python
from src.gal_goku.gal_goku.emus_multifid import HmfNativeBins

# GPU is enabled by default
hmf_emu = HmfNativeBins(
    data_dir="/path/to/data",
    z=1.0,
    num_latents=5,
    num_inducing=100,
    # use_gpu=True is the default
)

# Train on GPU
hmf_emu.train(model_file='my_model.pkl', force_train=True)
```

### Disable GPU (CPU Only)
```python
# Explicitly use CPU
hmf_emu = HmfNativeBins(
    data_dir="/path/to/data",
    z=1.0,
    num_latents=5,
    num_inducing=100,
    use_gpu=False  # Force CPU usage
)
```

### Check Device Being Used
```python
print(f"Training device: {hmf_emu.device_name}")
# Output: Training device: /GPU:0 or /CPU:0
```

## Benefits

1. **Automatic GPU Detection**: Automatically detects and uses available GPUs
2. **Graceful Fallback**: Falls back to CPU if GPU is unavailable
3. **Memory Efficient**: Enables GPU memory growth to avoid OOM errors
4. **Mixed Precision**: Uses mixed precision for better performance on modern GPUs
5. **Device Logging**: Clear logging about which device is being used
6. **Backward Compatible**: Existing code works without changes (GPU enabled by default)

## Performance Tips

1. **Monitor GPU Usage**: Use `nvidia-smi` to monitor GPU utilization during training
2. **Batch Size**: Increase batch size to better utilize GPU memory
3. **Data Loading**: Ensure data loading doesn't become the bottleneck
4. **Memory Management**: GPU memory growth is enabled to prevent allocation issues

## Troubleshooting

### GPU Not Detected
- Check CUDA installation: `nvidia-smi`
- Verify TensorFlow GPU support: `python -c "import tensorflow as tf; print(tf.config.experimental.list_physical_devices('GPU'))"`

### Out of Memory Errors
- Reduce `num_inducing` points
- Reduce `num_latents`
- Enable memory growth (done automatically)

### Slow Performance
- Ensure data is on GPU before training
- Monitor GPU utilization
- Check for CPU-GPU data transfer bottlenecks

## Testing

Use the provided test script to verify GPU functionality:
```bash
python test_gpu_training.py
```

This will:
- Test GPU availability
- Show example usage
- Demonstrate CPU fallback
- Provide performance tips

## Environment Requirements

Ensure your environment has:
- TensorFlow with GPU support
- CUDA-compatible GPU
- Proper CUDA/cuDNN installation
- GPflow package

Your current environment (.gal_venv) appears to have these requirements met based on the CUDA warnings during import.