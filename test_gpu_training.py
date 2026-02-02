#!/usr/bin/env python3
"""
Test script to demonstrate GPU-enabled multifidelity training.

This script shows how to use the updated BaseMFCoregEmu class with GPU support.
"""

import numpy as np
import tensorflow as tf
from src.gal_goku.gal_goku.emus_multifid import HmfNativeBins

def test_gpu_setup():
    """Test GPU configuration and availability."""
    print("=== GPU Configuration Test ===")
    
    # Check TensorFlow GPU availability
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {tf.config.experimental.list_physical_devices('GPU')}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    # Test basic GPU operations
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        with tf.device('/GPU:0'):
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
            c = tf.matmul(a, b)
            print(f"GPU matrix multiplication test: {c.numpy()}")
    else:
        print("No GPU available for testing")
    
    print()

def example_usage():
    """Example of how to use the GPU-enabled multifidelity training."""
    print("=== Example GPU Training Usage ===")
    
    # Example parameters - adjust these according to your data
    data_dir = "/path/to/your/data"  # Replace with actual data directory
    z = 1.0  # Redshift
    num_latents = 5
    num_inducing = 100
    noise_num_latents = 3
    
    # Create the emulator with GPU enabled (default)
    try:
        hmf_emu = HmfNativeBins(
            data_dir=data_dir,
            z=z,
            num_latents=num_latents,
            num_inducing=num_inducing,
            noise_num_latents=noise_num_latents,
            emu_type={'wide_and_narrow': True},
            norm_type='subtract_mean',
            noise_floor=0.01,
            use_gpu=True,  # Enable GPU
            logging_level='INFO'
        )
        
        print("GPU-enabled emulator created successfully!")
        print(f"Using device: {hmf_emu.device_name}")
        
        # Train the model (example parameters)
        opt_params = {
            'max_iters': 1000,
            'initial_lr': 0.01,
            'iter_save': 500
        }
        
        model_file = 'test_gpu_model.pkl'
        
        # This would start the training process on GPU
        # hmf_emu.train(
        #     model_file=model_file,
        #     opt_params=opt_params,
        #     force_train=True
        # )
        
        print("Training would proceed on GPU if data were available.")
        
    except Exception as e:
        print(f"Error creating emulator: {e}")
        print("This is expected if data_dir doesn't exist.")
    
    print()

def cpu_fallback_example():
    """Example showing CPU fallback when GPU is disabled."""
    print("=== CPU Fallback Example ===")
    
    # Example with GPU disabled
    data_dir = "/path/to/your/data"  # Replace with actual data directory
    
    try:
        hmf_emu_cpu = HmfNativeBins(
            data_dir=data_dir,
            z=1.0,
            num_latents=3,
            num_inducing=50,
            use_gpu=False,  # Explicitly disable GPU
            logging_level='INFO'
        )
        
        print("CPU-only emulator created successfully!")
        print(f"Using device: {hmf_emu_cpu.device_name}")
        
    except Exception as e:
        print(f"Error creating CPU emulator: {e}")
        print("This is expected if data_dir doesn't exist.")
    
    print()

def performance_tips():
    """Print performance tips for GPU training."""
    print("=== Performance Tips ===")
    
    tips = [
        "1. Use mixed precision training for better performance on modern GPUs",
        "2. Batch size should be large enough to utilize GPU memory efficiently",
        "3. Monitor GPU utilization using 'nvidia-smi' during training",
        "4. Consider using multiple GPUs for very large datasets",
        "5. Ensure your data loading pipeline doesn't become the bottleneck",
        "6. Use TensorFlow profiler to identify performance bottlenecks",
        "7. Enable GPU memory growth to avoid allocating all GPU memory at once"
    ]
    
    for tip in tips:
        print(tip)
    
    print()

if __name__ == "__main__":
    print("GPU-Enabled Multifidelity Training Test")
    print("=" * 50)
    
    test_gpu_setup()
    example_usage()
    cpu_fallback_example()
    performance_tips()
    
    print("Test completed. To actually run training:")
    print("1. Update data_dir to point to your actual data")
    print("2. Adjust hyperparameters as needed")
    print("3. Call the train() method on your emulator instance")