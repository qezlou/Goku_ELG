#!/bin/bash

# UV Installation Script for gal_goku with GPU support
# This script demonstrates the recommended UV workflow

set -e  # Exit on any error

echo "🚀 Setting up gal_goku with UV and GPU support"
echo "================================================"

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "✅ UV is already installed ($(uv --version))"
fi

# Navigate to the package directory
cd src/gal_goku

echo "🏗️  Creating virtual environment with UV..."
uv venv gal_goku_env --python 3.11

echo "🔌 Activating environment..."
source gal_goku_env/bin/activate

echo "📚 Installing gal_goku with GPU support..."
uv pip install -e ".[gpu,dev]"

echo "🧪 Verifying TensorFlow GPU setup..."
python -c "
import tensorflow as tf
print('✅ TensorFlow version:', tf.__version__)
print('✅ CUDA built:', tf.test.is_built_with_cuda())
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f'✅ GPU devices available: {len(gpus)}')
if gpus:
    print('🎯 GPU detected! Training will use GPU acceleration.')
    for i, gpu in enumerate(gpus):
        print(f'   GPU {i}: {gpu.name}')
else:
    print('⚠️  No GPU detected. Training will use CPU (still works fine).')
"

echo "🧪 Testing gal_goku import..."
python -c "
try:
    from gal_goku.emus_multifid import HmfNativeBins
    print('✅ gal_goku imported successfully!')
    print('✅ Ready for multifidelity training with GPU support!')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "🔧 To activate this environment in the future, run:"
echo "   source src/gal_goku/gal_goku_env/bin/activate"
echo ""
echo "📖 Next steps:"
echo "   1. Check GPU functionality: python test_gpu_training.py"
echo "   2. Start training your models with GPU acceleration!"
echo ""
echo "💡 For CPU-only installation, use: uv pip install -e '.[cpu]'"