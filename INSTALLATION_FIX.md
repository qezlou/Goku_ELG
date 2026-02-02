# Quick Installation Test Guide

## The Issue
The error you encountered was due to:
1. License format deprecation warnings (now fixed)
2. Package structure confusion (pyproject.toml is now correctly configured)

## Solution

From your remote machine, make sure you're in the right directory and run:

```bash
# Navigate to the gal_goku package directory
cd /path/to/Goku_ELG/src/gal_goku

# Clean any previous build artifacts
rm -rf build/ dist/ *.egg-info/

# Install with UV
uv pip install -e ".[gpu,dev]"
```

## If you're still in the parent src directory:
```bash
# From /path/to/Goku_ELG/src/
uv pip install -e gal_goku/[gpu,dev]
```

## Alternative: Use the traditional path approach
```bash
# From anywhere in the repo
uv pip install -e ./src/gal_goku/[gpu,dev]
```

## Verify Installation
```bash
python -c "
import tensorflow as tf
from gal_goku.emus_multifid import HmfNativeBins
print('✅ Installation successful!')
print('TensorFlow version:', tf.__version__)
print('GPU available:', len(tf.config.experimental.list_physical_devices('GPU')) > 0)
"
```

The key fixes made:
- ✅ Fixed license format (removed deprecation warning)
- ✅ Confirmed package structure is correct
- ✅ Clean installation commands provided