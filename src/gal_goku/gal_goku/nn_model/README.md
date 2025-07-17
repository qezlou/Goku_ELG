# Neural Network Emulator Package

A modular and maintainable neural network emulator for cosmological simulations, specifically designed for Halo Mass Function (HMF) emulation.

## Structure

## File Structure

```
nn_model/
├── __init__.py          # Package initialization and exports
├── emu_nn.py           # Neural network architecture (HMFNet)
├── data.py             # Data handling classes
├── training.py         # Training class and utilities
├── example.py          # Usage examples
└── README.md           # This documentation
```


The package is organized into three main modules:

### 1. `emu_nn.py` - Neural Network Architecture
- **HMFNet**: Fully connected neural network class with configurable architecture
- Clean, simple implementation with proper type hints
- Supports arbitrary hidden layer configurations

### 2. `data.py` - Data Handling
- **DataNormalizer**: Robust data normalization with proper error handling
- **CosmologyDataset**: PyTorch Dataset class for cosmological data with uncertainty handling
- **TrainingData**: Main data loading class with multi-fidelity support and logging

### 3. `training.py` - Training and Configuration
- **NeuralNetworkTrainer**: Complete training class with advanced features (NEW)
- **weighted_mse_loss**: Weighted MSE loss function with uncertainty weighting
- Legacy functions maintained for backward compatibility
- Configuration management and model persistence

#

## Usage Examples

1. For quick experiments: `trainer.train(SummaryStatClass)`
2. For production with checkpointing and plots: `trainer.fit(SummaryStatClass)`

### Method 1: New Class-Based Approach (Recommended)

```python
from gal_goku.nn_model import NeuralNetworkTrainer, save_config
from gal_goku import summary_stats

# Option A: Use default configuration
trainer = NeuralNetworkTrainer()

# Option B: Use configuration file
trainer = NeuralNetworkTrainer(config_path="config.json")

# Option C: Use configuration dictionary
config = {
    "model": {
        "input_dim": 9,
        "output_dim": 14,
        "hidden_dims": [64, 64]
    },
    "training": {
        "epochs": 100,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    },
    "data": {
        "data_dir": "placeholder_inputs.npy",
        "output_file": "placeholder_outputs.npy",
        "uncertainty_file": "placeholder_uncertainties.npy"
    }
}
save_config(config)

trainer = NeuralNetworkTrainer(config_path)

```
Quick test:
```python
# Train the model
history = trainer.train(summary_stats.HMF)
# Save the model
trainer.save_model("my_emulator.pth")
# Make predictions
predictions = trainer.predict(X_test)
```

Full run:

```python
trainer.fit()
```
#### Outut strucuture

```
save_dir/
├── training.log                    # Complete training log
├── training_history.json           # Loss history data  
├── training_loss_plot.png          # Training curves
├── validation_comparison.png       # Prediction plots
├── best_model.pth                  # Best model based on validation loss
├── checkpoint_epoch_10.pth         # Regular checkpoints
├── checkpoint_epoch_20.pth
└── ...
```
