"""
Neural Network Emulator Package

This package provides a modular neural network emulator for cosmological simulations.

Modules:
    emu_nn: Neural network architecture (HMFNet)
    data: Data handling classes (DataNormalizer, CosmologyDataset, TrainingData)
    training: Training utilities and trainer class (NNTrainer)

Example usage:
    from gal_goku.nn_model import NNTrainer, HMFNet
    from gal_goku.nn_model.data import DataNormalizer, CosmologyDataset
    
    # Create and train a model
    trainer = NNTrainer(config_path="config.json")
    history = trainer.fit(SummaryStatClass)
    trainer.save_model("model.pth")
"""

from .emu_nn import HMFNet
from .data import DataNormalizer, CosmologyDataset, TrainingData
from .training import NNTrainer, weighted_mse_loss, create_emu_config

__all__ = [
    'HMFNet',
    'DataNormalizer', 
    'CosmologyDataset', 
    'TrainingData',
    'NNTrainer',
    'weighted_mse_loss',
    'create_emu_config'
]

__version__ = '1.0.0'
