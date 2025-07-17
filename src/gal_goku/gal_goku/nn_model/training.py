import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
import logging
from typing import Tuple, List, Dict, Any, Optional
from . import data
from .emu_nn import HMFNet
import matplotlib.pyplot as plt


# Weighted MSE loss using uncertainties
def weighted_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, uncertainties: torch.Tensor) -> torch.Tensor:
    weights = 1.0 / (uncertainties ** 2 + 1e-8)  # Add small epsilon to avoid division by zero
    squared_errors = (predictions - targets) ** 2
    weighted_errors = weights * squared_errors
    return weighted_errors.mean()


class NNTrainer:
    """
    A comprehensive trainer class for neural network emulators.
    
    This class handles model initialization, data loading, training configuration,
    and the training loop for neural network emulators.
    
    Training Methods:
        train(): Simple training with basic logging (good for quick experiments)
        fit(): Enhanced training with checkpointing, plotting, and file logging (recommended for production)
    
    Example usage:
        # Quick training
        trainer = NNTrainer()
        history = trainer.train(SummaryStatClass, epochs=50)
        
        # Production training with all features
        trainer = NNTrainer(config_path="config.json")
        history = trainer.fit(SummaryStatClass, save_every=10, validate_every=5)
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize the trainer with either a config file or config dictionary.
        
        Args:
            config_path (str, optional): Path to JSON configuration file
            config_dict (dict, optional): Configuration dictionary
        """
        if config_path:
            self.config = self._load_config(config_path)
        elif config_dict:
            self.config = config_dict
        else:
            self.config = self._create_default_config()
            
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.normalizer = None
        self.optimizer = None
        self.device = self.config['training']['device']
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        return {
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
    
    def _setup_logging(self, log_to_file: bool = False, log_dir: str = None) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Always add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Add file handler if requested
        if log_to_file and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, 'training.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")
        
        return logger
    
    def setup_model(self) -> nn.Module:
        """Initialize and setup the neural network model."""
        self.model = HMFNet(**self.config['model'])
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['training']['learning_rate']
        )
        
        self.logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
        return self.model
    
    def setup_data(self, SummaryStatClass) -> Tuple[DataLoader, DataLoader, data.DataNormalizer]:
        """Setup data loaders for training and validation."""
        # Load training data
        training_data = data.TrainingData(
            SummaryStatClass=SummaryStatClass,
            data_dir=self.config['data']['data_dir']
        )
        
        self.train_loader, self.val_loader, self.normalizer = training_data.load_training_data()
        
        self.logger.info(f"Training data loaded: {len(self.train_loader)} batches for training, "
                        f"{len(self.val_loader)} batches for validation")
        
        return self.train_loader, self.val_loader, self.normalizer
    
    def train_epoch(self) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for X_batch, y_batch, y_err_batch in self.train_loader:
            # Move data to device
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            y_err_batch = y_err_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = weighted_mse_loss(predictions, y_batch, y_err_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        return epoch_loss / num_batches
    
    def validate(self) -> float:
        """Validate the model and return average validation loss."""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch, y_err_batch in self.val_loader:
                # Move data to device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_err_batch = y_err_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = weighted_mse_loss(predictions, y_batch, y_err_batch)
                
                val_loss += loss.item()
                num_batches += 1
        
        return val_loss / num_batches
    
    def train(self, SummaryStatClass, epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Simple training pipeline without checkpointing or plotting.
        
        Use this method for:
        - Quick experiments and prototyping
        - When you don't need checkpoints or plots
        - Minimal overhead training
        
        For production training with checkpointing, plotting, and logging, use fit() instead.
        
        Args:
            SummaryStatClass: Class for loading summary statistics
            epochs (int, optional): Number of epochs to train. Uses config if not provided.
            
        Returns:
            Dict containing training and validation loss history
        """
        if epochs is None:
            epochs = self.config['training']['epochs']
        
        # Setup model and data if not already done
        if self.model is None:
            self.setup_model()
        
        if self.train_loader is None:
            self.setup_data(SummaryStatClass)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        self.logger.info(f"Starting simple training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Log progress (less frequent than fit())
            if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == epochs - 1:
                self.logger.info(f"Epoch {epoch+1}/{epochs}, "
                               f"Train Loss: {train_loss:.6f}, "
                               f"Val Loss: {val_loss:.6f}")
        
        self.logger.info("Simple training completed!")
        return history
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'normalizer': self.normalizer
        }, filepath)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Update config if needed
        self.config = checkpoint.get('config', self.config)
        
        # Setup model if not already done
        if self.model is None:
            self.setup_model()
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load normalizer
        self.normalizer = checkpoint.get('normalizer', None)
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            predictions = self.model(X_tensor)
            
            # Denormalize if normalizer is available
            if self.normalizer is not None:
                predictions = self.normalizer.inverse_transform(predictions)
            
            return predictions.cpu().numpy()
    
    def fit(self, SummaryStatClass, save_every: int = 10, validate_every: int = 1) -> Dict[str, List[float]]:
        """
        Enhanced training loop with checkpointing, validation, and plotting.

        Args:
            SummaryStatClass: Class for loading summary statistics
            save_every (int): Save checkpoint every n epochs.
            validate_every (int): Validate every n epochs.
            
        Returns:
            Dict containing training and validation loss history
        """
        # Setup model and data if not already done
        if self.model is None:
            self.setup_model()
        
        if self.train_loader is None:
            self.setup_data(SummaryStatClass)
            
        num_epochs = self.config['training']['epochs']
        model_save_dir = self.config['training'].get('save_dir', './checkpoints')
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Setup file logging in the save directory
        self.logger = self._setup_logging(log_to_file=True, log_dir=model_save_dir)

        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        self.logger.info(f"Starting enhanced training for {num_epochs} epochs...")
        self.logger.info(f"Checkpoints will be saved to: {model_save_dir}")
        self.logger.info(f"Model config: {self.config['model']}")
        self.logger.info(f"Training config: {self.config['training']}")

        for epoch in range(1, num_epochs + 1):
            self.logger.info(f"\nEpoch {epoch}/{num_epochs}")

            train_loss = self.train_epoch()
            train_losses.append(train_loss)

            # Validate at specified intervals
            if epoch % validate_every == 0:
                val_loss = self.validate()
                val_losses.append(val_loss)
                
                self.logger.info(f"Epoch {epoch}/{num_epochs}, "
                               f"Train Loss: {train_loss:.6f}, "
                               f"Val Loss: {val_loss:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(model_save_dir, 'best_model.pth')
                    self.save_model(best_model_path)
                    self.logger.info(f"New best model saved to {best_model_path} (Val Loss: {val_loss:.6f})")
            else:
                self.logger.info(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.6f}")

            # Save checkpoints at specified intervals
            if epoch % save_every == 0:
                checkpoint_path = os.path.join(model_save_dir, f'checkpoint_epoch_{epoch}.pth')
                self.save_model(checkpoint_path)
                self.logger.info(f"Checkpoint saved to {checkpoint_path}")

        # Create training history dictionary
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses
        }
        
        # Save training history to file
        history_path = os.path.join(model_save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        self.logger.info(f"Training history saved to {history_path}")

        # Plot training and validation losses
        self._plot_training_losses(train_losses, val_losses, validate_every, model_save_dir)
        
        # Plot validation predictions vs ground truth
        self._plot_validation_comparison(model_save_dir)
        
        self.logger.info("Enhanced training completed!")
        self.logger.info(f"Final train loss: {train_losses[-1]:.6f}")
        self.logger.info(f"Final validation loss: {val_losses[-1]:.6f}")
        self.logger.info(f"Best validation loss: {best_val_loss:.6f}")
        
        return history
    
    def _plot_training_losses(self, train_losses: List[float], val_losses: List[float], 
                            validate_every: int, save_dir: str):
        """Plot and save training and validation loss curves."""
        try:
            plt.figure(figsize=(10, 6))
            epochs = range(1, len(train_losses) + 1)
            plt.plot(epochs, train_losses, label='Training Loss', color='blue', alpha=0.7)
            
            if val_losses:
                # Create epoch indices for validation losses
                val_epochs = list(range(validate_every, len(train_losses) + 1, validate_every))
                # Ensure we don't have more val_losses than val_epochs
                val_epochs = val_epochs[:len(val_losses)]
                plt.plot(val_epochs, val_losses, label='Validation Loss', color='red', alpha=0.7)
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(save_dir, 'training_loss_plot.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Training loss plot saved to {plot_path}")
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create training loss plot: {e}")
            plt.close()  # Ensure plot is closed even if error occurs
    
    def _plot_validation_comparison(self, save_dir: str):
        """Plot validation predictions vs ground truth for the first batch."""
        try:
            if self.val_loader is None:
                self.logger.warning("No validation loader available for plotting")
                return
                
            self.model.eval()
            with torch.no_grad():
                for X_val, y_val, y_err_val in self.val_loader:
                    X_val = X_val.to(self.device)
                    y_val = y_val.to(self.device)
                    y_err_val = y_err_val.to(self.device)
                    
                    preds = self.model(X_val)
                    
                    # Denormalize predictions and targets if normalizer is available
                    if self.normalizer is not None:
                        y_val_denorm = self.normalizer.inverse_transform(y_val).cpu().numpy()
                        preds_denorm = self.normalizer.inverse_transform(preds).cpu().numpy()
                        # For uncertainties, we need to rescale them back
                        y_err_denorm = (y_err_val * (self.normalizer.stds + self.normalizer.eps)).cpu().numpy()
                    else:
                        y_val_denorm = y_val.cpu().numpy()
                        preds_denorm = preds.cpu().numpy()
                        y_err_denorm = y_err_val.cpu().numpy()
                    
                    break  # Only use first batch
                    
            # Plot comparison for first sample
            plt.figure(figsize=(12, 8))
            
            # Main comparison plot
            plt.subplot(2, 1, 1)
            mass_bins = range(len(y_val_denorm[0]))
            plt.errorbar(mass_bins, y_val_denorm[0], yerr=y_err_denorm[0], 
                        label='Ground Truth', marker='o', capsize=3, alpha=0.7)
            plt.plot(mass_bins, preds_denorm[0], 
                    label='Prediction', marker='x', linestyle='--', alpha=0.8)
            plt.xlabel('Mass Bin Index')
            plt.ylabel('HMF Value (Denormalized)')
            plt.title('Validation: Prediction vs Ground Truth (First Sample)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Residuals plot
            plt.subplot(2, 1, 2)
            residuals = preds_denorm[0] - y_val_denorm[0]
            plt.plot(mass_bins, residuals, marker='o', linestyle='-', alpha=0.7)
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            plt.xlabel('Mass Bin Index')
            plt.ylabel('Residuals (Pred - True)')
            plt.title('Prediction Residuals')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            comparison_path = os.path.join(save_dir, 'validation_comparison.png')
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Validation comparison plot saved to {comparison_path}")
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create validation comparison plot: {e}")
            plt.close()  # Ensure plot is closed even if error occurs

def save_config(config: Dict[str, Any], fname: str = 'config.json'):
    """Save configuration dictionary to a JSON file."""
    filepath = config.get('training', 'save_dir')
    filepath = os.path.join(filepath, fname)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {filepath}")


# Example usage
if __name__ == "__main__":
    """
    Example of how to use the NNTrainer class for training a neural network emulator.
    """
    print("Neural Network Emulator Training Example")
    print("=" * 50)
    
    # Create a default configuration
    config = create_emu_config("example_config.json")
    print("Configuration created and saved to 'example_config.json'")
    
    # Method 1: Initialize trainer with config file
    print("\nMethod 1: Using config file")
    trainer_from_file = NNTrainer(config_path="example_config.json")
    print("Trainer initialized from config file")
    
    # Method 2: Initialize trainer with config dictionary
    print("\nMethod 2: Using config dictionary")
    custom_config = {
        "model": {
            "input_dim": 9,
            "output_dim": 14,
            "hidden_dims": [128, 64, 32]  # Deeper network
        },
        "training": {
            "epochs": 50,
            "learning_rate": 5e-4,
            "batch_size": 64,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "save_dir": "./my_checkpoints"
        },
        "data": {
            "data_dir": "/path/to/your/data"
        }
    }
    trainer_from_dict = NNTrainer(config_dict=custom_config)
    print("Trainer initialized from config dictionary")
    
    # Method 3: Initialize with defaults
    print("\nMethod 3: Using default configuration")
    trainer_default = NNTrainer()
    print("Trainer initialized with default configuration")
    
    print("\nTraining Methods:")
    print("1. For quick experiments: trainer.train(SummaryStatClass)")
    print("2. For production with checkpointing and plots: trainer.fit(SummaryStatClass)")
    print("   - fit() saves logs to file in save_dir")
    print("   - fit() creates training plots")
    print("   - fit() saves best model automatically")
    print("   - fit() saves training history as JSON")
    
    print("\nExample completed! Check the configuration file 'example_config.json'")
