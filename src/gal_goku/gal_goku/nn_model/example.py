"""
Example script demonstrating how to use the neural network emulator package.

This script shows both the new class-based approach and the legacy functions.
"""

import numpy as np
from typing import Dict, Any

# Example of how to use the new NNTrainer class
def example_with_trainer_class():
    """Example using the new NNTrainer class (recommended)."""
    
    from gal_goku.nn_model import NNTrainer, create_emu_config
    
    # Option 1: Use default config
    trainer = NNTrainer()
    
    # Option 2: Use config file
    # create_emu_config()  # Create default config file
    # trainer = NNTrainer(config_path="emu_training_config.json")
    
    # Option 3: Use config dictionary
    config = {
        "model": {
            "input_dim": 9,
            "output_dim": 14,
            "hidden_dims": [128, 64, 32]
        },
        "training": {
            "epochs": 50,
            "learning_rate": 5e-4,
            "batch_size": 64,
            "device": "cuda"
        },
        "data": {
            "data_dir": "/path/to/data"
        }
    }
    trainer_with_config = NNTrainer(config_dict=config)
    
    # Setup and train (you would need to provide SummaryStatClass)
    try:
        from gal_goku import summary_stats
        
        # Train the model
        history = trainer.train(summary_stats.HMF, epochs=10)
        
        # Save the trained model
        trainer.save_model("my_emulator.pth")
        
        # Make predictions
        # X_test = np.random.rand(100, 9)  # Example input
        # predictions = trainer.predict(X_test)
        
        print("Training completed successfully!")
        return trainer, history
        
    except ImportError as e:
        print(f"Could not import summary_stats: {e}")
        return None, None

def example_data_usage():
    """Example of using data classes independently."""
    
    from gal_goku.nn_model.data import DataNormalizer, CosmologyDataset
    import torch
    
    # Example data
    X = np.random.rand(1000, 9)
    Y = np.random.rand(1000, 14)
    Y_err = np.random.rand(1000, 14) * 0.1
    
    # Use DataNormalizer
    normalizer = DataNormalizer()
    normalizer.fit(torch.tensor(Y))
    
    # Create dataset
    dataset = CosmologyDataset(X, Y, Y_err, normalizer)
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Example of accessing data
    x_sample, y_sample, y_err_sample = dataset[0]
    print(f"Sample shapes: X={x_sample.shape}, Y={y_sample.shape}, Y_err={y_err_sample.shape}")
    
    return dataset, normalizer

if __name__ == "__main__":
    print("Neural Network Emulator Example")
    print("=" * 40)

    print("\n1. Example with NNTrainer class:")
    trainer, history = example_with_trainer_class()
    
    
    print("\n3. Example with data classes:")
    dataset, data_normalizer = example_data_usage()
    
    print("\nAll examples completed!")
