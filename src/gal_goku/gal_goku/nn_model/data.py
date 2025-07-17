import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
import logging
from typing import Tuple, List, Dict, Any


# --- Data Normalizer ---
class DataNormalizer:
    """
    Normalizes 1D data with multiple features.

    Methods:
        fit(x): Computes per-feature mean and std from input tensor.
        transform(x): Applies normalization using stored stats.
        inverse_transform(x): Reverts normalization to original scale.

    Args:
        eps (float): Small constant to avoid division by zero.
    """
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.means = None
        self.stds = None

    def fit(self, x: torch.Tensor):
        """
        Compute mean and std for each feature (channel).

        Args:
            x (Tensor): Input of shape [batch_size, seq_len, num_features]
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        flat = x.view(-1, x.size(-1))  # collapse batch and sequence
        self.means = flat.mean(dim=0)
        self.stds = flat.std(dim=0)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input using stored mean and std.

        Args:
            x (Tensor): Input of shape [batch_size, seq_len, num_features]

        Returns:
            Tensor: Normalized input, same shape.
        """
        if self.means is None or self.stds is None:
            raise RuntimeError("Call fit() before transform().")
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = (x - self.means) / (self.stds + self.eps)
        return x.float()

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Revert normalization.

        Args:
            x (Tensor): Normalized tensor.

        Returns:
            Tensor: Original-scale tensor.
        """
        return x * (self.stds + self.eps) + self.means

# Custom dataset class
class CosmologyDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, Y_err: np.ndarray, normalizer: DataNormalizer):
        self.normalizer = normalizer
        self.X = torch.tensor(X, dtype=torch.float32)
        
        # Normalize Y using the provided normalizer
        self.Y = self.normalizer.transform(torch.tensor(Y, dtype=torch.float32))
        
        # Normalize uncertainties in the same way as Y (scale only, do not shift)
        self.Y_err = torch.tensor(Y_err, dtype=torch.float32) / (self.normalizer.stds + self.normalizer.eps)
        
    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx], self.Y_err[idx]

class TrainingData:
    """
    Emulator for the Halo Mass Function using the native bins
    This does the full dimensionality reduction of the output space using
    `LatentMFCoregionalizationSVGP` which allows each output to have a different
    observational (simulation quality) uncertainty.
    """
    def __init__(self, SummaryStatClass, data_dir: str, emu_type: Dict[str, bool] = None, logging_level: str = 'INFO'):
        if emu_type is None:
            emu_type = {'wide_and_narrow': True}
            
        self.logging_level = logging_level
        self.logger = self.configure_logging(logging_level)
        self.data_dir = data_dir
        # Load the data
        self.X = []
        self.Y = []
        self.Y_err = []
        self.labels = []
        self.wide_array = np.array([])
        # Keeping the id for good HF sims
        self.good_sim_ids = []
        # Fix a few features of the emulator
        emu_type.update({'multi-fid':True, 'single-bin':False, 'linear':True, 'mf-svgp':True})
        self.emu_type = emu_type
        fids = ['L2', 'HF']
        for fd in fids:
            # Goku-wide sims
            stat_loader = SummaryStatClass(data_dir=data_dir, fid=fd, narrow=False, MPI=None, logging_level=logging_level)
            # Load xi((m1, m2), r) for wide
            self.mbins, Y_wide, err_wide, X_wide, labels_wide = stat_loader.get_wt_err()
            self.wide_array = np.append(self.wide_array, np.ones(Y_wide.shape[0]))
            self.logger.debug(f'Y_wide: {Y_wide.shape}')
            # Only use Goku-wide
            if not emu_type['wide_and_narrow']:
                self.Y.append(Y_wide)
                self.X.append(X_wide)
                self.Y_err.append(err_wide)
            # Use both Goku-wide and narrow
            else:
                # Goku-narrow sims
                stat_loader = SummaryStatClass(data_dir=data_dir, fid=fd, narrow=True, MPI=None, logging_level=logging_level)
                # Load xi((m1, m2), r) for wide
                _, Y_narrow, err_narrow, X_narrow, labels_narrow = stat_loader.get_wt_err()
                self.wide_array = np.append(self.wide_array, np.zeros(Y_narrow.shape[0]))
                self.logger.debug(f'Y_narrow: {Y_narrow.shape}')
                # For now, get rid of the lastbins with 0 value
                self.Y.append(np.concatenate((Y_wide, Y_narrow), axis=0))
                self.X.append(np.concatenate((X_wide, X_narrow), axis=0))
                self.Y_err.append(np.concatenate((err_wide, err_narrow), axis=0))
                if fd == 'HF':  # Only append labels for the last fidelity to avoid duplication
                    self.labels = np.concatenate((labels_wide, labels_narrow), axis=0)

        # Convert lists to numpy arrays
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        self.Y_err = np.array(self.Y_err)
        
        self.output_dim = self.Y[0].shape[1] if len(self.Y) > 0 else 0

        # scale the features between 0 and 1 given the low-fidelity's min and max values
        # for each feature
        if len(self.X) > 0:
            self.X_min, self.X_max = np.min(self.X[0], axis=0), np.max(self.X[0], axis=0)
            # Avoid division by zero
            range_mask = (self.X_max - self.X_min) > 1e-10
            for i in range(len(self.X)):
                self.X[i] = np.where(range_mask, 
                                   (self.X[i] - self.X_min) / (self.X_max - self.X_min), 
                                   0.5)  # Set constant features to 0.5
    
    def configure_logging(self, level: str) -> logging.Logger:
        """Configure logging with proper formatting."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        logger.setLevel(getattr(logging, level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def load_training_data(self, fid='L2', train_split=0.8, seed=42):
        """
        Load the training data for the emulator.
        Parameters:
        ---------------
        fid (str): Fidelity of the data to load, e.g., 'L2' or 'L2-HF', where
            the latter is the pair of the fidelities.
        train_split (float): Fraction of the data to use for training.
        seed (int): Random seed for reproducibility.
        Returns:
        ---------------
        X_train (np.ndarray): Training input features.
              They are already normalized between 0 and 1, 
              use self.X_min and self.X_max to convert them back 
              to the original cosmological parameters.
        Y_train (np.ndarray): Training output features.
        Y_err_train (np.ndarray): Uncertainties for the training outputs.
        Y_valid (np.ndarray): Validation output features.
        Y_err_valid (np.ndarray): Uncertainties for the validation outputs.
        """
        np.random.seed(seed)
        if fid == 'L2':
            # Shuffle the mock spectra
            data_size = self.X[0].shape[0]
            ind_shuf = np.random.permutation(data_size)
            split_index = int(data_size * train_split)
            Y_train = self.Y[0][ind_shuf][:split_index]
            Y_err_train = self.Y_err[0][ind_shuf][:split_index]
            X_train = self.X[0][ind_shuf][:split_index]
            Y_valid = self.Y[0][ind_shuf][split_index:]
            Y_err_valid = self.Y_err[0][ind_shuf][split_index:]
            X_valid = self.X[0][ind_shuf][split_index:]

        normalizer = DataNormalizer()
        normalizer.fit(Y_train)
        train_dataset = CosmologyDataset(X_train, Y_train, Y_err_train, normalizer)
        val_dataset = CosmologyDataset(X_valid, Y_valid, Y_err_valid, normalizer)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
        return train_loader, val_loader, normalizer
