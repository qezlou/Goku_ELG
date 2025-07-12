


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os


# --- Data Normalizer ---
class SpectrumNormalizer:
    """
    Normalizes 1D data with multiple features such as flux, sky spectrum, and noise.

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

    def fit(self, x):
        """
        Compute mean and std for each feature (channel).

        Args:
            x (Tensor): Input of shape [batch_size, seq_len, num_features]
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        flat = x.view(-1, x.size(-1))  # collapse batch and sequence
        self.means = flat.mean(dim=0)
        self.stds = flat.std(dim=0)

    def transform(self, x):
        """
        Normalize input using stored mean and std.

        Args:
            x (Tensor): Input of shape [batch_size, seq_len, num_features]

        Returns:
            Tensor: Normalized input, same shape.
        """
        if self.means is None or self.stds is None:
            raise RuntimeError("Call fit() before transform().")
        x = (x - self.means) / (self.stds + self.eps)
        return x.float()

    def inverse_transform(self, x):
        """
        Revert normalization.

        Args:
            x (Tensor): Normalized tensor.

        Returns:
            Tensor: Original-scale tensor.
        """
        return x * (self.stds + self.eps) + self.means

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    """
    Applies sinusoidal positional encoding to represent the position of each spectral bin.

    Args:
        d_model (int): Dimension of embedding.
        max_len (int): Maximum length of the sequence.

    Inputs:
        x (Tensor): Shape (batch_size, seq_len, d_model)

    Returns:
        Tensor: Same shape as input with position information added.
    """
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        # Create positional encodings for each spectral bin (wavelength position) using sinusoids of varying frequencies.
        # This encodes the relative position of each point in the 1D spectrum (important for transformer models).
        pe = torch.zeros(max_len, d_model)  # [seq_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Store the positional encodings as a non-trainable buffer so they move with the model across devices.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add position encoding to input embeddings: [batch, seq_len, emb_dim].
        # Each spectral bin now contains both the original information and its relative position.
        return x + self.pe[:x.size(1)]


class LowFidelityMassAwareTransformer(nn.Module):
    """
    Stage 1 model: Transformer-based predictor for low-fidelity power spectra.
    Inputs: 
        - θ: Tensor of shape (B, d_theta)
        - P_masked: Tensor of shape (B, N_m1m2, N_k), power spectrum values (with zeros where masked)
        - mask: Tensor of shape (B, N_m1m2, N_k), binary mask (1 if valid, 0 if missing)
        - m1m2: Tensor of shape (B, N_m1m2, 2), halo mass pairs
        - k_values: Tensor of shape (N_k,), global k-bin locations
    Output:
        - predicted P_low: shape (B, N_m1m2, N_k)
    """
    def __init__(self, d_theta, d_model=128, nhead=8, num_layers=4, dropout=0.1, n_k=64):
        super().__init__()
        self.n_k = n_k
        self.k_embed = nn.Linear(1, d_model)
        self.mass_embed = nn.Linear(2, d_model)
        self.theta_embed = nn.Linear(d_theta, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_k)
        )

    def forward(self, theta, m1m2, mask):
        """
        theta: (B, d_theta)
        m1m2: (B, N_m1m2, 2)
        mask: (B, N_m1m2, N_k)
        """
        B, N_m1m2, _ = m1m2.shape

        # Embed inputs
        theta_emb = self.theta_embed(theta)  # (B, d_model)
        theta_emb = theta_emb.unsqueeze(1).expand(-1, N_m1m2, -1)  # (B, N_m1m2, d_model)

        mass_emb = self.mass_embed(m1m2)  # (B, N_m1m2, d_model)

        tokens = theta_emb + mass_emb  # (B, N_m1m2, d_model)

        # Transformer encoding across mass bins
        tokens_out = self.encoder(tokens)  # (B, N_m1m2, d_model)

        # Output head per token (mass bin)
        preds = self.output_head(tokens_out)  # (B, N_m1m2, N_k)

        return preds
    
class HighFidelityCorrectionTransformer(nn.Module):
    """
    Stage 2 model: Learns correction from low-fidelity spectrum to high-fidelity.
    Inputs:
        - θ: Tensor of shape (B, d_theta)
        - m1m2: Tensor of shape (B, N_m1m2, 2)
        - P_low: Tensor of shape (B, N_m1m2, N_k), predicted low-fidelity spectrum
        - mask: Tensor of shape (B, N_m1m2, N_k), binary mask
    Output:
        - ΔP(k): predicted residuals to correct P_low into P_high
    """
    def __init__(self, d_theta, d_model=128, nhead=8, num_layers=4, dropout=0.1, n_k=64):
        super().__init__()
        self.n_k = n_k
        self.mass_embed = nn.Linear(2, d_model)
        self.theta_embed = nn.Linear(d_theta, d_model)
        self.plow_embed = nn.Linear(n_k, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_k)
        )

    def forward(self, theta, m1m2, plow, mask):
        """
        theta: (B, d_theta)
        m1m2: (B, N_m1m2, 2)
        plow: (B, N_m1m2, N_k)
        mask: (B, N_m1m2, N_k)
        """
        B, N_m1m2, _ = m1m2.shape

        # Embed each component
        theta_emb = self.theta_embed(theta).unsqueeze(1).expand(-1, N_m1m2, -1)  # (B, N_m1m2, d_model)
        mass_emb = self.mass_embed(m1m2)  # (B, N_m1m2, d_model)
        plow_emb = self.plow_embed(plow)  # (B, N_m1m2, d_model)

        tokens = theta_emb + mass_emb + plow_emb  # (B, N_m1m2, d_model)

        tokens_out = self.encoder(tokens)  # (B, N_m1m2, d_model)
        delta_p = self.output_head(tokens_out)  # (B, N_m1m2, N_k)

        return delta_p