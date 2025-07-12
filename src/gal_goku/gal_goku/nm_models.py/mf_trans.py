


import torch
import torch.nn as nn
import torch.nn.functional as F

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