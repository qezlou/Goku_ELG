import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
import logging
from typing import Tuple, List, Dict, Any




# Fully connected neural network
class HMFNet(nn.Module):
    def __init__(self, input_dim: int = 9, output_dim: int = 14, hidden_dims: List[int] = None):
        super(HMFNet, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

