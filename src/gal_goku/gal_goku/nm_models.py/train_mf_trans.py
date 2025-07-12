import json

import logging
import os
import matplotlib.pyplot as plt



import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_neurral_op import LowFidelityMassAwareTransformer, HighFidelityCorrectionTransformer

def masked_mse_loss(pred, target, mask):
    """
    Computes mean squared error only on valid (non-masked) elements.
    """
    loss = ((pred - target)**2 * mask).sum() / (mask.sum() + 1e-8)
    return loss

def train_stage1(model, dataset, val_dataset, d_theta, n_k, save_path, lr=1e-3, batch_size=32, epochs=50, device='cuda'):
    """
    Trains the LowFidelityMassAwareTransformer model for Stage 1.

    Parameters:
        model: the LowFidelityMassAwareTransformer instance
        dataset: a PyTorch Dataset returning (theta, m1m2, P_target, mask)
        d_theta: dimensionality of cosmology input
        n_k: number of k bins
        lr: learning rate
        batch_size: batch size
        epochs: number of training epochs
        device: 'cuda' or 'cpu'
    """
    os.makedirs(save_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(save_path, 'train.log'), level=logging.INFO, filemode='w')
    logging.info("Starting training...")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for theta, m1m2, P_target, mask in dataloader:
            theta = theta.to(device)
            m1m2 = m1m2.to(device)
            P_target = P_target.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            P_pred = model(theta, m1m2, mask)
            loss = masked_mse_loss(P_pred, P_target, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for theta, m1m2, P_target, mask in val_loader:
                theta = theta.to(device)
                m1m2 = m1m2.to(device)
                P_target = P_target.to(device)
                mask = mask.to(device)
                P_pred = model(theta, m1m2, mask)
                val_loss += masked_mse_loss(P_pred, P_target, mask).item()
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)
        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        torch.save(model.state_dict(), os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pt"))

    # Plot loss curves
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.close()


def train_stage2(model_stage1, model_stage2, dataset, val_dataset, d_theta, n_k, save_path, lr=1e-3, batch_size=32, epochs=50, device='cuda'):
    """
    Trains the HighFidelityCorrectionTransformer (Stage 2) model using frozen Stage 1 predictions.

    Parameters:
        model_stage1: Pretrained LowFidelityMassAwareTransformer (frozen)
        model_stage2: HighFidelityCorrectionTransformer to train
        dataset: a PyTorch Dataset returning (theta, m1m2, P_high, mask)
        d_theta: dimensionality of cosmology input
        n_k: number of k bins
        lr: learning rate
        batch_size: batch size
        epochs: number of training epochs
        device: 'cuda' or 'cpu'
    """
    os.makedirs(save_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(save_path, 'train.log'), level=logging.INFO, filemode='w')
    logging.info("Starting Stage 2 training...")

    model_stage1 = model_stage1.to(device)
    model_stage1.eval()  # Freeze Stage 1
    for param in model_stage1.parameters():
        param.requires_grad = False

    model_stage2 = model_stage2.to(device)
    optimizer = torch.optim.Adam(model_stage2.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model_stage2.train()
        total_loss = 0.0

        for theta, m1m2, P_high, mask in dataloader:
            theta = theta.to(device)
            m1m2 = m1m2.to(device)
            P_high = P_high.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                P_low_pred = model_stage1(theta, m1m2, mask)

            delta_target = P_high - P_low_pred
            delta_pred = model_stage2(theta, m1m2, P_low_pred, mask)

            loss = masked_mse_loss(delta_pred, delta_target, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)

        # Validation loop
        model_stage2.eval()
        val_loss = 0.0
        with torch.no_grad():
            for theta, m1m2, P_high, mask in val_loader:
                theta = theta.to(device)
                m1m2 = m1m2.to(device)
                P_high = P_high.to(device)
                mask = mask.to(device)
                P_low_pred = model_stage1(theta, m1m2, mask)
                delta_target = P_high - P_low_pred
                delta_pred = model_stage2(theta, m1m2, P_low_pred, mask)
                val_loss += masked_mse_loss(delta_pred, delta_target, mask).item()
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)
        logging.info(f"[Stage 2] Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        print(f"[Stage 2] Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        torch.save(model_stage2.state_dict(), os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pt"))

    # Plot loss curves
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Stage 2 Training Curve')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.close()

def save_config(config, path):
    """
    Save the configuration dictionary to a JSON file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info(f"Configuration saved to {path}")

def create_example_config():
    """
    Return a sample configuration dictionary.
    """
    example_config = {
        "d_theta": 6,
        "n_k": 64,
        "d_model": 128,
        "nhead": 8,
        "num_layers": 4,
        "dropout": 0.1,
        "batch_size": 32,
        "epochs": 50,
        "lr": 1e-3,
        "save_path": "./checkpoints/stage1"
    }
    return example_config


# === Config loader and training entrypoint ===
def load_config(path):
    """
    Load configuration dictionary from a JSON file.
    """
    with open(path, 'r') as f:
        config = json.load(f)
    return config

def train(config_path, stage, dataset, val_dataset, device='cuda'):
    """
    Load config and run training for specified stage.
    stage: 'stage1' or 'stage2'
    """
    config = load_config(config_path)
    d_theta = config['d_theta']
    n_k = config['n_k']
    save_path = config['save_path']
    lr = config.get('lr', 1e-3)
    batch_size = config.get('batch_size', 32)
    epochs = config.get('epochs', 50)

    if stage == 'stage1':
        model = LowFidelityMassAwareTransformer(
            d_theta=d_theta,
            d_model=config.get('d_model', 128),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 4),
            dropout=config.get('dropout', 0.1),
            n_k=n_k
        )
        train_stage1(model, dataset, val_dataset, d_theta, n_k, save_path, lr, batch_size, epochs, device)

    elif stage == 'stage2':
        model_stage1 = LowFidelityMassAwareTransformer(
            d_theta=d_theta,
            d_model=config.get('d_model', 128),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 4),
            dropout=config.get('dropout', 0.1),
            n_k=n_k
        )
        model_stage1.load_state_dict(torch.load(config['stage1_weights'], map_location=device))

        model_stage2 = HighFidelityCorrectionTransformer(
            d_theta=d_theta,
            d_model=config.get('d_model', 128),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 4),
            dropout=config.get('dropout', 0.1),
            n_k=n_k
        )
        train_stage2(model_stage1, model_stage2, dataset, val_dataset, d_theta, n_k, save_path, lr, batch_size, epochs, device)

    else:
        raise ValueError("Stage must be either 'stage1' or 'stage2'")