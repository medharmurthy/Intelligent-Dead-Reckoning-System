import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class TinyTCN(nn.Module):
    """
    Tiny Temporal Convolutional Network (Tiny TCN) with Multi-Task Heads:
    1. Forward Speed Estimation (v_f)
    2. Velocity Uncertainty Estimation (sigma_v^2 for EKF covariance matrix)
    """
    def __init__(self, in_channels=9):
        super().__init__()
        
        # Layer 1: Conv1D (kernel_size=5)
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        
        # Layer 2: Dilated Conv1D (dilation=2) with Residual Connection
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=2, dilation=2)
        
        # Layer 3: Dilated Conv1D (dilation=4)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=4, dilation=4)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Shared Dense representation
        self.fc_shared = nn.Linear(64, 64)
        
        # Multi-task Heads
        self.speed_head = nn.Linear(64, 1)        # Output: Forward Speed v_f (m/s)
        self.uncertainty_head = nn.Linear(64, 1)  # Output: log(sigma_v^2) for EKF
        
    def forward(self, x):
        # Input shape: (Batch, Channels=9, Seq_Len=20)
        h = self.relu(self.conv1(x))
        h_res = self.relu(self.conv2(h)) + h  # Residual connection
        h = self.relu(self.conv3(h_res))
        
        h = self.gap(h).squeeze(-1)  # Shape: (Batch, 64)
        feat = self.relu(self.fc_shared(h))
        
        pred_speed = self.speed_head(feat).squeeze(-1)
        log_var = self.uncertainty_head(feat).squeeze(-1)
        
        return pred_speed, log_var

def train_tiny_tcn():
    print("--- Training Model #4: Tiny TCN (Multi-Task Head) ---")
    
    # 1. Load windowed datasets
    train_data = np.load("data/processed/train_windows.npz")
    val_data = np.load("data/processed/val_windows.npz")
    
    X_train_raw, y_train_raw = train_data["X"], train_data["y"]
    X_val_raw, y_val_raw = val_data["X"], val_data["y"]
    
    # Reshape (N, T=20, C=9) to PyTorch Conv1D format (N, C=9, T=20)
    X_train = np.transpose(X_train_raw, (0, 2, 1))
    X_val = np.transpose(X_val_raw, (0, 2, 1))
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train_raw, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val_raw, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = TinyTCN(in_channels=9)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    huber_loss = nn.HuberLoss()
    
    epochs = 15
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred_speed, log_var = model(bx)
            
            # Loss = Speed Loss + Uncertainty Negative Log Likelihood
            loss_speed = huber_loss(pred_speed, by)
            # NLL loss term: (pred - target)^2 / exp(log_var) + log_var
            loss_uncertainty = torch.mean((pred_speed - by)**2 / torch.exp(log_var) + log_var)
            
            total_loss = loss_speed + 0.1 * loss_uncertainty
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item() * len(bx)
            
        train_loss /= len(train_dataset)
        
        # Validation Evaluation
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                pred_speed, _ = model(bx)
                val_mae += torch.sum(torch.abs(pred_speed - by)).item()
                
        val_mae /= len(val_dataset)
        val_mae_kmh = val_mae * 3.6
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Speed MAE: {val_mae:.2f} m/s ({val_mae_kmh:.2f} km/h)")
        
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/tiny_tcn.pth")
    print("\nTiny TCN Training complete! Saved checkpoint to models/tiny_tcn.pth")

if __name__ == "__main__":
    train_tiny_tcn()
