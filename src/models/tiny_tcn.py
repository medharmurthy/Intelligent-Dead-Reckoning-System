import os
import sys
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class TinyTCN(nn.Module):
    """
    Tiny Temporal Convolutional Network (Tiny TCN) with Multi-Task Heads:
    1. Forward Speed Estimation (v_f in m/s)
    2. Velocity Uncertainty Estimation (log_var = log(sigma_v^2) for EKF measurement covariance)
    3. Motion State Classification (5 classes: stationary, accelerating, cruising, braking, turning)
    
    Input shape: (Batch, Channels=9, Seq_Len=20)
    """
    def __init__(self, in_channels=9, num_motion_classes=5):
        super().__init__()
        
        # Layer 1: Conv1D (kernel_size=5, padding=2)
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        
        # Layer 2: Dilated Conv1D (dilation=2, padding=2) with Residual Connection
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=2, dilation=2)
        
        # Layer 3: Dilated Conv1D (dilation=4, padding=4)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=4, dilation=4)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Shared Dense representation
        self.fc_shared = nn.Linear(64, 64)
        
        # Multi-task Heads
        self.speed_head = nn.Linear(64, 1)              # Output: Forward Speed v_f (m/s)
        self.uncertainty_head = nn.Linear(64, 1)        # Output: log(sigma_v^2) for EKF covariance
        self.motion_head = nn.Linear(64, num_motion_classes) # Output: Motion Class Logits
        
    def forward(self, x):
        # Input shape: (Batch, Channels=9, Seq_Len=20)
        h = self.relu(self.conv1(x))
        h_res = self.relu(self.conv2(h)) + h  # Residual connection
        h = self.relu(self.conv3(h_res))
        
        h = self.gap(h).squeeze(-1)  # Shape: (Batch, 64)
        feat = self.relu(self.fc_shared(h))
        
        pred_speed = self.speed_head(feat).squeeze(-1)
        log_var = self.uncertainty_head(feat).squeeze(-1)
        motion_logits = self.motion_head(feat)
        
        return pred_speed, log_var, motion_logits


def apply_imu_augmentation(x_batch, noise_std=0.02, bias_max=0.05):
    """
    Applies domain randomization / IMU data augmentation:
    - Random Gaussian noise addition
    - Random sensor bias shift for accels and gyros
    """
    if not isinstance(x_batch, torch.Tensor):
        x_tensor = torch.tensor(x_batch, dtype=torch.float32)
    else:
        x_tensor = x_batch.clone()
        
    batch_size, channels, seq_len = x_tensor.shape
    
    # Add random noise
    noise = torch.randn_like(x_tensor) * noise_std
    
    # Add per-sample constant bias shift across temporal window
    bias = (torch.rand(batch_size, channels, 1) * 2 - 1) * bias_max
    
    return x_tensor + noise + bias


class PhysicsInformedMultiTaskLoss(nn.Module):
    """
    Physics-informed multi-task loss combining:
    1. Data Huber Loss (predicted speed vs ground truth)
    2. Uncertainty Negative Log-Likelihood Loss (adaptive variance learning)
    3. Physics Consistency Loss (|v_{t+1} - (v_t + a_x * dt)|^2)
    4. Smoothness Loss (|v_{t+1} - v_t|)
    5. Motion Classification Loss (Cross Entropy)
    """
    def __init__(self, lambda_unc=0.1, lambda_phys=0.05, lambda_smooth=0.05, lambda_motion=0.1):
        super().__init__()
        self.huber = nn.HuberLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.lambda_unc = lambda_unc
        self.lambda_phys = lambda_phys
        self.lambda_smooth = lambda_smooth
        self.lambda_motion = lambda_motion

    def forward(self, pred_speed, log_var, motion_logits, target_speed, target_motion=None, ax_last=None):
        # 1. Primary Data Loss
        loss_data = self.huber(pred_speed, target_speed)
        
        # 2. Heteroscedastic Uncertainty NLL Loss
        loss_unc = torch.mean((pred_speed - target_speed)**2 / (torch.exp(log_var) + 1e-6) + log_var)
        
        # 3. Smoothness Loss
        diffs = torch.abs(pred_speed[1:] - pred_speed[:-1]) if len(pred_speed) > 1 else torch.tensor(0.0)
        loss_smooth = torch.mean(diffs) if len(pred_speed) > 1 else torch.tensor(0.0)
        
        # 4. Physics Consistency Loss
        if ax_last is not None and len(pred_speed) > 1:
            dt = 0.1 # 10Hz sampling
            phys_expected = pred_speed[:-1] + ax_last[:-1] * dt
            loss_phys = torch.mean((pred_speed[1:] - phys_expected)**2)
        else:
            loss_phys = torch.tensor(0.0)
            
        # 5. Motion Classification Loss
        if target_motion is not None:
            loss_motion = self.cross_entropy(motion_logits, target_motion)
        else:
            loss_motion = torch.tensor(0.0)
            
        total_loss = loss_data + self.lambda_unc * loss_unc + self.lambda_smooth * loss_smooth + self.lambda_phys * loss_phys + self.lambda_motion * loss_motion
        return total_loss, {
            "data": loss_data.item(),
            "unc": loss_unc.item(),
            "smooth": loss_smooth.item(),
            "phys": loss_phys.item(),
            "motion": loss_motion.item() if isinstance(loss_motion, torch.Tensor) else 0.0
        }


def train_tiny_tcn(data_dir="data/processed", save_dir="models", epochs=20, batch_size=64, lr=1e-3, execute_training=False):
    """
    Prepares dataloaders and executes training loop with Cosine Annealing LR scheduler and best checkpoint saving.
    """
    print(f"--- Model #4: Tiny TCN Training ({epochs} Epochs, LR={lr}) ---")
    
    train_path = os.path.join(data_dir, "train_windows.npz")
    val_path = os.path.join(data_dir, "val_windows.npz")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Processed dataset not found at {train_path}. Run create_windows.py first!")
        
    train_data = np.load(train_path)
    val_data = np.load(val_path)
    
    X_train_raw, y_train_raw = train_data["X"], train_data["y"]
    X_val_raw, y_val_raw = val_data["X"], val_data["y"]
    
    # Reshape (N, T=20, C=9) to PyTorch Conv1D format (N, C=9, T=20)
    X_train = np.transpose(X_train_raw, (0, 2, 1))
    X_val = np.transpose(X_val_raw, (0, 2, 1))
    
    def derive_motion_class(speeds):
        classes = np.full(len(speeds), 2, dtype=np.int64) # Default cruising
        classes[speeds < 0.2] = 0 # Stationary
        return classes
        
    motion_train = derive_motion_class(y_train_raw)
    motion_val = derive_motion_class(y_val_raw)
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_raw, dtype=torch.float32),
        torch.tensor(motion_train, dtype=torch.int64)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val_raw, dtype=torch.float32),
        torch.tensor(motion_val, dtype=torch.int64)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = TinyTCN(in_channels=9, num_motion_classes=5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = PhysicsInformedMultiTaskLoss()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "tiny_tcn_best.pth")
    
    # Initial sanity check
    sample_x, sample_y, sample_m = next(iter(train_loader))
    sample_x_aug = apply_imu_augmentation(sample_x)
    pred_s, log_v, m_logits = model(sample_x_aug)
    loss, loss_dict = criterion(pred_s, log_v, m_logits, sample_y, sample_m, ax_last=sample_x_aug[:, 0, -1])
    
    if not os.path.exists(save_path):
        torch.save(model.state_dict(), save_path)
    
    best_val_mae = float('inf')
    best_epoch = 0
    
    if execute_training:
        print("\nStarting Training Loop...")
        for epoch in range(1, epochs + 1):
            model.train()
            total_train_loss = 0.0
            for bx, by, bm in train_loader:
                bx_aug = apply_imu_augmentation(bx)
                optimizer.zero_grad()
                pred_s, log_v, m_logits = model(bx_aug)
                loss, _ = criterion(pred_s, log_v, m_logits, by, bm, ax_last=bx_aug[:, 0, -1])
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * len(bx)
                
            scheduler.step()
            total_train_loss /= len(train_dataset)
            
            # Validation
            model.eval()
            val_mae = 0.0
            with torch.no_grad():
                for bx, by, _ in val_loader:
                    pred_s, _, _ = model(bx)
                    val_mae += torch.sum(torch.abs(pred_s - by)).item()
            val_mae /= len(val_dataset)
            val_mae_kmh = val_mae * 3.6
            
            is_best = ""
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path)
                is_best = " -> Saved Best Checkpoint!"
                
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {total_train_loss:.4f} | Val Speed MAE: {val_mae:.2f} m/s ({val_mae_kmh:.2f} km/h){is_best}")
            
        print(f"\nTraining Complete! Best model was from Epoch {best_epoch:02d} with Val MAE: {best_val_mae:.2f} m/s ({best_val_mae*3.6:.2f} km/h)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tiny TCN Model")
    parser.add_argument("--train", action="store_true", help="Execute full training loop")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    
    # Default to execute_training=True if --train or executed directly
    train_tiny_tcn(epochs=args.epochs, lr=args.lr, execute_training=args.train or len(sys.argv) == 1)
