import os
import numpy as np

def train_mlp_baseline():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("PyTorch is still installing in the background. Please re-run once installation finishes!")
        return

    print("--- Training Model #1: MLP Baseline ---")
    
    # 1. Load windowed datasets
    train_data = np.load("data/processed/train_windows.npz")
    val_data = np.load("data/processed/val_windows.npz")
    
    X_train_raw, y_train_raw = train_data["X"], train_data["y"]
    X_val_raw, y_val_raw = val_data["X"], val_data["y"]
    
    # Flatten (N, 20, 9) into (N, 180) for MLP input
    N_train, T, C = X_train_raw.shape
    N_val = X_val_raw.shape[0]
    
    X_train = X_train_raw.reshape(N_train, T * C)
    X_val = X_val_raw.reshape(N_val, T * C)
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train_raw, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val_raw, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 2. Define MLP Baseline Model
    class MLPBaseline(nn.Module):
        def __init__(self, input_dim=180):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            
        def forward(self, x):
            return self.net(x).squeeze(-1)
            
    model = MLPBaseline(input_dim=T * C)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 3. Training Loop
    epochs = 15
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
            
        train_loss /= len(train_dataset)
        
        # Validation Evaluation
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                pred = model(bx)
                val_mae += torch.sum(torch.abs(pred - by)).item()
                
        val_mae /= len(val_dataset)
        val_mae_kmh = val_mae * 3.6  # Convert m/s MAE to km/h MAE
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Speed MAE: {val_mae:.2f} m/s ({val_mae_kmh:.2f} km/h)")
        
    # Save model checkpoint
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/mlp_baseline.pth")
    print("\nMLP Baseline training complete! Saved checkpoint to models/mlp_baseline.pth")

if __name__ == "__main__":
    train_mlp_baseline()
