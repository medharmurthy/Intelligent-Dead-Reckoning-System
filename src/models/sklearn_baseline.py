import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def train_sklearn_baselines():
    print("--- Training Scikit-Learn Baselines ---")
    
    # 1. Load windowed data
    train_data = np.load("data/processed/train_windows.npz")
    val_data = np.load("data/processed/val_windows.npz")
    
    X_train_raw, y_train = train_data["X"], train_data["y"]
    X_val_raw, y_val = val_data["X"], val_data["y"]
    
    # Flatten (N, 20, 9) into (N, 180) for tabular regressors
    N_train, T, C = X_train_raw.shape
    N_val = X_val_raw.shape[0]
    
    X_train = X_train_raw.reshape(N_train, T * C)
    X_val = X_val_raw.reshape(N_val, T * C)
    
    print(f"Train samples: {N_train}, Val samples: {N_val}, Features per sample: {T * C}")
    
    # Baseline Model 1: Ridge Regression
    print("\n[Model Baseline 1A] Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_val)
    
    mae_ridge = mean_absolute_error(y_val, y_pred_ridge)
    rmse_ridge = root_mean_squared_error(y_val, y_pred_ridge)
    print(f"Ridge Regressor  -> Val MAE: {mae_ridge:.2f} m/s ({mae_ridge * 3.6:.2f} km/h) | RMSE: {rmse_ridge:.2f} m/s")
    
    # Baseline Model 2: Random Forest Regressor
    print("\n[Model Baseline 1B] Training Random Forest Regressor (50 trees)...")
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_val)
    
    mae_rf = mean_absolute_error(y_val, y_pred_rf)
    rmse_rf = root_mean_squared_error(y_val, y_pred_rf)
    print(f"Random Forest    -> Val MAE: {mae_rf:.2f} m/s ({mae_rf * 3.6:.2f} km/h) | RMSE: {rmse_rf:.2f} m/s")
    
    print("\nBaseline evaluation finished successfully!")

if __name__ == "__main__":
    train_sklearn_baselines()
