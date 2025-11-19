import torch
import numpy as np
from sklearn.metrics import mean_absolute_error

def calculate_mae(predictions, targets):
    """Calculate MAE for temperature and tint"""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    mae_temp = mean_absolute_error(targets[:, 0], predictions[:, 0])
    mae_tint = mean_absolute_error(targets[:, 1], predictions[:, 1])
    
    return mae_temp, mae_tint

def calculate_rmse(predictions, targets):
    """Calculate RMSE for temperature and tint"""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    rmse_temp = np.sqrt(np.mean((targets[:, 0] - predictions[:, 0]) ** 2))
    rmse_tint = np.sqrt(np.mean((targets[:, 1] - predictions[:, 1]) ** 2))
    
    return rmse_temp, rmse_tint

def calculate_composite_score(mae_temp, mae_tint, temp_weight=0.7, tint_weight=0.3):
    """Calculate composite score weighted by importance"""
    return (mae_temp * temp_weight + mae_tint * tint_weight) / (temp_weight + tint_weight)
