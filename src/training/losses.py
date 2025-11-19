import torch
import torch.nn as nn

def get_loss_function(loss_name):
    """Get loss function by name"""
    if loss_name == "l1":
        return nn.L1Loss()
    elif loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "huber":
        return nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

class AdaptiveLoss(nn.Module):
    """Adaptive loss that weights temperature and tint differently"""
    
    def __init__(self, temp_weight=1.0, tint_weight=1.0):
        super().__init__()
        self.temp_weight = temp_weight
        self.tint_weight = tint_weight
        self.l1_loss = nn.L1Loss(reduction='none')
    
    def forward(self, predictions, targets):
        losses = self.l1_loss(predictions, targets)
        temp_loss = losses[:, 0].mean() * self.temp_weight
        tint_loss = losses[:, 1].mean() * self.tint_weight
        return temp_loss + tint_loss
