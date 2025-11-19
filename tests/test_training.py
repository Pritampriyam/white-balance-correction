import pytest
import torch
from src.training.losses import get_loss_function, AdaptiveLoss

class TestTraining:
    
    def test_loss_functions(self):
        """Test loss function creation"""
        # Test L1 loss
        l1_loss = get_loss_function('l1')
        assert isinstance(l1_loss, torch.nn.L1Loss)
        
        # Test MSE loss
        mse_loss = get_loss_function('mse')
        assert isinstance(mse_loss, torch.nn.MSELoss)
        
        # Test Huber loss
        huber_loss = get_loss_function('huber')
        assert isinstance(huber_loss, torch.nn.HuberLoss)
        
        # Test invalid loss
        with pytest.raises(ValueError):
            get_loss_function('invalid')
    
    def test_adaptive_loss(self):
        """Test adaptive loss function"""
        loss_fn = AdaptiveLoss(temp_weight=2.0, tint_weight=1.0)
        
        predictions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = torch.tensor([[1.5, 2.5], [2.5, 3.5]])
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss > 0
