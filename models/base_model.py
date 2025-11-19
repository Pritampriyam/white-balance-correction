import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """Base class for all white balance models"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, image, metadata):
        """Forward pass
        
        Args:
            image: Input image tensor
            metadata: Metadata tensor
            
        Returns:
            Tuple of (temperature_residual, tint_residual)
        """
        pass
    
    def get_parameters(self):
        """Get model parameters for optimizer"""
        return self.parameters()
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        return self
