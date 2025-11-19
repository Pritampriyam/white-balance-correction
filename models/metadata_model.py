import torch
import torch.nn as nn
from .base_model import BaseModel

class MetadataOnlyModel(BaseModel):
    """Metadata-only model for white balance correction (fast baseline)"""
    
    def __init__(self, config):
        super().__init__(config)
        
        input_size = config.model.metadata_hidden_size
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)  # temperature_residual, tint_residual
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, image, metadata):
        # Ignore image, use only metadata
        return self.network(metadata)
