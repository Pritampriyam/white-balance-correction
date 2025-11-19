import torch
import torch.nn as nn
import torchvision.models as models
from .base_model import BaseModel

class ResNetWhiteBalanceModel(BaseModel):
    """ResNet-based white balance correction model"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Image branch
        backbone_name = config.model.name
        if backbone_name == "resnet18":
            backbone = models.resnet18(pretrained=config.model.pretrained)
            feature_size = 512
        elif backbone_name == "resnet34":
            backbone = models.resnet34(pretrained=config.model.pretrained)
            feature_size = 512
        elif backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=config.model.pretrained)
            feature_size = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        self.image_features = nn.Sequential(*list(backbone.children())[:-1])
        self.image_fc = nn.Linear(feature_size, config.model.hidden_size)
        
        # Metadata branch
        self.metadata_size = config.model.metadata_hidden_size
        self.metadata_fc = nn.Sequential(
            nn.Linear(self.metadata_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.model.dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.model.dropout * 0.5),
        )
        
        # Combined head
        combined_size = config.model.hidden_size + 128
        self.head = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.model.dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)  # temperature_residual, tint_residual
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for custom layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, image, metadata):
        # Image features
        img_feat = self.image_features(image)
        img_feat = img_feat.view(img_feat.size(0), -1)
        img_feat = self.image_fc(img_feat)
        
        # Metadata features
        meta_feat = self.metadata_fc(metadata)
        
        # Combine features
        combined = torch.cat([img_feat, meta_feat], dim=1)
        output = self.head(combined)
        
        return output
