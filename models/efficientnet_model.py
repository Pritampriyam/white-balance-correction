import torch
import torch.nn as nn
import torchvision.models as models
from .base_model import BaseModel

class EfficientNetWhiteBalanceModel(BaseModel):
    """EfficientNet-based white balance correction model"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Image branch
        backbone_name = config.model.name
        if backbone_name == "efficientnet_b0":
            backbone = models.efficientnet_b0(pretrained=config.model.pretrained)
            feature_size = 1280
        elif backbone_name == "efficientnet_b1":
            backbone = models.efficientnet_b1(pretrained=config.model.pretrained)
            feature_size = 1280
        elif backbone_name == "efficientnet_b2":
            backbone = models.efficientnet_b2(pretrained=config.model.pretrained)
            feature_size = 1408
        else:
            raise ValueError(f"Unsupported EfficientNet: {backbone_name}")
        
        self.image_features = backbone.features
        self.image_avgpool = backbone.avgpool
        self.image_fc = nn.Linear(feature_size, config.model.hidden_size)
        
        # Metadata branch
        self.metadata_size = config.model.metadata_hidden_size
        self.metadata_fc = nn.Sequential(
            nn.Linear(self.metadata_size, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(config.model.dropout),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(config.model.dropout * 0.5),
        )
        
        # Combined head
        combined_size = config.model.hidden_size + 128
        self.head = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(config.model.dropout),
            nn.Linear(256, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 2)  # temperature_residual, tint_residual
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, image, metadata):
        # Image features
        img_feat = self.image_features(image)
        img_feat = self.image_avgpool(img_feat)
        img_feat = img_feat.view(img_feat.size(0), -1)
        img_feat = self.image_fc(img_feat)
        
        # Metadata features
        meta_feat = self.metadata_fc(metadata)
        
        # Combine features
        combined = torch.cat([img_feat, meta_feat], dim=1)
        output = self.head(combined)
        
        return output
