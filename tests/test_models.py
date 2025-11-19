import pytest
import torch
from models.resnet_model import ResNetWhiteBalanceModel
from models.efficientnet_model import EfficientNetWhiteBalanceModel
from models.metadata_model import MetadataOnlyModel

class TestModels:
    
    def test_resnet_model(self):
        """Test ResNet model forward pass"""
        config = {
            'model': {
                'name': 'resnet34',
                'pretrained': False,
                'metadata_hidden_size': 10,
                'hidden_size': 512,
                'dropout': 0.3
            }
        }
        
        model = ResNetWhiteBalanceModel(config)
        
        # Test input
        batch_size = 2
        image = torch.randn(batch_size, 3, 224, 224)
        metadata = torch.randn(batch_size, 10)
        
        output = model(image, metadata)
        
        assert output.shape == (batch_size, 2)
        assert not torch.isnan(output).any()
    
    def test_efficientnet_model(self):
        """Test EfficientNet model forward pass"""
        config = {
            'model': {
                'name': 'efficientnet_b0',
                'pretrained': False,
                'metadata_hidden_size': 10,
                'hidden_size': 512,
                'dropout': 0.3
            }
        }
        
        model = EfficientNetWhiteBalanceModel(config)
        
        # Test input
        batch_size = 2
        image = torch.randn(batch_size, 3, 224, 224)
        metadata = torch.randn(batch_size, 10)
        
        output = model(image, metadata)
        
        assert output.shape == (batch_size, 2)
        assert not torch.isnan(output).any()
    
    def test_metadata_model(self):
        """Test metadata-only model"""
        config = {
            'model': {
                'metadata_hidden_size': 10
            }
        }
        
        model = MetadataOnlyModel(config)
        
        # Test input
        batch_size = 2
        image = torch.randn(batch_size, 3, 224, 224)  # Ignored
        metadata = torch.randn(batch_size, 10)
        
        output = model(image, metadata)
        
        assert output.shape == (batch_size, 2)
        assert not torch.isnan(output).any()
