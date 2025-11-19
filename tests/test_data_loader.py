import pytest
import torch
import pandas as pd
import os
from src.data.data_loader import WhiteBalanceDataset
from src.data.preprocessing import DataPreprocessor

class TestDataLoader:
    
    def test_dataset_creation(self):
        """Test that dataset can be created"""
        # This would require actual data to test properly
        pass
    
    def test_preprocessor_fit_transform(self):
        """Test data preprocessor"""
        config = {
            'data': {
                'numeric_features': ['feature1', 'feature2'],
                'categorical_features': ['category1']
            }
        }
        
        preprocessor = DataPreprocessor(config)
        
        # Test data
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'category1': ['A', 'B', 'A', 'C', 'B']
        })
        
        transformed = preprocessor.fit_transform(df)
        
        assert 'feature1' in transformed.columns
        assert 'category1' in transformed.columns
        assert len(transformed) == 5
    
    def test_metadata_tensor(self):
        """Test metadata tensor creation"""
        config = {
            'data': {
                'numeric_features': ['feature1'],
                'categorical_features': []
            }
        }
        
        preprocessor = DataPreprocessor(config)
        df = pd.DataFrame({'feature1': [1, 2, 3]})
        preprocessor.fit_transform(df)
        
        row = df.iloc[0]
        tensor = preprocessor.get_metadata_tensor(row)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
