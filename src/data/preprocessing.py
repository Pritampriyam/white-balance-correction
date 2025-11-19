import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

class DataPreprocessor:
    """Preprocess metadata for white balance model"""
    
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        
    def fit_transform(self, df):
        """Fit preprocessor and transform data"""
        return self._preprocess_data(df, fit=True)
    
    def transform(self, df):
        """Transform data using fitted preprocessor"""
        return self._preprocess_data(df, fit=False)
    
    def _preprocess_data(self, df, fit=False):
        df = df.copy()
        
        # Handle numeric features
        numeric_cols = self.config.data.numeric_features
        for col in numeric_cols:
            if col not in df.columns:
                warnings.warn(f"Numeric column {col} not found in data")
                continue
                
            # Fill missing values
            df[col] = df[col].fillna(df[col].median())
            
            if fit:
                self.scalers[col] = StandardScaler()
                df[col] = self.scalers[col].fit_transform(df[[col]]).flatten()
            else:
                if col in self.scalers:
                    df[col] = self.scalers[col].transform(df[[col]]).flatten()
                else:
                    warnings.warn(f"Scaler for {col} not fitted, using raw values")
        
        # Handle categorical features
        categorical_cols = self.config.data.categorical_features
        for col in categorical_cols:
            if col not in df.columns:
                warnings.warn(f"Categorical column {col} not found in data")
                continue
                
            # Fill missing with 'unknown'
            df[col] = df[col].fillna('unknown')
            
            if fit:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col])
            else:
                if col in self.encoders:
                    # Handle unseen categories
                    unseen_mask = ~df[col].isin(self.encoders[col].classes_)
                    if unseen_mask.any():
                        df.loc[unseen_mask, col] = 'unknown'
                    df[col] = self.encoders[col].transform(df[col])
                else:
                    warnings.warn(f"Encoder for {col} not fitted, using raw values")
        
        # Store feature columns for metadata tensor
        self.feature_columns = numeric_cols + categorical_cols
        
        return df
    
    def get_metadata_tensor(self, row):
        """Convert row to metadata tensor"""
        features = []
        
        for col in self.feature_columns:
            if col in row:
                features.append(float(row[col]))
            else:
                features.append(0.0)  # Default value for missing columns
        
        return torch.tensor(features, dtype=torch.float32)
    
    def get_metadata_size(self):
        """Get size of metadata vector"""
        return len(self.feature_columns)
