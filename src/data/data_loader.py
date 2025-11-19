import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2

class WhiteBalanceDataset(Dataset):
    """Dataset for white balance correction"""
    
    def __init__(self, image_dir, csv_path, is_train=True, transform=None, config=None):
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_path)
        self.is_train = is_train
        self.transform = transform
        self.config = config
        
        # Prepare metadata
        self.preprocessor = DataPreprocessor(config)
        self.df = self.preprocessor.fit_transform(self.df) if is_train else self.preprocessor.transform(self.df)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, f"{row['id_global']}.tiff")
        image = cv2.imread(img_path)
        
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image=image)['image']
        
        # Get metadata tensor
        metadata = self.preprocessor.get_metadata_tensor(row)
        
        if self.is_train:
            # Calculate residuals for training
            temp_residual = row['Temperature'] - row['currTemp']
            tint_residual = row['Tint'] - row['currTint']
            targets = torch.tensor([temp_residual, tint_residual], dtype=torch.float32)
            
            return image, metadata, targets, row['currTemp'], row['currTint']
        else:
            # For inference, return current values and id
            return image, metadata, row['currTemp'], row['currTint'], row['id_global']
