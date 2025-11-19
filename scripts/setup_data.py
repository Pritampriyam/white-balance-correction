#!/usr/bin/env python3
"""
Data setup and preparation script.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def check_data_structure():
    """Check if data is properly organized"""
    required_paths = [
        'data/raw/train/images',
        'data/raw/train/sliders.csv',
        'data/raw/validation/images', 
        'data/raw/validation/sliders_inputs.csv'
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path not found: {path}")
    
    # Check image counts
    train_images = [f for f in os.listdir('data/raw/train/images') if f.endswith('.tiff')]
    val_images = [f for f in os.listdir('data/raw/validation/images') if f.endswith('.tiff')]
    
    print(f"Found {len(train_images)} training images")
    print(f"Found {len(val_images)} validation images")
    
    # Check CSV files
    train_df = pd.read_csv('data/raw/train/sliders.csv')
    val_df = pd.read_csv('data/raw/validation/sliders_inputs.csv')
    
    print(f"Training CSV has {len(train_df)} rows")
    print(f"Validation CSV has {len(val_df)} rows")
    
    return True

def analyze_data():
    """Analyze dataset characteristics"""
    print("\nAnalyzing dataset...")
    
    # Training data analysis
    train_df = pd.read_csv('data/raw/train/sliders.csv')
    
    print("Training data columns:", train_df.columns.tolist())
    print("\nTraining data summary:")
    print(train_df.describe())
    
    if 'Temperature' in train_df.columns:
        print(f"\nTemperature range: {train_df['Temperature'].min()} to {train_df['Temperature'].max()}")
        print(f"Tint range: {train_df['Tint'].min()} to {train_df['Tint'].max()}")
    
    # Validation data analysis
    val_df = pd.read_csv('data/raw/validation/sliders_inputs.csv')
    print(f"\nValidation data columns: {val_df.columns.tolist()}")
    
    # Check for missing values
    print("\nMissing values in training data:")
    print(train_df.isnull().sum())
    
    print("\nMissing values in validation data:")
    print(val_df.isnull().sum())

def create_processed_dirs():
    """Create processed data directories"""
    os.makedirs('data/processed', exist_ok=True)
    print("Created processed data directory")

if __name__ == '__main__':
    print("Setting up white balance correction data...")
    
    try:
        check_data_structure()
        create_processed_dirs()
        analyze_data()
        print("\nData setup completed successfully!")
        
    except Exception as e:
        print(f"Error during data setup: {e}")
        print("Please ensure the data is organized according to the README")
