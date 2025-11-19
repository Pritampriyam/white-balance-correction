#!/usr/bin/env python3
"""
Main training script for white balance correction model.
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.trainer import WhiteBalanceTrainer
from src.utils.config import load_config
from src.data.data_loader import WhiteBalanceDataset
from src.data.transforms import get_transforms
from models.resnet_model import ResNetWhiteBalanceModel
from models.efficientnet_model import EfficientNetWhiteBalanceModel
from models.metadata_model import MetadataOnlyModel
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description='Train white balance model')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config file')
    parser.add_argument('--experiment', type=str, default='default',
                       help='Experiment name')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Prepare data
    print("Loading and preparing data...")
    
    # Load full training data
    full_df = pd.read_csv(config.data.train_csv)
    
    # Split into train and validation
    train_df, val_df = train_test_split(
        full_df, 
        test_size=config.data.val_size, 
        random_state=config.data.random_state,
        stratify=full_df[config.data.categorical_features[0]] if config.data.categorical_features else None
    )
    
    # Save splits
    os.makedirs('data/processed', exist_ok=True)
    train_df.to_csv('data/processed/train_split.csv', index=False)
    val_df.to_csv('data/processed/val_split.csv', index=False)
    
    # Create datasets
    train_dataset = WhiteBalanceDataset(
        image_dir=config.data.train_image_dir,
        csv_path='data/processed/train_split.csv',
        is_train=True,
        transform=get_transforms(config, 'train'),
        config=config
    )
    
    val_dataset = WhiteBalanceDataset(
        image_dir=config.data.train_image_dir,  # Use train dir but val split
        csv_path='data/processed/val_split.csv',
        is_train=True,
        transform=get_transforms(config, 'val'),
        config=config
    )
    
    # Create model based on config
    metadata_size = train_dataset.preprocessor.get_metadata_size()
    
    if config.model.name.startswith('resnet'):
        model = ResNetWhiteBalanceModel(config)
    elif config.model.name.startswith('efficientnet'):
        model = EfficientNetWhiteBalanceModel(config)
    else:
        model = MetadataOnlyModel(config)
    
    # Initialize trainer
    trainer = WhiteBalanceTrainer(config, args.experiment)
    trainer.setup_model(model, metadata_size)
    trainer.setup_data(train_dataset, val_dataset)
    
    # Train or resume training
    if args.resume:
        trainer.resume_training(args.resume)
    else:
        trainer.train()

if __name__ == '__main__':
    main()
