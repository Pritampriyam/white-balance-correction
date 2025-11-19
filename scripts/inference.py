#!/usr/bin/env python3
"""
Main inference script for white balance correction model.
"""

import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference.predictor import WhiteBalancePredictor
from src.utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize predictor
    predictor = WhiteBalancePredictor(config, args.checkpoint)
    
    # Run prediction
    predictions = predictor.predict()
    
    # Save results
    predictor.save_predictions(predictions, args.output)
    print("Inference completed!")

if __name__ == '__main__':
    main()
