#!/usr/bin/env python3
"""
Evaluation script for model performance.
"""

import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.utils.metrics import calculate_mae, calculate_rmse, calculate_composite_score

def evaluate_predictions(pred_csv, true_csv=None):
    """Evaluate predictions against ground truth"""
    
    pred_df = pd.read_csv(pred_csv)
    
    if true_csv:
        # Compare with ground truth
        true_df = pd.read_csv(true_csv)
        
        # Merge predictions with ground truth
        merged_df = true_df.merge(pred_df, on='id_global', suffixes=('_true', '_pred'))
        
        # Calculate metrics
        mae_temp, mae_tint = calculate_mae(
            merged_df[['Temperature_pred', 'Tint_pred']].values,
            merged_df[['Temperature_true', 'Tint_true']].values
        )
        
        rmse_temp, rmse_tint = calculate_rmse(
            merged_df[['Temperature_pred', 'Tint_pred']].values,
            merged_df[['Temperature_true', 'Tint_true']].values
        )
        
        composite_score = calculate_composite_score(mae_temp, mae_tint)
        
        print("Evaluation Results:")
        print(f"MAE - Temperature: {mae_temp:.2f}, Tint: {mae_tint:.2f}")
        print(f"RMSE - Temperature: {rmse_temp:.2f}, Tint: {rmse_tint:.2f}")
        print(f"Composite Score: {composite_score:.2f}")
        
        return {
            'mae_temp': mae_temp,
            'mae_tint': mae_tint,
            'rmse_temp': rmse_temp,
            'rmse_tint': rmse_tint,
            'composite_score': composite_score
        }
    
    else:
        # Just show prediction statistics
        print("Prediction Statistics:")
        print(f"Temperature - Min: {pred_df['Temperature'].min()}, Max: {pred_df['Temperature'].max()}, Mean: {pred_df['Temperature'].mean():.2f}")
        print(f"Tint - Min: {pred_df['Tint'].min()}, Max: {pred_df['Tint'].max()}, Mean: {pred_df['Tint'].mean():.2f}")
        print(f"Total predictions: {len(pred_df)}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate model predictions')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions CSV')
    parser.add_argument('--ground-truth', type=str, default=None,
                       help='Path to ground truth CSV (optional)')
    
    args = parser.parse_args()
    
    evaluate_predictions(args.predictions, args.ground_truth)

if __name__ == '__main__':
    main()
