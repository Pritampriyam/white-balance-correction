import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from ..data.data_loader import WhiteBalanceDataset
from ..data.transforms import get_transforms

class WhiteBalancePredictor:
    """Predictor for white balance correction"""
    
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Setup data
        self.dataset = WhiteBalanceDataset(
            image_dir=config.data.val_image_dir,
            csv_path=config.data.val_csv,
            is_train=False,
            transform=get_transforms(config, 'val'),
            config=config
        )
    
    def _load_model(self, checkpoint_path):
        """Load model from checkpoint"""
        # Determine model type from config
        if self.config.model.name.startswith('resnet'):
            from models.resnet_model import ResNetWhiteBalanceModel
            model_class = ResNetWhiteBalanceModel
        elif self.config.model.name.startswith('efficientnet'):
            from models.efficientnet_model import EfficientNetWhiteBalanceModel
            model_class = EfficientNetWhiteBalanceModel
        else:
            from models.metadata_model import MetadataOnlyModel
            model_class = MetadataOnlyModel
        
        # Create model
        metadata_size = self.dataset.preprocessor.get_metadata_size() if hasattr(self, 'dataset') else 64
        model = model_class(self.config)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model.to(self.device)
    
    def predict(self):
        """Generate predictions for validation set"""
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(self.dataset, desc="Generating predictions"):
                if len(batch) == 5:  # validation data
                    image, metadata, curr_temp, curr_tint, id_global = batch
                else:
                    continue
                
                # Add batch dimension
                image = image.unsqueeze(0).to(self.device)
                metadata = metadata.unsqueeze(0).to(self.device)
                
                # Get residuals
                residuals = self.model(image, metadata)
                residuals = residuals.cpu().numpy()[0]
                
                # Convert to actual values
                pred_temp = residuals[0] + curr_temp
                pred_tint = residuals[1] + curr_tint
                
                # Round if configured
                if self.config.inference.round_predictions:
                    pred_temp = int(round(pred_temp))
                    pred_tint = int(round(pred_tint))
                
                predictions.append({
                    'id_global': id_global,
                    'Temperature': pred_temp,
                    'Tint': pred_tint
                })
        
        return predictions
    
    def save_predictions(self, predictions, output_path=None):
        """Save predictions to CSV"""
        if output_path is None:
            output_path = self.config.inference.output_csv
        
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(predictions)
        
        # Ensure we have exactly 493 predictions in correct order
        original_val_df = pd.read_csv(self.config.data.val_csv)
        final_df = original_val_df[['id_global']].merge(df, on='id_global', how='left')
        
        # Fill any missing predictions with current values as fallback
        missing_mask = final_df['Temperature'].isna()
        if missing_mask.any():
            print(f"Warning: {missing_mask.sum()} missing predictions, filling with current values")
            temp_cols = [col for col in original_val_df.columns if 'temp' in col.lower()]
            tint_cols = [col for col in original_val_df.columns if 'tint' in col.lower()]
            
            curr_temp_col = temp_cols[0] if temp_cols else 'currTemp'
            curr_tint_col = tint_cols[0] if tint_cols else 'currTint'
            
            final_df.loc[missing_mask, 'Temperature'] = original_val_df.loc[missing_mask, curr_temp_col]
            final_df.loc[missing_mask, 'Tint'] = original_val_df.loc[missing_mask, curr_tint_col]
        
        # Ensure correct data types and rounding
        final_df['Temperature'] = final_df['Temperature'].round().astype(int)
        final_df['Tint'] = final_df['Tint'].round().astype(int)
        
        # Save to CSV
        final_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        print(f"Generated {len(final_df)} predictions")
        
        return final_df
