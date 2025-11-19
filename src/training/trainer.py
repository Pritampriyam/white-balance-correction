import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import os
import time

from ..utils.metrics import calculate_mae
from .losses import get_loss_function
from .callbacks import EarlyStopping, ModelCheckpoint

class WhiteBalanceTrainer:
    """Training class for white balance models"""
    
    def __init__(self, config, experiment_name):
        self.config = config
        self.experiment_name = experiment_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup model, data, optimizer
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae_temp': [],
            'train_mae_tint': [],
            'val_mae_temp': [],
            'val_mae_tint': []
        }
        
        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=1e-4
        )
        
        self.model_checkpoint = ModelCheckpoint(
            save_dir=config.checkpoint.save_dir,
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
            save_best_only=config.checkpoint.save_best_only
        )
        
        # Experiment tracking
        if config.experiment.use_wandb:
            wandb.init(
                project="white-balance",
                name=experiment_name,
                config=dict(config)
            )
    
    def setup_model(self, model, metadata_size):
        """Setup model, optimizer, and scheduler"""
        self.model = model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Setup scheduler
        if self.config.training.scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=self.config.training.scheduler_patience,
                factor=self.config.training.scheduler_factor,
                verbose=True
            )
        elif self.config.training.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs
            )
        
        # Loss function
        self.criterion = get_loss_function(self.config.training.loss)
    
    def setup_data(self, train_dataset, val_dataset):
        """Setup data loaders"""
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_targets = []
        all_predictions = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} - Training")
        
        for batch_idx, batch in enumerate(pbar):
            images, metadata, targets, curr_temp, curr_tint = batch
            
            images = images.to(self.device, non_blocking=True)
            metadata = metadata.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images, metadata)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            
            # Convert predictions back to actual values for metrics
            with torch.no_grad():
                pred_temp = outputs[:, 0] + curr_temp.to(self.device)
                pred_tint = outputs[:, 1] + curr_tint.to(self.device)
                actual_temp = targets[:, 0] + curr_temp.to(self.device)
                actual_tint = targets[:, 1] + curr_tint.to(self.device)
                
                all_predictions.append(torch.stack([pred_temp, pred_tint], dim=1))
                all_targets.append(torch.stack([actual_temp, actual_tint], dim=1))
            
            # Update progress bar
            if batch_idx % self.config.experiment.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{running_loss / (batch_idx + 1):.4f}'
                })
                
                if self.config.experiment.use_wandb:
                    wandb.log({
                        'batch_loss': loss.item(),
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        mae_temp, mae_tint = calculate_mae(all_predictions, all_targets)
        
        return epoch_loss, mae_temp, mae_tint
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} - Validation")
            
            for batch in pbar:
                images, metadata, targets, curr_temp, curr_tint = batch
                
                images = images.to(self.device, non_blocking=True)
                metadata = metadata.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(images, metadata)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                
                # Convert predictions back to actual values
                pred_temp = outputs[:, 0] + curr_temp.to(self.device)
                pred_tint = outputs[:, 1] + curr_tint.to(self.device)
                actual_temp = targets[:, 0] + curr_temp.to(self.device)
                actual_tint = targets[:, 1] + curr_tint.to(self.device)
                
                all_predictions.append(torch.stack([pred_temp, pred_tint], dim=1))
                all_targets.append(torch.stack([actual_temp, actual_tint], dim=1))
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.val_loader)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        mae_temp, mae_tint = calculate_mae(all_predictions, all_targets)
        
        return epoch_loss, mae_temp, mae_tint
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.experiment_name}")
        print(f"Using device: {self.device}")
        
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train and validate
            train_loss, train_mae_temp, train_mae_tint = self.train_epoch()
            val_loss, val_mae_temp, val_mae_tint = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler and self.config.training.scheduler == "plateau":
                self.scheduler.step(val_loss)
            elif self.scheduler:
                self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae_temp'].append(train_mae_temp)
            self.history['train_mae_tint'].append(train_mae_tint)
            self.history['val_mae_temp'].append(val_mae_temp)
            self.history['val_mae_tint'].append(val_mae_tint)
            
            # Print epoch results
            print(f"\nEpoch {epoch + 1}/{self.config.training.epochs}:")
            print(f"Train Loss: {train_loss:.4f}, Train MAE - Temp: {train_mae_temp:.2f}, Tint: {train_mae_tint:.2f}")
            print(f"Val Loss: {val_loss:.4f}, Val MAE - Temp: {val_mae_temp:.2f}, Tint: {val_mae_tint:.2f}")
            
            # Log to wandb
            if self.config.experiment.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_mae_temp': train_mae_temp,
                    'train_mae_tint': train_mae_tint,
                    'val_mae_temp': val_mae_temp,
                    'val_mae_tint': val_mae_tint,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Check early stopping
            if self.early_stopping(val_loss):
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save checkpoint
            self.model_checkpoint(self.model, val_loss, epoch)
            
            # Update best loss
            if val_loss < self.best_loss:
                self.best_loss = val_loss
        
        print(f"Training completed. Best validation loss: {self.best_loss:.4f}")
        
        if self.config.experiment.use_wandb:
            wandb.finish()
    
    def resume_training(self, checkpoint_path):
        """Resume training from checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        print(f"Resumed training from epoch {self.current_epoch}")
