import torch
import os

class EarlyStopping:
    """Early stopping callback"""
    
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

class ModelCheckpoint:
    """Model checkpoint callback"""
    
    def __init__(self, save_dir, monitor='val_loss', mode='min', save_best_only=True):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
        os.makedirs(save_dir, exist_ok=True)
    
    def __call__(self, model, value, epoch):
        if self.save_best_only:
            if (self.mode == 'min' and value < self.best_value) or \
               (self.mode == 'max' and value > self.best_value):
                self.best_value = value
                self._save_model(model, epoch, is_best=True)
        else:
            self._save_model(model, epoch, is_best=False)
    
    def _save_model(self, model, epoch, is_best=True):
        filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
        path = os.path.join(self.save_dir, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_value': self.best_value
        }, path)
