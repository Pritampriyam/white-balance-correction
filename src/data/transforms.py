import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(config, phase='train'):
    """Get data transforms for training or validation"""
    
    if phase == 'train':
        augmentations = config.augmentation.train
        transform_list = []
        
        for aug_config in augmentations:
            aug_name = aug_config.name
            params = {k: v for k, v in aug_config.items() if k != 'name'}
            
            if aug_name == 'RandomResizedCrop':
                transform_list.append(A.RandomResizedCrop(**params))
            elif aug_name == 'HorizontalFlip':
                transform_list.append(A.HorizontalFlip(**params))
            elif aug_name == 'RandomBrightnessContrast':
                transform_list.append(A.RandomBrightnessContrast(**params))
            elif aug_name == 'GaussianBlur':
                transform_list.append(A.GaussianBlur(**params))
            elif aug_name == 'Resize':
                transform_list.append(A.Resize(**params))
        
        # Always add normalization and tensor conversion
        transform_list.extend([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
        
    else:  # validation or test
        augmentations = config.augmentation.val
        transform_list = []
        
        for aug_config in augmentations:
            aug_name = aug_config.name
            params = {k: v for k, v in aug_config.items() if k != 'name'}
            
            if aug_name == 'Resize':
                transform_list.append(A.Resize(**params))
        
        transform_list.extend([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    
    return A.Compose(transform_list)
