# src/data/image_dataset.py
"""
PyTorch Image DataLoaders for Cats vs Dogs (MLOps Assignment 2)
Handles data from data/PetImages/ folder structure
"""
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_train_val_test_splits(source_dir, output_dir, val_split=0.1, test_split=0.1, random_state=42):
    """
    Create train/val/test splits from PetImages/Cat and PetImages/Dog structure
    
    Expected input structure:
        source_dir/
            Cat/
                image1.jpg
                image2.jpg
                ...
            Dog/
                image1.jpg
                image2.jpg
                ...
    
    Output structure:
        output_dir/
            train/
                Cat/
                Dog/
            val/
                Cat/
                Dog/
            test/
                Cat/
                Dog/
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        for cls in ['Cat', 'Dog']:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for cls in ['Cat', 'Dog']:
        cls_source = source_dir / cls
        if not cls_source.exists():
            print(f"[WARN] {cls_source} not found")
            continue
        
        # Get all images
        images = [f for f in os.listdir(cls_source) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            print(f"[WARN] No images found in {cls_source}")
            continue
        
        print(f"Found {len(images)} {cls} images")
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            images, test_size=test_split, random_state=random_state
        )
        
        # Second split: train vs val
        train, val = train_test_split(
            train_val, test_size=val_split / (1 - test_split), random_state=random_state
        )
        
        # Copy files
        for split_name, split_images in [('train', train), ('val', val), ('test', test)]:
            for img in split_images:
                src = cls_source / img
                dst = output_dir / split_name / cls / img
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"[ERROR] Failed to copy {src}: {e}")
        
        print(f"[OK] {cls}: train={len(train)}, val={len(val)}, test={len(test)}")
    
    return str(output_dir)


def get_dataloaders(data_dir='data/PetImages', batch_size=32, num_workers=0, 
                    splits_output='data/splits', val_split=0.1, test_split=0.1):
    """
    Create train/val/test dataloaders from PetImages folder
    
    Args:
        data_dir: Path to PetImages folder (contains Cat/ and Dog/ subfolders)
        batch_size: Batch size for dataloaders
        num_workers: Number of workers (0 for CPU on Windows/GitHub)
        splits_output: Where to save the splits
        val_split: Validation set fraction
        test_split: Test set fraction
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create splits if they don't exist
    splits_dir = Path(splits_output)
    if not splits_dir.exists() or len(list((splits_dir / 'train').glob('*/*'))) == 0:
        print(f"Creating train/val/test splits from {data_dir}...")
        create_train_val_test_splits(
            source_dir=data_dir,
            output_dir=splits_output,
            val_split=val_split,
            test_split=test_split
        )
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dataloaders
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        split_path = splits_dir / split
        
        if not split_path.exists() or len(list(split_path.glob('*/*'))) == 0:
            print(f"[WARN] Split {split} is empty")
            continue
        
        transform = train_transform if split == 'train' else eval_transform
        dataset = datasets.ImageFolder(str(split_path), transform=transform)
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
        print(f"[OK] {split} dataloader: {len(dataset)} images, {len(dataloaders[split])} batches")
    
    return dataloaders['train'], dataloaders['val'], dataloaders['test']
