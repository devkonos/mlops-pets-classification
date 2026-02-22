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


def preprocess_dataset(data_dir):
    """
    Validate images in dataset and optionally resize to 224x224
    
    Args:
        data_dir: Path to dataset directory (train/val/test/class structure)
    
    Returns:
        True if successful
    """
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    data_dir = Path(data_dir)
    valid_count = 0
    removed_count = 0
    
    # Iterate through all splits and classes
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            
            # Find all image files (case-insensitive)
            for img_file in list(cls_dir.iterdir()):
                if not img_file.is_file() or img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                try:
                    # Try to open and load image
                    img = Image.open(img_file)
                    img.load()
                    
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize to 224x224
                    img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
                    img.close()
                    
                    # Save with good quality
                    img_resized.save(str(img_file), 'JPEG', quality=95)
                    
                    valid_count += 1
                    
                except Exception as e:
                    # Remove corrupt images
                    print(f"[WARN] Removing corrupt image: {img_file}: {type(e).__name__}")
                    try:
                        img_file.unlink()
                        removed_count += 1
                    except:
                        pass
    
    print(f"[OK] Validation: {valid_count} valid, {removed_count} removed")
    return True


def get_dataloaders(data_dir, batch_size=32, num_workers=0, splits_output=None, 
                    val_split=0.1, test_split=0.1):
    """
    Create train/val/test dataloaders from dataset folder
    
    Args:
        data_dir: Path to dataset or PetImages folder
        batch_size: Batch size for dataloaders
        num_workers: Number of workers (0 for CPU on Windows/GitHub)
        splits_output: Optional path where to save splits (for PetImages)
        val_split: Validation set fraction (used for split creation)
        test_split: Test set fraction (used for split creation)
    
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    data_dir = Path(data_dir)
    
    # Check if this is PetImages folder (has Cat/Dog at root)
    is_petimages = (data_dir / 'Cat').exists() and (data_dir / 'Dog').exists()
    
    # Check if already split (has train/val/test at root)
    is_presplit = (data_dir / 'train').exists() and (data_dir / 'val').exists() and (data_dir / 'test').exists()
    
    if is_petimages and splits_output:
        # Create splits if they don't exist
        splits_dir = Path(splits_output)
        if not splits_dir.exists() or len(list((splits_dir / 'train').glob('*/*'))) == 0:
            print(f"Creating train/val/test splits from {data_dir}...")
            create_train_val_test_splits(
                source_dir=str(data_dir),
                output_dir=splits_output,
                val_split=val_split,
                test_split=test_split
            )
        data_dir = splits_dir
    
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
        split_path = data_dir / split
        
        if not split_path.exists():
            print(f"[WARN] Split directory {split_path} does not exist")
            dataloaders[split] = None
            continue
        
        # Check if there are any images in this split
        images = list(split_path.glob('*/*.jpg')) + list(split_path.glob('*/*.png')) + list(split_path.glob('*/*.jpeg'))
        if len(images) == 0:
            print(f"[WARN] Split {split} has no images")
            dataloaders[split] = None
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
    
    return dataloaders
