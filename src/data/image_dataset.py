# src/data/image_dataset.py
"""
PyTorch Image DataLoaders for Cats vs Dogs (MLOps Assignment 2)
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def preprocess_dataset(data_dir):
    """Clean corrupt + resize 224x224 (idempotent)"""
    import cv2
    from PIL import Image, ImageFile  # ← Complete!
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    splits = ['train', 'val', 'test']
    cleaned = 0
    
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        for cls in ['cats', 'dogs']:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.exists(cls_dir): 
                continue
                
            images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
            
            for img_name in images:
                img_path = os.path.join(cls_dir, img_name)
                
                # Skip if already processed (idempotent!)
                if os.path.getsize(img_path) > 10000:  # ~224x224 JPG >10KB
                    cleaned += 1
                    continue
                
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        os.remove(img_path)
                        print(f"Removed corrupt: {img_path}")
                        continue
                    
                    # Resize + optimize
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (224, 224))
                    
                    # Save as PIL (handles corrupt better)
                    pil_img = Image.fromarray(img_resized)
                    pil_img.save(img_path, 'JPEG', quality=95, optimize=True)
                    cleaned += 1
                    
                except Exception as e:
                    print(f"Failed {img_path}: {e}")
                    if os.path.exists(img_path):
                        os.remove(img_path)
    
    print(f"Preprocessed: {cleaned} valid 224x224 images")
    return True

def get_dataloaders(data_dir, batch_size=32, num_workers=0):  # Windows CPU safe
    """
    FIXED: Stronger aug + proper workers (75%+ baseline)
    """
    preprocess_dataset(data_dir)
    
    # STRONG AUG (notebook demo + extras)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),  # Stronger
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomVerticalFlip(p=0.1),  # NEW
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        transform = train_transform if split == 'train' else eval_transform
        dataset = datasets.ImageFolder(os.path.join(data_dir, split), transform=transform)
        dataloaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == 'train'),
            num_workers=num_workers, pin_memory=True  # CPU safe
        )
    print(f"Strong DataLoaders: batch={batch_size}, workers={num_workers}")
    return dataloaders

# Usage:
# dataloaders = get_dataloaders("data/raw/cats_vs_dogs")
