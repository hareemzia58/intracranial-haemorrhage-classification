import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

# Augmentation Pipelines
def get_train_transforms(img_size: int = 384) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.05, 0.05),
            rotate=(-12, 12),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.1, 
            contrast_limit=0.1, 
            p=0.4
        ),
        A.Resize(height=img_size, width=img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transforms(img_size: int = 384) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

class RSNADataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
        # Target label columns
        self.label_cols = ['any', 'epidural', 'intraparenchymal', 
                           'intraventricular', 'subarachnoid', 'subdural']
        
        # Pre-extract label matrix as float32 numpy array for faster indexing
        self.labels = self.df[self.label_cols].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img_id = self.df.iloc[idx]['Image_ID']
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
            
        # convert to RGB for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply Augmentation + Normalization + ToTensor
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

# 3. Train/Val/Test Split & DataLoader Helper
def prepare_dataloaders(
    df: pd.DataFrame, 
    img_dir: str, 
    batch_size: int = 32, 
    num_workers: int = 4,
    img_size: int = 384
):
    stratify_col = df['any']
    
    # split into Train (80%) and Temp (20%)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.20, 
        random_state=42, 
        stratify=stratify_col
    )
    
    temp_stratify = temp_df['any']
    
    # split Temp into 10% Val and 10% Test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        random_state=42, 
        stratify=temp_stratify
    )
    
    print(f"Dataset Split Summary:")
    print(f" - Train samples: {len(train_df)}")
    print(f" - Val samples:   {len(val_df)}")
    print(f" - Test samples:  {len(test_df)}")

    # Create PyTorch Datasets
    train_dataset = RSNADataset(train_df, img_dir, transform=get_train_transforms(img_size))
    val_dataset   = RSNADataset(val_df, img_dir, transform=get_val_transforms(img_size))
    test_dataset  = RSNADataset(test_df, img_dir, transform=get_val_transforms(img_size))

    # Create PyTorch DataLoaders
    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=use_pin
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=use_pin
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=use_pin
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    PNG_DIR = r"C:\Users\haree\OneDrive\Documents\rsna_dataset_png"
    METADATA_CSV = r"C:\Users\haree\OneDrive\Documents\rsna_train_sample_final.csv"
    
    df = pd.read_csv(METADATA_CSV)
    
    train_loader, val_loader, test_loader = prepare_dataloaders(
        df=df, 
        img_dir=PNG_DIR, 
        batch_size=16, 
        num_workers=2,
        img_size=384
    )
    
    images, labels = next(iter(train_loader))
    
    print(f"Images batch tensor shape: {images.shape}") 
    print(f"Labels batch tensor shape: {labels.shape}") 
    print(f"Sample label vector: {labels[0]}")