%%writefile /kaggle/working/data_loader.py
import os
import cv2
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedGroupKFold


def get_train_transforms(img_size=384):

    return A.Compose([

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.05,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),

        A.GaussNoise(
            std_range=(0.01, 0.03),
            mean_range=(0.0, 0.0),
            p=0.25,
        ),

        A.GaussianBlur(
            blur_limit=(3, 3),
            p=0.15,
        ),

        A.ElasticTransform(
            alpha=1,
            sigma=20,
            p=0.10,
        ),

        A.Resize(img_size, img_size),

        A.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        ),

        ToTensorV2()

    ])


def get_val_transforms(img_size=384):

    return A.Compose([

        A.Resize(img_size, img_size),

        A.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        ),

        ToTensorV2()

    ])


class RSNADataset(Dataset):

    def __init__(self, df, img_dir, transform=None):

        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        self.label_cols = [
            "any",
            "epidural",
            "intraparenchymal",
            "intraventricular",
            "subarachnoid",
            "subdural"
        ]

        self.labels = self.df[self.label_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image_id = self.df.iloc[idx]["Image_ID"]

        path = os.path.join(
            self.img_dir,
            image_id + ".png"
        )

        image = cv2.imread(path)

        if image is None:
            raise FileNotFoundError(path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.float32
        )

        return image, label


def prepare_dataloaders(
    df,
    img_dir,
    batch_size,
    num_workers,
    img_size,
    patient_col="PatientID",
):

    groups = df[patient_col]
    y = df["any"]

    outer = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    train_idx, temp_idx = next(
        outer.split(df, y, groups)
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    inner = StratifiedGroupKFold(
        n_splits=2,
        shuffle=True,
        random_state=42
    )

    val_idx, test_idx = next(

        inner.split(

            temp_df,

            temp_df["any"],

            temp_df[patient_col]

        )

    )

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    print()

    print("Dataset Summary")

    print("--------------------------")

    print("Train :", len(train_df))
    print("Val   :", len(val_df))
    print("Test  :", len(test_df))

    train_dataset = RSNADataset(
        train_df,
        img_dir,
        get_train_transforms(img_size)
    )

    val_dataset = RSNADataset(
        val_df,
        img_dir,
        get_val_transforms(img_size)
    )

    test_dataset = RSNADataset(
        test_df,
        img_dir,
        get_val_transforms(img_size)
    )

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader
