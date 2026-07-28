import os
import cv2
import pydicom
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

#Convert DICOM pixel values to Hounsfield Units (HU)
def dicom_to_hu(dicom_ds):
    image = dicom_ds.pixel_array.astype(np.float32)
    
    intercept = getattr(dicom_ds, 'RescaleIntercept', -1024)
    slope = getattr(dicom_ds, 'RescaleSlope', 1.0)
    
    # Standardize padding values below -1000 to air density
    image[image < -1000] = -1000
    
    hu_image = (image * slope) + intercept
    return hu_image

# Apply windowing to the HU image
def apply_window(hu_image, center, width):
    min_hu = center - (width / 2.0)
    max_hu = center + (width / 2.0)
    
    windowed = np.clip(hu_image, min_hu, max_hu)
    
    normalized = (windowed - min_hu) / (max_hu - min_hu)
    return normalized


# Convert DICOM to 3-channel PNG

def convert_dicom_to_3channel_png(dicom_path):
    dicom_ds = pydicom.dcmread(dicom_path)
    hu_img = dicom_to_hu(dicom_ds)
    
    # Subdural / Blood Window (80, 200)
    chan_subdural = apply_window(hu_img, center=80, width=200)
    # Brain Window (40, 80)
    chan_brain = apply_window(hu_img, center=40, width=80)
    # Bone Window (600, 2800)
    chan_bone = apply_window(hu_img, center=600, width=2800)
    
    # Stack along third dimension and rescale to 8-bit [0, 255]
    rgb_img = np.dstack((chan_subdural, chan_brain, chan_bone))
    rgb_8bit = (rgb_img * 255.0).astype(np.uint8)
    
    return rgb_8bit 


# Data Augmentation Pipeline
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


# 3. Dicom to PNG conversion
def process_dicom_folder(dicom_dir: str, output_dir: str, target_size: int = 384):
    os.makedirs(output_dir, exist_ok=True)
    dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    
    print(f"Found {len(dicom_files)} DICOM files. Processing to '{output_dir}'...")

    for filename in tqdm(dicom_files):
        img_id = filename.replace('.dcm', '')
        dcm_path = os.path.join(dicom_dir, filename)
        png_path = os.path.join(output_dir, f"{img_id}.png")
        
        if os.path.exists(png_path):
            continue
            
        try:
            rgb_8bit = convert_dicom_to_3channel_png(dcm_path)
            
            if target_size:
                rgb_8bit = cv2.resize(rgb_8bit, (target_size, target_size), interpolation=cv2.INTER_AREA)
            
            # Convert RGB -> BGR for OpenCV image saving
            bgr_img = cv2.cvtColor(rgb_8bit, cv2.COLOR_RGB2BGR)
            cv2.imwrite(png_path, bgr_img)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("Preprocessing completed successfully")


if __name__ == "__main__":
    INPUT_DICOM_DIR = r"C:\Users\haree\Downloads\rsna_sample_dataset"
    OUTPUT_PNG_DIR = r"C:\Users\haree\OneDrive\Documents\rsna_dataset_png"

    process_dicom_folder(INPUT_DICOM_DIR, OUTPUT_PNG_DIR, target_size=384)