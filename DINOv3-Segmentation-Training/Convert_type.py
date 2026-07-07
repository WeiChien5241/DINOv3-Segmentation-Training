import json
import numpy as np
import cv2
from pathlib import Path
import os
from PIL import Image

def convert_coco_to_masks(json_path, image_folder, output_mask_folder):
    """
    Convert COCO JSON to segmentation masks.
    Saves as .png extension with JPEG compression for fast loading.
    """
    os.makedirs(output_mask_folder, exist_ok=True)
    
    # Load JSON
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Map category IDs to class values
    # Ensure these are mapped Correctly!!!
    # Traversable -> class 1
    # Untraversable -> class 2  
    # Sky -> class 0
    class_mapping = {
        1: 0,   
        2: 1,   
        3: 2    
    }
    
    # Get image info
    images = {img['id']: img for img in coco_data['images']}
    
    # Process each image
    for img_id, img_info in images.items():
        # Create empty mask (all zeros = ignore class)
        height = img_info['height']
        width = img_info['width']
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Find annotations for this image
        img_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
        
        # Draw each polygon
        for ann in img_annotations:
            cat_id = ann['category_id']
            class_id = class_mapping.get(cat_id, 0)
            
            if 'segmentation' in ann and ann['segmentation']:
                for seg in ann['segmentation']:
                    points = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [points], class_id)
        
        # Save as .png (but use JPEG compression for speed)
        img_filename = Path(img_info['file_name']).stem
        mask_path = Path(output_mask_folder) / f"{img_filename}.png"
        mask_image = Image.fromarray(mask)
        # mask_image.save(str(mask_path), 'JPEG', quality=95)
        mask_image.save(str(mask_path))
    
    print(f"✓ Saved {len(images)} masks to {output_mask_folder}")

# ============================================================
# Path Directories
# ============================================================

# JSON_PATH = r"Annotated/labels_my-project-name_2026-04-13-07-40-34.json"
# IMAGE_FOLDER = r"Annotated/Images"
# OUTPUT_MASK_FOLDER = r"Annotated/Output"

JSON_PATH = r"Annotated/labels_my-project-name_2026-06-30-03-23-31.json"
IMAGE_FOLDER = r"Annotated/Images/2026-06-16-11-18-09"
OUTPUT_MASK_FOLDER = r"Annotated/Output/2026-06-16-11-18-09"

convert_coco_to_masks(JSON_PATH, IMAGE_FOLDER, OUTPUT_MASK_FOLDER)