import lightly_train
import torch

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        out="out/corn_field_navigation",
        overwrite=True,  # This allows overwriting existing files
        steps=2500,  # <-- Roughly 500 steps for every 20 images
        batch_size=2,  # <-- SET THIS TO 1 OR 2
        model="dinov3/vits16-eomt",  # Uses the ViT-S/16 backbone
        precision="16-mixed",
        num_workers=0,
        torch_compile_args={"disable": False},
        transform_args={
            "image_size": (224, 224),  # Smaller = faster training
        },
        loader_args={
            "persistent_workers": False,  # Disable persistent workers
            "pin_memory": False,  # Disable pin memory for small dataset
        },
        data={
            "train": {
                "images": r"C:/Users/Paul/Documents/Cornfield_data/Train/Images",
                "masks": r"C:/Users/Paul/Documents/Cornfield_data/Train/Masks",
            },
            "val": {
                "images": r"C:/Users/Paul/Documents/Cornfield_data/Val/Images",
                "masks": r"C:/Users/Paul/Documents/Cornfield_data/Val/Masks",
            },
            "classes": {           # MATCH YOUR JSON CLASSES
                0: "sky",          # Class 0 from your JSON
                1: "traversable",  # Class 1 from your JSON  
                2: "obstacle",     # Class 2 from your JSON
            },
            "ignore_classes": [],  # Skip ignore class during training
        },
    )