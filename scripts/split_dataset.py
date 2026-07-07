import random, shutil
from pathlib import Path

IMAGES = Path.home() / "ag_bot/src/DINOv3-Segmentation-Training/annotations_combined/images"
MASKS  = Path.home() / "ag_bot/src/DINOv3-Segmentation-Training/annotations_combined/masks"
OUT    = Path.home() / "ag_bot/src/DINOv3-Segmentation-Training/dataset_v3"
VAL_FRACTION = 0.2
random.seed(0)

pairs = []
for img in sorted(IMAGES.glob("*.jpg")):
    mask = MASKS / (img.stem + ".png")
    if mask.exists():
        pairs.append((img, mask))
    else:
        print("WARNING: no mask for", img.name)

random.shuffle(pairs)
n_val = max(1, round(len(pairs) * VAL_FRACTION))
val, train = pairs[:n_val], pairs[n_val:]

for split, items in [("train", train), ("val", val)]:
    (OUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT / split / "masks").mkdir(parents=True, exist_ok=True)
    for img, mask in items:
        shutil.copy(img,  OUT / split / "images" / img.name)
        shutil.copy(mask, OUT / split / "masks"  / mask.name)

print(f"train: {len(train)} pairs, val: {len(val)} pairs")