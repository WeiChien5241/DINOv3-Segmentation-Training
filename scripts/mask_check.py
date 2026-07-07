import numpy as np
from PIL import Image
m = np.array(Image.open("/home/weichien241/ag_bot/DINOv3-Segmentation-Training/Annotated/Output/2026-06-11-11-53-20/frame_000120.png"))
print(np.unique(m))   # should print exactly [0 1 2]