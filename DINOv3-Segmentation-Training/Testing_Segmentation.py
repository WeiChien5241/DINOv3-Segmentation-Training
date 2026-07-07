import torch
import lightly_train
from PIL import Image
import cv2
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = "out/corn_field_navigation/exported_models/exported_best.pt"
VIDEO_PATH = r"Images/2024-08-21-14-19-47.mp4"
OUTPUT_VIDEO_PATH = r"output_segmentation.mp4"
PROCESS_EVERY_N_FRAMES = 5
MAX_FRAMES = 38000
ALPHA = 0.5

# ============================================================
# LOAD MODEL
# ============================================================
print(f"Loading model...")
model = lightly_train.load_model(MODEL_PATH)
model.eval()
print("Model loaded!")

def process_frame(frame_bgr):
    """Process frame and return overlay"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    
    # model.predict() returns the mask tensor directly
    mask = model.predict(img_pil)
    
    # Debug first frame only
    if not hasattr(process_frame, '_debug_done'):
        print("\n=== MASK FORMAT DEBUG ===")
        print(f"  mask type: {type(mask)}")
        print(f"  mask shape: {mask.shape}")
        process_frame._debug_done = True
    
    # Convert to numpy if tensor
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    # Squeeze if needed (remove singleton dimensions)
    mask = np.squeeze(mask)
    
    # Ensure integer type
    mask = mask.astype(np.uint8)
    
    # Create overlay (class 1 = traversable)
    traversable = (mask == 1)
    overlay = frame_bgr.copy()
    overlay[traversable] = [0, 255, 0]
    result_frame = cv2.addWeighted(frame_bgr, 1 - ALPHA, overlay, ALPHA, 0)
    
    return result_frame

# ============================================================
# PROCESS VIDEO
# ============================================================
print(f"\nOpening video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Cannot open {VIDEO_PATH}")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height}, {fps} fps, {total_frames} frames")
print(f"Processing every {PROCESS_EVERY_N_FRAMES} frame")

# Setup output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

frame_count = 0
processed_count = 0
last_result = None

print("\nProcessing...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if MAX_FRAMES > 0 and frame_count >= MAX_FRAMES:
        break
    
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        result_frame = process_frame(frame)
        last_result = result_frame
        processed_count += 1
    else:
        result_frame = last_result if last_result is not None else frame
    
    out.write(result_frame)
    frame_count += 1
    
    if frame_count % 100 == 0:
        print(f"Frame {frame_count}/{total_frames} (processed: {processed_count})")

cap.release()
out.release()
# cv2.destroyAllWindows()  # Commented out - not needed for headless OpenCV

print(f"\n✓ Done! Processed {frame_count} frames")
print(f"✓ Model ran on {processed_count} frames")
print(f"✓ Output saved to: {OUTPUT_VIDEO_PATH}")