import cv2
import os
from pathlib import Path

def extract_frames_smart(video_path, output_dir, start_time_sec=0, end_time_sec=2000, 
                         every_n_frames=30, skip_existing=True):
    """
    Extract frames from a specific time range with skip existing option
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        start_time_sec: Start time in seconds (e.g., 60 = start at 1 minute)
        end_time_sec: End time in seconds (None = go to end)
        every_n_frames: Extract every Nth frame
        skip_existing: If True, skip frames that already exist
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of existing frames
    existing_frames = set()
    if skip_existing:
        for f in Path(output_dir).glob("frame_*.jpg"):
            existing_frames.add(f.stem)
        print(f"Found {len(existing_frames)} existing frames, will skip them")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    
    # Convert time to frame numbers
    start_frame = int(start_time_sec * fps)
    end_frame = int(end_time_sec * fps) if end_time_sec else total_frames
    
    print(f"Video info: {fps:.2f} fps, {duration_sec:.1f} sec total")
    print(f"Extracting frames from {start_time_sec:.1f}s to {end_time_sec:.1f}s")
    print(f"Frame range: {start_frame} to {end_frame}")
    print(f"Extracting every {every_n_frames} frame(s)")
    
    # Jump to start position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    saved_count = 0
    skipped_count = 0
    
    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if this frame should be saved (every Nth frame)
        if (frame_count - start_frame) % every_n_frames == 0:
            # Generate filename
            filename = f"frame_{frame_count:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Check if already exists
            if skip_existing and os.path.exists(filepath):
                skipped_count += 1
            else:
                cv2.imwrite(filepath, frame)
                saved_count += 1
                if saved_count % 10 == 0:
                    print(f"Saved: {saved_count} frames (skipped: {skipped_count})")
        
        frame_count += 1
        
        # Progress update every 500 frames
        if (frame_count - start_frame) % 500 == 0:
            progress = (frame_count - start_frame) / (end_frame - start_frame) * 100
            print(f"Progress: {progress:.1f}% (saved: {saved_count})")
    
    cap.release()
    print(f"\n✓ Done! Saved {saved_count} new frames")
    print(f"✓ Skipped {skipped_count} existing frames")
    print(f"✓ Total frames in folder: {saved_count + skipped_count}")
    
    return saved_count

# ============================================================
# Path Directories
# ============================================================
#VIDEO_PATH = r"/home/weichien241/ag_bot/src/videos/2026-06-16-11-18-09.mp4"
VIDEO_PATH = r"/home/weichien241/ag_bot/src/videos/sim_run_v3.mp4"
# OUTPUT_DIR = r"frames_to_annotate/2026-06-16-11-18-09"
OUTPUT_DIR = r"frames_to_annotate/sim"

# ============================================================
# Extract from the whole video
# ============================================================
extract_frames_smart(
    video_path=VIDEO_PATH,
    output_dir=OUTPUT_DIR,
    start_time_sec=0,        # start at the beginning
    end_time_sec=100,        # set to None first to figure how long the video is
    every_n_frames=30,        # grab ~1 frame/sec; set this ≈ your video's fps
    skip_existing=True,
)

# # ============================================================
# # Path Directories
# # ============================================================

# VIDEO_PATH = r"Images/2024-08-14-12-20-40.mp4"
# OUTPUT_DIR = r"frames_to_annotate"

# # ============================================================
# # BATCH 1: First frames 
# # ============================================================
# print("\n" + "="*50)
# print("BATCH 1:")
# print("="*50)

# extract_frames_smart(
#     video_path=VIDEO_PATH,
#     output_dir=OUTPUT_DIR,
#     start_time_sec=0,        # Start at beginning
#     end_time_sec=30,         # End at 30 seconds
#     every_n_frames=30,       # Every 30th frame (about 1 frame per second at 30fps)
#     skip_existing=True       # Don't re-extract existing frames
# )

# # ============================================================
# # BATCH 2: Next frames
# # ============================================================
# print("\n" + "="*50)
# print("BATCH 2:")
# print("="*50)

# extract_frames_smart(
#     video_path=VIDEO_PATH,
#     output_dir=OUTPUT_DIR,
#     start_time_sec=210,       # Start 
#     end_time_sec=360,         # End 
#     every_n_frames=30,
#     skip_existing=True
# )

# # ============================================================
# # BATCH 3: Next frames 
# # ============================================================
# print("\n" + "="*50)
# print("BATCH 3:")
# print("="*50)

# extract_frames_smart(
#     video_path=VIDEO_PATH,
#     output_dir=OUTPUT_DIR,
#     start_time_sec=390,       # Start at 1 minute
#     end_time_sec=540,         # End at 1.5 minutes
#     every_n_frames=30,
#     skip_existing=True
# )
