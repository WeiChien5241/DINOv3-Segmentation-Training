# DINOv3-Segmentation-Training

DINOv3-based semantic segmentation pipeline for Purdue's P-AgBot agricultural robot.
It trains a model (via [`lightly_train`](https://github.com/lightly-ai/lightly-train), `dinov3/vits16-eomt`) to classify camera frames from real corn/sorghum rows into three classes:

| Class index | Meaning |
|---|---|
| 0 | sky |
| 1 | traversable |
| 2 | obstacle / untraversable |

This class mapping is load-bearing and must stay consistent across annotation conversion, training, and inference.

## Repository layout

- **`DINOv3-Segmentation-Training/`** — the training pipeline itself (frame extraction, COCO-JSON → mask conversion, training, inference testing) plus the hand-annotated dataset (`Annotated/`, `annotations_combined/`). See its [README](DINOv3-Segmentation-Training/README.md) for the step-by-step workflow.
- **`scripts/`** — supporting pipeline scripts:
  - `bag2video.py` — converts rosbag recordings to mp4 (pure Python via the `rosbags` lib, no ROS install needed).
  - `split_dataset.py` — splits `annotations_combined/` into versioned `train`/`val` datasets.
  - `mask_check.py` — sanity-checks that a generated mask PNG contains only class values `{0, 1, 2}`.
  - `naming_frames` — the file-naming convention used when merging new annotations into the master pool.

Raw extracted frames, the derived `dataset_v*` train/val splits, and training outputs (`out/`, including model checkpoints) are not tracked — they are regenerated from the annotated data using the scripts above.

## Raw rosbag recordings

The raw rosbags the pipeline starts from (~13 GB total) are too large for regular git files, so they are attached as assets to the [`rosbags-v1` release](https://github.com/WeiChien5241/DINOv3-Segmentation-Training/releases/tag/rosbags-v1):

- Date-stamped bags (`2026-06-11-*`, `2026-06-16-*`, `2026-07-07-*`) — real corn/sorghum row recordings from the robot's camera (`/usb_cam/image_raw/compressed`).
- `sim_run*.bag` — Gazebo simulation runs.

Download everything into `ros_bags/` with the [GitHub CLI](https://cli.github.com/):

```bash
gh release download rosbags-v1 --repo WeiChien5241/DINOv3-Segmentation-Training --dir ros_bags
```

`sim_run_v2.bag` exceeds GitHub's 2 GiB per-asset cap, so it is uploaded as two chunks — rejoin them after downloading:

```bash
cd ros_bags
cat sim_run_v2.bag.00.part sim_run_v2.bag.01.part > sim_run_v2.bag
rm sim_run_v2.bag.??.part
sha256sum -c SHA256SUMS   # verify all bags
```

## Quick start

```bash
# 1. Convert recorded rosbags to video (needs: pip install rosbags opencv-python-headless numpy)
python3 scripts/bag2video.py <bag-or-folder> videos

# 2. Extract frames to annotate (edit path constants at the bottom of the file first)
python3 DINOv3-Segmentation-Training/Extract_Frames.py

# 3. Annotate in MakeSense AI (polygons: traversable=blue, untraversable=green, sky=pink), export COCO JSON

# 4. Convert annotations to single-channel PNG masks (edit path constants first)
python3 DINOv3-Segmentation-Training/Convert_type.py

# 5. Split into train/val (edit constants at the top first)
python3 scripts/split_dataset.py

# 6. Train (note: overwrites out/, including exported_best.pt — move checkpoints you want to keep)
python3 DINOv3-Segmentation-Training/Train.py

# 7. Render a segmentation-overlay test video with the trained model
python3 DINOv3-Segmentation-Training/Testing_Segmentation.py
```

Training has so far been run on Google Colab. Dataset images/masks are 640×480; the model trains at 224×224.
