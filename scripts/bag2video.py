#!/usr/bin/env python3
"""
Convert ROS1 .bag camera recordings to .mp4 on a ROS2 / Ubuntu 22.04 (WSL) box,
with NO ROS1 install required. Reads bags directly with the pure-Python `rosbags`
library and auto-detects whether the camera topic is compressed or raw.

--- One-time setup ----------------------------------------------------------
    python3 -m venv ~/ag_bot/.venv
    source ~/ag_bot/.venv/bin/activate
    pip install rosbags opencv-python-headless numpy

--- Usage -------------------------------------------------------------------
    source ~/ag_bot/.venv/bin/activate

    # convert ONE bag:
    python3 bag2video.py ~/ag_bot/src/ros_bags/2026-06-10-16-46-02.bag ~/ag_bot/src/videos

    # convert EVERY .bag in a folder (the .active file is skipped automatically):
    python3 bag2video.py ~/ag_bot/ros_bags ~/ag_bot/videos
-----------------------------------------------------------------------------
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

FALLBACK_FPS = 8.0  # used only if a bag's timestamps can't give an fps


def pick_image_connection(reader):
    """Find an image topic in the bag; prefer a compressed one.
    Returns (connections_for_that_topic, is_compressed) or (None, None)."""
    image_conns = [c for c in reader.connections if "image" in c.topic.lower()]
    if not image_conns:
        return None, None
    compressed = [c for c in image_conns
                  if "Compressed" in c.msgtype or "compressed" in c.topic.lower()]
    chosen = compressed[0] if compressed else image_conns[0]
    conns = [c for c in reader.connections if c.topic == chosen.topic]
    return conns, bool(compressed)


def decode(msg, is_compressed):
    """ROS image message -> BGR numpy image (OpenCV)."""
    if is_compressed:
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    # raw sensor_msgs/Image
    img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, -1)
    enc = (getattr(msg, "encoding", "") or "").lower()
    if enc == "rgb8":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif enc in ("mono8", "8uc1"):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def convert_one(bag_path: Path, out_dir: Path):
    if bag_path.name.endswith(".active"):
        print(f"  skipping {bag_path.name} (unfinished .active recording)")
        return
    try:
        with AnyReader([bag_path]) as reader:
            conns, is_compressed = pick_image_connection(reader)
            if not conns:
                topics = sorted({c.topic for c in reader.connections})
                print(f"  no image topic in {bag_path.name}. Topics present: {topics}")
                return

            kind = "compressed" if is_compressed else "raw"
            print(f"  {bag_path.name}: using '{conns[0].topic}' ({kind})")

            # pass 1: collect timestamps (ns) for fps, decode first frame for size
            timestamps, width, height = [], None, None
            for conn, t, raw in reader.messages(connections=conns):
                timestamps.append(t)
                if width is None:
                    frame = decode(reader.deserialize(raw, conn.msgtype), is_compressed)
                    if frame is None:
                        print("  could not decode first frame; skipping")
                        return
                    height, width = frame.shape[:2]

            if len(timestamps) < 2:
                print("  fewer than 2 frames; skipping")
                return
            dur = (timestamps[-1] - timestamps[0]) / 1e9
            fps = (len(timestamps) - 1) / dur if dur > 0 else FALLBACK_FPS

            # pass 2: encode the video
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (bag_path.stem + ".mp4")
            writer = cv2.VideoWriter(str(out_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (width, height))
            try:
                for conn, t, raw in reader.messages(connections=conns):
                    frame = decode(reader.deserialize(raw, conn.msgtype), is_compressed)
                    if frame is not None:
                        writer.write(frame)
            finally:
                writer.release()
            print(f"  -> {out_path}  ({len(timestamps)} frames, {fps:.2f} fps, {width}x{height})")
    except Exception as e:
        print(f"  ERROR on {bag_path.name}: {e}")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    inp = Path(sys.argv[1]).expanduser()
    out_dir = Path(sys.argv[2]).expanduser()
    bags = sorted(inp.glob("*.bag")) if inp.is_dir() else [inp]
    if not bags:
        print(f"No .bag files found in {inp}")
        sys.exit(1)
    print(f"Converting {len(bags)} bag(s) -> {out_dir}")
    for b in bags:
        convert_one(b, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()