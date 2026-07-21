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
    
    # only the Brio:
    python3 bag2video.py front_compare.bag ~/ag_bot/src/videos /brio_front/image_raw/compressed

    # only the original camera:
    python3 bag2video.py front_compare.bag ~/ag_bot/src/videos /usb_cam/image_raw/compressed

    # no topic → one mp4 per camera, e.g. front_compare_usb_cam.mp4 and front_compare_brio_front.mp4
    python3 bag2video.py ~/ag_bot/src/ros_bags/2026-07-14-10-50.bag ~/ag_bot/src/videos
-----------------------------------------------------------------------------
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

FALLBACK_FPS = 8.0  # used only if a bag's timestamps can't give an fps


def image_topics(reader, want_topic=None):
    """Return {topic: (connections, is_compressed)} for image topics in the bag.
    If want_topic is given, only that topic (error if absent)."""
    conns_by_topic = {}
    for c in reader.connections:
        if "image" not in c.topic.lower():
            continue
        conns_by_topic.setdefault(c.topic, []).append(c)
    if want_topic is not None:
        if want_topic not in conns_by_topic:
            print(f"  topic '{want_topic}' not in bag. Image topics present: "
                  f"{sorted(conns_by_topic)}")
            return {}
        conns_by_topic = {want_topic: conns_by_topic[want_topic]}
    return {
        topic: (conns, "Compressed" in conns[0].msgtype or "compressed" in topic.lower())
        for topic, conns in conns_by_topic.items()
    }


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


def convert_one(bag_path: Path, out_dir: Path, want_topic=None):
    if bag_path.name.endswith(".active"):
        print(f"  skipping {bag_path.name} (unfinished .active recording)")
        return
    try:
        with AnyReader([bag_path]) as reader:
            topics = image_topics(reader, want_topic)
            if not topics:
                print(f"  no image topic in {bag_path.name}")
                return
            for topic, (conns, is_compressed) in topics.items():
                kind = "compressed" if is_compressed else "raw"
                print(f"  {bag_path.name}: '{topic}' ({kind})")

                # pass 1: collect timestamps (ns) for fps, decode first frame for size
                timestamps, width, height = [], None, None
                for conn, t, raw in reader.messages(connections=conns):
                    timestamps.append(t)
                    if width is None:
                        frame = decode(reader.deserialize(raw, conn.msgtype), is_compressed)
                        if frame is None:
                            print("  could not decode first frame; skipping")
                            break
                        height, width = frame.shape[:2]

                if len(timestamps) < 2 or width is None:
                    print("  fewer than 2 frames; skipping")
                    continue
                dur = (timestamps[-1] - timestamps[0]) / 1e9
                fps = (len(timestamps) - 1) / dur if dur > 0 else FALLBACK_FPS

                # pass 2: encode the video
                out_dir.mkdir(parents=True, exist_ok=True)
                suffix = topic.strip("/").replace("/image_raw", "").replace("/compressed", "").replace("/", "_")
                out_path = out_dir / f"{bag_path.stem}_{suffix}.mp4"
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
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        sys.exit(1)
    inp = Path(sys.argv[1]).expanduser()
    out_dir = Path(sys.argv[2]).expanduser()
    topic = sys.argv[3] if len(sys.argv) == 4 else None
    bags = sorted(inp.glob("*.bag")) if inp.is_dir() else [inp]
    if not bags:
        print(f"No .bag files found in {inp}")
        sys.exit(1)
    print(f"Converting {len(bags)} bag(s) -> {out_dir}")
    for b in bags:
        convert_one(b, out_dir, topic)
    print("Done.")


if __name__ == "__main__":
    main()