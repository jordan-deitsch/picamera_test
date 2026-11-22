#!/usr/bin/env python3
"""Collect training images for Edge Impulse with Picamera2.

Runs ON THE RASPBERRY PI.

Images are saved to dataset/<label>/<label>.<session>_<n>.jpg. The
"<label>." filename prefix is the convention Edge Impulse uses to infer
labels, and upload_dataset.py on the PC also uses the folder name.

Two modes:
  Interactive (default): press Enter to take a picture, type q + Enter to quit.
      Lets you reposition the object between shots. Over SSH, use -t:
          ssh -t pi "cd ~/picamera_test && python3 collect_data.py --label cup"
  Burst: take --count images, one every --interval seconds.
          python3 collect_data.py --label cup --count 50 --interval 0.5

Tip: capture at the same resolution, framing, and lighting the robot
will see at run time, and include varied backgrounds, angles, and
distances. For object detection, also capture some images with no
target objects in them.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from picamera2 import Picamera2

# libcamera control enums (autofocus modes); ship with Picamera2 on Pi OS.
try:
    from libcamera import controls
except ImportError:
    controls = None

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "dataset"


def parse_size(text: str) -> tuple[int, int]:
    """argparse type for "WIDTHxHEIGHT" strings, e.g. "640x480" -> (640, 480)."""
    try:
        width, height = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("size must look like 640x480")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect labeled training images with Picamera2.")
    parser.add_argument("--label", required=True, help="Class label, e.g. 'cup' (used for folder and filenames)")
    parser.add_argument(
        "--size",
        type=parse_size,
        default=(640, 480),
        help="Image size WIDTHxHEIGHT (default: 640x480). Keep it the same for inference later.",
    )
    parser.add_argument("--count", type=int, help="Burst mode: number of images to take")
    parser.add_argument("--interval", type=float, default=1.0, help="Burst mode: seconds between images (default: 1.0)")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Root folder for the dataset (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality 1-100 (default: 95)")
    args = parser.parse_args()

    # The label becomes part of folder and file names, so allow only safe
    # characters (this also blocks things like "../" path tricks).
    if not args.label.replace("_", "").replace("-", "").isalnum():
        parser.error("label may only contain letters, numbers, '-' and '_'")
    if args.count is not None and args.count < 1:
        parser.error("--count must be at least 1")
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")
    return args


def start_camera(size: tuple[int, int], quality: int) -> Picamera2:
    """Open, configure, and start the camera. Caller must stop()/close() it."""
    try:
        picam2 = Picamera2()
    except (RuntimeError, IndexError) as exc:
        raise SystemExit(f"Could not open the camera ({exc}). Is another camera script still running?")

    # A still configuration at the requested size. buffer_count=2 keeps memory
    # use low; we only ever need the latest frame.
    config = picam2.create_still_configuration(main={"size": size}, buffer_count=2)
    picam2.configure(config)
    picam2.options["quality"] = quality  # JPEG quality used by capture_file()
    picam2.start()

    # Camera Module 3: continuous autofocus keeps refocusing in the background
    # as you move the object between shots (no blocking focus cycle needed).
    if controls is not None and "AfMode" in picam2.camera_controls:
        picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

    time.sleep(1.0)  # let exposure, white balance and focus settle
    return picam2


def main() -> int:
    args = parse_args()
    label_dir: Path = (args.dataset_dir.expanduser() / args.label).resolve()
    label_dir.mkdir(parents=True, exist_ok=True)

    # One timestamp per run ("session"), plus a running number per image.
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = 0       # images saved in this run
    next_index = 1  # number used in the next file name

    def capture(picam2: Picamera2) -> None:
        """Save one image with a file name that is never reused."""
        nonlocal saved, next_index
        # Skip names that already exist, e.g. from another run started in the
        # same second, so earlier images are never overwritten.
        while (path := label_dir / f"{args.label}.{session}_{next_index:04d}.jpg").exists():
            next_index += 1
        picam2.capture_file(str(path))
        saved += 1
        next_index += 1
        print(f"[{saved}] {path.name}", flush=True)  # flush: show progress immediately over SSH

    picam2 = start_camera(args.size, args.quality)
    width, height = picam2.camera_configuration()["main"]["size"]
    print(f"Saving {width}x{height} images for label '{args.label}' to {label_dir}", flush=True)

    try:
        if args.count:
            # Burst mode: fixed number of images at a fixed interval.
            print(f"Burst mode: {args.count} images, {args.interval}s apart. Ctrl+C to stop early.", flush=True)
            for i in range(args.count):
                capture(picam2)
                if i < args.count - 1:
                    time.sleep(args.interval)
        else:
            # Interactive mode: each Enter press takes one image.
            print("Interactive mode: press Enter to capture, 'q' + Enter to quit.", flush=True)
            while True:
                try:
                    command = input()
                except EOFError:
                    # stdin closed, e.g. ssh without -t: nothing can be typed.
                    print("No interactive input (use 'ssh -t' or --count). Stopping.", file=sys.stderr)
                    break
                if command.strip().lower() == "q":
                    break
                capture(picam2)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        # Always release the camera so other scripts can use it.
        picam2.stop()
        picam2.close()

    total = len(list(label_dir.glob("*.jpg")))
    print(f"Saved {saved} new image(s). '{args.label}' now has {total} image(s) in {label_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
