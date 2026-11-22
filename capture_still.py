#!/usr/bin/env python3
"""Capture a single still image with Picamera2 and save it on the Pi.

Runs ON THE RASPBERRY PI.

Usage:
    python3 capture_still.py                    # saves to ./captures/still_<timestamp>.jpg
    python3 capture_still.py --output-dir /tmp  # custom folder
    python3 capture_still.py --no-autofocus     # skip the autofocus cycle

The last line printed is "SAVED: <absolute path>". remote_capture.py (on the
PC) looks for that line to find out which file to copy back.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from picamera2 import Picamera2

# libcamera's control enums (e.g. autofocus modes). They ship with Picamera2 on
# Raspberry Pi OS; the fallback only matters if the script is imported elsewhere.
try:
    from libcamera import controls
except ImportError:
    controls = None

# Default output folder: captures/ next to this script (ignored by git).
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "captures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a still image with Picamera2.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder to save the image in (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-autofocus",
        action="store_true",
        help="Skip the autofocus cycle (for cameras without AF or fixed-focus setups)",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=1.0,
        help="Seconds to let auto-exposure/white-balance settle (default: 1.0)",
    )
    return parser.parse_args()


def open_camera() -> Picamera2:
    """Open the camera, turning the common 'camera busy' failure into a clear message."""
    try:
        return Picamera2()
    except (RuntimeError, IndexError) as exc:
        # RuntimeError: another process (e.g. live_detect.py) holds the camera.
        # IndexError: no camera detected at all.
        raise SystemExit(f"Could not open the camera ({exc}). Is another camera script still running?")


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped file name so repeated captures never overwrite each other.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"still_{timestamp}.jpg"

    picam2 = open_camera()
    try:
        # A "still" configuration defaults to the sensor's full resolution
        # (4608x2592 on the Camera Module 3).
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()

        # Auto-exposure and auto-white-balance need a few frames to converge.
        time.sleep(args.warmup)

        # The Camera Module 3 has autofocus: run one blocking focus cycle
        # (autofocus_cycle() returns True if focus locked).
        has_af = "AfMode" in picam2.camera_controls
        if has_af and not args.no_autofocus and controls is not None:
            picam2.set_controls({"AfMode": controls.AfModeEnum.Auto})
            focused = picam2.autofocus_cycle()
            print(f"Autofocus {'succeeded' if focused else 'did not lock (capturing anyway)'}")

        # Encodes the next frame as JPEG (format chosen from the file extension).
        picam2.capture_file(str(output_path))
    finally:
        # Always release the camera, even on errors, so the next script can use it.
        picam2.stop()
        picam2.close()

    size_kb = output_path.stat().st_size / 1024
    width, height = config["main"]["size"]
    print(f"Captured {width}x{height} image ({size_kb:.0f} KB)")
    print(f"SAVED: {output_path}")  # machine-readable line for remote_capture.py
    return 0


if __name__ == "__main__":
    sys.exit(main())
