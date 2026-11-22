#!/usr/bin/env python3
"""Copy the training dataset from the Raspberry Pi to this PC.

Runs ON THE PC. Uses the "pi" host alias from ~/.ssh/config.

Copies <remote project>/dataset/ from the Pi into ./dataset/ here,
keeping the dataset/<label>/ folder layout. Files that already exist
locally are overwritten with the Pi's copy; nothing is deleted on
either side.

Usage:
    python fetch_dataset.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_HOST = "pi"
# Relative to the Pi user's home directory (scp resolves relative remote paths from there).
DEFAULT_REMOTE_DATASET = "picamera_test/dataset"

# Never prompt for a password; give up quickly if the Pi is unreachable.
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy the dataset folder from the Pi to this PC.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"SSH host or alias (default: {DEFAULT_HOST})")
    parser.add_argument(
        "--remote-dataset",
        default=DEFAULT_REMOTE_DATASET,
        help=f"Dataset folder on the Pi, relative to home (default: {DEFAULT_REMOTE_DATASET})",
    )
    return parser.parse_args()


def count_images(root: Path) -> dict[str, int]:
    """Return {label: number of .jpg files} for each label folder under root."""
    if not root.is_dir():
        return {}
    return {d.name: len(list(d.glob("*.jpg"))) for d in sorted(root.iterdir()) if d.is_dir()}


def main() -> int:
    args = parse_args()
    local_dataset = PROJECT_DIR / "dataset"
    local_dataset.mkdir(exist_ok=True)

    print(f"Copying {args.host}:{args.remote_dataset}/ -> {local_dataset}")
    # Copy the folder's *contents* ("<dir>/.") so they merge into ./dataset/
    # instead of creating ./dataset/dataset/.  -r = recursive, -q = no progress bars.
    result = subprocess.run(
        ["scp", "-r", "-q", *SSH_OPTS, f"{args.host}:{args.remote_dataset}/.", str(local_dataset)]
    )
    if result.returncode != 0:
        raise SystemExit(f"scp failed (exit {result.returncode}). Has collect_data.py been run on the Pi yet?")

    counts = count_images(local_dataset)
    print("Local dataset:")
    for label, n in counts.items():
        print(f"  {label}: {n} image(s)")
    print(f"  total: {sum(counts.values())} image(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
