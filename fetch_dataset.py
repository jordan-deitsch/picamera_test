#!/usr/bin/env python3
"""Copy the training dataset from the Raspberry Pi to this PC.

Runs ON THE PC. Uses the "pi" host alias from ~/.ssh/config.

Copies <remote project>/dataset/ from the Pi into ./dataset/ here,
keeping the dataset/<label>/ folder layout. Files that already exist
locally are overwritten with the Pi's copy; nothing is deleted on
either side unless you pass --delete-remote.

With --delete-remote, images are removed from the Pi after the copy, but
only those verified on the PC (same relative path and same file size).
Anything that failed to copy, or was captured during the fetch, stays on
the Pi. Label folders are kept (the web page suggests them as labels).
After this the PC holds the only copy until you upload to Edge Impulse.

Usage:
    python fetch_dataset.py                  # copy only
    python fetch_dataset.py --delete-remote  # copy, verify, then free space on the Pi
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_HOST = "pi"
# Relative to the Pi user's home directory (scp and ssh both start there).
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
    parser.add_argument(
        "--delete-remote",
        action="store_true",
        help="After copying, delete images from the Pi that are verified on this PC",
    )
    return parser.parse_args()


def count_images(root: Path) -> dict[str, int]:
    """Return {label: number of .jpg files} for each label folder under root.

    Folders starting with "_" (like _uploaded/, used by upload_dataset.py) are not labels.
    """
    if not root.is_dir():
        return {}
    return {
        d.name: len(list(d.glob("*.jpg")))
        for d in sorted(root.iterdir())
        if d.is_dir() and not d.name.startswith("_")
    }


def ssh(host: str, command: str, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    """Run a shell command on the Pi and capture its output."""
    return subprocess.run(["ssh", *SSH_OPTS, host, command], input=stdin, capture_output=True)


def remote_images(host: str, remote_dataset: str) -> dict[str, int]:
    """Return {"<label>/<file>.jpg": size in bytes} for every image on the Pi."""
    # find prints "<size> <path relative to the dataset folder>" per image.
    result = ssh(host, f"cd {shlex.quote(remote_dataset)} && find . -type f -name '*.jpg' -printf '%s %P\\n'")
    if result.returncode != 0:
        raise SystemExit(f"Could not list images on the Pi: {result.stderr.decode(errors='replace').strip()}")
    images = {}
    for line in result.stdout.decode().splitlines():
        size, _, rel = line.partition(" ")
        if rel:
            images[rel] = int(size)
    return images


def delete_verified(host: str, remote_dataset: str, local_dataset: Path) -> int:
    """Delete images on the Pi whose copy on this PC has the same size. Returns the count deleted."""
    remote = remote_images(host, remote_dataset)
    verified, missing = [], []
    for rel, size in remote.items():
        local = local_dataset / rel
        if local.is_file() and local.stat().st_size == size:
            verified.append(rel)
        else:
            missing.append(rel)

    if missing:
        print(f"Keeping {len(missing)} image(s) on the Pi that are not verified on this PC:")
        for rel in missing[:10]:
            print(f"  {rel}")
    if not verified:
        print("Nothing to delete on the Pi.")
        return 0

    # Send the file list on stdin, NUL-separated, so names never pass through
    # the remote shell. "--" stops rm from reading a name as an option.
    names = b"\0".join(rel.encode() for rel in verified) + b"\0"
    result = ssh(host, f"cd {shlex.quote(remote_dataset)} && xargs -0 rm -f --", stdin=names)
    if result.returncode != 0:
        raise SystemExit(f"Deleting on the Pi failed: {result.stderr.decode(errors='replace').strip()}")
    print(f"Deleted {len(verified)} verified image(s) from the Pi.")
    return len(verified)


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
        # Never delete anything after a failed copy.
        raise SystemExit(f"scp failed (exit {result.returncode}). Has collect_data.py been run on the Pi yet?")

    # Fetched images land in dataset/<label>/, i.e. "pending" for upload_dataset.py,
    # even if an identical copy was uploaded before (Edge Impulse skips those).
    pending = count_images(local_dataset)
    uploaded = count_images(local_dataset / "_uploaded")
    print("Local dataset:")
    for label in sorted(set(pending) | set(uploaded)):
        print(f"  {label}: {pending.get(label, 0)} pending upload, {uploaded.get(label, 0)} already uploaded")
    print(f"  total: {sum(pending.values())} pending, {sum(uploaded.values())} already uploaded")

    if args.delete_remote:
        delete_verified(args.host, args.remote_dataset, local_dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
