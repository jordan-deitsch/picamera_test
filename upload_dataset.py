#!/usr/bin/env python3
"""Upload the local dataset to an Edge Impulse project via the ingestion API.

Runs ON THE PC (or the Pi). Needs the `requests` package.

Reads images from dataset/<label>/*.jpg and uploads them with the folder
name as the label. For object detection projects, you then draw bounding
boxes in Edge Impulse Studio (Data acquisition -> Labeling queue).

The API key is read from the EI_API_KEY environment variable so it never
ends up in the code, git, or your shell history. Get it from your project's
Dashboard -> Keys page in Edge Impulse Studio, then in PowerShell:
    $env:EI_API_KEY = "ei_..."        # current terminal only

Usage:
    python upload_dataset.py --dry-run             # list what would be uploaded
    python upload_dataset.py                       # upload; Edge Impulse splits train/test
    python upload_dataset.py --category training   # force all into training
    python upload_dataset.py --labels cup ball     # only some labels
"""

import argparse
import os
import sys
from pathlib import Path

import requests

PROJECT_DIR = Path(__file__).resolve().parent
INGESTION_URL = "https://ingestion.edgeimpulse.com/api/{category}/files"
BATCH_SIZE = 20  # files per request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload dataset/<label>/*.jpg to Edge Impulse.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_DIR / "dataset",
        help="Root dataset folder (default: ./dataset)",
    )
    parser.add_argument(
        "--category",
        choices=["split", "training", "testing"],
        default="split",
        help="'split' lets Edge Impulse divide ~80/20 into training/testing (default)",
    )
    parser.add_argument("--labels", nargs="+", help="Only upload these labels (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded, upload nothing")
    return parser.parse_args()


def find_images(dataset_dir: Path, labels: list[str] | None) -> dict[str, list[Path]]:
    if not dataset_dir.is_dir():
        raise SystemExit(f"No dataset folder at {dataset_dir}. Run fetch_dataset.py first.")
    found = {}
    for label_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        if labels and label_dir.name not in labels:
            continue
        images = sorted(label_dir.glob("*.jpg"))
        if images:
            found[label_dir.name] = images
    return found


def upload_batch(api_key: str, category: str, label: str, paths: list[Path]) -> tuple[int, list[str]]:
    """Upload one batch; return (number accepted, list of error messages)."""
    handles = [open(p, "rb") for p in paths]
    try:
        files = [("data", (p.name, h, "image/jpeg")) for p, h in zip(paths, handles)]
        response = requests.post(
            INGESTION_URL.format(category=category),
            headers={
                "x-api-key": api_key,
                "x-label": label,
                # Skip images Edge Impulse already has, so re-running is safe.
                "x-disallow-duplicates": "1",
            },
            files=files,
            timeout=120,
        )
    finally:
        for h in handles:
            h.close()

    if response.status_code != 200:
        return 0, [f"HTTP {response.status_code}: {response.text[:300]}"]

    body = response.json()
    errors = []
    accepted = 0
    for item in body.get("files", []):
        if item.get("success"):
            accepted += 1
        else:
            errors.append(f"{item.get('fileName', '?')}: {item.get('error', 'unknown error')}")
    if not body.get("files") and not body.get("success", True):
        errors.append(body.get("error", "upload failed"))
    return accepted, errors


def main() -> int:
    args = parse_args()
    images = find_images(args.dataset_dir.resolve(), args.labels)
    if not images:
        raise SystemExit("No images found to upload.")

    total = sum(len(v) for v in images.values())
    print(f"Found {total} image(s) in {len(images)} label(s); category: {args.category}")
    for label, paths in images.items():
        print(f"  {label}: {len(paths)}")

    if args.dry_run:
        print("Dry run: nothing uploaded.")
        return 0

    api_key = os.environ.get("EI_API_KEY")
    if not api_key:
        raise SystemExit("Set the EI_API_KEY environment variable to your Edge Impulse project API key first.")

    uploaded = 0
    all_errors = []
    for label, paths in images.items():
        for start in range(0, len(paths), BATCH_SIZE):
            batch = paths[start : start + BATCH_SIZE]
            accepted, errors = upload_batch(api_key, args.category, label, batch)
            uploaded += accepted
            all_errors.extend(errors)
            print(f"  {label}: {min(start + BATCH_SIZE, len(paths))}/{len(paths)} sent, {accepted} accepted")

    print(f"Uploaded {uploaded}/{total} image(s).")
    if all_errors:
        print(f"{len(all_errors)} not uploaded (duplicates are expected on re-runs):")
        for err in all_errors[:20]:
            print(f"  {err}")
    # Fail if a whole request was rejected (e.g. bad API key); per-file duplicates are fine.
    return 1 if any(err.startswith("HTTP ") for err in all_errors) else 0


if __name__ == "__main__":
    sys.exit(main())
