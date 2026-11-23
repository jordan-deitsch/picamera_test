#!/usr/bin/env python3
"""Upload the local dataset to an Edge Impulse project via the ingestion API.

Runs ON THE PC (or the Pi). Needs the `requests` package.

Folder layout (inside dataset/):
    <label>/*.jpg             pending: not uploaded yet (fetch_dataset.py puts new images here)
    _uploaded/<label>/*.jpg   already in Edge Impulse

By default only pending images are uploaded. After each batch, images that
Edge Impulse accepted, or rejected because it already has them, are moved to
_uploaded/<label>/. Images that failed for any other reason (network, bad key,
unrecognized error) stay pending and are tried again next time.

--all also re-sends everything in _uploaded/ (Edge Impulse skips the ones it
already has). Useful after deleting data in Studio or switching projects.

For object detection projects, you then draw bounding boxes in Edge Impulse
Studio (Data acquisition -> Labeling queue).

The API key never goes in the code or git. Get it from your project's
Dashboard -> Keys page in Edge Impulse Studio, then either:
  - save it once in .secrets/edge_impulse.env (git-ignored) as a line
        EI_API_KEY=ei_...
  - or set it for the current PowerShell terminal only (this wins over the file):
        $env:EI_API_KEY = "ei_..."

Usage:
    python upload_dataset.py --dry-run             # list what would be uploaded
    python upload_dataset.py                       # upload pending images; Edge Impulse splits train/test
    python upload_dataset.py --all                 # re-send everything, including _uploaded/
    python upload_dataset.py --category training   # force all into training
    python upload_dataset.py --labels cup ball     # only some labels
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import requests

PROJECT_DIR = Path(__file__).resolve().parent
# Edge Impulse ingestion endpoint; {category} is "training", "testing" or "split".
INGESTION_URL = "https://ingestion.edgeimpulse.com/api/{category}/files"
BATCH_SIZE = 20  # files per HTTP request (keeps each request a reasonable size)
# Subfolder of dataset/ for images already in Edge Impulse. The leading "_"
# keeps it from being mistaken for a label (labels must start with a letter or digit).
UPLOADED_DIR = "_uploaded"
# Words that identify a per-file "Edge Impulse already has this" rejection.
DUPLICATE_HINTS = ("already exist", "duplicate", "same hash")
# Git-ignored file holding KEY=VALUE lines, e.g. EI_API_KEY=ei_...
SECRETS_FILE = PROJECT_DIR / ".secrets" / "edge_impulse.env"


@dataclass
class Image:
    path: Path
    pending: bool  # True: in dataset/<label>/; False: already in _uploaded/<label>/


@dataclass
class BatchResult:
    accepted: list[str] = field(default_factory=list)    # file names newly stored by Edge Impulse
    duplicates: list[str] = field(default_factory=list)  # file names Edge Impulse already had
    errors: list[str] = field(default_factory=list)      # messages for everything else
    request_failed: bool = False                         # whole request failed (network, HTTP error)


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
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Also re-send images in {UPLOADED_DIR}/ (default: only pending images)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded, upload or move nothing")
    return parser.parse_args()


def label_dirs(root: Path) -> dict[str, Path]:
    """{label: folder} for each label folder directly under root (skips _uploaded etc.)."""
    if not root.is_dir():
        return {}
    return {d.name: d for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith("_")}


def find_images(dataset_dir: Path, labels: list[str] | None, include_uploaded: bool) -> dict[str, list[Image]]:
    """Return {label: [images to send]}; pending images first, then (with --all) uploaded ones."""
    if not dataset_dir.is_dir():
        raise SystemExit(f"No dataset folder at {dataset_dir}. Run fetch_dataset.py first.")
    pending = label_dirs(dataset_dir)
    uploaded = label_dirs(dataset_dir / UPLOADED_DIR) if include_uploaded else {}

    found: dict[str, list[Image]] = {}
    for label in sorted(set(pending) | set(uploaded)):
        if labels and label not in labels:
            continue
        images = [Image(p, True) for p in sorted(pending[label].glob("*.jpg"))] if label in pending else []
        if label in uploaded:
            # Skip uploaded copies of names that are pending again (e.g. fetched
            # from the Pi a second time); the pending copy is sent instead.
            names = {img.path.name for img in images}
            images += [Image(p, False) for p in sorted(uploaded[label].glob("*.jpg")) if p.name not in names]
        if images:
            found[label] = images
    return found


def count_uploaded(dataset_dir: Path, labels: list[str] | None) -> dict[str, int]:
    """{label: number of images in _uploaded/<label>/}."""
    return {
        label: len(list(d.glob("*.jpg")))
        for label, d in label_dirs(dataset_dir / UPLOADED_DIR).items()
        if not labels or label in labels
    }


def read_secrets(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE lines; blank lines and # comments are ignored."""
    values = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():  # -sig: tolerate a Notepad BOM
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip("'\"")  # allow optional quotes
    return values


def load_api_key() -> str:
    """API key from $env:EI_API_KEY, else from .secrets/edge_impulse.env. Never printed."""
    key = os.environ.get("EI_API_KEY") or read_secrets(SECRETS_FILE).get("EI_API_KEY", "")
    if not key or key == "ei_...":  # empty, or the placeholder left unchanged
        raise SystemExit(
            f"No Edge Impulse API key found. Put a line EI_API_KEY=ei_... in {SECRETS_FILE}\n"
            "(Dashboard -> Keys in Edge Impulse Studio), or set $env:EI_API_KEY for this terminal."
        )
    return key


def is_duplicate(message: str) -> bool:
    return any(hint in message.lower() for hint in DUPLICATE_HINTS)


def upload_batch(api_key: str, category: str, label: str, paths: list[Path]) -> BatchResult:
    """Upload one batch of images (multipart form, one "data" field per file)."""
    result = BatchResult()
    handles = [open(p, "rb") for p in paths]
    try:
        files = [("data", (p.name, h, "image/jpeg")) for p, h in zip(paths, handles)]
        response = requests.post(
            INGESTION_URL.format(category=category),
            headers={
                "x-api-key": api_key,
                "x-label": label,
                # Skip images Edge Impulse already has, so re-sending is safe.
                "x-disallow-duplicates": "1",
            },
            files=files,
            timeout=120,
        )
    except requests.RequestException as exc:
        # Network problem (no internet, DNS, timeout): report it and let the caller continue.
        result.request_failed = True
        result.errors.append(f"Request failed: {exc}")
        return result
    finally:
        for h in handles:
            h.close()

    if response.status_code != 200:
        result.request_failed = True
        result.errors.append(f"HTTP {response.status_code}: {response.text[:300]}")
        return result

    try:
        body = response.json()
    except ValueError:
        result.request_failed = True
        result.errors.append(f"HTTP 200 but response was not JSON: {response.text[:300]}")
        return result

    # The response lists each file with its own success flag.
    for item in body.get("files", []):
        name = item.get("fileName", "?")
        message = str(item.get("error", "unknown error"))
        if item.get("success"):
            result.accepted.append(name)
        elif is_duplicate(message):
            result.duplicates.append(name)
        else:
            result.errors.append(f"{name}: {message}")
    if not body.get("files") and not body.get("success", True):
        result.request_failed = True
        result.errors.append(str(body.get("error", "upload failed")))
    return result


def mark_uploaded(images: list[Image], names: set[str], dataset_dir: Path, label: str) -> int:
    """Move pending images whose names are in `names` to _uploaded/<label>/. Returns the count moved."""
    target_dir = dataset_dir / UPLOADED_DIR / label
    moved = 0
    for img in images:
        if img.pending and img.path.name in names:
            target_dir.mkdir(parents=True, exist_ok=True)
            # replace() overwrites an older copy with the same name (same image re-fetched).
            img.path.replace(target_dir / img.path.name)
            moved += 1
    return moved


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    images = find_images(dataset_dir, args.labels, args.all)
    already = count_uploaded(dataset_dir, args.labels)

    total = sum(len(v) for v in images.values())
    mode = "pending + already uploaded" if args.all else "pending only"
    print(f"Sending {total} image(s) ({mode}); category: {args.category}")
    for label in sorted(set(images) | set(already)):
        batch = images.get(label, [])
        n_pending = sum(img.pending for img in batch)
        print(f"  {label}: {n_pending} pending, {already.get(label, 0)} already uploaded")
    if not images:
        print("Nothing to upload." + ("" if args.all else " Use --all to re-send uploaded images."))
        return 0

    if args.dry_run:
        print("Dry run: nothing uploaded or moved.")
        return 0

    api_key = load_api_key()

    accepted = duplicates = moved = 0
    all_errors: list[str] = []
    request_failed = False
    for label, label_images in images.items():
        # Send each label's images in batches of BATCH_SIZE.
        for start in range(0, len(label_images), BATCH_SIZE):
            batch = label_images[start : start + BATCH_SIZE]
            result = upload_batch(api_key, args.category, label, [img.path for img in batch])
            accepted += len(result.accepted)
            duplicates += len(result.duplicates)
            all_errors.extend(result.errors)
            request_failed |= result.request_failed
            # Move after every batch, so an interrupted run keeps its progress.
            moved += mark_uploaded(batch, set(result.accepted) | set(result.duplicates), dataset_dir, label)
            print(
                f"  {label}: {min(start + BATCH_SIZE, len(label_images))}/{len(label_images)} sent, "
                f"{len(result.accepted)} new, {len(result.duplicates)} already in Edge Impulse"
            )

    print(f"Done: {accepted} new, {duplicates} already in Edge Impulse, {len(all_errors)} error(s).")
    print(f"Moved {moved} image(s) to {UPLOADED_DIR}/.")
    if all_errors:
        print("Errors (these images stay pending and are retried next time):")
        for err in all_errors[:20]:
            print(f"  {err}")
    # Exit code 1 if anything failed; duplicates count as success.
    return 1 if request_failed or all_errors else 0


if __name__ == "__main__":
    sys.exit(main())
