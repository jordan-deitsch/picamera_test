#!/usr/bin/env python3
"""Trigger a still capture on the Raspberry Pi and copy the image to this PC.

Runs ON THE PC. Uses the "pi" host alias from ~/.ssh/config.

Steps:
  1. (optional) `git pull` on the Pi so it runs the latest committed code
  2. Run capture_still.py on the Pi over SSH
  3. scp the saved image back to ./captures/ in this project

Usage:
    python remote_capture.py           # capture + copy
    python remote_capture.py --pull    # git pull on the Pi first
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_LOCAL_DIR = PROJECT_DIR / "captures"
DEFAULT_HOST = "pi"
DEFAULT_REMOTE_PROJECT = "~/picamera_test"

# Never prompt for a password; fail fast if key auth isn't working.
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a still on the Pi and copy it to this PC.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"SSH host or alias (default: {DEFAULT_HOST})")
    parser.add_argument(
        "--remote-project",
        default=DEFAULT_REMOTE_PROJECT,
        help=f"Project folder on the Pi (default: {DEFAULT_REMOTE_PROJECT})",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR,
        help=f"Where to save the image on this PC (default: {DEFAULT_LOCAL_DIR})",
    )
    parser.add_argument("--pull", action="store_true", help="Run `git pull` on the Pi before capturing")
    return parser.parse_args()


def run_remote(host: str, command: str) -> str:
    """Run a shell command on the Pi, stream its output, and return stdout."""
    result = subprocess.run(
        ["ssh", *SSH_OPTS, host, command],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, end="", file=sys.stderr)
        raise SystemExit(f"Remote command failed (exit {result.returncode}): {command}")
    return result.stdout


def main() -> int:
    args = parse_args()
    # The remote project path may start with "~", which the remote shell must expand,
    # so it is passed unquoted; it comes from this script's own arguments, not from the Pi.
    remote_dir = args.remote_project

    if args.pull:
        print(f"[1/3] git pull on {args.host}")
        run_remote(args.host, f"cd {remote_dir} && git pull --ff-only")

    print(f"[2/3] Capturing still on {args.host}")
    output = run_remote(args.host, f"cd {remote_dir} && python3 capture_still.py")

    saved_lines = [line for line in output.splitlines() if line.startswith("SAVED: ")]
    if not saved_lines:
        raise SystemExit("Could not find 'SAVED: <path>' in the capture script output.")
    remote_path = saved_lines[-1].removeprefix("SAVED: ").strip()

    local_dir: Path = args.local_dir.resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / Path(remote_path).name

    print(f"[3/3] Copying {remote_path} -> {local_path}")
    # scp's remote side goes through a shell, so quote the path for it.
    result = subprocess.run(["scp", *SSH_OPTS, f"{args.host}:{shlex.quote(remote_path)}", str(local_path)])
    if result.returncode != 0:
        raise SystemExit(f"scp failed (exit {result.returncode})")

    print(f"Done. Image saved to {local_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
