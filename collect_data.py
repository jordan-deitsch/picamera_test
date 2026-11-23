#!/usr/bin/env python3
"""Collect training images for Edge Impulse with Picamera2.

Runs ON THE RASPBERRY PI.

Images are saved to dataset/<label>/<label>.<session>_<n>.jpg. The
"<label>." filename prefix is the convention Edge Impulse uses to infer
labels, and upload_dataset.py on the PC also uses the folder name. If the
label folder already exists, new images are added to it (names never clash).

Three modes:
  Web (default when no --label is given): live video in the browser, plus a
      form for the label, image count and interval, a Start/Stop button and
      a live counter. Open http://<pi-address>:8000 after starting:
          ssh -t pi "cd ~/picamera_test && python3 collect_data.py"
  Interactive (--label only): press Enter to take a picture, q + Enter to quit.
          ssh -t pi "cd ~/picamera_test && python3 collect_data.py --label cup"
  Burst (--label and --count): take --count images, one every --interval seconds.
          python3 collect_data.py --label cup --count 50 --interval 0.5

All modes use the same camera setup as live_detect.py (full sensor field of
view, same output size), so training images are framed exactly like the
frames the model sees at run time.

Web mode architecture (two threads share one Collector object):

    camera thread (Collector.run)              Flask web server (main thread)
    -----------------------------              ------------------------------
    capture frame                              /            -> page with form
    -> if a run is active and an image is due:  /stream      -> MJPEG live video
         save full-quality JPEG, count++       /api/status  -> run state as JSON
    -> encode preview JPEG, notify viewers     /api/start   -> begin a run (POST)
                                               /api/stop    -> end a run early (POST)

Tip: capture at the same resolution, framing, and lighting the robot
will see at run time, and include varied backgrounds, angles, and
distances. For object detection, also capture some images with no
target objects in them.
"""

import _thread
import argparse
import logging
import re
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from picamera2 import Picamera2

# libcamera control enums (autofocus modes); ship with Picamera2 on Pi OS.
try:
    from libcamera import controls
except ImportError:
    controls = None

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "dataset"
# Labels become folder and file names: only letters, digits, '-' and '_'
# (this also blocks path tricks like "../").
LABEL_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_COUNT = 10000          # web mode: upper limit on images per run
PREVIEW_JPEG_QUALITY = 70  # live stream only; saved images use --quality


def parse_size(text: str) -> tuple[int, int]:
    """argparse type for "WIDTHxHEIGHT" strings, e.g. "640x480" -> (640, 480)."""
    try:
        width, height = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("size must look like 640x480")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect labeled training images with Picamera2.")
    parser.add_argument(
        "--label",
        help="Class label, e.g. 'cup' (folder and file prefix). Omit it to start the web page instead.",
    )
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
    parser.add_argument("--host", default="0.0.0.0", help="Web mode: address to listen on (default: all)")
    parser.add_argument("--port", type=int, default=8000, help="Web mode: port (default: 8000)")
    args = parser.parse_args()

    if args.label is not None and not LABEL_RE.match(args.label):
        parser.error("label may only contain letters, numbers, '-' and '_' (max 64 characters)")
    if args.count is not None:
        if args.label is None:
            parser.error("--count needs --label (or omit both for web mode)")
        if args.count < 1:
            parser.error("--count must be at least 1")
    if args.interval < 0:
        parser.error("--interval cannot be negative")
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")
    return args


# =========================================================================== camera and saving


def start_camera(size: tuple[int, int]) -> Picamera2:
    """Open, configure, and start the camera. Caller must stop()/close() it.

    Same configuration as live_detect.py, so training images and live frames
    have the same field of view and size.
    """
    try:
        picam2 = Picamera2()
    except (RuntimeError, IndexError) as exc:
        raise RuntimeError(f"Could not open the camera ({exc}). Is another camera script still running?") from exc

    # Smallest sensor mode that still covers the FULL sensor area (on the
    # Camera Module 3 the fastest mode is a zoomed-in center crop).
    modes = picam2.sensor_modes
    full_area = max(m["crop_limits"][2] * m["crop_limits"][3] for m in modes)
    full_fov = [m for m in modes if m["crop_limits"][2] * m["crop_limits"][3] == full_area]
    sensor_mode = min(full_fov, key=lambda m: m["size"][0] * m["size"][1])

    config = picam2.create_video_configuration(
        # "RGB888" gives BGR byte order, which is what OpenCV expects.
        main={"size": size, "format": "RGB888"},
        raw={"size": sensor_mode["size"], "format": sensor_mode["unpacked"]},
        buffer_count=4,
    )
    picam2.configure(config)
    picam2.start()
    # Continuous autofocus keeps refocusing as you move the object between shots.
    if controls is not None and "AfMode" in picam2.camera_controls:
        picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    time.sleep(1.0)  # let exposure, white balance and focus settle
    return picam2


class ImageWriter:
    """Saves frames for one label and one run ("session") with unique names."""

    def __init__(self, dataset_dir: Path, label: str, quality: int):
        self.label = label
        self.quality = quality
        self.dir = (dataset_dir.expanduser() / label).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)  # reuse the folder if it exists
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.next_index = 1

    def save(self, frame) -> Path:
        """Write one BGR frame as JPEG and return its path."""
        # Skip names that already exist, e.g. from another run started in the
        # same second, so earlier images are never overwritten.
        while (path := self.dir / f"{self.label}.{self.session}_{self.next_index:04d}.jpg").exists():
            self.next_index += 1
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality]):
            raise OSError(f"could not write {path}")
        self.next_index += 1
        return path

    def total(self) -> int:
        """Number of images in this label's folder (all runs)."""
        return len(list(self.dir.glob("*.jpg")))


# =========================================================================== command-line modes


def run_cli(args: argparse.Namespace) -> int:
    """Interactive (Enter per image) or burst (--count) collection in the terminal."""
    writer = ImageWriter(args.dataset_dir, args.label, args.quality)
    saved = 0

    def capture(picam2: Picamera2) -> None:
        nonlocal saved
        path = writer.save(picam2.capture_array("main"))
        saved += 1
        print(f"[{saved}] {path.name}", flush=True)  # flush: show progress immediately over SSH

    try:
        picam2 = start_camera(args.size)
    except RuntimeError as exc:
        raise SystemExit(str(exc))
    width, height = picam2.camera_configuration()["main"]["size"]
    print(f"Saving {width}x{height} images for label '{args.label}' to {writer.dir}", flush=True)

    try:
        if args.count:
            print(f"Burst mode: {args.count} images, {args.interval}s apart. Ctrl+C to stop early.", flush=True)
            for i in range(args.count):
                capture(picam2)
                if i < args.count - 1:
                    time.sleep(args.interval)
        else:
            print("Interactive mode: press Enter to capture, 'q' + Enter to quit.", flush=True)
            while True:
                try:
                    command = input()
                except EOFError:
                    # stdin closed, e.g. ssh without -t: nothing can be typed.
                    print("No interactive input (use 'ssh -t', --count, or web mode). Stopping.", file=sys.stderr)
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

    print(f"Saved {saved} new image(s). '{args.label}' now has {writer.total()} image(s) in {writer.dir}", flush=True)
    return 0


# =========================================================================== web mode: camera thread


class Collector(threading.Thread):
    """Camera thread for web mode: streams preview frames and runs collection runs.

    All shared state is guarded by self.cond (a lock plus "new frame" signal).
    Web handlers call start_run()/stop_run()/status(); only this thread touches
    the camera, so saves happen on exact frames with no camera contention.
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(daemon=True)
        self.args = args
        self.stop_event = threading.Event()  # set by main() to shut down
        self.ready = threading.Event()       # set once streaming starts (or failed)
        self.error: str | None = None
        self.cond = threading.Condition()
        self.jpeg: bytes | None = None       # latest preview frame
        self.frame_id = 0
        self.frame_size = args.size
        # Current/last run. "state" is idle | running | done | stopped | error.
        self.run_info = {"state": "idle", "label": None, "count": 0, "interval": 0.0,
                         "saved": 0, "last_file": None, "folder_total": None, "message": ""}
        self._writer: ImageWriter | None = None
        self._next_due = 0.0  # monotonic time of the next capture in a run

    # ---- called from web handler threads ---------------------------------

    def start_run(self, label: str, count: int, interval: float) -> None:
        """Begin a run; raises ValueError if one is already in progress."""
        with self.cond:
            if self.run_info["state"] == "running":
                raise ValueError("a run is already in progress")
            self._writer = ImageWriter(self.args.dataset_dir, label, self.args.quality)
            self._next_due = time.monotonic()  # first image right away
            self.run_info = {"state": "running", "label": label, "count": count, "interval": interval,
                             "saved": 0, "last_file": None, "folder_total": self._writer.total(),
                             "message": f"Saving to dataset/{label}/"}
        logging.info("Run started: label=%s count=%d interval=%.2fs", label, count, interval)

    def stop_run(self) -> None:
        with self.cond:
            if self.run_info["state"] == "running":
                self._finish("stopped", "Stopped early")

    def status(self) -> dict:
        with self.cond:
            return dict(self.run_info)

    def wait_for_frame(self, last_id: int, timeout: float = 2.0) -> tuple[int, bytes | None]:
        """Block until a frame newer than last_id exists (or timeout/stop/error)."""
        with self.cond:
            self.cond.wait_for(lambda: self.frame_id != last_id or self.stop_event.is_set() or self.error, timeout)
            return self.frame_id, self.jpeg

    # ---- camera thread ----------------------------------------------------

    def _finish(self, state: str, message: str) -> None:
        """End the current run (caller holds self.cond)."""
        self.run_info["state"] = state
        self.run_info["message"] = message
        if self._writer is not None:
            self.run_info["folder_total"] = self._writer.total()
        self._writer = None
        logging.info("Run %s: %d image(s) saved", state, self.run_info["saved"])

    def _maybe_save(self, frame) -> None:
        """Save this frame if a run is active and the next image is due."""
        with self.cond:
            if self.run_info["state"] != "running" or time.monotonic() < self._next_due:
                return
            info = self.run_info
            try:
                path = self._writer.save(frame)
            except OSError as exc:
                self._finish("error", str(exc))
                return
            info["saved"] += 1
            info["last_file"] = path.name
            info["folder_total"] += 1
            # Schedule from the previous due time (not "now") so the spacing
            # doesn't drift; if we fell behind, don't try to catch up in a burst.
            self._next_due = max(self._next_due + info["interval"], time.monotonic())
            if info["saved"] >= info["count"]:
                self._finish("done", "Run complete")

    def run(self) -> None:
        try:
            picam2 = start_camera(self.args.size)
        except Exception as exc:
            self.error = str(exc)
            self.ready.set()
            return
        try:
            self.frame_size = tuple(picam2.camera_configuration()["main"]["size"])
            self.ready.set()
            while not self.stop_event.is_set():
                frame = picam2.capture_array("main")  # waits for the next camera frame
                self._maybe_save(frame)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
                if ok:
                    with self.cond:
                        self.jpeg = buf.tobytes()
                        self.frame_id += 1
                        self.cond.notify_all()  # wake /stream handlers
        except Exception as exc:
            logging.exception("Camera loop crashed")
            with self.cond:
                self.error = str(exc)
                if self.run_info["state"] == "running":
                    self._finish("error", f"Camera error: {exc}")
                self.cond.notify_all()
            _thread.interrupt_main()  # stop the web server too (exit code 1)
        finally:
            picam2.stop()
            picam2.close()


# =========================================================================== web mode: page and server

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data Collection</title>
<style>
  :root { color-scheme: light dark; --bg:#f4f4f5; --fg:#18181b; --card:#fff; --muted:#71717a;
          --line:#d4d4d8; --accent:#2563eb; --ok:#16a34a; --warn:#d97706; --bad:#dc2626; }
  @media (prefers-color-scheme: dark) { :root { --bg:#18181b; --fg:#f4f4f5; --card:#27272a;
          --muted:#a1a1aa; --line:#3f3f46; --accent:#60a5fa; --ok:#4ade80; --warn:#fbbf24; --bad:#f87171; } }
  body { margin:0; font-family: system-ui, sans-serif; background:var(--bg); color:var(--fg); }
  main { max-width: 1000px; margin: 0 auto; padding: 16px; display:grid; gap:16px; grid-template-columns: 1fr; }
  @media (min-width: 920px) { main { grid-template-columns: 640px 1fr; } }
  h1 { font-size: 1.1rem; margin: 0 0 8px; }
  img { width: 100%; max-width: 640px; border-radius: 8px; background:#000; display:block;
        outline: 4px solid transparent; transition: outline-color .15s; }
  img.flash { outline-color: var(--ok); transition: none; }
  .card { background: var(--card); border-radius: 8px; padding: 16px; display:grid; gap:12px; align-content:start; }
  label { display:grid; gap:4px; font-size:.9rem; color:var(--muted); }
  input { font: inherit; padding: 8px 10px; border: 1px solid var(--line); border-radius: 6px;
          background: var(--bg); color: var(--fg); }
  input:disabled { opacity: .6; }
  .buttons { display:flex; gap:8px; }
  button { font: inherit; font-weight: 600; padding: 10px 16px; border: 0; border-radius: 6px; cursor: pointer; flex:1; }
  #start { background: var(--accent); color: #fff; }
  #stop { background: var(--line); color: var(--fg); }
  button:disabled { opacity: .45; cursor: default; }
  .counter { font-size: 2.4rem; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1; }
  .counter small { font-size: 1rem; color: var(--muted); font-weight: 400; }
  progress { width: 100%; height: 10px; }
  dl { display:grid; grid-template-columns: auto 1fr; gap: 4px 12px; margin:0; font-size:.9rem; }
  dt { color: var(--muted); } dd { margin:0; overflow-wrap:anywhere; }
  #msg.running { color: var(--accent); } #msg.done { color: var(--ok); }
  #msg.stopped { color: var(--warn); } #msg.error { color: var(--bad); }
</style>
</head>
<body>
<main>
  <div>
    <h1>Training data collection</h1>
    <img id="video" src="/stream" alt="Live camera stream">
  </div>
  <form class="card" id="form">
    <label>Label (folder and file prefix)
      <input id="label" required maxlength="64" pattern="[A-Za-z0-9_\\-]+" list="labels"
             placeholder="e.g. pen" autocomplete="off">
      <datalist id="labels"></datalist>
    </label>
    <label>Images per run
      <input id="count" type="number" min="1" max="__MAX_COUNT__" value="20" required>
    </label>
    <label>Seconds between images
      <input id="interval" type="number" min="0" step="0.1" value="1" required>
    </label>
    <div class="buttons">
      <button id="start" type="submit">Start</button>
      <button id="stop" type="button" disabled>Stop</button>
    </div>
    <div class="counter"><span id="saved">0</span> <small>/ <span id="total">0</span> images</small></div>
    <progress id="bar" value="0" max="1"></progress>
    <div id="msg">Ready. Letters, numbers, - and _ only.</div>
    <dl>
      <dt>Folder</dt><dd id="folder">–</dd>
      <dt>In folder</dt><dd id="infolder">–</dd>
      <dt>Last file</dt><dd id="last">–</dd>
    </dl>
  </form>
</main>
<script>
  const $ = id => document.getElementById(id);
  let lastSaved = 0;

  async function api(path, body) {
    const r = await fetch(path, { method: body ? "POST" : "GET",
      headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || r.statusText);
    return data;
  }

  function render(s) {
    const running = s.state === "running";
    for (const id of ["label", "count", "interval", "start"]) $(id).disabled = running;
    $("stop").disabled = !running;
    $("saved").textContent = s.saved;
    $("total").textContent = s.count;
    $("bar").max = s.count || 1;
    $("bar").value = s.saved;
    $("msg").textContent = s.message || "Ready. Letters, numbers, - and _ only.";
    $("msg").className = s.state;
    $("folder").textContent = s.label ? `dataset/${s.label}/` : "–";
    $("infolder").textContent = s.folder_total ?? "–";
    $("last").textContent = s.last_file ?? "–";
    if (running && s.saved > lastSaved) {           // flash the video border on each new image
      $("video").classList.add("flash");
      setTimeout(() => $("video").classList.remove("flash"), 150);
    }
    lastSaved = s.saved;
    $("labels").innerHTML = (s.labels || []).map(l => `<option value="${l}">`).join("");
  }

  async function poll() {
    try { render(await api("/api/status")); } catch (e) { /* server restarting; keep polling */ }
    setTimeout(poll, 250);
  }

  $("form").addEventListener("submit", async ev => {
    ev.preventDefault();
    try {
      lastSaved = 0;
      render(await api("/api/start", { label: $("label").value.trim(),
        count: Number($("count").value), interval: Number($("interval").value) }));
    } catch (e) { $("msg").textContent = e.message; $("msg").className = "error"; }
  });
  $("stop").addEventListener("click", async () => {
    try { render(await api("/api/stop", {})); } catch (e) { $("msg").textContent = e.message; }
  });
  poll();
</script>
</body>
</html>
""".replace("__MAX_COUNT__", str(MAX_COUNT))


def existing_labels(dataset_dir: Path) -> list[str]:
    """Label folders already in the dataset (offered as suggestions on the page)."""
    root = dataset_dir.expanduser()
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and LABEL_RE.match(d.name))


def create_app(collector: Collector):
    """Build the Flask app (imported here so CLI modes don't need Flask)."""
    from flask import Flask, Response, jsonify, request

    app = Flask(__name__)

    def status_json():
        return {**collector.status(), "labels": existing_labels(collector.args.dataset_dir)}

    @app.get("/")
    def index():
        return PAGE

    @app.get("/stream")
    def stream():
        # MJPEG: browsers show a stream of JPEG "parts" as live video in an <img>.
        def frames():
            last_id = -1
            while not collector.stop_event.is_set() and collector.error is None:
                last_id, jpeg = collector.wait_for_frame(last_id)
                if jpeg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/status")
    def status():
        return jsonify(status_json())

    @app.post("/api/start")
    def start():
        # Validate everything server-side; the page's checks are only a convenience.
        body = request.get_json(silent=True) or {}
        label = str(body.get("label", "")).strip()
        try:
            count = int(body.get("count"))
            interval = float(body.get("interval"))
        except (TypeError, ValueError):
            return jsonify(error="count and interval must be numbers"), 400
        if not LABEL_RE.match(label):
            return jsonify(error="Label may only contain letters, numbers, - and _ (max 64)"), 400
        if not 1 <= count <= MAX_COUNT:
            return jsonify(error=f"Images per run must be 1-{MAX_COUNT}"), 400
        if not 0 <= interval <= 3600:
            return jsonify(error="Seconds between images must be 0-3600"), 400
        try:
            collector.start_run(label, count, interval)
        except ValueError as exc:
            return jsonify(error=str(exc)), 409
        except OSError as exc:  # e.g. the folder could not be created
            return jsonify(error=str(exc)), 500
        return jsonify(status_json())

    @app.post("/api/stop")
    def stop():
        collector.stop_run()
        return jsonify(status_json())

    return app


def handle_sigterm(signum, frame) -> None:
    """Treat `kill` / `pkill` (SIGTERM) like Ctrl+C so cleanup code runs."""
    raise KeyboardInterrupt


def run_web(args: argparse.Namespace) -> int:
    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # hide per-request logs
    signal.signal(signal.SIGTERM, handle_sigterm)
    require_flask()  # fail fast, before opening the camera
    collector = Collector(args)
    try:
        collector.start()
        collector.ready.wait()
        if collector.error:
            logging.error("%s", collector.error)
            return 1
        w, h = collector.frame_size
        logging.info("Collecting %dx%d images into %s", w, h, args.dataset_dir.expanduser().resolve())
        logging.info("Open http://<pi-address>:%d in a browser  (Ctrl+C to stop)", args.port)
        # Flask's built-in server; threaded=True gives each viewer its own thread.
        # Blocks until Ctrl+C / SIGTERM / a camera crash (KeyboardInterrupt).
        create_app(collector).run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        collector.stop_event.set()
        if collector.is_alive():
            collector.join(timeout=5)
        logging.info("Stopped.")
    return 1 if collector.error else 0


def require_flask() -> None:
    """Exit with an install hint if Flask is missing."""
    try:
        import flask  # noqa: F401
    except ImportError:
        raise SystemExit("Web mode needs Flask: sudo apt install python3-flask")


# =========================================================================== main


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    return run_cli(args) if args.label else run_web(args)


if __name__ == "__main__":
    sys.exit(main())
