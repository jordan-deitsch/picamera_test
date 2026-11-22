#!/usr/bin/env python3
"""Live pen-orientation detection with an Edge Impulse FOMO model, streamed to a web page.

Runs ON THE RASPBERRY PI, inside the project virtual environment:
    cd ~/picamera_test
    .venv/bin/python live_detect.py
Then open http://<pi-ip-address>:8000 in a browser on the same network.

Pipeline (one background thread):
    Picamera2 frame (640x480) -> center square crop -> 96x96 -> FOMO model
      -> best "pen_top" and "pen_bottom" centroids, mapped back to frame pixels
      -> overlay: both points, a line between them, the pen's center point,
         and its orientation angle
      -> JPEG -> web page (MJPEG stream) + JSON API

Orientation convention: angle of the vector from pen_bottom to pen_top,
in degrees, measured counter-clockwise from the +x axis (to the right) with
y pointing up on screen. 90 = top pointing straight up, 0 = pointing right,
180/-180 = pointing left, -90 = pointing down.

Web endpoints:
    /                 viewer page (stream + live numbers)
    /stream           MJPEG video with overlays
    /snapshot.jpg     single annotated frame
    /api/detections   latest detection result as JSON (for robot control later)
"""

import argparse
import logging
import math
import signal
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
from edge_impulse_linux.image import ImageImpulseRunner
from flask import Flask, Response, jsonify
from picamera2 import Picamera2

try:
    from libcamera import controls
except ImportError:
    controls = None

PROJECT_DIR = Path(__file__).resolve().parent
TOP_LABEL = "pen_top"
BOTTOM_LABEL = "pen_bottom"

# Overlay colors (OpenCV uses BGR order).
COLOR_TOP = (60, 220, 60)        # green
COLOR_BOTTOM = (0, 140, 255)     # orange
COLOR_LINE = (255, 220, 0)       # cyan
COLOR_CENTER = (255, 0, 255)     # magenta
COLOR_CROP = (160, 160, 160)     # gray: the square region the model sees
COLOR_TEXT = (255, 255, 255)


# --------------------------------------------------------------------------- data


@dataclass
class Point:
    label: str
    x: float          # frame pixel coordinates (origin top-left)
    y: float
    confidence: float


@dataclass
class PenPose:
    center_x: float
    center_y: float
    angle_deg: float  # see module docstring for the convention
    length_px: float


# --------------------------------------------------------------------------- args


def parse_size(text: str) -> tuple[int, int]:
    try:
        width, height = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("size must look like 640x480")
    return width, height


def newest_model() -> Path | None:
    models = sorted((PROJECT_DIR / "models").glob("*.eim"), key=lambda p: p.stat().st_mtime)
    return models[-1] if models else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live FOMO pen detection with a web stream.")
    parser.add_argument("--model", type=Path, help="Path to the .eim model (default: newest in ./models)")
    parser.add_argument(
        "--size",
        type=parse_size,
        default=(640, 480),
        help="Camera frame size (default: 640x480, same as the training images)",
    )
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Ignore detections below this (default: 0.5)")
    parser.add_argument("--host", default="0.0.0.0", help="Web server address (default: all interfaces)")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="Stream JPEG quality (default: 80)")
    parser.add_argument("--print", dest="print_results", action="store_true", help="Also print detections to the console")
    args = parser.parse_args()

    args.model = args.model or newest_model()
    if args.model is None or not args.model.is_file():
        parser.error("no .eim model found; put one in ./models or pass --model")
    return args


# --------------------------------------------------------------------------- camera


def start_camera(size: tuple[int, int]) -> Picamera2:
    picam2 = Picamera2()

    # Use the smallest sensor mode that still covers the FULL sensor area, so the
    # field of view matches the full-FOV training images. (On the Camera Module 3
    # the fastest 1536x864 mode is a center crop, which would zoom in.)
    modes = picam2.sensor_modes
    full_area = max(m["crop_limits"][2] * m["crop_limits"][3] for m in modes)
    full_fov = [m for m in modes if m["crop_limits"][2] * m["crop_limits"][3] == full_area]
    sensor_mode = min(full_fov, key=lambda m: m["size"][0] * m["size"][1])

    config = picam2.create_video_configuration(
        # "RGB888" in libcamera means BGR byte order in the numpy array, which is what OpenCV uses.
        main={"size": size, "format": "RGB888"},
        raw={"size": sensor_mode["size"], "format": sensor_mode["unpacked"]},
        buffer_count=4,
    )
    picam2.configure(config)
    picam2.start()
    if controls is not None and "AfMode" in picam2.camera_controls:
        picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    time.sleep(1.0)  # let exposure / white balance / focus settle
    return picam2


# --------------------------------------------------------------------------- detection


def model_crop(frame_w: int, frame_h: int) -> tuple[int, int, int]:
    """The 'fit-shortest' resize mode center-crops a square. Return (x0, y0, side)."""
    side = min(frame_w, frame_h)
    return (frame_w - side) // 2, (frame_h - side) // 2, side


def extract_points(result: dict, frame_w: int, frame_h: int, model_w: int, model_h: int, min_conf: float) -> list[Point]:
    """Convert FOMO boxes (in model-input pixels) to centroids in frame pixels."""
    x0, y0, side = model_crop(frame_w, frame_h)
    sx, sy = side / model_w, side / model_h
    points = []
    for box in result.get("result", {}).get("bounding_boxes", []):
        if box["value"] < min_conf:
            continue
        cx = (box["x"] + box["width"] / 2) * sx + x0
        cy = (box["y"] + box["height"] / 2) * sy + y0
        points.append(Point(box["label"], cx, cy, box["value"]))
    return points


def best_per_label(points: list[Point]) -> dict[str, Point]:
    best: dict[str, Point] = {}
    for p in points:
        if p.label not in best or p.confidence > best[p.label].confidence:
            best[p.label] = p
    return best


def pen_pose(top: Point, bottom: Point) -> PenPose:
    dx = top.x - bottom.x
    dy = bottom.y - top.y  # flip so that "up" on screen is positive
    return PenPose(
        center_x=(top.x + bottom.x) / 2,
        center_y=(top.y + bottom.y) / 2,
        angle_deg=math.degrees(math.atan2(dy, dx)),
        length_px=math.hypot(dx, dy),
    )


# --------------------------------------------------------------------------- overlay


def put_text(img, text: str, org: tuple[int, int], color=COLOR_TEXT, scale: float = 0.5) -> None:
    """Text with a dark background box for readability on a white background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 2, y - h - 4), (x + w + 2, y + baseline), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, text, (x, y), font, scale, color, 1, cv2.LINE_AA)


def draw_overlay(frame, points: list[Point], best: dict[str, Point], pose: PenPose | None, fps: float, infer_ms: float) -> None:
    h, w = frame.shape[:2]
    x0, y0, side = model_crop(w, h)
    cv2.rectangle(frame, (x0, y0), (x0 + side - 1, y0 + side - 1), COLOR_CROP, 1)

    # Weaker extra detections (other FOMO cells) as small hollow dots.
    for p in points:
        if best.get(p.label) is not p:
            color = COLOR_TOP if p.label == TOP_LABEL else COLOR_BOTTOM
            cv2.circle(frame, (round(p.x), round(p.y)), 4, color, 1, cv2.LINE_AA)

    top, bottom = best.get(TOP_LABEL), best.get(BOTTOM_LABEL)

    if pose is not None:
        cv2.line(frame, (round(bottom.x), round(bottom.y)), (round(top.x), round(top.y)), COLOR_LINE, 2, cv2.LINE_AA)
        c = (round(pose.center_x), round(pose.center_y))
        cv2.drawMarker(frame, c, COLOR_CENTER, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        cv2.circle(frame, c, 5, COLOR_CENTER, 1, cv2.LINE_AA)
        put_text(frame, f"center ({c[0]}, {c[1]})", (c[0] + 12, c[1] + 18), COLOR_CENTER)

    for p, color, name in ((top, COLOR_TOP, "top"), (bottom, COLOR_BOTTOM, "bottom")):
        if p is None:
            continue
        pt = (round(p.x), round(p.y))
        cv2.circle(frame, pt, 7, color, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(frame, pt, 7, (0, 0, 0), 1, cv2.LINE_AA)
        put_text(frame, f"{name} ({pt[0]}, {pt[1]}) {p.confidence:.2f}", (pt[0] + 10, pt[1] - 10), color)

    put_text(frame, f"{fps:4.1f} FPS | inference {infer_ms:4.1f} ms", (8, 20))
    if pose is not None:
        status = f"angle {pose.angle_deg:6.1f} deg | length {pose.length_px:5.0f} px"
    elif top or bottom:
        status = f"only {'top' if top else 'bottom'} found"
    else:
        status = "no pen detected"
    put_text(frame, status, (8, h - 10))


# --------------------------------------------------------------------------- detector thread


class Detector(threading.Thread):
    def __init__(self, args: argparse.Namespace):
        super().__init__(daemon=True, name="detector")
        self.args = args
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.cond = threading.Condition()
        self.jpeg: bytes | None = None
        self.frame_id = 0
        self.latest: dict = {"status": "starting"}

    def run(self) -> None:
        try:
            self._loop()
        except BaseException as exc:  # surface errors to the main thread
            self.error = exc
            logging.exception("Detector thread failed")
        finally:
            self.ready.set()
            with self.cond:
                self.cond.notify_all()

    def _loop(self) -> None:
        args = self.args
        with ImageImpulseRunner(str(args.model.resolve())) as runner:
            info = runner.init()
            params = info["model_parameters"]
            model_w, model_h = params["image_input_width"], params["image_input_height"]
            logging.info(
                "Model: %s v%s | %dx%d | labels %s",
                info["project"]["name"], info["project"]["deploy_version"], model_w, model_h, params["labels"],
            )

            picam2 = start_camera(args.size)
            try:
                frame_w, frame_h = picam2.camera_configuration()["main"]["size"]
                logging.info("Camera: %dx%d frames", frame_w, frame_h)
                self.ready.set()

                fps = 0.0
                last = time.perf_counter()
                while not self.stop_event.is_set():
                    frame = picam2.capture_array("main")  # BGR
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    t0 = time.perf_counter()
                    features, _cropped = runner.get_features_from_image_auto_studio_settings(rgb)
                    result = runner.classify(features)
                    infer_ms = (time.perf_counter() - t0) * 1000

                    points = extract_points(result, frame_w, frame_h, model_w, model_h, args.min_confidence)
                    best = best_per_label(points)
                    top, bottom = best.get(TOP_LABEL), best.get(BOTTOM_LABEL)
                    pose = pen_pose(top, bottom) if top and bottom else None

                    now = time.perf_counter()
                    fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)) if fps else 1.0 / max(now - last, 1e-6)
                    last = now

                    draw_overlay(frame, points, best, pose, fps, infer_ms)
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

                    latest = {
                        "timestamp": time.time(),
                        "frame_size": [frame_w, frame_h],
                        "fps": round(fps, 1),
                        "inference_ms": round(infer_ms, 1),
                        "top": asdict(top) if top else None,
                        "bottom": asdict(bottom) if bottom else None,
                        "pose": {k: round(v, 1) for k, v in asdict(pose).items()} if pose else None,
                        "all_points": [asdict(p) for p in points],
                    }
                    with self.cond:
                        if ok:
                            self.jpeg = buf.tobytes()
                        self.latest = latest
                        self.frame_id += 1
                        self.cond.notify_all()

                    if args.print_results and (top or bottom):
                        msg = " | ".join(f"{p.label} ({p.x:.0f},{p.y:.0f}) {p.confidence:.2f}" for p in (top, bottom) if p)
                        if pose:
                            msg += f" | center ({pose.center_x:.0f},{pose.center_y:.0f}) angle {pose.angle_deg:.1f}"
                        print(msg, flush=True)
            finally:
                picam2.stop()
                picam2.close()

    def wait_for_frame(self, last_id: int, timeout: float = 2.0) -> tuple[int, bytes | None]:
        with self.cond:
            self.cond.wait_for(lambda: self.frame_id != last_id or self.stop_event.is_set() or self.error, timeout)
            return self.frame_id, self.jpeg


# --------------------------------------------------------------------------- web


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pen Detection</title>
<style>
  :root { color-scheme: light dark; --bg:#f4f4f5; --fg:#18181b; --card:#fff; --muted:#71717a; }
  @media (prefers-color-scheme: dark) { :root { --bg:#18181b; --fg:#f4f4f5; --card:#27272a; --muted:#a1a1aa; } }
  body { margin:0; font-family: system-ui, sans-serif; background:var(--bg); color:var(--fg); }
  main { max-width: 980px; margin: 0 auto; padding: 16px; display:grid; gap:16px; grid-template-columns: 1fr; }
  @media (min-width: 900px) { main { grid-template-columns: 640px 1fr; } }
  h1 { font-size: 1.1rem; margin: 0 0 4px; }
  img { width: 100%; max-width: 640px; border-radius: 8px; background:#000; display:block; }
  .card { background: var(--card); border-radius: 8px; padding: 12px 16px; }
  dl { display:grid; grid-template-columns: auto 1fr; gap: 6px 12px; margin:0; font-variant-numeric: tabular-nums; }
  dt { color: var(--muted); }
  dd { margin:0; }
  .top { color:#3cdc3c; } .bottom { color:#ff8c00; } .center { color:#ff00ff; }
</style>
</head>
<body>
<main>
  <div>
    <h1>Pen detection (FOMO)</h1>
    <img src="/stream" alt="Live camera stream with detection overlay">
  </div>
  <div class="card">
    <dl>
      <dt>FPS</dt><dd id="fps">–</dd>
      <dt>Inference</dt><dd id="inf">–</dd>
      <dt class="top">Top</dt><dd id="top">–</dd>
      <dt class="bottom">Bottom</dt><dd id="bottom">–</dd>
      <dt class="center">Center</dt><dd id="center">–</dd>
      <dt>Angle</dt><dd id="angle">–</dd>
      <dt>Length</dt><dd id="length">–</dd>
    </dl>
  </div>
</main>
<script>
  const fmt = p => p ? `(${p.x.toFixed(0)}, ${p.y.toFixed(0)})  ${(p.confidence * 100).toFixed(0)}%` : "not found";
  const set = (id, v) => document.getElementById(id).textContent = v;
  async function poll() {
    try {
      const d = await (await fetch("/api/detections")).json();
      set("fps", d.fps ?? "–");
      set("inf", d.inference_ms != null ? d.inference_ms + " ms" : "–");
      set("top", fmt(d.top));
      set("bottom", fmt(d.bottom));
      set("center", d.pose ? `(${d.pose.center_x.toFixed(0)}, ${d.pose.center_y.toFixed(0)})` : "–");
      set("angle", d.pose ? d.pose.angle_deg.toFixed(1) + "°" : "–");
      set("length", d.pose ? d.pose.length_px.toFixed(0) + " px" : "–");
    } catch (e) { /* server restarting; keep polling */ }
    setTimeout(poll, 200);
  }
  poll();
</script>
</body>
</html>
"""


def create_app(detector: Detector) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return PAGE

    @app.get("/stream")
    def stream():
        def frames():
            last_id = -1
            while not detector.stop_event.is_set() and detector.error is None:
                last_id, jpeg = detector.wait_for_frame(last_id)
                if jpeg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/snapshot.jpg")
    def snapshot():
        _, jpeg = detector.wait_for_frame(-1, timeout=0)
        if not jpeg:
            return "no frame yet", 503
        return Response(jpeg, mimetype="image/jpeg")

    @app.get("/api/detections")
    def detections():
        with detector.cond:
            return jsonify(detector.latest)

    return app


# --------------------------------------------------------------------------- main


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # hide per-request logs
    args = parse_args()

    detector = Detector(args)
    # Stop cleanly on Ctrl+C or `kill` (SIGTERM), releasing the camera and model.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    detector.start()
    detector.ready.wait()
    if detector.error:
        raise SystemExit(f"Detector failed to start: {detector.error}")

    logging.info("Streaming on http://%s:%d  (Ctrl+C to stop)", args.host, args.port)
    try:
        create_app(detector).run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop_event.set()
        detector.join(timeout=5)
        logging.info("Stopped.")


if __name__ == "__main__":
    main()
