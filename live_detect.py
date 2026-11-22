#!/usr/bin/env python3
"""Live pen-orientation detection with an Edge Impulse FOMO model, streamed to a web page.

Runs ON THE RASPBERRY PI, inside the project virtual environment:
    cd ~/picamera_test
    .venv/bin/python live_detect.py
Then open http://<pi-ip-address>:8000 in a browser on the same network.

Pipeline (one background thread):
    Picamera2 frame (640x480) -> center square crop -> 96x96 -> FOMO model
      -> "pen_top" / "pen_bottom" cell centroids, mapped back to frame pixels
      -> jitter reduction:
           1. spatial: confidence-weighted average of the strongest cell and its
              neighbouring cells of the same label (sub-cell precision instead of
              snapping to FOMO's coarse grid)
           2. temporal: One Euro filter per point (smooth when still, responsive
              when moving), plus a short hold so a point that drops out for a
              frame or two doesn't flicker
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
    size: float = 0.0  # FOMO cell size in frame pixels
    held: bool = False  # True if not detected this frame and showing the last position


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

    smooth = parser.add_argument_group("jitter reduction")
    smooth.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable all jitter reduction (raw strongest cell per label, no filtering or hold)",
    )
    smooth.add_argument(
        "--min-cutoff",
        type=float,
        default=0.5,
        help="One Euro filter cutoff (Hz) when still. Lower = smoother but laggier (default: 0.5)",
    )
    smooth.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="One Euro filter speed coefficient. Higher = less lag when moving, more jitter (default: 0.01)",
    )
    smooth.add_argument(
        "--hold-frames",
        type=int,
        default=5,
        help="Keep showing a point for this many frames after it is lost (default: 5)",
    )
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
        points.append(Point(box["label"], cx, cy, box["value"], size=box["width"] * sx))
    return points


def strongest(points: list[Point], label: str) -> Point | None:
    candidates = [p for p in points if p.label == label]
    return max(candidates, key=lambda p: p.confidence) if candidates else None


def refined_point(points: list[Point], label: str, radius_cells: float = 1.5) -> Point | None:
    """Spatial jitter reduction: confidence-weighted centroid of the strongest cell
    and its neighbouring same-label cells (within radius_cells cell widths).

    FOMO usually lights up 1-4 adjacent cells per object. Averaging them places the
    point between grid cells instead of jumping from cell to cell. Cells further
    away are treated as separate objects/false positives and ignored.
    """
    best = strongest(points, label)
    if best is None:
        return None
    radius = radius_cells * best.size
    near = [p for p in points if p.label == label and math.hypot(p.x - best.x, p.y - best.y) <= radius]
    weight = sum(p.confidence for p in near)
    return Point(
        label,
        sum(p.x * p.confidence for p in near) / weight,
        sum(p.y * p.confidence for p in near) / weight,
        best.confidence,
        best.size,
    )


class OneEuroFilter:
    """One Euro filter (Casiez et al., 2012): a low-pass filter whose cutoff rises
    with speed. Still objects get heavy smoothing (no jitter); moving objects get
    light smoothing (little lag)."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev: float | None = None
        self.dx_prev = 0.0
        self.t_prev = 0.0

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self.x_prev = None

    def __call__(self, x: float, t: float) -> float:
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, 0.0, t
            return x
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        a = self._alpha(dt, self.min_cutoff + self.beta * abs(dx_hat))
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class PointTracker:
    """Temporal jitter reduction for one label: One Euro filter on x and y, and a
    short hold so brief detection dropouts don't make the point flicker."""

    def __init__(self, min_cutoff: float, beta: float, hold_frames: int, enabled: bool = True):
        self.enabled = enabled
        self.hold_frames = hold_frames if enabled else 0
        self.fx = OneEuroFilter(min_cutoff, beta)
        self.fy = OneEuroFilter(min_cutoff, beta)
        self.last: Point | None = None
        self.missed = 0

    def update(self, p: Point | None, t: float) -> Point | None:
        if p is not None:
            self.missed = 0
            if self.enabled:
                p = Point(p.label, self.fx(p.x, t), self.fy(p.y, t), p.confidence, p.size)
            self.last = p
            return p

        self.missed += 1
        if self.last is not None and self.missed <= self.hold_frames:
            held = Point(**{**asdict(self.last), "held": True})
            return held
        # Lost for too long: forget it so the next detection starts fresh
        # instead of sliding in from the old position.
        self.last = None
        self.fx.reset()
        self.fy.reset()
        return None


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


def put_label(img, text: str, anchor: tuple[int, int], color, below: bool = False, gap: int = 10, scale: float = 0.5) -> None:
    """Text label next to a point that always stays fully inside the frame.

    Placed to the right of the anchor when there is room; flipped to the LEFT when
    it would run off the right edge. Placed above (or below, if below=True) and
    flipped vertically at the top/bottom edges. Finally clamped to the frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    img_h, img_w = img.shape[:2]
    ax, ay = anchor
    pad = 2

    # Horizontal: right of the point, or left if it would overflow the right edge.
    x = ax + gap
    if x + tw + pad > img_w:
        x = ax - gap - tw
    x = max(pad, min(x, img_w - tw - pad))

    # Vertical (y is the text baseline): above or below, flipping at the edges.
    above_y, below_y = ay - gap, ay + gap + th
    y = below_y if below else above_y
    if y - th - 2 * pad < 0:
        y = below_y
    elif y + baseline + pad > img_h:
        y = above_y
    y = max(th + 2 * pad, min(y, img_h - baseline - pad))

    cv2.rectangle(img, (x - pad, y - th - 2 * pad), (x + tw + pad, y + baseline), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, text, (x, y), font, scale, color, 1, cv2.LINE_AA)


def draw_overlay(
    frame,
    raw_points: list[Point],
    top: Point | None,
    bottom: Point | None,
    pose: PenPose | None,
    fps: float,
    infer_ms: float,
) -> None:
    h, w = frame.shape[:2]
    x0, y0, side = model_crop(w, h)
    cv2.rectangle(frame, (x0, y0), (x0 + side - 1, y0 + side - 1), COLOR_CROP, 1)

    # Raw FOMO cells for this frame as small hollow dots (shows what was smoothed).
    for p in raw_points:
        color = COLOR_TOP if p.label == TOP_LABEL else COLOR_BOTTOM
        cv2.circle(frame, (round(p.x), round(p.y)), 4, color, 1, cv2.LINE_AA)

    if pose is not None:
        cv2.line(frame, (round(bottom.x), round(bottom.y)), (round(top.x), round(top.y)), COLOR_LINE, 2, cv2.LINE_AA)
        c = (round(pose.center_x), round(pose.center_y))
        cv2.drawMarker(frame, c, COLOR_CENTER, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        cv2.circle(frame, c, 5, COLOR_CENTER, 1, cv2.LINE_AA)
        put_label(frame, f"center ({c[0]}, {c[1]})", c, COLOR_CENTER, below=True)

    for p, color, name in ((top, COLOR_TOP, "top"), (bottom, COLOR_BOTTOM, "bottom")):
        if p is None:
            continue
        pt = (round(p.x), round(p.y))
        if p.held:  # not detected this frame: hollow marker
            cv2.circle(frame, pt, 7, color, 2, cv2.LINE_AA)
            text = f"{name} ({pt[0]}, {pt[1]}) held"
        else:
            cv2.circle(frame, pt, 7, color, cv2.FILLED, cv2.LINE_AA)
            cv2.circle(frame, pt, 7, (0, 0, 0), 1, cv2.LINE_AA)
            text = f"{name} ({pt[0]}, {pt[1]}) {p.confidence:.2f}"
        put_label(frame, text, pt, color)

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

                smoothing = not args.no_smoothing
                trackers = {
                    label: PointTracker(args.min_cutoff, args.beta, args.hold_frames, enabled=smoothing)
                    for label in (TOP_LABEL, BOTTOM_LABEL)
                }
                logging.info(
                    "Jitter reduction: %s",
                    f"on (min_cutoff={args.min_cutoff}, beta={args.beta}, hold={args.hold_frames} frames)"
                    if smoothing else "off",
                )

                fps = 0.0
                last = time.perf_counter()
                while not self.stop_event.is_set():
                    frame = picam2.capture_array("main")  # BGR
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    t0 = time.perf_counter()
                    features, _cropped = runner.get_features_from_image_auto_studio_settings(rgb)
                    result = runner.classify(features)
                    infer_ms = (time.perf_counter() - t0) * 1000

                    now = time.perf_counter()
                    points = extract_points(result, frame_w, frame_h, model_w, model_h, args.min_confidence)
                    measure = refined_point if smoothing else strongest
                    raw_top, raw_bottom = measure(points, TOP_LABEL), measure(points, BOTTOM_LABEL)
                    top = trackers[TOP_LABEL].update(raw_top, now)
                    bottom = trackers[BOTTOM_LABEL].update(raw_bottom, now)
                    pose = pen_pose(top, bottom) if top and bottom else None

                    fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)) if fps else 1.0 / max(now - last, 1e-6)
                    last = now

                    draw_overlay(frame, points, top, bottom, pose, fps, infer_ms)
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

                    def rounded(p: Point | None) -> dict | None:
                        if p is None:
                            return None
                        d = asdict(p)
                        d.update(x=round(p.x, 1), y=round(p.y, 1), confidence=round(p.confidence, 3), size=round(p.size, 1))
                        return d

                    latest = {
                        "timestamp": time.time(),
                        "frame_size": [frame_w, frame_h],
                        "fps": round(fps, 1),
                        "inference_ms": round(infer_ms, 1),
                        "smoothing": smoothing,
                        # top/bottom/pose are the smoothed values (use these for control);
                        # raw_* are this frame's unsmoothed measurements.
                        "top": rounded(top),
                        "bottom": rounded(bottom),
                        "pose": {k: round(v, 1) for k, v in asdict(pose).items()} if pose else None,
                        "raw_top": rounded(raw_top),
                        "raw_bottom": rounded(raw_bottom),
                        "all_points": [rounded(p) for p in points],
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
  const fmt = p => p ? `(${p.x.toFixed(0)}, ${p.y.toFixed(0)})  ${p.held ? "held" : (p.confidence * 100).toFixed(0) + "%"}` : "not found";
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
