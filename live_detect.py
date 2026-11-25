#!/usr/bin/env python3
"""Live pen-orientation detection with an Edge Impulse FOMO model, streamed to a web page.

Runs ON THE RASPBERRY PI, inside the project virtual environment:
    cd ~/picamera_test
    .venv/bin/python live_detect.py
Then open the address it prints (http://<pi-ip-address>:8000) in a browser on the same network.

ARCHITECTURE
------------
Two threads share one "latest result" (protected by a threading.Condition):

  Detector thread (class Detector)              Main thread (Flask web server)
  --------------------------------              ------------------------------
  loop:                                         /                -> viewer page (PAGE)
    capture frame (Picamera2, 640x480 BGR)      /stream          -> MJPEG: sends each new JPEG
    run FOMO model (.eim) on the frame          /snapshot.jpg    -> latest JPEG
    -> raw cells per label                      /api/detections  -> latest result as JSON
    -> jitter reduction (refined_point +
       PointTracker)
    -> pen pose (center, angle, length)
    -> draw overlay, encode JPEG
    -> publish JPEG + JSON  ------------------>  (read by the web handlers)

  HealthMonitor thread: every 2 s reads the CPU temperature, clock and
  throttle flags; added to /api/detections as "system" and logged to the
  terminal when they need attention (see "system health" below).

Processing steps per frame:
  1. Model input: the model's resize mode decides which part of the frame it
     sees. Ours is "fit-shortest": center square crop (480x480) -> 96x96.
  2. FOMO output: small boxes (one per grid cell, 8x8 model px) with a label and
     a confidence. We use each box's center and map it back to frame pixels.
  3. Jitter reduction, per label ("pen_top", "pen_bottom"):
       a. spatial  - confidence-weighted average of the strongest cell and its
                     neighbouring cells (sub-cell precision, no grid snapping)
       b. temporal - One Euro filter (smooth when still, responsive when moving)
       c. hold     - keep the last position for a few frames if a point drops out
  4. Pose: center = midpoint of top and bottom; angle and length of bottom->top.

Orientation convention: angle of the vector from pen_bottom to pen_top,
in degrees, measured counter-clockwise from the +x axis (to the right) with
y pointing up on screen. 90 = top pointing straight up, 0 = pointing right,
180/-180 = pointing left, -90 = pointing down.
"""

import _thread
import argparse
import logging
import math
import signal
import socket
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
from edge_impulse_linux.image import ImageImpulseRunner
from flask import Flask, Response, jsonify
from picamera2 import Picamera2

# libcamera control enums (autofocus modes); ship with Picamera2 on Pi OS.
try:
    from libcamera import controls
except ImportError:
    controls = None

PROJECT_DIR = Path(__file__).resolve().parent

# Model labels this script tracks (must match the labels trained in Edge Impulse).
TOP_LABEL = "pen_top"
BOTTOM_LABEL = "pen_bottom"

# Overlay colors. OpenCV uses BGR order, not RGB.
COLOR_TOP = (60, 220, 60)        # green
COLOR_BOTTOM = (0, 140, 255)     # orange
COLOR_LINE = (255, 220, 0)       # cyan
COLOR_CENTER = (255, 0, 255)     # magenta
COLOR_CROP = (160, 160, 160)     # gray: the region the model sees
COLOR_TEXT = (255, 255, 255)


# =========================================================================== data


@dataclass
class Point:
    """One detected point (a FOMO cell, or a refined/smoothed position)."""

    label: str
    x: float           # frame pixel coordinates (origin top-left, y grows downward)
    y: float
    confidence: float  # model score 0..1
    size: float = 0.0  # FOMO cell size in frame pixels (used as the neighbour radius unit)
    held: bool = False  # True if not detected this frame and showing the last position


@dataclass
class PenPose:
    """Pen position and orientation derived from the top and bottom points."""

    center_x: float
    center_y: float
    angle_deg: float  # see module docstring for the convention
    length_px: float


@dataclass
class ModelRegion:
    """Part of the camera frame that the model actually sees, in frame pixels.

    Model coordinates (0..model_w, 0..model_h) map linearly onto this rectangle.
    With "fit-longest" the region is larger than the frame (it includes padding),
    so x0/y0 can be negative.
    """

    x0: float
    y0: float
    width: float
    height: float

    def is_crop(self, frame_w: int, frame_h: int) -> bool:
        """True if the region is a crop *inside* the frame (worth outlining)."""
        return self.x0 > 0 or self.y0 > 0 or self.width < frame_w or self.height < frame_h


# =========================================================================== command line


def parse_size(text: str) -> tuple[int, int]:
    """argparse type for "WIDTHxHEIGHT" strings, e.g. "640x480" -> (640, 480)."""
    try:
        width, height = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("size must look like 640x480")
    return width, height


def newest_model() -> Path | None:
    """Most recently modified .eim file in ./models, or None."""
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
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Detection threshold 0-1, applied inside the model (default: 0.5)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Web server address (default: all interfaces)")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="Stream JPEG quality 1-100 (default: 80)")
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

    # Validate ranges up front so mistakes fail with a clear message.
    if not 0.0 < args.min_confidence <= 1.0:
        parser.error("--min-confidence must be between 0 and 1")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.min_cutoff <= 0 or args.beta < 0 or args.hold_frames < 0:
        parser.error("--min-cutoff must be > 0, --beta and --hold-frames must be >= 0")

    args.model = args.model or newest_model()
    if args.model is None or not args.model.is_file():
        parser.error("no .eim model found; put one in ./models or pass --model")
    return args


# =========================================================================== camera


def start_camera(size: tuple[int, int]) -> Picamera2:
    """Open, configure, and start the camera for continuous video frames."""
    try:
        picam2 = Picamera2()
    except (RuntimeError, IndexError) as exc:
        raise RuntimeError(f"Could not open the camera ({exc}). Is another camera script still running?") from exc

    # Pick the smallest sensor mode that still covers the FULL sensor area, so
    # the field of view matches the full-FOV training images. (On the Camera
    # Module 3 the fastest 1536x864 mode is a center crop, which would zoom in.)
    # crop_limits = (x, y, width, height) of the sensor area a mode reads.
    modes = picam2.sensor_modes
    full_area = max(m["crop_limits"][2] * m["crop_limits"][3] for m in modes)
    full_fov = [m for m in modes if m["crop_limits"][2] * m["crop_limits"][3] == full_area]
    sensor_mode = min(full_fov, key=lambda m: m["size"][0] * m["size"][1])

    config = picam2.create_video_configuration(
        # "RGB888" in libcamera means BGR byte order in the numpy array,
        # which is exactly what OpenCV expects.
        main={"size": size, "format": "RGB888"},
        # Forcing the raw stream size selects the sensor mode chosen above.
        raw={"size": sensor_mode["size"], "format": sensor_mode["unpacked"]},
        buffer_count=4,  # a few buffers so capture never waits on processing
    )
    picam2.configure(config)
    picam2.start()
    # Continuous autofocus: refocuses automatically as the scene changes.
    if controls is not None and "AfMode" in picam2.camera_controls:
        picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    time.sleep(1.0)  # let exposure / white balance / focus settle
    return picam2


# =========================================================================== model output -> points


def model_region(frame_w: int, frame_h: int, resize_mode: str) -> ModelRegion:
    """Which part of the frame the model sees, for Edge Impulse's resize modes.

    fit-shortest: center square crop (side = shorter frame edge), then scale.
    fit-longest:  pad the shorter edge to make a square (side = longer edge), then scale.
    squash:       the whole frame, stretched to the model's aspect ratio.
    """
    if resize_mode == "squash":
        return ModelRegion(0, 0, frame_w, frame_h)
    if resize_mode in ("fit-shortest", "fit-longest"):
        side = min(frame_w, frame_h) if resize_mode == "fit-shortest" else max(frame_w, frame_h)
        return ModelRegion((frame_w - side) / 2, (frame_h - side) / 2, side, side)
    raise ValueError(f"Unsupported model resize mode: {resize_mode!r}")


def extract_points(result: dict, region: ModelRegion, model_w: int, model_h: int) -> list[Point]:
    """Convert FOMO boxes (model-input pixels) to cell-center Points in frame pixels.

    The model has already dropped boxes below the threshold (see set_model_threshold).
    """
    sx, sy = region.width / model_w, region.height / model_h  # frame px per model px
    points = []
    for box in result.get("result", {}).get("bounding_boxes", []):
        cx = (box["x"] + box["width"] / 2) * sx + region.x0
        cy = (box["y"] + box["height"] / 2) * sy + region.y0
        points.append(Point(box["label"], cx, cy, box["value"], size=box["width"] * sx))
    return points


def strongest(points: list[Point], label: str) -> Point | None:
    """Highest-confidence point with the given label (raw FOMO output, no smoothing)."""
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
    if weight <= 0:  # only possible with zero-confidence cells; avoid dividing by zero
        return best
    return Point(
        label,
        sum(p.x * p.confidence for p in near) / weight,
        sum(p.y * p.confidence for p in near) / weight,
        best.confidence,
        best.size,
    )


# =========================================================================== temporal smoothing


class OneEuroFilter:
    """One Euro filter (Casiez et al., 2012) for one coordinate.

    An exponential low-pass filter whose cutoff frequency rises with speed:
      cutoff = min_cutoff + beta * |smoothed speed|
    Still objects -> low cutoff -> heavy smoothing (no jitter).
    Moving objects -> higher cutoff -> light smoothing (little lag).
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev: float | None = None  # last filtered value (None = no history)
        self.dx_prev = 0.0                # last filtered speed
        self.t_prev = 0.0                 # time of the last sample (seconds)

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        """Smoothing factor of a first-order low-pass filter for this time step."""
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self.x_prev = None

    def __call__(self, x: float, t: float) -> float:
        """Filter a new sample x taken at time t (seconds); returns the smoothed value."""
        if self.x_prev is None:  # first sample: nothing to smooth against yet
            self.x_prev, self.dx_prev, self.t_prev = x, 0.0, t
            return x
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
        # 1. Estimate speed, smoothed with a fixed cutoff (d_cutoff).
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        # 2. Smooth the value with a cutoff that grows with that speed.
        a = self._alpha(dt, self.min_cutoff + self.beta * abs(dx_hat))
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class PointTracker:
    """Temporal jitter reduction for one label: One Euro filter on x and y, and a
    short hold so brief detection dropouts don't make the point flicker.

    With enabled=False it passes points straight through (no filter, no hold).
    """

    def __init__(self, min_cutoff: float, beta: float, hold_frames: int, enabled: bool = True):
        self.enabled = enabled
        self.hold_frames = hold_frames if enabled else 0
        self.fx = OneEuroFilter(min_cutoff, beta)
        self.fy = OneEuroFilter(min_cutoff, beta)
        self.last: Point | None = None  # last output, re-shown while "held"
        self.missed = 0                 # consecutive frames without a detection

    def update(self, p: Point | None, t: float) -> Point | None:
        """Feed this frame's measurement (None if not detected); returns the point to use."""
        if p is not None:
            self.missed = 0
            if self.enabled:
                p = Point(p.label, self.fx(p.x, t), self.fy(p.y, t), p.confidence, p.size)
            self.last = p
            return p

        # Not detected this frame: show the last position for up to hold_frames frames.
        self.missed += 1
        if self.last is not None and self.missed <= self.hold_frames:
            return Point(**{**asdict(self.last), "held": True})
        # Lost for too long: forget it so the next detection starts fresh
        # instead of sliding in from the old position.
        self.last = None
        self.fx.reset()
        self.fy.reset()
        return None


def pen_pose(top: Point, bottom: Point) -> PenPose:
    """Center, orientation angle, and length of the pen from its two end points."""
    dx = top.x - bottom.x
    dy = bottom.y - top.y  # flip sign: image y grows downward, but we want "up" positive
    return PenPose(
        center_x=(top.x + bottom.x) / 2,
        center_y=(top.y + bottom.y) / 2,
        angle_deg=math.degrees(math.atan2(dy, dx)),
        length_px=math.hypot(dx, dy),
    )


# =========================================================================== overlay drawing


def put_text(img, text: str, org: tuple[int, int], color=COLOR_TEXT, scale: float = 0.5) -> None:
    """Text at a fixed position (org = baseline start), on a dark background box."""
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
    region: ModelRegion,
    raw_points: list[Point],
    top: Point | None,
    bottom: Point | None,
    pose: PenPose | None,
    fps: float,
    infer_ms: float,
) -> None:
    """Draw all detection graphics onto the frame (in place)."""
    h, w = frame.shape[:2]

    # Outline the area the model sees, if it's a crop of the frame.
    if region.is_crop(w, h):
        x0, y0 = round(region.x0), round(region.y0)
        cv2.rectangle(frame, (x0, y0), (x0 + round(region.width) - 1, y0 + round(region.height) - 1), COLOR_CROP, 1)

    # Raw FOMO cells for this frame as small hollow dots (shows what was smoothed).
    for p in raw_points:
        color = COLOR_TOP if p.label == TOP_LABEL else COLOR_BOTTOM
        cv2.circle(frame, (round(p.x), round(p.y)), 4, color, 1, cv2.LINE_AA)

    # Pen line (bottom -> top) and center marker, when both ends are known.
    if pose is not None:
        cv2.line(frame, (round(bottom.x), round(bottom.y)), (round(top.x), round(top.y)), COLOR_LINE, 2, cv2.LINE_AA)
        c = (round(pose.center_x), round(pose.center_y))
        cv2.drawMarker(frame, c, COLOR_CENTER, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        cv2.circle(frame, c, 5, COLOR_CENTER, 1, cv2.LINE_AA)
        put_label(frame, "center", c, COLOR_CENTER, below=True)

    # End points: filled dot if detected this frame, hollow ring if "held".
    for p, color, name in ((top, COLOR_TOP, "top"), (bottom, COLOR_BOTTOM, "bottom")):
        if p is None:
            continue
        pt = (round(p.x), round(p.y))
        if p.held:
            cv2.circle(frame, pt, 7, color, 2, cv2.LINE_AA)
            text = f"{name} held"
        else:
            cv2.circle(frame, pt, 7, color, cv2.FILLED, cv2.LINE_AA)
            cv2.circle(frame, pt, 7, (0, 0, 0), 1, cv2.LINE_AA)
            text = f"{name} {p.confidence:.2f}"
        put_label(frame, text, pt, color)

    # Status lines: performance (top-left) and pose summary (bottom-left).
    put_text(frame, f"{fps:4.1f} FPS | inference {infer_ms:4.1f} ms", (8, 20))
    if pose is not None:
        status = f"angle {pose.angle_deg:6.1f} deg | length {pose.length_px:5.0f} px"
    elif top or bottom:
        status = f"only {'top' if top else 'bottom'} found"
    else:
        status = "no pen detected"
    put_text(frame, status, (8, h - 10))


# =========================================================================== detector thread


def point_json(p: Point | None) -> dict | None:
    """Point -> JSON-friendly dict with rounded numbers (None stays None)."""
    if p is None:
        return None
    d = asdict(p)
    d.update(x=round(p.x, 1), y=round(p.y, 1), confidence=round(p.confidence, 3), size=round(p.size, 1))
    return d


def set_model_threshold(runner: ImageImpulseRunner, params: dict, min_score: float) -> None:
    """Apply --min-confidence inside the model.

    The .eim model drops detections below its built-in threshold (0.5 by default)
    before Python sees them, so filtering in Python alone could only *raise* the
    threshold. Setting it in the model works in both directions.
    """
    for threshold in params.get("thresholds", []):
        if threshold.get("type") == "object_detection":
            if threshold.get("min_score") != min_score:
                runner.set_threshold({"id": threshold["id"], "min_score": min_score})
                logging.info("Detection threshold: %.2f (model default %.2f)", min_score, threshold.get("min_score"))
            return
    logging.warning("Model reports no object-detection threshold; using its built-in default")


class Detector(threading.Thread):
    """Background thread: camera -> model -> smoothing -> overlay -> latest result.

    Shared with the web handlers (always accessed while holding self.cond):
      self.jpeg      latest annotated frame as JPEG bytes
      self.latest    latest result dict (served as JSON)
      self.frame_id  increments on every new frame, so waiters can tell "new" from "old"
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(daemon=True, name="detector")
        self.args = args
        self.stop_event = threading.Event()  # set by main() to ask the loop to exit
        self.ready = threading.Event()       # set once streaming starts (or startup failed)
        self.error: BaseException | None = None
        self.cond = threading.Condition()    # lock + "new frame" notifications
        self.jpeg: bytes | None = None
        self.frame_id = 0
        self.latest: dict = {"status": "starting"}

    def run(self) -> None:
        """Thread entry point: run the loop and report any crash to the main thread."""
        try:
            self._loop()
        except BaseException as exc:
            self.error = exc
            logging.exception("Detector thread failed")
            if self.ready.is_set():
                # Failed while streaming: stop the whole program instead of
                # serving a frozen image (raises KeyboardInterrupt in main()).
                _thread.interrupt_main()
        finally:
            self.ready.set()
            with self.cond:
                self.cond.notify_all()  # wake any web handlers waiting for a frame

    def _loop(self) -> None:
        args = self.args
        # The .eim model runs as a separate process; the runner talks to it over a
        # socket and stops it when the `with` block exits.
        with ImageImpulseRunner(str(args.model.resolve())) as runner:
            info = runner.init()
            params = info["model_parameters"]
            model_w, model_h = params["image_input_width"], params["image_input_height"]
            resize_mode = params.get("image_resize_mode", "fit-shortest")
            logging.info(
                "Model: %s v%s | %dx%d %s | labels %s",
                info["project"]["name"], info["project"]["deploy_version"],
                model_w, model_h, resize_mode, params["labels"],
            )
            missing = {TOP_LABEL, BOTTOM_LABEL} - set(params["labels"])
            if missing:
                logging.warning("Model has no %s label(s); those points will never be detected", sorted(missing))
            set_model_threshold(runner, params, args.min_confidence)

            picam2 = start_camera(args.size)
            try:
                frame_w, frame_h = picam2.camera_configuration()["main"]["size"]
                region = model_region(frame_w, frame_h, resize_mode)
                logging.info("Camera: %dx%d frames", frame_w, frame_h)

                smoothing = not args.no_smoothing
                trackers = {
                    label: PointTracker(args.min_cutoff, args.beta, args.hold_frames, enabled=smoothing)
                    for label in (TOP_LABEL, BOTTOM_LABEL)
                }
                # Raw measurement per label: refined (neighbour-averaged) or plain strongest cell.
                measure = refined_point if smoothing else strongest
                logging.info(
                    "Jitter reduction: %s",
                    f"on (min_cutoff={args.min_cutoff}, beta={args.beta}, hold={args.hold_frames} frames)"
                    if smoothing else "off",
                )
                self.ready.set()

                fps = 0.0
                last = time.perf_counter()
                while not self.stop_event.is_set():
                    # ---- 1. capture
                    frame = picam2.capture_array("main")  # BGR numpy array, shape (h, w, 3)
                    t_frame = time.perf_counter()         # timestamp for the smoothing filters
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # the model expects RGB

                    # ---- 2. inference (crop/resize exactly as configured in Studio, then classify)
                    features, _cropped = runner.get_features_from_image_auto_studio_settings(rgb)
                    result = runner.classify(features)
                    infer_ms = (time.perf_counter() - t_frame) * 1000

                    # ---- 3. points: raw cells -> per-label measurement -> smoothed/held
                    points = extract_points(result, region, model_w, model_h)
                    raw_top, raw_bottom = measure(points, TOP_LABEL), measure(points, BOTTOM_LABEL)
                    top = trackers[TOP_LABEL].update(raw_top, t_frame)
                    bottom = trackers[BOTTOM_LABEL].update(raw_bottom, t_frame)
                    pose = pen_pose(top, bottom) if top and bottom else None

                    # ---- 4. frame rate: exponential moving average of 1/frame-time
                    instant_fps = 1.0 / max(t_frame - last, 1e-6)
                    fps = 0.9 * fps + 0.1 * instant_fps if fps else instant_fps
                    last = t_frame

                    # ---- 5. overlay + JPEG for the stream
                    draw_overlay(frame, region, points, top, bottom, pose, fps, infer_ms)
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

                    # ---- 6. publish (JSON layout documented in the README)
                    latest = {
                        "timestamp": time.time(),
                        "frame_size": [frame_w, frame_h],
                        "fps": round(fps, 1),
                        "inference_ms": round(infer_ms, 1),
                        "smoothing": smoothing,
                        # top/bottom/pose are the smoothed values (use these for control);
                        # raw_* are this frame's unsmoothed measurements.
                        "top": point_json(top),
                        "bottom": point_json(bottom),
                        "pose": {k: round(v, 1) for k, v in asdict(pose).items()} if pose else None,
                        "raw_top": point_json(raw_top),
                        "raw_bottom": point_json(raw_bottom),
                        "all_points": [point_json(p) for p in points],
                    }
                    with self.cond:
                        if ok:
                            self.jpeg = buf.tobytes()
                        self.latest = latest
                        self.frame_id += 1
                        self.cond.notify_all()  # wake /stream handlers waiting for a new frame

                    if args.print_results and (top or bottom):
                        msg = " | ".join(f"{p.label} ({p.x:.0f},{p.y:.0f}) {p.confidence:.2f}" for p in (top, bottom) if p)
                        if pose:
                            msg += f" | center ({pose.center_x:.0f},{pose.center_y:.0f}) angle {pose.angle_deg:.1f}"
                        print(msg, flush=True)
            finally:
                # Always release the camera, even after errors.
                picam2.stop()
                picam2.close()

    def wait_for_frame(self, last_id: int, timeout: float = 2.0) -> tuple[int, bytes | None]:
        """Block until a frame newer than last_id exists (or timeout/stop/error).

        Returns (current frame id, latest JPEG). Called from web handler threads.
        """
        with self.cond:
            self.cond.wait_for(lambda: self.frame_id != last_id or self.stop_event.is_set() or self.error, timeout)
            return self.frame_id, self.jpeg


# =========================================================================== system health

# The Pi 4 starts throttling (lowering the CPU clock) at 80 °C and throttles
# hard at 85 °C. Warn a little earlier so there's time to react.
TEMP_WARN_C = 75.0
TEMP_THROTTLE_C = 80.0
# Bits of `vcgencmd get_throttled`. Bits 0-3 are "happening now"; the same
# conditions shifted by 16 (bits 16-19) mean "has happened since boot".
THROTTLE_BITS = {0: "under-voltage", 1: "frequency capped", 2: "throttled", 3: "soft temperature limit"}


def read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def read_throttled() -> int | None:
    """Raw throttle flags from the firmware (e.g. "throttled=0x50000" -> 0x50000)."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2).stdout
        return int(out.strip().split("=")[1], 16)
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None


def read_health() -> dict:
    """One reading of temperature, CPU clock and throttle state, as JSON-ready values."""
    millideg = read_int("/sys/class/thermal/thermal_zone0/temp")         # 55017 -> 55.0 °C
    khz = read_int("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")  # current CPU clock
    flags = read_throttled()
    temp_c = round(millideg / 1000, 1) if millideg is not None else None
    return {
        "temp_c": temp_c,
        "temp_warning": temp_c is not None and temp_c >= TEMP_WARN_C,
        "arm_mhz": round(khz / 1000) if khz is not None else None,
        "throttled": f"0x{flags:x}" if flags is not None else None,
        "now": [name for bit, name in THROTTLE_BITS.items() if flags and flags & (1 << bit)],
        "since_boot": [name for bit, name in THROTTLE_BITS.items() if flags and flags & (1 << (bit + 16))],
    }


class HealthMonitor(threading.Thread):
    """Background thread: refreshes read_health() every `interval` seconds.

    Logs a warning in the terminal whenever the state gets worse (and a note
    when the temperature recovers), so problems show up even with no browser open.
    """

    def __init__(self, interval: float = 2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest: dict = {}

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.latest)

    def run(self) -> None:
        was_hot, seen_now, seen_boot = False, set(), set()
        while not self.stop_event.is_set():
            h = read_health()
            with self.lock:
                self.latest = h
            if h["temp_warning"] and not was_hot:
                logging.warning("CPU temperature %.1f °C (throttling starts at %.0f °C)", h["temp_c"], TEMP_THROTTLE_C)
            elif was_hot and not h["temp_warning"]:
                logging.info("CPU temperature back to %.1f °C", h["temp_c"])
            was_hot = h["temp_warning"]
            # Report each new condition once.
            if new := set(h["now"]) - seen_now:
                logging.warning("Pi reports now: %s (flags %s)", ", ".join(sorted(new)), h["throttled"])
            if new := set(h["since_boot"]) - seen_boot:
                logging.warning("Pi reports since boot: %s (flags %s)", ", ".join(sorted(new)), h["throttled"])
            seen_now, seen_boot = set(h["now"]), seen_boot | set(h["since_boot"])
            self.stop_event.wait(self.interval)


# =========================================================================== web server

# Viewer page: the <img> shows the MJPEG stream; the script polls
# /api/detections 5x per second and fills in the summary panel.
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
  .ok { color:#16a34a; } .warn { color:#d97706; font-weight:600; } .bad { color:#dc2626; font-weight:700; }
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
      <dt>CPU temp</dt><dd id="temp">–</dd>
      <dt>Throttling</dt><dd id="throttle">–</dd>
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
      // System health: color the temperature (ok < 75, warn < 80, hot >= 80 °C).
      const s = d.system || {};
      const temp = document.getElementById("temp");
      temp.textContent = s.temp_c != null ? `${s.temp_c.toFixed(1)} °C  @ ${s.arm_mhz ?? "?"} MHz` : "–";
      temp.className = s.temp_c == null ? "" : s.temp_c >= __THROTTLE_C__ ? "bad" : s.temp_c >= __WARN_C__ ? "warn" : "ok";
      const thr = document.getElementById("throttle");
      if (s.throttled == null) { thr.textContent = "–"; thr.className = ""; }
      else if (s.now.length) { thr.textContent = "NOW: " + s.now.join(", "); thr.className = "bad"; }
      else if (s.since_boot.length) { thr.textContent = "earlier: " + s.since_boot.join(", "); thr.className = "warn"; }
      else { thr.textContent = "none"; thr.className = "ok"; }
    } catch (e) { /* server restarting; keep polling */ }
    setTimeout(poll, 200);
  }
  poll();
</script>
</body>
</html>
""".replace("__WARN_C__", str(TEMP_WARN_C)).replace("__THROTTLE_C__", str(TEMP_THROTTLE_C))


def create_app(detector: Detector, health: HealthMonitor) -> Flask:
    """Build the Flask app. Handlers only read the detector's and monitor's shared state."""
    app = Flask(__name__)

    @app.get("/")
    def index():
        return PAGE

    @app.get("/stream")
    def stream():
        # MJPEG: one long HTTP response made of JPEG "parts" separated by a
        # boundary marker; browsers show it as live video in a plain <img>.
        def frames():
            last_id = -1
            while not detector.stop_event.is_set() and detector.error is None:
                last_id, jpeg = detector.wait_for_frame(last_id)
                if jpeg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/snapshot.jpg")
    def snapshot():
        _, jpeg = detector.wait_for_frame(-1, timeout=0)  # don't wait; just take the latest
        if not jpeg:
            return "no frame yet", 503
        return Response(jpeg, mimetype="image/jpeg")

    @app.get("/api/detections")
    def detections():
        with detector.cond:
            latest = dict(detector.latest)
        latest["system"] = health.snapshot()  # temperature / clock / throttle flags
        return jsonify(latest)

    return app


# =========================================================================== main


def page_addresses(host: str) -> list[str]:
    """Addresses a browser can use to reach this Pi, for the startup message.

    Looked up at run time, so no address is ever stored in the code.
    """
    if host not in ("0.0.0.0", ""):
        return [host]  # listening on one specific address
    # Listening on all interfaces: list this Pi's IPv4 addresses (Wi-Fi, Ethernet).
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2).stdout
        addresses = [a for a in out.split() if "." in a]  # keep IPv4, skip IPv6
    except (OSError, subprocess.SubprocessError):
        addresses = []
    return addresses or [f"{socket.gethostname()}.local"]  # fallback: mDNS name


def handle_sigterm(signum, frame) -> None:
    """Treat `kill` / `pkill` (SIGTERM) like Ctrl+C so cleanup code runs."""
    raise KeyboardInterrupt


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # hide per-request logs
    args = parse_args()
    signal.signal(signal.SIGTERM, handle_sigterm)

    detector = Detector(args)
    health = HealthMonitor()
    health.start()  # temperature/throttle readings, logged even with no browser open
    try:
        # Start the camera/model thread and wait until it is streaming (or failed).
        detector.start()
        detector.ready.wait()
        if detector.error:
            logging.error("Detector failed to start: %s", detector.error)
            return 1

        for address in page_addresses(args.host):
            logging.info("Streaming on http://%s:%d", address, args.port)
        logging.info("Ctrl+C to stop")
        # Flask's built-in server; threaded=True gives each viewer its own thread.
        # Blocks until Ctrl+C / SIGTERM / a detector crash (KeyboardInterrupt).
        create_app(detector, health).run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        # Ask the detector loop to exit; it releases the camera and model itself.
        health.stop_event.set()
        detector.stop_event.set()
        if detector.is_alive():
            detector.join(timeout=5)
        logging.info("Stopped.")
    return 1 if detector.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
