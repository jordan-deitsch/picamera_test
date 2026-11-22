# PiCamera Test

Camera and machine-vision experiments on a Raspberry Pi 4 with a Camera Module 3 (IMX708, autofocus), using **Picamera2**. The current goal is to collect training data for an **Edge Impulse** object-detection model. Later, the trained model will run live on the Pi and control a robot.

Development happens on a Windows PC. Code reaches the Pi through GitHub, and the PC runs the Pi's scripts over SSH.

```
 Windows PC (dev)                         GitHub                    Raspberry Pi 4
 Documents/RaspberryPi/PiCamera_Test ──git push──► picamera_test ──git pull──► ~/picamera_test
            ▲                                                                 │
            └──────────────── scp (images, datasets) ◄────────────────────────┘
                           ssh pi "..." (run scripts remotely)
```

## Scripts

| Script | Runs on | Purpose |
|---|---|---|
| [`capture_still.py`](capture_still.py) | Pi | Take one full-resolution still (4608x2592) and save it to `captures/` |
| [`remote_capture.py`](remote_capture.py) | PC | Run `capture_still.py` on the Pi and copy the image into `captures/` on the PC |
| [`collect_data.py`](collect_data.py) | Pi | Collect labeled training images into `dataset/<label>/` |
| [`fetch_dataset.py`](fetch_dataset.py) | PC | Copy `dataset/` from the Pi to the PC |
| [`upload_dataset.py`](upload_dataset.py) | PC | Upload `dataset/` to an Edge Impulse project |
| [`live_detect.py`](live_detect.py) | Pi | Run the FOMO model on the live camera feed and stream it with overlays to a web page |

Git ignores `captures/`, `dataset/`, `models/`, and `.venv/`, so images, model files, and the virtual environment never go into the repo.

---

## Setup

### Raspberry Pi
- Raspberry Pi OS Bookworm (64-bit), Python 3.11, with Picamera2 already installed (`python3-picamera2`).
- Check that the camera is detected:
  ```bash
  rpicam-hello --list-cameras
  ```
- The repo is cloned at `~/picamera_test`. The Pi's SSH key is added to GitHub so it can pull and push.
- Live detection needs OpenCV, Flask, and PyAudio from apt, and the Edge Impulse Python SDK in a project virtual environment. The virtual environment uses `--system-site-packages`, so it can still see Picamera2, OpenCV, and the system NumPy:
  ```bash
  sudo apt install -y python3-opencv python3-flask python3-pyaudio
  cd ~/picamera_test
  python3 -m venv --system-site-packages .venv
  .venv/bin/pip install edge_impulse_linux==1.2.2
  ```
  The SDK imports PyAudio even for image models, so it's needed even though this project doesn't use audio. Don't let pip upgrade NumPy inside `.venv`: the apt OpenCV and Picamera2 are built against the system NumPy (1.24).

### Windows PC
- Python 3.13 with `requests` (only needed by `upload_dataset.py`):
  ```powershell
  python -m pip install requests
  ```
- Windows' built-in OpenSSH client.
- An SSH host alias called `pi` in `~/.ssh/config`, so `ssh pi` logs in without a password:
  ```
  Host pi
      HostName <pi-ip-address>
      User <pi-username>
      IdentityFile ~/.ssh/id_ed25519_pi
      IdentitiesOnly yes
      ServerAliveInterval 30
  ```
  The PC scripts use this `pi` alias, and they never prompt for a password. If the Pi's IP address changes, update `HostName`. A DHCP reservation in the router keeps the address fixed.

### Line endings
`.gitattributes` makes git store all text files with Linux (LF) line endings. Scripts edited on Windows then still run on the Pi; Windows (CRLF) line endings break shebang lines and shell scripts.

---

## Development workflow

1. Edit the code on the PC, then commit and push:
   ```powershell
   git add -A
   git commit -m "..."
   git push
   ```
2. Update the Pi:
   ```powershell
   ssh pi "cd ~/picamera_test && git pull"
   ```
3. Run a script on the Pi from the PC:
   ```powershell
   ssh pi "cd ~/picamera_test && python3 <script>.py"
   ```
   Add `-t` (`ssh -t pi ...`) for scripts that wait for keyboard input.

You can also use **VS Code → Remote-SSH → Connect to Host → `pi`** to edit and debug directly on the Pi.

---

## Taking a single still image

From the PC (`--pull` runs `git pull` on the Pi first):
```powershell
python remote_capture.py --pull
```
This takes a full-resolution picture on the Pi, saves it to `~/picamera_test/captures/still_<timestamp>.jpg`, and copies it to `captures/` on the PC.

To run it directly on the Pi:
```bash
python3 capture_still.py                    # saves to ./captures/
python3 capture_still.py --output-dir /tmp  # custom folder
python3 capture_still.py --no-autofocus     # skip the autofocus cycle
python3 capture_still.py --warmup 2         # longer settle time for exposure/white balance
```

---

## Collecting training data for Edge Impulse

The full process: **collect on the Pi → fetch to the PC → upload to Edge Impulse → label → train → deploy.**

### 1. Collect images on the Pi

Images are saved as `dataset/<label>/<label>.<session>_<n>.jpg`.

**Interactive mode** (default): press **Enter** to take a picture, and type **`q` + Enter** to quit. Move the object between shots. The `-t` flag is required over SSH so your keypresses reach the Pi:
```powershell
ssh -t pi "cd ~/picamera_test && python3 collect_data.py --label cup"
```

**Burst mode**: take `--count` images, one every `--interval` seconds. This example takes 50 images 0.5 s apart:
```powershell
ssh pi "cd ~/picamera_test && python3 collect_data.py --label cup --count 50 --interval 0.5"
```

Options:

| Option | Default | Notes |
|---|---|---|
| `--label` | (required) | Letters, numbers, `-` and `_` only |
| `--size` | `640x480` | `WIDTHxHEIGHT`. See the note on image size below |
| `--count` | – | Turns on burst mode |
| `--interval` | `1.0` | Seconds between burst shots |
| `--quality` | `95` | JPEG quality |
| `--dataset-dir` | `./dataset` | Root folder for the dataset |

Autofocus runs continuously during collection.

**Image size and framing:** the sensor is 16:9. A 4:3 size like `640x480` crops the sides, giving a narrower, more zoomed-in view. Use `--size 640x360` to keep the full width. Whatever size you choose, **use the same size and framing when you run the model later**.

**Tips for a good dataset:**
- Capture from the robot's actual camera position, in the lighting it will actually work in.
- Vary the angle, distance, background, and position in the frame.
- Aim for balanced numbers per class. Start with about 50–100+ images per object.
- For object detection, also capture some images with **no** target objects in them.

### 2. Copy the dataset to the PC

```powershell
python fetch_dataset.py
```
This copies `~/picamera_test/dataset/` from the Pi into `./dataset/`, keeping the label folders, and prints how many images each label has. It never deletes anything on either machine.

### 3. Upload to Edge Impulse

1. Create a project at <https://studio.edgeimpulse.com>. Choose **Images** and then **Object detection** (or **Classification** if you only need to know what's in the picture).
2. Copy the project's **API key** from **Dashboard → Keys** in Edge Impulse Studio.
3. Set it in your PowerShell session. It only lasts for that terminal, so it never goes into the code or git:
   ```powershell
   $env:EI_API_KEY = "ei_..."
   ```
4. Check what will be uploaded, then upload:
   ```powershell
   python upload_dataset.py --dry-run
   ```
   ```powershell
   python upload_dataset.py
   ```

Options:

| Option | Default | Notes |
|---|---|---|
| `--category` | `split` | `split` lets Edge Impulse divide the images about 80/20 into training and testing; or use `training` / `testing` |
| `--labels` | all | Only upload some labels, e.g. `--labels cup ball` |
| `--dataset-dir` | `./dataset` | Root folder for the dataset |
| `--dry-run` | – | List what would be uploaded without uploading |

The script uses Edge Impulse's **ingestion API**, so Node.js and the Edge Impulse CLI aren't needed. Each folder name becomes that image's label. The script tells Edge Impulse to **skip images it already has**, so re-running after collecting more is safe; skipped duplicates are reported but aren't errors.

### 4. Label, train, and deploy (in Edge Impulse Studio)

1. **Label:** for object detection, open **Data acquisition → Labeling queue** and draw bounding boxes. AI-assisted labeling speeds this up.
2. **Create impulse:** Image input → Image processing block → Object Detection learning block. Set the input size to match your plans (e.g. 96x96 or 160x160 for FOMO, 320x320 for MobileNetV2 SSD).
3. **Choose a model with the Pi 4 in mind.** The Pi 4 runs models on its processor only, with no AI accelerator:
   - **FOMO:** very fast (tens of frames per second). Gives object *centers*, not box sizes, which is often enough for steering a robot.
   - **MobileNetV2 SSD / YOLO-style:** full bounding boxes, but roughly 2–10 frames per second depending on input size.
   - Studio's estimated speed for **Raspberry Pi 4** helps you compare.
4. **Train**, then check results under **Model testing**.
5. **Deploy:** choose **Linux (AARCH64)** to get a `.eim` model file for the Pi. (The **C++ library** export is for microcontrollers and can't be loaded from Python.) Save it in `models/` on the PC, then copy it to the Pi. See below.

---

## Live detection and streaming

[`live_detect.py`](live_detect.py) runs the Edge Impulse FOMO model on every camera frame and streams the annotated video to a web page. The current model finds the **top** and **bottom** of a pen on a white background. From those two points the script works out the pen's **center** and **orientation**, so a fast FOMO model, which only finds object centers, still gives the full pose.

### 1. Copy the model to the Pi

Model files (`.eim`) are kept out of the public repo, so copy them with `scp`:
```powershell
scp "models/<model-name>.eim" pi:picamera_test/models/
```
```powershell
ssh pi "chmod +x ~/picamera_test/models/*.eim"
```
The script uses the **newest** `.eim` in `models/` unless you pass `--model`.

### 2. Start it

**From the PC (recommended):** always use **`-t`**. It gives the Pi a terminal, so Ctrl+C reaches the script:
```powershell
ssh -t pi "cd ~/picamera_test && .venv/bin/python live_detect.py"
```
To update the code on the Pi first:
```powershell
ssh -t pi "cd ~/picamera_test && git pull && .venv/bin/python live_detect.py"
```
**Directly on the Pi** (local terminal, or after `ssh pi`):
```bash
cd ~/picamera_test
.venv/bin/python live_detect.py
```
**In the background**, so it keeps running after you close the terminal. Output goes to `/tmp/live_detect.log`:
```powershell
ssh pi "cd ~/picamera_test && nohup .venv/bin/python live_detect.py > /tmp/live_detect.log 2>&1 &"
```

Then open **`http://<pi-ip-address>:8000`** in a browser on the same network.

### 3. Stop it

The script stops cleanly on either Ctrl+C or `kill`: it stops the camera, shuts down the model, and exits.

**Started in the foreground:** press **Ctrl+C** in that terminal.

> **Without `-t`, Ctrl+C only closes your SSH connection.** The script can keep running on the Pi and keep holding the camera. The next run then fails with a "camera busy" error. Use the command below to stop it.

**Started in the background, or left running:** from any PC terminal:
```powershell
ssh pi "pkill -f '[.]venv/bin/python live_detect'"
```
Or directly on the Pi:
```bash
pkill -f '[.]venv/bin/python live_detect'
```
The brackets in `[.]venv` stop `pkill` from matching, and killing, the shell command that runs it. Without them, `pkill -f live_detect.py` over SSH ends your own SSH session.

**Check whether it's running:**
```powershell
ssh pi "pgrep -af '[.]venv/bin/python live_detect' || echo 'not running'"
```
**View the log of a background run:**
```powershell
ssh pi "tail -n 30 /tmp/live_detect.log"
```

If it ignores a normal stop (it shouldn't), force it with `pkill -9 -f '[.]venv/bin/python live_detect'`. The camera is freed when the process exits.

Options:

| Option | Default | Notes |
|---|---|---|
| `--model` | newest `models/*.eim` | Path to a specific model |
| `--size` | `640x480` | Camera frame size. Keep it the same as the training images |
| `--min-confidence` | `0.5` | Ignore weaker detections |
| `--port` | `8000` | Web server port |
| `--jpeg-quality` | `80` | Stream quality versus bandwidth |
| `--print` | – | Also print detections to the console |
| `--no-smoothing` | – | Turn off all jitter reduction (raw strongest cell per label) |
| `--min-cutoff` | `0.5` | One Euro filter: smoothing while still. Lower = steadier, but more lag |
| `--beta` | `0.01` | One Euro filter: response to motion. Higher = less lag when moving, more jitter |
| `--hold-frames` | `5` | Keep showing a point for this many frames after it's lost |

### Jitter reduction

FOMO detects on a coarse grid. Each cell is 8x8 model pixels, about **40 frame pixels**, so raw points snap between grid cells and flicker when a detection drops for a frame. `live_detect.py` applies three fixes to each label (`pen_top`, `pen_bottom`). The center point and angle are then calculated from the smoothed points:

1. **Within a frame (sub-cell precision):** FOMO usually marks 1–4 neighboring cells for one object. The point is the confidence-weighted average of the strongest cell and its neighbors (within 1.5 cells), so it can land between grid cells. Cells further away are ignored, as a separate object or a false detection.
2. **Over time (One Euro filter):** a smoothing filter that adapts to speed. When the pen is still, it smooths heavily, which removes jitter. When the pen moves, it smooths less, which keeps lag low.
3. **Dropout hold:** if a point isn't detected for up to `--hold-frames` frames, its last position stays on screen with a **hollow** marker labeled `held`, instead of disappearing. After that it's cleared, so the next detection starts fresh rather than sliding in from the old position.

In a simulation of FOMO's grid output at 30 FPS, these cut a still pen's jitter by roughly 60–70%. They also filled every gap when 15% of frames had no detection.

**Tuning on the real setup:**
- **Steadier while still:** lower `--min-cutoff` (e.g. `0.3`).
- **Less lag when moving:** raise `--beta` (e.g. `0.02`).
- **Compare with the raw output:** run with `--no-smoothing`.

The simulation showed this trade-off:

| `--min-cutoff` / `--beta` | Still jitter | Lag at ~160 px/s |
|---|---|---|
| 0.3 / 0.005 | ~6 px | ~11 px |
| **0.5 / 0.01 (default)** | **~7 px** | **~7 px** |
| 1.0 / 0.02 | ~9 px | ~4 px |
| raw (`--no-smoothing`) | ~20 px | – |

### What the overlay shows

| Element | Meaning |
|---|---|
| Green dot | `pen_top`: smoothed position, labeled `top` and its confidence |
| Orange dot | `pen_bottom`: the same, for the bottom |
| Hollow green/orange ring with `held` | That point wasn't detected this frame; showing its last position |
| Small hollow dots | This frame's raw FOMO cells, before smoothing |
| Cyan line | Bottom → top: the pen's orientation |
| Magenta crosshair | The pen's center (midpoint of top and bottom) |
| Gray square | The center square of the frame that the model actually sees. Objects outside it aren't detected |
| Top-left | Stream FPS and model inference time |
| Bottom-left | Orientation angle and length in pixels, or which point is missing |

The video only labels each point with its name and confidence, to keep the image uncluttered. The **coordinates** of the top, bottom, and center points, plus the angle and length, are in the summary panel next to the video and in `/api/detections`.

Labels sit to the upper right of their point, and flip to the **left** (and below, at the top edge) when they would run off the image, so they're always fully visible.

**Angle convention:** the direction from bottom to top, in degrees, counter-clockwise from pointing right, with "up" on screen positive. **90°** = top pointing straight up, **0°** = right, **180°** = left, **−90°** = down. All coordinates are frame pixels with the origin at the top-left (y increases downward).

### Web endpoints

| URL | Returns |
|---|---|
| `/` | Viewer page: live stream plus live coordinates, angle, and FPS |
| `/stream` | MJPEG video with overlays |
| `/snapshot.jpg` | One annotated frame |
| `/api/detections` | Latest result as JSON. Intended for robot control later. `top`, `bottom`, and `pose` are the **smoothed** values to use for control. Each point has `x`, `y`, `confidence`, `size`, and `held`. `raw_top` / `raw_bottom` are this frame's unsmoothed measurements. Also includes `all_points` (raw FOMO cells), `smoothing`, `fps`, and `inference_ms` |

### Performance and framing notes
- On the Pi 4 the 96x96 FOMO model takes about **5 ms per frame**. The stream runs at the camera's **30 FPS**, leaving most of the processor free for robot control.
- The model uses the **fit-shortest** resize mode: each frame is center-cropped to a square (480x480 from 640x480) and scaled to 96x96. The script maps the detections back to full-frame pixels.
- The script forces a **full-field-of-view sensor mode** (2304x1296). On the Camera Module 3, the faster 1536x864 mode is a center crop, which would zoom in compared to the training images.
- The web page has no login. Use it only on your home network; for remote access, use a VPN such as Tailscale instead of opening a port on your router.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Permission denied (publickey)` from `ssh pi` | Check that `~/.ssh/id_ed25519_pi.pub` is in `~/.ssh/authorized_keys` on the Pi and that the `pi` entry in `~/.ssh/config` is correct |
| `ssh: connect to host ... timed out` | The Pi is off, or its IP address changed. Find it on the router (or check the ARP table with `arp -a`) and update `HostName` |
| `Permission denied (publickey)` on `git pull` on the Pi | The Pi's `~/.ssh/id_ed25519.pub` must be added to GitHub. Test with `ssh -T git@github.com` on the Pi |
| Camera errors such as `Device or resource busy` | Another program is using the camera, often a `live_detect.py` left running after Ctrl+C without `ssh -t`. Stop it with `ssh pi "pkill -f '[.]venv/bin/python live_detect'"` (or `pkill -f '[r]picam'` for camera apps), then try again |
| `No interactive input (use 'ssh -t' or --count)` | Run interactive collection with `ssh -t`, or use burst mode |
| `Set the EI_API_KEY environment variable...` | Set `$env:EI_API_KEY` in the same terminal before running `upload_dataset.py` |
| Upload returns `HTTP 401/403` | The API key is wrong or belongs to a different project |
| Scripts fail on the Pi with `\r` or `bad interpreter` | The file has Windows line endings. `.gitattributes` should prevent this. Re-checkout with `git rm --cached -r . && git reset --hard` (this discards uncommitted changes on the Pi) |
