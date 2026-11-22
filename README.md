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

Git ignores `captures/` and `dataset/`, so images never go into the repo.

---

## Setup

### Raspberry Pi
- Raspberry Pi OS Bookworm (64-bit), Python 3.11, with Picamera2 already installed (`python3-picamera2`).
- Check that the camera is detected:
  ```bash
  rpicam-hello --list-cameras
  ```
- The repo is cloned at `~/picamera_test`. The Pi's SSH key is added to GitHub so it can pull and push.

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
5. **Deploy:** choose **Linux (AARCH64)** to get a `.eim` model file for the Pi. Load it in Python with the [`edge_impulse_linux`](https://github.com/edgeimpulse/linux-sdk-python) SDK. The next stage of this project will add the live object-recognition script.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Permission denied (publickey)` from `ssh pi` | Check that `~/.ssh/id_ed25519_pi.pub` is in `~/.ssh/authorized_keys` on the Pi and that the `pi` entry in `~/.ssh/config` is correct |
| `ssh: connect to host ... timed out` | The Pi is off, or its IP address changed. Find it on the router (or check the ARP table with `arp -a`) and update `HostName` |
| `Permission denied (publickey)` on `git pull` on the Pi | The Pi's `~/.ssh/id_ed25519.pub` must be added to GitHub. Test with `ssh -T git@github.com` on the Pi |
| Camera errors such as `Device or resource busy` | Another program is using the camera. Stop it (`pkill -f rpicam`, or another Python script) and try again |
| `No interactive input (use 'ssh -t' or --count)` | Run interactive collection with `ssh -t`, or use burst mode |
| `Set the EI_API_KEY environment variable...` | Set `$env:EI_API_KEY` in the same terminal before running `upload_dataset.py` |
| Upload returns `HTTP 401/403` | The API key is wrong or belongs to a different project |
| Scripts fail on the Pi with `\r` or `bad interpreter` | The file has Windows line endings. `.gitattributes` should prevent this. Re-checkout with `git rm --cached -r . && git reset --hard` (this discards uncommitted changes on the Pi) |
