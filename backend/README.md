# PLR Pupillometry System
## Raspberry Pi 4 — Setup & Test Guide

---

## File Overview

```
plr_system/
├── main.py               ← Entry point. Run this.
├── config_wizard.py      ← Interactive terminal config (mode, pins, schedule)
├── led_controller.py     ← Per-LED polarity-safe GPIO HIGH/LOW control
├── orchestrator.py       ← Flash sequencer + fills 4 queues
├── camera_controller.py  ← IR camera recording (continuous, background thread)
├── segmenter.py          ← Cuts full video into per-flash clips via ffmpeg
├── eye_cropper.py        ← Crops full camera clips to one-eye model inputs
├── model_caller.py       ← Runs bundled INT8 ONNX inference
├── plr_metrics.py        ← Initial pixel-based PLR formulas
├── requirements.txt      ← Python dependencies
│
├── recordings/           ← Full session videos saved here (auto-created)
├── clips/
│   ├── led1/             ← Clips for LED 1, named by hex code
│   └── led2/             ← Clips for LED 2, named by hex code
└── results/              ← JSON results per clip + session_summary.json
```

---

## Step 1 — OS dependencies

```bash
sudo apt update
sudo apt install -y python3-pip ffmpeg
```

Verify ffmpeg works:
```bash
ffmpeg -version
```

---

## Step 2 — Python dependencies

```bash
pip3 install -r requirements.txt
```

`requirements.txt` installs GPIO/Flask plus the direct ONNX inference stack:
`onnxruntime`, headless OpenCV, NumPy, Pandas, and tqdm.

> **Note:** `opencv-python` can be slow to install on Pi. If it times out,
> use the pre-built package instead:
> ```bash
> sudo apt install -y python3-opencv
> ```

---

## Step 3 — Enable camera

If you are using the Pi camera module (not a USB webcam):
```bash
sudo raspi-config
# → Interface Options → Camera → Enable
sudo reboot
```

For a USB IR webcam, no extra steps needed. Check it is detected:
```bash
ls /dev/video*
# Should show /dev/video0 (or video1, video2 …)
```

Test the camera opens:
```bash
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
print('Camera opened:', cap.isOpened())
cap.release()
"
```

---

## Step 4 — Wire the LEDs

Each LED is a 4-pin RGB LED with a 6Ω resistor on the common pin.

```
LED pins:
  Common (long leg) → 6Ω resistor → GND  (common cathode)
                  OR → 6Ω resistor → 3.3V (common anode)
  R pin → GPIO (BCM)
  G pin → GPIO (BCM)
  B pin → GPIO (BCM)
```

**Suggested GPIO assignment (BCM numbering):**

| Channel     | LED 1 (Left) | LED 2 (Right) |
|-------------|-------------|--------------|
| R           | GPIO 17     | GPIO 22      |
| G           | GPIO 27     | GPIO 23      |
| B           | GPIO 22     | GPIO 24      |

> You can use any available BCM GPIO pins. You will enter them
> in the terminal wizard each session. Avoid GPIO 0,1 (I2C) and
> GPIO 14,15 (UART).

**Quick GPIO test before running the full system:**
```bash
python3 - << 'EOF'
import RPi.GPIO as GPIO, time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Change these to your actual pins
PINS = [17, 27, 22]   # LED 1: R, G, B

for pin in PINS:
    GPIO.setup(pin, GPIO.OUT)

print("Cycling R → G → B → off on LED 1...")
for pin in PINS:
    GPIO.output(pin, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(pin, GPIO.LOW)

GPIO.cleanup()
print("Done.")
EOF
```

If colors look inverted (LED stays on when it should be off), your LED
is common anode — select option 2 in the wizard.

---

## Step 5 — Run the system

```bash
cd plr_system/
python3 main.py
```

### Terminal control modes

The terminal wizard always offers these three hardware-control modes:

1. **Dual** — both LEDs flash simultaneously with the same color. Three
   flashes are collected by default; each flash has its own RGB hex color and
   duration. More flashes may be added, followed by one configurable
   all-LEDs-off gap between flashes.
2. **Left → Right** — LED 1 flashes, the configured inner pause elapses, then
   LED 2 flashes. The default is three rounds, and the wizard allows changing
   the round count, color, flash duration, inner pause, and gap between rounds.
3. **Right → Left** — the same sequential schedule in reverse: LED 2, inner
   pause, then LED 1.

The mobile API's optional `active_led` field is not emitted by the terminal
wizard, so both LEDs always participate in either terminal sequential mode.

### What you will see

```
╔════════════════════════════════════════════════════════╗
║        PLR Pupillometry — Session Configuration        ║
╚════════════════════════════════════════════════════════╝

──────────────────────────────────────────────────────────
  GPIO Pin Assignment  (BCM numbering)
──────────────────────────────────────────────────────────
  Left LED  = LED 1    Right LED = LED 2

  LED 1  (Left ):
    R pin  [2]: 17
    G pin  [2]: 27
    B pin  [2]: 22

  LED 2  (Right):
    R pin  [2]: 23
    G pin  [2]: 24
    B pin  [2]: 25

──────────────────────────────────────────────────────────
  LED Wiring Type
──────────────────────────────────────────────────────────
  Common cathode : common pin → GND   (most common)
  Common anode   : common pin → 3.3V/5V

  LED type  [1 = common cathode,  2 = common anode]  [1]: 1

──────────────────────────────────────────────────────────
  Flash Mode
──────────────────────────────────────────────────────────
  1.  Dual          (both LEDs same color, simultaneously)
  2.  Left → Right  (left eye first, pause, right eye)
  3.  Right → Left  (right eye first, pause, left eye)

  Select mode  [1]: 2
```

---

## Testing without the model (Mock Mode)

Real bundled ONNX inference is the default. For an orchestration-only test:

```bash
export PLR_MOCK_MODE=1
```

To disable mock mode and restore real inference:

```bash
export PLR_MOCK_MODE=0
```

Mock mode produces simulated pixel-diameter curves. It does not validate the
model, crop alignment, or measurement accuracy.

---

## Testing each stage independently

### Test GPIO only (no camera, no model)
```bash
python3 - << 'EOF'
from led_controller import LedController
import time

config = {
    "common_anode": False,
    "led1_r": 17, "led1_g": 27, "led1_b": 22,
    "led2_r": 23, "led2_g": 24, "led2_b": 25,
}

leds = LedController(config)

print("LED 1 — Red 500ms")
leds.led1.flash("#FF0000", 500)
time.sleep(0.3)

print("LED 1 — Green 500ms")
leds.led1.flash("#00FF00", 500)
time.sleep(0.3)

print("LED 2 — Blue 500ms")
leds.led2.flash("#0000FF", 500)
time.sleep(0.3)

print("Both — White 500ms")
import threading
t1 = threading.Thread(target=lambda: leds.led1.flash("#FFFFFF", 500))
t2 = threading.Thread(target=lambda: leds.led2.flash("#FFFFFF", 500))
t1.start(); t2.start(); t1.join(); t2.join()

leds.cleanup()
print("Done.")
EOF
```

### Test camera only (no GPIO, no model)
```bash
python3 - << 'EOF'
from camera_controller import CameraController
import time

cam = CameraController(device_index=0, output_dir="recordings")
cam.start()
print("Recording for 5 seconds...")
time.sleep(5)
cam.stop()
print("Check recordings/ for the output file.")
EOF
```

### Test segmenter on an existing recording
```bash
python3 - << 'EOF'
import queue, glob, os
from camera_controller import CameraController
from segmenter import Segmenter

# Find the most recent full recording
recordings = sorted(glob.glob("recordings/full_recording_*.avi"))
if not recordings:
    print("No recordings found. Run the camera test first.")
    raise SystemExit(1)

recording = recordings[-1]
print(f"Using: {recording}")

# Build fake queues with made-up timestamps
# (pretend 2 flashes per LED at known offsets in the video)
cam = CameraController(device_index=0, output_dir="recordings")
cam.recording_start_time = 0.0          # treat video start as t=0
cam.output_path = recording

hq1, tq1 = queue.Queue(), queue.Queue()
hq2, tq2 = queue.Queue(), queue.Queue()

hq1.put("#FF0000");  tq1.put((1.0, 1.3))   # LED1 flash 1: 1.0s–1.3s
hq1.put("#00FF00");  tq1.put((3.0, 3.5))   # LED1 flash 2: 3.0s–3.5s
hq2.put("#0000FF");  tq2.put((2.0, 2.4))   # LED2 flash 1: 2.0s–2.4s
hq2.put("#FFFF00");  tq2.put((4.0, 4.3))   # LED2 flash 2: 4.0s–4.3s

seg = Segmenter(
    recording_path=recording,
    camera_controller=cam,
    hex_queue_1=hq1, ts_queue_1=tq1,
    hex_queue_2=hq2, ts_queue_2=tq2,
    clips_root="clips",
    pre_flash_s=1.0,
    post_flash_s=3.0,
)

paths = seg.run()
print("\nClips created:")
for led, clip_list in paths.items():
    for p in clip_list:
        print(f"  LED {led}: {p}")
EOF
```

---

## Expected output folder structure after a full run

```
plr_system/
├── recordings/
│   └── full_recording_20250610_143022.avi   ← full continuous video
│
├── clips/
│   ├── led1/
│   │   ├── FF0000.mp4     ← red flash clip, LED 1
│   │   ├── 00FF00.mp4     ← green flash clip, LED 1
│   │   └── 0000FF.mp4     ← blue flash clip, LED 1
│   └── led2/
│       ├── FF0000.mp4
│       └── FFFF00.mp4
│
└── results/
    ├── FF0000.json             ← per-clip model result (or mock result)
    ├── 00FF00.json
    └── session_summary.json    ← all results combined
```

---

## Common problems

| Symptom | Fix |
|---|---|
| `RuntimeError: No access to /dev/mem` | Run with `sudo python3 main.py` or add user to `gpio` group: `sudo usermod -aG gpio $USER` |
| Camera opens but records black frames | IR camera may need illuminator on. Check `v4l2-ctl --list-devices` |
| `cv2` not found | `sudo apt install python3-opencv` |
| ffmpeg clips are 0 bytes | Flash timestamps fell outside video duration. Check pre-roll in camera_controller.py |
| LED wrong color / stays on | Flip common anode setting. Select option 2 in wizard instead of 1 |
| `ModuleNotFoundError: RPi.GPIO` | Running on non-Pi machine — the mock GPIO class handles this automatically for GPIO, but camera still needs a real device |

---

## Model mode

The bundled INT8 ONNX models run directly in the backend. Real inference is
the default:

```bash
export PLR_MOCK_MODE=0
```

For a temporary orchestration-only test:

```bash
export PLR_MOCK_MODE=1
```

See the root `README.md` for crop ROI configuration and `TESTING.md` for the
video integration harness.
