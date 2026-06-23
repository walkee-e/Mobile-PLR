# PLR Pupillometry — Prototype

A pupil light-response (PLR) measurement prototype.

- **Backend** runs on a Raspberry Pi 4. It drives two RGB LEDs (GPIO HIGH/LOW),
  records the eye on an IR Pi Camera Module, segments per-flash clips,
  and runs them through a pupillometry model.
- **Mobile app** is the operator interface. It collects participant
  details, configures the experiment, triggers the session over LAN,
  and displays results.

```
device/
├── backend/         ← runs on the Raspberry Pi 4
│   ├── server.py            (Flask HTTP entry — what the app talks to)
│   ├── main.py              (legacy terminal-wizard entry — still works)
│   ├── config_wizard.py
│   ├── config_adapter.py
│   ├── orchestrator.py
│   ├── led_controller.py
│   ├── camera_controller.py
│   ├── segmenter.py
│   ├── model_caller.py
│   ├── requirements.txt
│   ├── README.md            (hardware-level setup notes)
│   └── INTEGRATION.md       (payload shape, mapping rules)
│
└── mobile/          ← React Native app (operator interface)
    ├── App.js
    ├── api.js               (talks to the Pi over HTTP)
    ├── screens/             (Home, Details, Config, Experiment, Results, Dashboard)
    ├── assets/
    ├── android/  ios/
    └── package.json
```

---

## 1. Hardware

| Item                                | Notes                                                                 |
|-------------------------------------|-----------------------------------------------------------------------|
| Raspberry Pi 4 (any RAM tier)       | No active cooler required — code is tuned to keep thermal load low.   |
| IR Pi Camera Module (CSI ribbon)    | Connected to the camera connector. NoIR / IR-cut variants both work.  |
| 2 × 4-pin RGB LEDs                  | May be common-anode or common-cathode — even one of each is fine.     |
| Resistors                           | One series resistor per LED (≈ 220 Ω on each R/G/B channel works well). |
| Jumper wires + breadboard           |                                                                       |
| Phone or tablet on the same Wi-Fi   | Android dev easiest; iOS works too but needs Xcode.                   |

**Wiring sketch** (BCM pin numbers are placeholders — edit them, see §3):

```
LED 1 (Left eye)            LED 2 (Right eye)
  R → GPIO 17 → 220Ω           R → GPIO 23 → 220Ω
  G → GPIO 27 → 220Ω           G → GPIO 24 → 220Ω
  B → GPIO 22 → 220Ω           B → GPIO 25 → 220Ω
  common → GND  (cathode)      common → GND  (cathode)
           or 3.3V (anode)              or 3.3V (anode)
```

Avoid GPIO 0/1 (I²C) and 14/15 (UART).

---

## 2. Raspberry Pi — initial OS setup

Tested on Raspberry Pi OS Bookworm (64-bit).

```bash
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-picamera2 python3-libcamera \
    ffmpeg

# Enable the camera (Bookworm enables it by default; older OS need raspi-config)
ls /dev/video*       # should list at least one device
libcamera-hello -t 1000   # quick camera sanity check (window pops up briefly)
```

Clone or copy the repository to `~/Documents/Mobile-PLR`, then create a
64-bit Python virtual environment:

```bash
cd ~/Documents/Mobile-PLR
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

`--system-site-packages` lets the virtual environment use the APT-installed
`picamera2` and `libcamera` modules. The requirements install Flask, GPIO,
OpenCV, NumPy/Pandas, and ONNX Runtime for direct INT8 model inference.

---

## 3. Configure GPIO + LED wiring (one-time, per device)

Open [`backend/led_controller.py`](backend/led_controller.py) and edit the
two placeholder blocks near the top to match your wiring:

```python
DEFAULT_CONFIG = {
    "led1_common_anode": False,  # common cathode → GND, HIGH = ON
    "led1_r": 17, "led1_g": 27, "led1_b": 22,    # ← LED 1 (left eye)
    "led2_common_anode": True,   # common anode → 3.3V, LOW = ON
    "led2_r": 23, "led2_g": 24, "led2_b": 25,    # ← LED 2 (right eye)
}
```

The two LEDs may have different wiring — one CA and one CC works fine.

**Quick smoke test** (with the LEDs wired):

```bash
cd ~/device/backend
python3 - << 'EOF'
from led_controller import LedController
import time
leds = LedController({})              # uses DEFAULT_CONFIG
leds.led1.flash("#FF0000", 400, 60)   # red, 400 ms, 60 % intensity
time.sleep(0.3)
leds.led2.flash("#0000FF", 400, 60)   # blue, 400 ms, 60 % intensity
leds.cleanup()
EOF
```

If the colors look inverted (LED is *on* when it should be *off*), flip
that LED's `led1_common_anode` or `led2_common_anode` flag.

The timing-safe controller uses direct GPIO HIGH/LOW output rather than PWM.
The mobile `intensity` value remains in the API for compatibility but is not
applied by this controller.

---

## 4. Start the Pi backend

```bash
cd ~/Documents/Mobile-PLR
source .venv/bin/activate
cd backend
python3 server.py
#  * Serving Flask app …
#  * Running on http://0.0.0.0:5000
```

The server accepts one active session at a time, but remains running during
and after inference. Every experiment is stored under a unique
`sessions/<timestamp>_<participant>/` directory with its recording, clips,
cropped eye videos, prediction CSVs, result JSON, metadata, and `session.log`.

```ini
# /etc/systemd/system/plr.service
[Unit]
Description=PLR Pupillometry HTTP server
After=network-online.target

[Service]
User=pi
WorkingDirectory=/home/pi/Documents/Mobile-PLR/backend
ExecStart=/home/pi/Documents/Mobile-PLR/.venv/bin/python /home/pi/Documents/Mobile-PLR/backend/server.py
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now plr.service
```

Find the Pi's address:

```bash
hostname -I        # e.g. 192.168.1.42
```

---

## 5. Mobile app — first run

Prerequisites on your dev machine:

- Node.js 18+ and either npm or yarn.
- Android Studio with an Android SDK + emulator, **or** a USB-connected
  Android phone with USB debugging enabled.
- (iOS only) macOS with Xcode and CocoaPods.

```bash
cd device/mobile
npm install
# or: yarn install
```

**Point the app at your Pi.** Edit [`mobile/api.js`](mobile/api.js):

```js
export const PI_BASE_URL = 'http://192.168.1.42:5000';   // ← your Pi IP
```

`http://eyezer.local:5000` also works if your network has mDNS.

### Android

```bash
cd device/mobile
npx react-native start                 # leave Metro running in this terminal
# in a second terminal:
npx react-native run-android
```

### iOS

```bash
cd device/mobile/ios
pod install
cd ..
npx react-native run-ios
```

---

## 6. Using the prototype end-to-end

1. **Home** → tap **Start**.
2. **Details** — enter participant name, age, sex.
3. **Config** — choose Dual, Left → Right, or Right → Left and configure
   the mode-specific flash schedule. Tap **Start Experiment**.
   - The phone POSTs the config to `http://<pi>:5000/session`.
   - The Pi runs the flash sequence while recording IR video.
4. **Experiment** — looping background while the app polls `/status`.
   The server remains responsive while it crops eye regions and runs ONNX
   inference.
5. **Results** — the Pi's `session_summary.json` is fetched, rendered
   as per-flash cards, pixel-diameter response curves, and a baseline/min
   comparison chart. It is also **cached locally
   to `Downloads/eyezer/`** on the phone so the Dashboard can show it
   later offline.
6. **Dashboard** (PIN 2580) — browse past sessions stored on the phone.

---

## 7. Troubleshooting

| Symptom                                   | Fix                                                                                  |
|-------------------------------------------|--------------------------------------------------------------------------------------|
| `RuntimeError: No access to /dev/mem`     | `sudo usermod -aG gpio $USER` then re-login, or run server with `sudo`.              |
| Camera opens but frames are black         | IR illuminator off, or wrong sensor. `libcamera-hello -t 2000` should show a preview.|
| LED stays on when it should be off        | Flip that LED's `common_anode` flag in `led_controller.py`.                          |
| App says "Network request failed"         | Wrong `PI_BASE_URL`, phone on a different Wi-Fi, or firewall blocking port 5000.     |
| Cropped video misses the eye              | Adjust `PLR_LEFT_EYE_ROI` / `PLR_RIGHT_EYE_ROI`; inspect the session's `cropped/` videos. |
| Inference fails                           | Inspect the session's `session.log` and per-clip error in `session_summary.json`. |
| Pi gets hot fast                          | Lower `fps` / `bitrate` in `camera_controller.py`, or shorten the session.           |
| `ModuleNotFoundError: picamera2`          | `sudo apt install -y python3-picamera2 python3-libcamera` (not pip).                 |
| ffmpeg clips are 0 bytes                  | Flash timestamps fell outside the recording window — increase pre/post-roll.         |

---

## 8. Eye cropping, inference, and mock mode

The model training code performs no crop: every training frame is already an
eye region. The application therefore creates an eye-only video before calling
the bundled INT8 ONNX inference script.

Default normalized camera ROIs:

```bash
export PLR_LEFT_EYE_ROI="0,0,0.5,1"
export PLR_RIGHT_EYE_ROI="0.5,0,0.5,1"
```

The format is `x,y,width,height`, each in the range 0–1. These defaults split
the image in half and must be verified against the physical camera orientation.
The exact ROI is recorded in each result.

Real ONNX inference is the default. To explicitly disable mock mode:

```bash
export PLR_MOCK_MODE=0
```

To temporarily enable simulated predictions for an orchestration test:

```bash
export PLR_MOCK_MODE=1
```

Results remain in pixels. No pixel-to-mm conversion is applied.

The initial analysis windows are one second before flash onset and three
seconds after it. The backend enforces at least a three-second inter-flash gap
to avoid the next flash contaminating the response window. Initial formulas:

- baseline: median of the smoothed pre-flash predictions
- minimum: minimum smoothed prediction after flash onset
- amplitude: baseline minus minimum
- latency: flash onset to the minimum

These formulas are an initial implementation and require clinical validation.

---

## 9. File map cheat-sheet

| Path                                    | Purpose                                              |
|-----------------------------------------|------------------------------------------------------|
| `backend/server.py`                     | HTTP entry — talk to this from the app.              |
| `backend/led_controller.py`             | **Edit pins + CA/CC here.** Polarity-safe GPIO driver. |
| `backend/camera_controller.py`          | picamera2 IR camera wrapper.                         |
| `backend/orchestrator.py`               | Flash sequencer.                                     |
| `backend/segmenter.py`                  | Cuts per-flash clips via ffmpeg.                     |
| `backend/eye_cropper.py`                | Creates configured eye-region videos for inference.  |
| `backend/model_caller.py`               | Runs bundled INT8 ONNX models and builds results.     |
| `backend/plr_metrics.py`                | Initial aggregate metrics + per-frame response data.  |
| `backend/config_adapter.py`             | Mobile payload → backend config translation.         |
| `mobile/api.js`                         | **Edit `PI_BASE_URL` here.**                         |
| `mobile/screens/ConfigScreen.js`        | Posts session to Pi.                                 |
| `mobile/screens/ExperimentScreen.js`    | Polls `/status` + `/results`.                        |
| `mobile/screens/ResultsScreen.js`       | Renders per-flash results, caches to phone storage.  |

See [`TESTING.md`](TESTING.md) for metric tests and the video integration
harness that can be used once a validation video is available.
