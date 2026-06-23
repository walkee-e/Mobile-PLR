# Eyezer Mobile ↔ Pi Backend Integration

End-to-end flow:

```
ConfigScreen (RN)  ──POST /session──►  server.py
                                         │
                                         ▼
                              Orchestrator → IR cam + LEDs
                                         │
                                         ▼
                                    Segmenter
                                         │
                                         ▼
                                    EyeCropper
                                         │
                                         ▼
                              INT8 ONNX ModelCaller
                                         │
                           sessions/<id>/results/session_summary.json
ResultsScreen (RN) ──GET /results───────┘   (+ cached to Downloads/eyezer/)
```

## Pi side — running the server

```bash
cd ~/Documents/Mobile-PLR
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
sudo apt install -y python3-picamera2 python3-libcamera ffmpeg
cd backend
python3 server.py             # listens on 0.0.0.0:5000
```

The server permits one active session at a time and remains running during
inference and after results are written.

## GPIO + wiring — set per device

Pins and common-anode flags are placeholders in `led_controller.py`:

```python
DEFAULT_CONFIG = {
    "led1_common_anode": False,
    "led1_r": 17, "led1_g": 27, "led1_b": 22,    # ← edit
    "led2_common_anode": True,
    "led2_r": 23, "led2_g": 24, "led2_b": 25,    # ← edit
}
```

Either edit the file or pass overrides in the mobile payload:

```json
{
  "gpio": {"led1_r": 5, "led1_g": 6, ...},
  "common_anode": {"led1": true, "led2": false}
}
```

Two LEDs of mixed wiring (one CA, one CC) are supported — set the dict form.

## IR Pi Camera Module (RPi 4)

`camera_controller.py` uses **picamera2** with the H.264 hardware encoder
(2 Mbps, 640×480 @ 24 fps). The camera is opened only during a session and
fully released the moment the flash sequence ends — important on a Pi 4
without a cooler because the IR module heats up quickly.

Falls back to a mock file on dev machines without picamera2.

## Mobile payload shape

```json
{
  "participant": {"name":"...", "age":30, "sex":"M"},
  "controlMode": "dual" | "left_to_right" | "right_to_left",
  "schedule": {}
}
```

Translation rules (`config_adapter.py`):

| Mobile field | Backend mapping |
|---|---|
| `controlMode = dual` | Both LEDs flash simultaneously; `schedule.flashes` carries each RGB hex and duration |
| `controlMode = left_to_right` | LED 1, inner pause, LED 2 for each round |
| `controlMode = right_to_left` | LED 2, inner pause, LED 1 for each round |
| `schedule.gap` | Break between flashes/rounds; minimum 3 seconds for analysis |
| `schedule.innerPause` | Pause between eyes in sequential modes |

Dual schedule example:

```json
{
  "controlMode": "dual",
  "schedule": {
    "flashes": [
      {"hex": "#FF0000", "duration": 1.0},
      {"hex": "#00FF00", "duration": 1.0},
      {"hex": "#0000FF", "duration": 1.0}
    ],
    "gap": 3
  }
}
```

Sequential schedule example:

```json
{
  "controlMode": "left_to_right",
  "schedule": {
    "rounds": 3,
    "hex": "#FFFFFF",
    "duration": 1.0,
    "innerPause": 1.0,
    "gap": 3
  }
}
```

## Mobile — set the Pi address

Edit `eyezer/mobile/api.js`:

```js
export const PI_BASE_URL = 'http://eyezer.local:5000';   // ← your Pi
```

If mDNS isn't available on your LAN, use the Pi's IP directly.

## Pi 4 thermal budget (single IR cam, no cooler)

- LEDs use direct GPIO HIGH/LOW output with safe OFF initialization.
- Camera resolution capped at 640×480 @ 24 fps with H.264 HW encoder.
- Camera and GPIO are released as soon as flashes finish.
- ONNX Runtime uses four inference threads by default.

## Model crop and result contract

The trained model does not crop an eye; it expects the complete frame to
already be an eye region. The backend therefore crops each segmented clip:

- LED 1 uses `PLR_LEFT_EYE_ROI` and the left-eye ONNX model.
- LED 2 uses `PLR_RIGHT_EYE_ROI` and the right-eye ONNX model.

ROIs use normalized `x,y,width,height` values. Defaults split the camera frame
into left and right halves. Inspect each session's `cropped/` videos and tune
the ROIs for the physical camera orientation.

Results use pixels and include aggregate metrics plus the full per-frame
diameter series. See the root README for the initial formulas and mock-mode
environment variables.
