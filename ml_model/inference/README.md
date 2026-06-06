# Pupil Diameter Inference — Raspberry Pi 4

Batch inference pipeline that runs two fine-tuned ResNet18 models (one per
eye) over a folder of video clips and writes per-clip pupil diameter
predictions as CSV files. Fully optimised to run on the **Raspberry Pi 4
CPU** — no GPU required.

Two inference backends are provided:

| Script | Backend | Requires on Pi | Speed (est.) |
|---|---|---|---|
| `inference.py` | PyTorch + dynamic INT8 (Linear only) | torch, torchvision | ~8–15 fps |
| `inference_onnx.py` | ONNX Runtime + static INT8 (all layers) | onnxruntime only | ~20–35 fps |

**Recommendation: use the ONNX path.** It is 2–3× faster because static INT8
quantisation covers Conv2d layers (which dominate ResNet18 compute), and ONNX
Runtime's ARM-tuned MLAS kernels outperform PyTorch's ATen backend on
Cortex-A72. PyTorch does not need to be installed on the Pi at all.

---

## Project layout

```
project_root/
├── export_to_onnx.py      ← run once on PC to convert .pth → .onnx
├── inference.py           ← PyTorch inference (fallback / comparison)
├── inference_onnx.py      ← ONNX Runtime inference (recommended for Pi 4)
│
├── model_weights/
│   ├── left_eye.pth           ← original PyTorch checkpoint
│   ├── right_eye.pth
│   ├── left_eye_int8.onnx     ← produced by export_to_onnx.py (copy to Pi)
│   └── right_eye_int8.onnx
│
└── model_inputs/
    ├── ID_001/
    │   ├── Left/
    │   │   ├── clip_A.mp4
    │   │   └── clip_B.mp4
    │   └── Right/
    │       ├── clip_C.mp4
    │       └── clip_D.mp4
    └── ID_002/
        └── ...
```

After running, outputs appear at:

```
project_root/
└── model_outputs/
    ├── ID_001/
    │   ├── left/
    │   │   ├── clip_A.csv
    │   │   └── clip_B.csv
    │   └── right/
    │       ├── clip_C.csv
    │       └── clip_D.csv
    └── ID_002/
        └── ...
```

Each CSV has two columns:

| column | description |
|---|---|
| `frame` | Frame index (0-based) |
| `pred_diameter_px` | Predicted pupil diameter in pixels |

---

## Step 1 — Export models on your PC (one-time)

This step runs on your **PC or laptop**, not the Pi. It converts the PyTorch
`.pth` checkpoints to INT8-quantised ONNX models.

### Install PC dependencies

```bash
pip install torch torchvision onnx onnxruntime onnxruntime-tools
```

### Run the export

```bash
python export_to_onnx.py
```

This reads `model_weights/left_eye.pth` and `model_weights/right_eye.pth`
and produces:

```
model_weights/
├── left_eye.onnx          # FP32 intermediate (can delete)
├── left_eye_int8.onnx     # ← copy this to the Pi
├── right_eye.onnx
└── right_eye_int8.onnx    # ← copy this to the Pi
```

### Export options

```
python export_to_onnx.py [OPTIONS]

  --weights PATH              Folder with .pth files. Default: model_weights
  --calibration_batches INT   Synthetic calibration batches for INT8 (default: 20)
                              More = better scale estimates, slower export.
  --skip_quantize             Export FP32 ONNX only, skip INT8 step.
```

### Copy models to the Pi

```bash
scp model_weights/left_eye_int8.onnx  pi@<PI_IP>:~/project_root/model_weights/
scp model_weights/right_eye_int8.onnx pi@<PI_IP>:~/project_root/model_weights/
```

---

## Step 2 — Set up the Raspberry Pi 4

### OS requirement

Use **Raspberry Pi OS 64-bit** (Bookworm or later). The `onnxruntime`
`aarch64` wheel is only published for 64-bit ARM. The 32-bit OS will not work.

Check your architecture:
```bash
uname -m    # must print: aarch64
```

### Install system dependencies

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libopencv-dev
```

### Create a virtual environment

```bash
cd ~/project_root
python3 -m venv venv
source venv/bin/activate
```

### Install Python packages (Pi — ONNX path)

```bash
pip install --upgrade pip
pip install onnxruntime opencv-python pandas tqdm
```

> **Important:** Install `onnxruntime`, not `onnxruntime-gpu`.
> The GPU variant does not exist for ARM and will fail to install.

### Install Python packages (Pi — PyTorch fallback only)

Only needed if you want to use `inference.py` instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python pandas tqdm
```

---

## Step 3 — Run inference on the Pi

Activate your virtual environment first:

```bash
source venv/bin/activate
```

### ONNX Runtime inference (recommended)

```bash
python inference_onnx.py
```

### All options

```
python inference_onnx.py [OPTIONS]

  --inputs  PATH    Input folder with subject ID sub-folders. Default: model_inputs
  --outputs PATH    Output folder for CSVs. Default: model_outputs
  --weights PATH    Folder containing .onnx files. Default: model_weights
  --left_model  PATH   Direct path to left-eye .onnx (overrides --weights).
  --right_model PATH   Direct path to right-eye .onnx (overrides --weights).
  --fp32            Use FP32 .onnx instead of _int8.onnx.
  --batch INT       Frames per forward pass. Default: 4. Keep ≤ 8 on Pi 4.
  --threads INT     ORT intra-op threads. Default: 4. Do not exceed 4 on Pi 4.
```

### Examples

```bash
# Custom paths
python inference_onnx.py --inputs /data/eyes --outputs /data/results

# Use less RAM (if other processes are running)
python inference_onnx.py --batch 2

# Use FP32 models (no INT8 — for accuracy comparison)
python inference_onnx.py --fp32

# Point directly to model files
python inference_onnx.py \
  --left_model  /models/left_eye_int8.onnx \
  --right_model /models/right_eye_int8.onnx
```

### PyTorch inference (fallback)

```bash
python inference.py
```

```
  --inputs  PATH    Default: model_inputs
  --outputs PATH    Default: model_outputs
  --weights PATH    Folder with .pth files. Default: model_weights
  --left_weights  PATH
  --right_weights PATH
  --batch INT       Default: 4
  --threads INT     Default: 4
  --no_quantize     Disable INT8 quantisation of Linear layers.
```

---

## Pi 4 optimisations explained

### ONNX Runtime (`inference_onnx.py`)

**Static INT8 quantisation of all layers**
The export script quantises every layer — including all Conv2d layers — to
INT8 using ORT's static quantisation with calibration. PyTorch's
`quantize_dynamic` cannot quantise Conv2d. Since Conv2d accounts for the
majority of ResNet18's compute, this is the largest single speedup.

**MLAS kernels**
ONNX Runtime uses Microsoft's MLAS library for matrix operations, which
includes hand-tuned ARM NEON assembly for Cortex-A72. It outperforms
PyTorch's ATen backend for CPU inference on ARM.

**No PyTorch import**
PyTorch is not imported at all on the Pi. This saves ~3 seconds of startup
time and ~200 MB of RAM.

**ORT session settings**
```python
opts.intra_op_num_threads     = 4    # all 4 Cortex-A72 cores
opts.inter_op_num_threads     = 1    # no cross-op parallelism (no SMT on Pi)
opts.execution_mode           = ORT_SEQUENTIAL   # lower thread overhead at small batch sizes
opts.graph_optimization_level = ORT_ENABLE_ALL   # constant folding, node fusion
providers = ["CPUExecutionProvider"]             # no CUDA/TensorRT probing
```

**Constant folding at export**
`do_constant_folding=True` during ONNX export folds BatchNorm statistics into
the preceding Conv2d weight tensors. This eliminates BatchNorm as a separate
runtime op — reducing the number of graph nodes ORT executes per frame.

### Both scripts

**OpenCV preprocessing (no PIL)**
Every frame goes through `cv2.cvtColor` → `cv2.resize` (bilinear) →
NumPy normalise → CHW transpose. OpenCV's resize on ARM uses NEON intrinsics
and is roughly 3× faster than PIL's bicubic resize.

**Sequential single-process frame decode**
No multiprocessing `DataLoader` workers. Forking processes on ARM Linux has
high startup cost and unpredictable memory usage. Frames are decoded in a
plain `while cap.read()` loop.

**Batch size 4 default**
Each batch of 4 frames at 224×224 float32 is ~2.4 MB. Keeping batch size
small ensures peak RAM stays well within 4 GB even for high-resolution clips.

---

## Performance expectations on Pi 4

Approximate figures — actual speed depends on clip resolution and SD-card
read speed.

| Backend | Batch | Approx. throughput |
|---|---|---|
| ONNX INT8 (recommended) | 4 | ~20–35 fps |
| ONNX FP32 | 4 | ~10–18 fps |
| PyTorch INT8 Linear | 4 | ~8–15 fps |
| PyTorch FP32 | 4 | ~4–8 fps |

For a 30-second clip at 30 fps (900 frames), the ONNX INT8 path takes
roughly **25–45 seconds**.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'onnxruntime'`**
Activate the virtual environment: `source venv/bin/activate`, then
`pip install onnxruntime`.

**`ERROR: ONNX model not found`**
Run `export_to_onnx.py` on your PC and copy the `_int8.onnx` files to
`model_weights/` on the Pi.

**`onnxruntime-gpu` install fails**
The GPU variant does not exist for ARM. Install `onnxruntime` (CPU only).

**`uname -m` shows `armv7l` instead of `aarch64`**
You are running 32-bit Raspberry Pi OS. Re-flash with the 64-bit image from
https://www.raspberrypi.com/software/

**`Cannot open video`**
Ensure the clip is not corrupted and that OpenCV was installed with codec
support (`libopencv-dev` via apt provides this).

**Script killed / out of memory**
Reduce batch size: `--batch 2`. Check available RAM: `free -h`.

**Predictions differ between ONNX and PyTorch**
A small numerical difference (< 0.5 px) is expected from INT8 rounding.
Run `inference_onnx.py --fp32` to compare against FP32 ONNX; if that matches
PyTorch, the INT8 quantisation calibration is the source of any larger
discrepancy. Re-run `export_to_onnx.py --calibration_batches 50` for better
calibration.
