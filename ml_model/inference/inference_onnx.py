"""
inference_onnx.py — Raspberry Pi 4 inference using ONNX Runtime
================================================================
Drop-in replacement for inference.py that uses the INT8 ONNX models
produced by export_to_onnx.py instead of PyTorch.

Why this is faster than the PyTorch version on Pi 4
─────────────────────────────────────────────────────
  • ONNX Runtime uses MLAS (Microsoft's ARM NEON-tuned GEMM kernels)
    instead of PyTorch's ATen backend — faster matrix ops on Cortex-A72
  • Static INT8 quantisation covers ALL layers including Conv2d, which
    PyTorch's quantize_dynamic cannot touch
  • PyTorch is not imported at all — saves ~3s startup and ~200 MB RAM
  • ORT's graph optimiser folds BatchNorm into Conv weights at load time
    (do_constant_folding=True was set during export)

Pi 4-specific ORT session settings applied
───────────────────────────────────────────
  • intra_op_num_threads = 4       (all 4 Cortex-A72 cores)
  • inter_op_num_threads = 1       (no cross-op parallelism — no SMT on Pi)
  • execution_mode = SEQUENTIAL    (avoids thread pool overhead for small batches)
  • graph_optimization_level = ORT_ENABLE_ALL
  • CPUExecutionProvider only      (no fallback to CUDA/TensorRT probing)

Directory layout expected
─────────────────────────
project_root/
├── inference_onnx.py
├── model_weights/
│   ├── left_eye_int8.onnx    ← produced by export_to_onnx.py
│   └── right_eye_int8.onnx
└── model_inputs/
    ├── ID_001/
    │   ├── Left/
    │   └── Right/
    └── ...

Output — identical format to inference.py
─────────────────────────────────────────
model_outputs/
└── ID_001/
    ├── left/clip_A.csv    (columns: frame, pred_diameter_px)
    └── right/clip_C.csv

Usage
─────
  python inference_onnx.py
  python inference_onnx.py --batch 4 --threads 4
  python inference_onnx.py --fp32   # use FP32 .onnx instead of INT8

Dependencies (Pi 4)
────────────────────
  pip install onnxruntime opencv-python pandas tqdm
  (torch / torchvision NOT required on the Pi)
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import onnxruntime as ort
except ImportError:
    sys.exit(
        "[ERROR] onnxruntime not installed.\n"
        "Run: pip install onnxruntime\n"
        "Do NOT install onnxruntime-gpu — it does not exist for ARM."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match train.py
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE      = 224
VIDEO_EXTS    = {".mp4", ".avi", ".mov", ".mkv", ".MOV", ".MP4"}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ONNX Runtime session — tuned for Pi 4
# ─────────────────────────────────────────────────────────────────────────────

def create_session(model_path: str, num_threads: int = 4) -> ort.InferenceSession:
    """
    Create an ORT InferenceSession with settings tuned for Pi 4.

    intra_op_num_threads  — parallelism inside one op (matmul, conv).
                            Set to 4 to use all Cortex-A72 cores.
    inter_op_num_threads  — parallelism across independent graph nodes.
                            Kept at 1: Pi 4 has no hyperthreading, and
                            cross-node parallelism adds scheduling overhead
                            that outweighs the benefit at batch sizes ≤ 8.
    ORT_SEQUENTIAL        — run graph nodes in sequence, not in parallel.
                            Reduces thread pool overhead for small batches.
    ORT_ENABLE_ALL        — apply all graph optimisations at session load:
                            constant folding, node fusion, layout transforms.
    CPUExecutionProvider  — explicit list prevents ORT from probing for
                            CUDA/TensorRT/CoreML, which wastes startup time.
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads        = num_threads
    opts.inter_op_num_threads        = 1
    opts.execution_mode              = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level    = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Suppress ORT's verbose initialisation log
    opts.log_severity_level = 3     # 0=VERBOSE … 3=ERROR

    session = ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )

    print(f"  [ort] loaded {Path(model_path).name}")
    print(f"  [ort] input  : {session.get_inputs()[0].name}  "
          f"{session.get_inputs()[0].shape}")
    print(f"  [ort] output : {session.get_outputs()[0].name}  "
          f"{session.get_outputs()[0].shape}")
    return session


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Frame preprocessing — pure OpenCV/NumPy, no PyTorch
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_frame(bgr_frame: np.ndarray) -> np.ndarray:
    """
    BGR numpy frame → normalised (3, 224, 224) float32 array (CHW).

    ORT takes numpy arrays directly — no tensor conversion needed.
    Full pipeline runs in OpenCV/NumPy using ARM NEON intrinsics.

      BGR → RGB → resize 224×224 (bilinear) → /255 → normalise → CHW
    """
    rgb     = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    arr     = resized.astype(np.float32) / 255.0
    arr     = (arr - IMAGENET_MEAN) / IMAGENET_STD      # broadcast over HWC
    arr     = np.transpose(arr, (2, 0, 1))              # HWC → CHW
    # np.ascontiguousarray ensures the array is C-contiguous for ORT
    return np.ascontiguousarray(arr)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Per-clip inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_on_clip(
    clip_path: Path,
    session: ort.InferenceSession,
    batch_size: int = 4,
) -> pd.DataFrame:
    """
    Decode every frame of *clip_path* and run ORT inference in batches.

    Returns a DataFrame with columns: frame (int), pred_diameter_px (float).

    The input node name is read from the session so this works whether the
    ONNX was exported as "input" or with any other name.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {clip_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_name   = session.get_inputs()[0].name

    frame_indices:  list[int]   = []
    pred_diameters: list[float] = []
    batch_arrays:   list[np.ndarray] = []
    batch_idx_buf:  list[int]   = []

    def _flush() -> None:
        if not batch_arrays:
            return
        # Stack into (B, 3, 224, 224) — ORT expects NCHW float32
        batch = np.stack(batch_arrays, axis=0)
        preds = session.run(None, {input_name: batch})[0]   # (B, 1) or (B,)
        preds = preds.reshape(-1)                            # ensure 1-D
        frame_indices.extend(batch_idx_buf)
        pred_diameters.extend(preds.tolist())
        batch_arrays.clear()
        batch_idx_buf.clear()

    pbar = tqdm(
        total=total_frames,
        desc=f"    {clip_path.name}",
        unit="fr",
        leave=False,
        dynamic_ncols=True,
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        batch_arrays.append(preprocess_frame(frame))
        batch_idx_buf.append(frame_idx)
        frame_idx += 1
        pbar.update(1)
        if len(batch_arrays) >= batch_size:
            _flush()

    _flush()
    cap.release()
    pbar.close()

    return (
        pd.DataFrame({"frame": frame_indices, "pred_diameter_px": pred_diameters})
        .sort_values("frame")
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Directory walk & orchestration
# ─────────────────────────────────────────────────────────────────────────────

def find_clips(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix in VIDEO_EXTS)


def run_batch_inference(
    inputs_dir: Path,
    outputs_dir: Path,
    left_session: ort.InferenceSession,
    right_session: ort.InferenceSession,
    batch_size: int,
) -> None:
    subject_dirs = sorted(d for d in inputs_dir.iterdir() if d.is_dir())
    if not subject_dirs:
        print(f"[WARN] No subject folders found in {inputs_dir}")
        return

    total_clips  = 0
    failed_clips: list[str] = []

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        print(f"\n{'─'*60}")
        print(f"  Subject: {subject_id}")
        print(f"{'─'*60}")

        for eye_label, session in [("Left", left_session), ("Right", right_session)]:
            eye_input_dir = subject_dir / eye_label
            if not eye_input_dir.is_dir():
                print(f"  [SKIP] {eye_label}/ folder not found in {subject_dir}")
                continue

            clips = find_clips(eye_input_dir)
            if not clips:
                print(f"  [SKIP] No video clips found in {eye_input_dir}")
                continue

            eye_output_dir = outputs_dir / subject_id / eye_label.lower()
            eye_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"  [{eye_label}] {len(clips)} clip(s) → {eye_output_dir}")

            for clip_path in clips:
                out_csv = eye_output_dir / (clip_path.stem + ".csv")
                if out_csv.exists():
                    print(f"    [SKIP] {clip_path.name} — CSV already exists")
                    continue

                t0 = time.perf_counter()
                try:
                    df      = run_inference_on_clip(clip_path, session, batch_size)
                    df.to_csv(out_csv, index=False)
                    elapsed = time.perf_counter() - t0
                    fps     = len(df) / elapsed if elapsed > 0 else float("inf")
                    print(f"    ✓ {clip_path.name} → {out_csv.name}"
                          f"  ({len(df)} frames, {elapsed:.1f}s, {fps:.1f} fps)")
                    total_clips += 1
                except Exception as exc:
                    print(f"    ✗ {clip_path.name} — ERROR: {exc}")
                    failed_clips.append(str(clip_path))

    print(f"\n{'═'*60}")
    print(f"  Done.  {total_clips} clip(s) processed successfully.")
    if failed_clips:
        print(f"  Failed ({len(failed_clips)}):")
        for p in failed_clips:
            print(f"    • {p}")
    print(f"{'═'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pupil-diameter ONNX inference — optimised for Raspberry Pi 4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--inputs",  default="model_inputs",
                   help="Root folder containing subject ID sub-folders.")
    p.add_argument("--outputs", default="model_outputs",
                   help="Root folder where CSV predictions are written.")
    p.add_argument("--weights", default="model_weights",
                   help="Folder containing the .onnx model files.")
    p.add_argument("--left_model",  default=None,
                   help="Direct path to left-eye .onnx (overrides --weights).")
    p.add_argument("--right_model", default=None,
                   help="Direct path to right-eye .onnx (overrides --weights).")
    p.add_argument("--fp32", action="store_true",
                   help="Use FP32 .onnx models instead of _int8.onnx.")
    p.add_argument("--batch",   type=int, default=4,
                   help="Frames per ORT forward pass. Keep ≤ 8 on Pi 4.")
    p.add_argument("--threads", type=int, default=4,
                   help="ORT intra-op threads. Pi 4 has 4 cores — do not exceed 4.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Resolve model paths ───────────────────────────────────────────────────
    weights_dir = Path(args.weights)
    suffix      = ".onnx" if args.fp32 else "_int8.onnx"

    left_model_path  = Path(args.left_model)  if args.left_model  else weights_dir / f"left_eye{suffix}"
    right_model_path = Path(args.right_model) if args.right_model else weights_dir / f"right_eye{suffix}"

    for p in (left_model_path, right_model_path):
        if not p.is_file():
            sys.exit(
                f"[ERROR] ONNX model not found: {p.resolve()}\n"
                f"        Run export_to_onnx.py on your PC first, then copy the "
                f"        .onnx files to the Pi's {weights_dir}/ folder."
            )

    inputs_dir  = Path(args.inputs)
    outputs_dir = Path(args.outputs)

    if not inputs_dir.is_dir():
        sys.exit(f"[ERROR] Inputs directory not found: {inputs_dir.resolve()}")

    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ORT sessions ─────────────────────────────────────────────────────
    print(f"\n[INFO] ORT version : {ort.__version__}")
    print(f"[INFO] Providers   : {ort.get_available_providers()}")
    print(f"[INFO] Threads     : intra-op={args.threads}, inter-op=1")
    print(f"[INFO] Batch size  : {args.batch}")
    print(f"[INFO] Model type  : {'FP32' if args.fp32 else 'INT8'}\n")

    print("[INFO] Loading LEFT eye session …")
    left_session  = create_session(str(left_model_path),  num_threads=args.threads)

    print("\n[INFO] Loading RIGHT eye session …")
    right_session = create_session(str(right_model_path), num_threads=args.threads)

    print(f"\n[INFO] Inputs  : {inputs_dir.resolve()}")
    print(f"[INFO] Outputs : {outputs_dir.resolve()}\n")

    # ── Run ───────────────────────────────────────────────────────────────────
    run_batch_inference(
        inputs_dir=inputs_dir,
        outputs_dir=outputs_dir,
        left_session=left_session,
        right_session=right_session,
        batch_size=args.batch,
    )


if __name__ == "__main__":
    main()
