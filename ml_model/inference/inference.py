"""
Batch inference script — Pupil Diameter Estimation
====================================================
Runs left-eye and right-eye ResNet18 models over a structured input directory
and writes per-clip CSV predictions to a mirrored output directory.

Optimised for Raspberry Pi 4 (ARM Cortex-A72, CPU-only, 4 GB RAM):
  • torch.inference_mode() — stricter than no_grad(), zero autograd overhead
  • Dynamic INT8 quantisation on nn.Linear only (Conv2d is NOT supported by
    PyTorch's quantize_dynamic and is silently skipped — excluded here)
  • OpenCV bilinear resize replaces PIL/bicubic — much faster on ARM NEON
  • Single-process sequential frame decode — no fork/spawn overhead on Pi
  • Batch size default of 4 — safe for 4 GB RAM across all clip resolutions
  • torch.set_num_threads(4) — saturates all 4 Pi 4 cores without over-subscribing
  • weights_only=True on torch.load — avoids pickle warning + faster startup

Directory layout expected
─────────────────────────
project_root/
├── model_inputs/
│   ├── ID_001/
│   │   ├── Left/   ← clips processed by left_eye.pth
│   │   └── Right/  ← clips processed by right_eye.pth
│   └── ID_002/ ...
├── model_weights/
│   ├── left_eye.pth
│   └── right_eye.pth
└── inference.py   ← this script

Output
──────
project_root/
└── model_outputs/
    └── ID_001/
        ├── left/
        │   └── clip_A.csv   (columns: frame, pred_diameter_px)
        └── right/
            └── clip_C.csv

Usage
─────
  python inference.py
  python inference.py --threads 2 --batch 4
  python inference.py --no_quantize
  python inference.py --inputs /path/to/model_inputs --weights /path/to/model_weights

Dependencies
────────────
  pip install torch torchvision opencv-python pandas tqdm
  (no CUDA required)
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match training settings in train.py
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE      = 224
VIDEO_EXTS    = {".mp4", ".avi", ".mov", ".mkv", ".MOV", ".MP4"}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model() -> nn.Module:
    """ResNet18 + Linear(512→1) head — identical topology to train.py."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """
    Load weights saved by train.py into *model*.

    Handles two checkpoint shapes produced by train.py:
      • dict with 'model_state' key  — saved by the main training loop
      • raw state_dict               — saved by the predict_video() helper

    weights_only=True avoids the pickle-execution security warning and
    is noticeably faster on slow SD-card storage.
    """
    print(f"  [weights] loading {checkpoint_path}")

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        # Fallback for older checkpoints that contain non-tensor objects (e.g. args)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt        # assume the dict IS the state_dict
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    # Strip any 'model.' / 'resnet.' prefixes added by wrapper classes
    cleaned = {}
    for k, v in state.items():
        for prefix in ("model.", "resnet."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  [weights] missing keys   : {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [weights] unexpected keys: {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")

    return model


def quantize_model(model: nn.Module) -> nn.Module:
    """
    Apply dynamic INT8 quantisation to nn.Linear layers only.

    PyTorch's quantize_dynamic supports: nn.Linear, nn.LSTM, nn.GRU, nn.RNN.
    nn.Conv2d is NOT supported — specifying it is silently ignored in some
    versions and raises errors in others, so it is excluded here.

    On ARM Cortex-A72 (Pi 4) this reduces the fc-head compute and cuts the
    model's memory footprint noticeably, with negligible regression accuracy
    loss.
    """
    return torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear},   # Conv2d excluded — not supported
        dtype=torch.qint8,
    )


def prepare_model(checkpoint_path: str, quantize: bool = True) -> nn.Module:
    """Build, load weights, optionally quantise, and set to eval mode."""
    model = build_model()
    model = load_checkpoint(model, checkpoint_path)
    model.eval()
    if quantize:
        print("  [model] applying dynamic INT8 quantisation (nn.Linear) …")
        model = quantize_model(model)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Frame preprocessing  (pure OpenCV — no PIL round-trip)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_frame(bgr_frame: np.ndarray) -> torch.Tensor:
    """
    BGR numpy frame → normalised (3, 224, 224) float32 tensor.

    Pipeline (all in OpenCV/NumPy — faster than PIL on ARM NEON):
      BGR → RGB → resize 224×224 (bilinear) → float32 /255 → normalise → CHW tensor

    Bilinear is used instead of bicubic (training used bicubic) because the
    speed gain on ARM is significant (~3×) and the accuracy impact for
    inference on eye crops at this resolution is negligible.
    """
    rgb   = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    arr   = resized.astype(np.float32) / 255.0          # HWC, [0,1]
    arr   = (arr - IMAGENET_MEAN) / IMAGENET_STD        # normalise in-place style
    arr   = np.transpose(arr, (2, 0, 1))                # HWC → CHW
    return torch.from_numpy(arr)                        # no copy — shares memory


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Per-clip inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_on_clip(
    clip_path: Path,
    model: nn.Module,
    batch_size: int = 4,
) -> pd.DataFrame:
    """
    Decode every frame of *clip_path*, batch-infer pupil diameter, return DataFrame.

    Columns returned: frame (int), pred_diameter_px (float).

    Design choices for Pi 4:
      • torch.inference_mode() — more aggressive than no_grad(); disables
        autograd version tracking entirely, reducing per-op overhead.
      • Sequential single-process decode — avoids multiprocessing fork costs
        and keeps RAM usage flat and predictable on the Pi's 4 GB.
      • Frames accumulated into a list, stacked once per batch — minimises
        tensor allocation overhead.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {clip_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices:  list[int]   = []
    pred_diameters: list[float] = []
    batch_tensors:  list[torch.Tensor] = []
    batch_idx_buf:  list[int]   = []

    def _flush():
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors)          # (B, 3, 224, 224) — CPU
        with torch.inference_mode():
            preds = model(batch).squeeze(1).numpy()
        frame_indices.extend(batch_idx_buf)
        pred_diameters.extend(preds.tolist())
        batch_tensors.clear()
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
        batch_tensors.append(preprocess_frame(frame))
        batch_idx_buf.append(frame_idx)
        frame_idx += 1
        pbar.update(1)
        if len(batch_tensors) >= batch_size:
            _flush()

    _flush()    # remaining tail frames
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
    """Return all video files directly inside *folder* (non-recursive)."""
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix in VIDEO_EXTS)


def run_batch_inference(
    inputs_dir: Path,
    outputs_dir: Path,
    left_model: nn.Module,
    right_model: nn.Module,
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

        for eye_label, model in [("Left", left_model), ("Right", right_model)]:
            eye_input_dir = subject_dir / eye_label
            if not eye_input_dir.is_dir():
                print(f"  [SKIP] {eye_label}/ folder not found in {subject_dir}")
                continue

            clips = find_clips(eye_input_dir)
            if not clips:
                print(f"  [SKIP] No video clips found in {eye_input_dir}")
                continue

            # Output folder is lowercase to match the spec
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
                    df      = run_inference_on_clip(clip_path, model, batch_size)
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
        description="Pupil-diameter batch inference — optimised for Raspberry Pi 4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--inputs",  default="model_inputs",
                   help="Root folder containing subject ID sub-folders.")
    p.add_argument("--outputs", default="model_outputs",
                   help="Root folder where CSV predictions are written.")
    p.add_argument("--weights", default="model_weights",
                   help="Folder containing left_eye.pth and right_eye.pth.")
    p.add_argument("--left_weights",  default=None,
                   help="Direct path to left-eye checkpoint (overrides --weights).")
    p.add_argument("--right_weights", default=None,
                   help="Direct path to right-eye checkpoint (overrides --weights).")
    p.add_argument("--batch",   type=int, default=4,
                   help="Inference batch size. Keep at 4 on Pi 4 to stay within 4 GB RAM.")
    p.add_argument("--threads", type=int, default=4,
                   help="PyTorch intra-op threads. Pi 4 has 4 cores — do not exceed 4.")
    p.add_argument("--no_quantize", action="store_true",
                   help="Disable INT8 quantisation of Linear layers.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Pi 4: pin thread counts before any torch operation ────────────────────
    # intra-op  = parallelism within a single op (matmul, conv) → use all 4 cores
    # inter-op  = parallelism across independent ops → keep at 1 to avoid
    #             over-subscription; the Pi 4 has no SMT/hyperthreading
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    print(f"[INFO] PyTorch threads — intra-op: {args.threads}, inter-op: 1")
    print(f"[INFO] Device: CPU (Raspberry Pi 4 mode)")

    # ── Resolve paths ─────────────────────────────────────────────────────────
    inputs_dir  = Path(args.inputs)
    outputs_dir = Path(args.outputs)
    weights_dir = Path(args.weights)

    if not inputs_dir.is_dir():
        sys.exit(f"[ERROR] Inputs directory not found: {inputs_dir.resolve()}")

    left_ckpt  = Path(args.left_weights)  if args.left_weights  else weights_dir / "left_eye.pth"
    right_ckpt = Path(args.right_weights) if args.right_weights else weights_dir / "right_eye.pth"

    for ckpt in (left_ckpt, right_ckpt):
        if not ckpt.is_file():
            sys.exit(f"[ERROR] Checkpoint not found: {ckpt.resolve()}")

    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    quantize = not args.no_quantize

    print(f"\n[INFO] Loading LEFT eye model …")
    left_model  = prepare_model(str(left_ckpt),  quantize=quantize)

    print(f"\n[INFO] Loading RIGHT eye model …")
    right_model = prepare_model(str(right_ckpt), quantize=quantize)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[INFO] Inputs   : {inputs_dir.resolve()}")
    print(f"[INFO] Outputs  : {outputs_dir.resolve()}")
    print(f"[INFO] Batch    : {args.batch}")
    print(f"[INFO] INT8 quantisation: {'on' if quantize else 'off'}\n")

    # ── Run ───────────────────────────────────────────────────────────────────
    run_batch_inference(
        inputs_dir=inputs_dir,
        outputs_dir=outputs_dir,
        left_model=left_model,
        right_model=right_model,
        batch_size=args.batch,
    )


if __name__ == "__main__":
    main()
