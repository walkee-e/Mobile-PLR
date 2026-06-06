"""
export_to_onnx.py — Run this ONCE on your PC/laptop, not on the Pi.
=====================================================================
Converts left_eye.pth and right_eye.pth to INT8-quantised ONNX models
ready for fast inference on the Raspberry Pi 4.

What this script produces
──────────────────────────
  model_weights/
    left_eye.onnx        ← FP32 ONNX (intermediate, can delete after)
    left_eye_int8.onnx   ← INT8 static-quantised  (copy this to the Pi)
    right_eye.onnx
    right_eye_int8.onnx

Steps performed
───────────────
  1. Load each .pth checkpoint (same logic as inference.py)
  2. Export to ONNX (opset 17, dynamic batch axis)
  3. Run ONNX Runtime's static INT8 quantisation with a synthetic
     calibration dataset (224×224 random frames, matching inference input)
     Static quantisation calibrates Conv2d scales — something PyTorch's
     quantize_dynamic cannot do — giving the largest speed-up on ARM.

Usage
─────
  pip install torch torchvision onnx onnxruntime onnxruntime-tools
  python export_to_onnx.py
  python export_to_onnx.py --weights model_weights --calibration_batches 20

Dependencies (PC only — not needed on the Pi)
─────────────────────────────────────────────
  torch torchvision onnx onnxruntime onnxruntime-tools
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# onnxruntime quantisation — PC only
try:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
except ImportError:
    sys.exit(
        "[ERROR] onnxruntime quantisation tools not found.\n"
        "Run: pip install onnxruntime onnxruntime-tools"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match train.py
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE      = 224


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Model — identical to inference.py
# ─────────────────────────────────────────────────────────────────────────────

def build_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    print(f"  [weights] loading {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

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


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ONNX export
# ─────────────────────────────────────────────────────────────────────────────

def export_onnx(model: nn.Module, output_path: str) -> None:
    """
    Export *model* to ONNX with a dynamic batch axis.

    opset 17: stable, fully supported by onnxruntime ≥ 1.15 on aarch64.
    Dynamic batch axis lets us run any batch size on the Pi without
    re-exporting.
    """
    model.eval()
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)   # (1, C, H, W)

    print(f"  [onnx] exporting → {output_path}")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            output_path,
            opset_version=18,
            input_names=["input"],
            output_names=["pred_diameter"],
            dynamic_axes={
                "input":         {0: "batch_size"},
                "pred_diameter": {0: "batch_size"},
            },
            do_constant_folding=True,   # folds BatchNorm into Conv weights
        )
    print(f"  [onnx] export complete: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Calibration data reader (synthetic — no real clips needed on PC)
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticCalibrationReader(CalibrationDataReader):
    """
    Generates random normalised frames to calibrate INT8 quantisation.

    Static quantisation needs representative data to compute per-channel
    activation scales for Conv2d layers.  Synthetic data is sufficient here
    because:
      • We are quantising activations of a pretrained visual backbone
      • The distribution of intermediate activations is dominated by the
        weight distribution, not the specific input content
      • Real eye-crop data would be preferable but requires the clips on
        the PC; synthetic removes that dependency without meaningful
        accuracy loss for regression tasks

    If you have clips available on the PC, replace this with a reader that
    loads real frames for better calibration accuracy.
    """

    def __init__(self, n_batches: int = 20, batch_size: int = 4):
        self.n_batches  = n_batches
        self.batch_size = batch_size
        self._idx       = 0

    def get_next(self) -> dict | None:
        if self._idx >= self.n_batches:
            return None
        self._idx += 1
        # Random image in the normalised input space of the model
        imgs = np.random.randn(
            self.batch_size, 3, IMG_SIZE, IMG_SIZE
        ).astype(np.float32)
        return {"input": imgs}

    def rewind(self) -> None:
        self._idx = 0


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Static INT8 quantisation
# ─────────────────────────────────────────────────────────────────────────────

def quantize_onnx(fp32_path: str, int8_path: str, n_calibration_batches: int) -> None:
    """
    Run ONNX Runtime static INT8 quantisation on *fp32_path*.

    quant_pre_process is intentionally skipped — it calls SymbolicShapeInference
    internally regardless of flags on current onnxruntime versions, and that
    step fails on graphs produced by PyTorch 2.4+'s dynamo exporter.
    ResNet18 has a simple static graph so quantize_static works correctly
    directly on the raw exported ONNX without pre-processing.

    QDQ format (QuantizeLinear / DequantizeLinear nodes) is best supported
    by ORT's CPUExecutionProvider on ARM.
    QuantType.QInt8 weights + QUInt8 activations is the standard CPU pairing.
    """
    print(f"  [quant] calibrating and quantising → {int8_path}")
    calibration_reader = SyntheticCalibrationReader(
        n_batches=n_calibration_batches,
        batch_size=4,
    )

    quantize_static(
        model_input=fp32_path,
        model_output=int8_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    print(f"  [quant] INT8 model saved → {int8_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export .pth checkpoints → INT8 ONNX for Raspberry Pi 4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights", default="model_weights",
                   help="Folder containing left_eye.pth and right_eye.pth.")
    p.add_argument("--calibration_batches", type=int, default=20,
                   help="Number of synthetic calibration batches (batch_size=4 each). "
                        "More batches = better scale estimates, but slower export.")
    p.add_argument("--skip_quantize", action="store_true",
                   help="Export FP32 ONNX only — skip INT8 quantisation step.")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    w_dir  = Path(args.weights)

    checkpoints = [
        ("left_eye",  w_dir / "left_eye.pth"),
        ("right_eye", w_dir / "right_eye.pth"),
    ]

    for name, ckpt_path in checkpoints:
        if not ckpt_path.is_file():
            sys.exit(f"[ERROR] Checkpoint not found: {ckpt_path.resolve()}")

        fp32_path = str(w_dir / f"{name}.onnx")
        int8_path = str(w_dir / f"{name}_int8.onnx")

        print(f"\n{'═'*60}")
        print(f"  Processing: {name}")
        print(f"{'═'*60}")

        # Load PyTorch model
        model = build_model()
        model = load_checkpoint(model, str(ckpt_path))
        model.eval()

        # Export FP32 ONNX
        export_onnx(model, fp32_path)

        # Quantise to INT8
        if not args.skip_quantize:
            quantize_onnx(fp32_path, int8_path, args.calibration_batches)
        else:
            print("  [quant] skipped (--skip_quantize)")

    print(f"\n{'═'*60}")
    print("  Export complete.")
    if not args.skip_quantize:
        print("  Copy these files to the Pi's model_weights/ folder:")
        for name, _ in checkpoints:
            print(f"    • {name}_int8.onnx")
    else:
        print("  Copy these files to the Pi's model_weights/ folder:")
        for name, _ in checkpoints:
            print(f"    • {name}.onnx")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()