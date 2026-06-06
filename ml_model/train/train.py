"""
Standalone fine-tuning pipeline for pupil diameter estimation.

Based on the EyeDentify / eyedentify architecture (vijulshah/eyedentify),
adapted to use a lighter ResNet18 backbone:
  - ResNet18 backbone with a single linear regression head
  - Input: 224x224 eye-crop images
  - Target: pupil diameter in pixels (float regression)

Your data layout expected:
  video  : GS_F_07_032-1_left.mp4   (already a cropped eye recording —
                                      each frame is fed directly to the model)
  labels : GS_F_07_032-1_left.csv   (columns: Frame, diameter_px, ...)

Usage:
  # Fine-tune from pretrained weights:
  python finetune_pupil.py \
      --video   GS_F_07_032-1_left.mp4 \
      --csv     GS_F_07_032-1_left.csv \
      --weights path/to/resnet18_weights.pth \
      --output  finetuned_pupil_model.pth

  # Train from ImageNet-pretrained ResNet18 (no custom weights):
  python finetune_pupil.py \
      --video  GS_F_07_032-1_left.mp4 \
      --csv    GS_F_07_032-1_left.csv \
      --output finetuned_pupil_model.pth

Dependencies:
  pip install torch torchvision opencv-python pandas scikit-learn tqdm
"""

import argparse
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# 1.  MODEL  (ResNet18 + simple fc(512→1) regression head)
# ─────────────────────────────────────────────────────────────────────────────

def build_model(weights_path: str | None = None, freeze_backbone: bool = False) -> nn.Module:
    """
    Build ResNet18 with a single Linear(512→1) regression head.

    The pretrained backbone (conv1 through layer4) is loaded from weights_path;
    the fc head is always freshly initialised and trained at a higher LR.

    Args:
        weights_path:    Path to a .pth checkpoint whose keys are prefixed with
                         "resnet." (e.g. the left_eye.pt weights).
                         Pass None to start from ImageNet init only.
        freeze_backbone: If True, only the FC head is trained.
                         Useful when fine-tuning data is very limited (<500 samples).
    """
    # Standard ResNet18 backbone from torchvision
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Fresh single-output regression head — no ImageNet class bottleneck
    in_features = model.fc.in_features   # 512 for ResNet18
    model.fc = nn.Linear(in_features, 1)

    # ── Load pretrained backbone weights ─────────────────────────────────────
    if weights_path is not None:
        print(f"[INFO] Loading pretrained weights from: {weights_path}")
        state = torch.load(weights_path, map_location="cpu")

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Strip known wrapper prefixes so bare ResNet keys remain
        STRIP_PREFIXES = ("model.", "resnet.")
        cleaned = {}
        for k, v in state.items():
            new_k = k
            for prefix in STRIP_PREFIXES:
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
                    break
            cleaned[new_k] = v

        # Drop all head keys from the checkpoint — fc(512→1000) and
        # regression_head(1000→1) are both irrelevant to our new head.
        cleaned = {k: v for k, v in cleaned.items()
                   if not k.startswith("fc.") and not k.startswith("regression_head.")}

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"[INFO] Weights loaded — missing keys: {len(missing)}, unexpected: {len(unexpected)}")
        if missing:
            print(f"       Missing  : {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"       Unexpected: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    else:
        print("[INFO] No pretrained weights supplied — starting from ImageNet init.")

    # ── Optional backbone freeze ─────────────────────────────────────────────
    if freeze_backbone:
        print("[INFO] Freezing backbone — only the FC head will be trained.")
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATASET
# ─────────────────────────────────────────────────────────────────────────────

# The video is already a cropped eye recording, so each frame is used as-is.
# Pipeline: BGR frame → RGB → resize to 224×224 (bicubic) → normalize.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = 224

def make_transforms(augment: bool = False) -> transforms.Compose:
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC)]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(5),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


class PupilVideoDataset(Dataset):
    """
    Loads frames on-the-fly from an eye video.
    No cropping — the full frame is the eye region.

    Each sample returns:
        img    : (3, 224, 224) float tensor
        target : scalar float  — diameter_px
        frame  : int           — original frame index
    """

    def __init__(
        self,
        video_path: str,
        csv_path: str,
        frame_indices: list[int] | None = None,
        augment: bool = False,
    ):
        self.video_path = video_path
        self.transform  = make_transforms(augment=augment)

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["diameter_px"])
        df = df[df["diameter_px"].astype(str).str.strip() != ""]
        df["Frame"]       = df["Frame"].astype(int)
        df["diameter_px"] = df["diameter_px"].astype(float)

        if frame_indices is not None:
            df = df[df["Frame"].isin(frame_indices)]

        self.df = df.reset_index(drop=True)

        cap = cv2.VideoCapture(video_path)
        self.total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    def __len__(self) -> int:
        return len(self.df)

    def _get_frame(self, frame_idx: int) -> np.ndarray | None:
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image
        row       = self.df.iloc[idx]
        frame_idx = int(row["Frame"])
        diameter  = float(row["diameter_px"])

        frame = self._get_frame(frame_idx)
        if frame is None:
            frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        img_t = self.transform(Image.fromarray(frame))

        return {
            "img":    img_t,
            "target": torch.tensor(diameter, dtype=torch.float32),
            "frame":  frame_idx,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,          # GradScaler or None
) -> dict:
    model.train()
    total_loss = total_mae = 0.0

    for batch in tqdm(loader, desc="  train", leave=False):
        imgs    = batch["img"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)   # (B, 1)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                preds = model(imgs)
                loss  = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(imgs)
            loss  = criterion(preds, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(imgs)
        total_mae  += (preds.detach() - targets).abs().sum().item()

    n = len(loader.dataset)
    return {"loss": total_loss / n, "mae": total_mae / n}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = total_mae = total_mape = 0.0

    for batch in tqdm(loader, desc="  eval ", leave=False):
        imgs    = batch["img"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)

        preds = model(imgs)
        loss  = criterion(preds, targets)

        total_loss += loss.item() * len(imgs)
        total_mae  += (preds - targets).abs().sum().item()
        total_mape += ((preds - targets).abs() / (targets.abs() + 1e-8)).sum().item() * 100

    n = len(loader.dataset)
    return {"loss": total_loss / n, "mae": total_mae / n, "mape": total_mape / n}


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune ResNet18 for pupil diameter estimation.")

    # ── Data ─────────────────────────────────────────────────────────────────
    p.add_argument("--video",   required=True,  help="Path to the eye video (.mp4).")
    p.add_argument("--csv",     required=True,  help="Path to the label CSV.")

    # ── Weights ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--weights", default=None,
        help="Path to ResNet18-compatible pretrained .pth weights. "
             "Omit to start from ImageNet init.",
    )
    p.add_argument(
        "--freeze_backbone", action="store_true",
        help="Freeze backbone — only train the FC head. "
             "Recommended when fine-tuning data is very small.",
    )

    # ── Training hyper-parameters ────────────────────────────────────────────
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4,
                   help="Learning rate (EyeDentify used 1e-3; lower for fine-tune).")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--lr_step",    type=int,   default=10,
                   help="Reduce LR by lr_gamma every this many epochs.")
    p.add_argument("--lr_gamma",   type=float, default=0.2)
    p.add_argument("--val_split",  type=float, default=0.15,
                   help="Fraction of data used for validation.")
    p.add_argument("--num_workers", type=int,  default=2)

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--output", default="finetuned_pupil_model.pth",
                   help="Where to save the best checkpoint.")
    p.add_argument("--no_amp", action="store_true",
                   help="Disable automatic mixed precision (AMP).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    use_amp = torch.cuda.is_available() and not args.no_amp
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("[INFO] Mixed precision (AMP) enabled.")

    # ── Split frame indices into train / val ──────────────────────────────────
    df_all    = pd.read_csv(args.csv)
    df_all    = df_all.dropna(subset=["diameter_px"])
    df_all    = df_all[df_all["diameter_px"].astype(str).str.strip() != ""]
    all_frames = df_all["Frame"].astype(int).tolist()

    train_frames, val_frames = train_test_split(
        all_frames, test_size=args.val_split, random_state=42
    )
    print(f"[INFO] Train frames: {len(train_frames)} | Val frames: {len(val_frames)}")

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_ds = PupilVideoDataset(args.video, args.csv, frame_indices=train_frames, augment=True)
    val_ds   = PupilVideoDataset(args.video, args.csv, frame_indices=val_frames,   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(
        weights_path=args.weights,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {n_params:,}")

    # ── Loss / optimiser / scheduler  ────────────────────────────────────────
    criterion = nn.L1Loss()          # MAE — same as EyeDentify

    # Differential learning rates: backbone gets args.lr, head gets 10x higher.
    # The backbone already knows how to see; the head needs to learn fast.
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith('fc')]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and n.startswith('fc')]
    optimizer = torch.optim.AdamW(
        [
            {'params': backbone_params, 'lr': args.lr},
            {'params': head_params,     'lr': args.lr * 10},
        ],
        weight_decay=args.weight_decay,
    )
    print(f'[INFO] Differential LR — backbone: {args.lr:.1e}  head: {args.lr*10:.1e}')
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_mae = float("inf")
    history      = []

    print(f"\n{'─'*60}")
    print(f"  Starting fine-tuning for {args.epochs} epochs")
    print(f"{'─'*60}\n")

    for epoch in range(1, args.epochs + 1):
        backbone_lr = optimizer.param_groups[0]["lr"]
        head_lr     = optimizer.param_groups[1]["lr"]
        print(f"Epoch {epoch:03d}/{args.epochs}  (backbone_lr={backbone_lr:.2e}  head_lr={head_lr:.2e})")

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"  train  loss={train_metrics['loss']:.4f}  mae={train_metrics['mae']:.4f} px"
        )
        print(
            f"  val    loss={val_metrics['loss']:.4f}  mae={val_metrics['mae']:.4f} px"
            f"  mape={val_metrics['mape']:.2f}%"
        )

        history.append({"epoch": epoch, **train_metrics,
                         "val_loss": val_metrics["loss"],
                         "val_mae":  val_metrics["mae"],
                         "val_mape": val_metrics["mape"]})

        # Save best checkpoint
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_mae":     best_val_mae,
                    "args":        vars(args),
                },
                args.output,
            )
            print(f"  ✓ New best val MAE={best_val_mae:.4f} px — checkpoint saved → {args.output}")
        print()

    print("─" * 60)
    print(f"Training complete.  Best val MAE = {best_val_mae:.4f} px")
    print(f"Checkpoint saved to: {args.output}")

    # Save training history as CSV
    hist_path = args.output.replace(".pth", "_history.csv")
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print(f"Training history  → {hist_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  INFERENCE HELPER  (run after training)
# ─────────────────────────────────────────────────────────────────────────────

def predict_video(
    video_path: str,
    csv_path: str,
    checkpoint_path: str,
    output_csv: str = "predictions.csv",
) -> pd.DataFrame:
    """
    Run the fine-tuned model on every frame in the CSV and return predictions.

    Example:
        df = predict_video(
            "GS_F_07_032-1_left.mp4",
            "GS_F_07_032-1_left.csv",
            "finetuned_pupil_model.pth",
        )
        print(df.head())
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds     = PupilVideoDataset(video_path, csv_path, augment=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    all_frames, all_preds, all_targets = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            preds = model(batch["img"].to(device)).squeeze(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(batch["target"].numpy().tolist())
            all_frames.extend(batch["frame"].numpy().tolist())

    results = pd.DataFrame({
        "frame":          all_frames,
        "true_diameter":  all_targets,
        "pred_diameter":  all_preds,
        "abs_error":      [abs(p - t) for p, t in zip(all_preds, all_targets)],
    }).sort_values("frame").reset_index(drop=True)

    mae  = results["abs_error"].mean()
    mape = (results["abs_error"] / results["true_diameter"] * 100).mean()
    print(f"[Inference] MAE={mae:.4f} px | MAPE={mape:.2f}%")

    results.to_csv(output_csv, index=False)
    print(f"Predictions saved → {output_csv}")
    return results


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()