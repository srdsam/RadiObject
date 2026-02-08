"""Standalone training script for fast iteration on segmentation model.

Run with:
    eval $(aws configure export-credentials --profile souzy-s3 --format env)
    PYTHONPATH=src .venv/bin/python3 scripts/train_debug.py
"""

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Compose, NormalizeIntensityd, RandFlipd, RandRotate90d

from radiobject import RadiObject, S3Config, configure
from radiobject.data import S3_REGION, get_msd_lung_uri
from radiobject.ml import create_segmentation_dataloader

# --- Config ---
MSD_LUNG_URI = get_msd_lung_uri()
BATCH_SIZE = 2
PATCH_SIZE = (96, 96, 96)
NUM_EPOCHS = 150
LEARNING_RATE = 1e-3
MODEL_PATH = "scripts/model_checkpoint.pt"

# --- Device ---
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# --- Data ---
if MSD_LUNG_URI.startswith("s3://"):
    configure(s3=S3Config(region=S3_REGION, max_parallel_ops=8))

radi = RadiObject(MSD_LUNG_URI)
all_ids = list(radi.obs_subject_ids)
np.random.seed(42)
np.random.shuffle(all_ids)
split_idx = int(0.8 * len(all_ids))
train_ids, val_ids = all_ids[:split_idx], all_ids[split_idx:]
radi_train, radi_val = radi.loc[train_ids], radi.loc[val_ids]

print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

spatial_keys = ["image", "mask"]
spatial_transform = Compose(
    [
        RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=[0, 1, 2]),
        RandRotate90d(keys=spatial_keys, prob=0.3, spatial_axes=(0, 1)),
    ]
)

print("Creating train loader (pre-computing foreground coords)...")
train_loader = create_segmentation_dataloader(
    image=radi_train.CT,
    mask=radi_train.seg,
    batch_size=BATCH_SIZE,
    patch_size=PATCH_SIZE,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    image_transform=NormalizeIntensityd(keys="image"),
    spatial_transform=spatial_transform,
    foreground_sampling=True,
    foreground_threshold=0.005,
    patches_per_volume=2,
)

print("Creating val loader (pre-computing foreground coords)...")
val_loader = create_segmentation_dataloader(
    image=radi_val.CT,
    mask=radi_val.seg,
    batch_size=BATCH_SIZE,
    patch_size=PATCH_SIZE,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    image_transform=NormalizeIntensityd(keys="image"),
    foreground_sampling=True,
    foreground_threshold=0.005,
)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# --- Verify foreground is present in training data ---
batch = next(iter(train_loader))
fg_frac = (batch["mask"] > 0).float().mean().item()
print(f"Sample batch foreground fraction: {fg_frac:.4f}")

# --- Model ---
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(32, 64, 128, 256),
    strides=(2, 2, 2),
    num_res_units=2,
).to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# --- Training ---
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
criterion = DiceFocalLoss(to_onehot_y=True, softmax=True)
dice_metric = DiceMetric(include_background=False, reduction="mean")

best_val_dice = 0.0
patience_counter = 0
PATIENCE = 40

print(f"\nTraining for up to {NUM_EPOCHS} epochs...\n")
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    dice_metric.reset()
    n_fg_batches = 0

    for batch in train_loader:
        images = batch["image"].to(DEVICE)
        labels = batch["mask"].long().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1, keepdim=True)
        dice_metric(preds, labels)

        if (labels > 0).any():
            n_fg_batches += 1

    train_dice = dice_metric.aggregate().item()

    model.eval()
    val_loss = 0.0
    dice_metric.reset()

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(DEVICE)
            labels = batch["mask"].long().to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1, keepdim=True)
            dice_metric(preds, labels)

    val_dice = dice_metric.aggregate().item()
    scheduler.step()

    improved = val_dice > best_val_dice
    if improved:
        best_val_dice = val_dice
        torch.save(model.state_dict(), MODEL_PATH)
        patience_counter = 0
    else:
        patience_counter += 1

    if (epoch + 1) % 5 == 0 or epoch == 0 or improved:
        print(
            f"Epoch {epoch+1:3d}/{NUM_EPOCHS}: "
            f"Loss={train_loss/len(train_loader):.4f}, "
            f"Train Dice={train_dice:.4f}, "
            f"Val Dice={val_dice:.4f}, "
            f"FG batches={n_fg_batches}/{len(train_loader)} "
            f"{'*BEST*' if improved else ''}"
        )

    if patience_counter >= PATIENCE and best_val_dice > 0.3:
        print(f"Early stopping at epoch {epoch+1}, best Val Dice: {best_val_dice:.4f}")
        break

print(f"\nBest Val Dice: {best_val_dice:.4f}")
print(f"Model saved to {MODEL_PATH}")

# --- Quick inference check on first val subject ---
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

subject_id = val_ids[0]
ct_data = radi.loc[subject_id].CT.iloc[0].to_numpy().astype(np.float32)
seg_data = radi.loc[subject_id].seg.iloc[0].to_numpy()

ct_tensor = torch.from_numpy(ct_data).unsqueeze(0).unsqueeze(0)
ct_tensor = (ct_tensor - ct_tensor.mean()) / (ct_tensor.std() + 1e-8)

with torch.no_grad():
    pred_vol = sliding_window_inference(
        ct_tensor.to(DEVICE),
        roi_size=PATCH_SIZE,
        sw_batch_size=4,
        predictor=model,
        overlap=0.25,
    )
    pred_mask = torch.argmax(pred_vol, dim=1).squeeze().cpu().numpy()

gt_flat = (seg_data > 0).astype(np.float32).ravel()
pred_flat = (pred_mask > 0).astype(np.float32).ravel()
intersection = (gt_flat * pred_flat).sum()
subject_dice = 2 * intersection / (gt_flat.sum() + pred_flat.sum() + 1e-8)

print(f"\nFull-volume inference on {subject_id}:")
print(f"  GT tumor voxels:   {int(gt_flat.sum()):,}")
print(f"  Pred tumor voxels: {int(pred_flat.sum()):,}")
print(f"  Subject Dice:      {subject_dice:.4f}")
