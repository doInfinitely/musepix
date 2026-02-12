"""
Learned pixel mask segmentation network for content/background separation.

Architecture: LRASPP + MobileNetV3-Large (~3M params)
- Pretrained MobileNetV3 backbone (ImageNet)
- Modified final classifier: 21-class (COCO) -> 1-class (binary content mask)
- Output: per-pixel logits -> sigmoid -> binary mask

Training data comes from synthetic sheet music images where GT masks are
trivially derivable (any non-white pixel = content). Heavy augmentation
bridges the domain gap to real scanned sheet music.

Usage:
    python3 mask_network.py --data_dir data/sheet_music --epochs 50
"""

import os
import io
import json
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
import torchvision.models.segmentation as seg_models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageFilter


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class MaskNetwork(nn.Module):
    """Lightweight binary segmentation network based on LRASPP + MobileNetV3."""

    def __init__(self):
        super().__init__()
        lraspp = seg_models.lraspp_mobilenet_v3_large(
            weights_backbone=tv_models.MobileNet_V3_Large_Weights.DEFAULT,
        )
        # Replace 21-class COCO heads with 1-class binary heads
        lraspp.classifier.low_classifier = nn.Conv2d(40, 1, 1)
        lraspp.classifier.high_classifier = nn.Conv2d(128, 1, 1)
        self.net = lraspp

        # ImageNet normalisation constants
        self.register_buffer(
            'img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) RGB tensor in [0, 1]
        Returns:
            logits: (B, 1, H, W) per-pixel logits (apply sigmoid for probs)
        """
        x = (x - self.img_mean) / self.img_std
        return self.net(x)['out']

    @torch.no_grad()
    def segment(self, pil_image, device, threshold=0.5):
        """Run segmentation on a PIL Image.

        Returns a PIL Image in 'L' mode: 255 = content, 0 = background.
        """
        self.eval()
        orig_w, orig_h = pil_image.size

        # Pad to preserve aspect ratio, resize to 512x512
        img_tensor = _pil_to_padded_tensor(pil_image, 512).unsqueeze(0).to(device)

        logits = self.forward(img_tensor)  # (1, 1, 512, 512)
        probs = torch.sigmoid(logits[0, 0])  # (512, 512)
        mask_512 = (probs > threshold).cpu().numpy().astype(np.uint8) * 255

        # Unpad and resize back to original dimensions
        mask_pil = Image.fromarray(mask_512, mode='L')
        # Determine padding that was applied
        scale = 512 / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        # Crop away padding
        mask_cropped = mask_pil.crop((0, 0, new_w, new_h))
        # Resize to original
        mask_out = mask_cropped.resize((orig_w, orig_h), Image.NEAREST)
        return mask_out


def _pil_to_padded_tensor(pil_image, target_size):
    """Convert PIL image to padded square tensor of given size.

    Pads with white (1.0) to preserve aspect ratio.
    Returns (3, target_size, target_size) tensor in [0, 1].
    """
    orig_w, orig_h = pil_image.size
    scale = target_size / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    resized = pil_image.resize((new_w, new_h), Image.BILINEAR)
    # Create white canvas
    canvas = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    canvas.paste(resized, (0, 0))

    arr = np.array(canvas, dtype=np.float32) / 255.0  # (H, W, 3)
    return torch.from_numpy(arr.transpose(2, 0, 1))  # (3, H, W)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset with augmentation
# ──────────────────────────────────────────────────────────────────────────────

class MaskDataset(Dataset):
    """Dataset for training the mask segmentation network.

    Loads synthetic sheet music images and derives GT masks
    (any non-white pixel = content). Applies heavy augmentation to the
    image only (mask stays clean) to bridge domain gap.
    """

    def __init__(self, data_dir, target_size=512, augment=True):
        self.target_size = target_size
        self.augment = augment

        ann_path = os.path.join(data_dir, 'annotations.json')
        with open(ann_path) as f:
            annotations = json.load(f)

        self.image_paths = []
        img_dir = os.path.join(data_dir, 'images')
        for ann in annotations:
            path = os.path.join(img_dir, ann['image'])
            if os.path.isfile(path):
                self.image_paths.append(path)

        print(f'MaskDataset: {len(self.image_paths)} images from {data_dir}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        arr = np.array(img)

        # GT mask: any non-white pixel is content
        mask = (arr != 255).any(axis=2).astype(np.float32)  # (H, W), 0/1

        # Apply augmentation to image only
        if self.augment:
            img = self._augment(img, arr)

        # Pad and resize to target_size
        img_tensor = _pil_to_padded_tensor(img, self.target_size)

        # Pad and resize mask similarly
        orig_w, orig_h = mask.shape[1], mask.shape[0]
        scale = self.target_size / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_resized = mask_pil.resize((new_w, new_h), Image.NEAREST)
        mask_canvas = Image.new('L', (self.target_size, self.target_size), 0)
        mask_canvas.paste(mask_resized, (0, 0))

        mask_tensor = torch.from_numpy(
            np.array(mask_canvas, dtype=np.float32) / 255.0
        ).unsqueeze(0)  # (1, H, W)

        return img_tensor, mask_tensor

    def _augment(self, img, arr):
        """Apply augmentation pipeline to image (not mask)."""
        from torchvision import transforms

        # 1. Background tinting: shift white background toward yellow/cream/gray
        if random.random() < 0.5:
            img = self._tint_background(img, arr)

        # 2. Color jitter
        jitter = transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.1, hue=0.014)
        img = jitter(img)

        # 3. Gaussian noise
        if random.random() < 0.5:
            sigma = random.uniform(0, 15)
            img_arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, sigma, img_arr.shape).astype(np.float32)
            img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_arr)

        # 4. Gaussian blur
        if random.random() < 0.3:
            radius = random.choice([1, 2])
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # 5. JPEG compression artifacts
        if random.random() < 0.3:
            quality = random.randint(70, 95)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')

        # 6. Random perspective distortion
        if random.random() < 0.2:
            persp = transforms.RandomPerspective(
                distortion_scale=0.05, p=1.0, fill=255)
            img = persp(img)

        return img

    @staticmethod
    def _tint_background(img, arr):
        """Tint white background pixels toward a random off-white color."""
        # Identify background pixels (close to white)
        is_bg = (arr > 240).all(axis=2)

        # Random tint color: yellowish, cream, or grayish
        tint_type = random.choice(['yellow', 'cream', 'gray'])
        if tint_type == 'yellow':
            tint = np.array([random.randint(230, 250),
                             random.randint(225, 245),
                             random.randint(190, 220)], dtype=np.uint8)
        elif tint_type == 'cream':
            tint = np.array([random.randint(240, 255),
                             random.randint(235, 250),
                             random.randint(220, 240)], dtype=np.uint8)
        else:  # gray
            g = random.randint(220, 245)
            tint = np.array([g, g, g], dtype=np.uint8)

        result = arr.copy()
        result[is_bg] = tint
        return Image.fromarray(result)


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) raw logits
            targets: (B, 1, H, W) binary targets in [0, 1]
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    device = (
        torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu')
    )
    print(f'Device: {device}', flush=True)

    full_dataset = MaskDataset(args.data_dir, target_size=512, augment=True)

    # Train / validation split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed))

    # Disable augmentation for validation
    val_ds_wrapper = _ValWrapper(val_ds)

    print(f'Train: {train_size}, Val: {val_size}', flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds_wrapper, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = MaskNetwork().to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,} total, {n_trainable:,} trainable',
          flush=True)

    # Differential learning rates
    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': classifier_params, 'lr': args.lr},
    ], weight_decay=1e-2)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    # Loss
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    # Training loop
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = 'checkpoints/mask_segmentation.pt'

    best_val_loss = float('inf')
    epochs_no_improve = 0
    num_batches = len(train_loader)
    log_interval = max(1, num_batches // 5)

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for batch_idx, (imgs, masks) in enumerate(train_loader, 1):
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            loss = bce_loss(logits, masks) + dice_loss(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs

            if batch_idx % log_interval == 0:
                print(f'  Epoch {epoch:3d}  batch {batch_idx}/{num_batches}  '
                      f'loss={loss.item():.4f}', flush=True)

        train_loss = train_loss_sum / train_n

        # ── Validate ──
        model.eval()
        val_loss_sum = 0.0
        val_iou_sum = 0.0
        val_acc_sum = 0.0
        val_n = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = model(imgs)
                loss = bce_loss(logits, masks) + dice_loss(logits, masks)

                bs = imgs.size(0)
                val_loss_sum += loss.item() * bs
                val_n += bs

                # Metrics
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_acc_sum += ((preds == masks).float().mean().item() * bs)

                # IoU
                intersection = (preds * masks).sum().item()
                union = ((preds + masks) > 0).float().sum().item()
                if union > 0:
                    val_iou_sum += (intersection / union) * bs
                else:
                    val_iou_sum += 1.0 * bs

        val_loss = val_loss_sum / val_n
        val_acc = val_acc_sum / val_n
        val_iou = val_iou_sum / val_n

        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[1]['lr']

        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'train loss={train_loss:.4f} | '
              f'val loss={val_loss:.4f} acc={val_acc:.4f} IoU={val_iou:.4f} | '
              f'lr={cur_lr:.2e}', flush=True)

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f'  -> saved best model ({ckpt_path})', flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f'  Early stopping: val loss not improved for '
                      f'{args.patience} epochs (best={best_val_loss:.4f})',
                      flush=True)
                break

    print(f'Training complete. Best val loss: {best_val_loss:.4f}')


class _ValWrapper(Dataset):
    """Wraps a Subset to disable augmentation during validation."""

    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Temporarily disable augmentation
        ds = self.subset.dataset
        orig_augment = ds.augment
        ds.augment = False
        item = self.subset[idx]
        ds.augment = orig_augment
        return item


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train the mask segmentation network.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory with annotations.json and images/.')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
