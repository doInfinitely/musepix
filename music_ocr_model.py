"""
Music OCR model — retina-based hierarchical detector with teacher forcing.

Architecture (adapted from tiny-tessarachnid)
──────────────────────────────────────────────
Input  : 1024 × 1024 RGB retina image  +  prev_bbox (5-d)
Output : (x1, y1, x2, y2) normalised bbox  +  class logits

Backbone : pretrained ResNet-18 → AdaptiveAvgPool → 512-d features
Concat   : 512 + 5 (prev_bbox) = 517
FC shared: 517 → 256 → 128 (with dropout)
Heads    : bbox (128 → 64 → 4, Sigmoid)  +  cls (128 → 64 → num_classes)

Key improvements over v1:
  • Pretrained ResNet-18 backbone (vs custom CNN from scratch)
  • Previous bbox as autoregressive input (helps model learn to stop)
  • SmoothL1 loss for bbox (only on non-none samples)
  • AdamW with differential LR + ReduceLROnPlateau + early stopping
  • Validation split for proper generalisation monitoring

Usage:
    python3 music_ocr_model.py --data_dir data/sheet_music --epochs 50
"""

import os
import json
import argparse
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

from retina import fit_to_retina, image_to_tensor
from generate_sheet_music import (
    CLASSES, CLASS_TO_ID, BBox, SheetMusicGenerator,
    extract_training_samples,
)

NONE_CLASS_ID = CLASS_TO_ID['none']
RETINA_SIZE = 1024
PREV_BBOX_NONE = (0.0, 0.0, 0.0, 0.0, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class MusicOCRModel(nn.Module):
    """
    Retina-based detector with autoregressive previous-bbox conditioning.

    For a given retina image + prev_bbox, predicts:
        • bbox — (x1, y1, x2, y2) in [0, 1]
        • cls  — logits over CLASSES
    """

    def __init__(self, num_classes: int = len(CLASSES)):
        super().__init__()
        self.num_classes = num_classes

        # Pretrained ResNet-18 backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ImageNet normalisation constants
        self.register_buffer(
            'img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Shared FC trunk: 512 (CNN) + 5 (prev_bbox) = 517
        self.fc_shared = nn.Sequential(
            nn.Linear(517, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
        )

        # Bbox head → normalised [0, 1]
        self.bbox_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 4), nn.Sigmoid(),
        )

        # Classification head → raw logits
        self.cls_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, img, prev_bbox):
        """
        Args:
            img:       (B, 3, 1024, 1024) RGB tensor in [0, 1]
            prev_bbox: (B, 5) — (x1, y1, x2, y2, class_id), all normalised
        Returns:
            bbox_pred: (B, 4) in [0, 1]
            cls_pred:  (B, num_classes) raw logits
        """
        # ImageNet normalise
        x = (img - self.img_mean) / self.img_std

        # Backbone
        x = self.backbone(x)       # (B, 512, H, W)
        x = self.pool(x)           # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 512)

        # Normalise prev_bbox: coords already in [0,1], class_id / num_classes
        prev_norm = prev_bbox.clone()
        prev_norm[:, 4] = prev_norm[:, 4] / self.num_classes

        # Concat and pass through heads
        x = torch.cat([x, prev_norm], dim=1)  # (B, 517)
        x = self.fc_shared(x)                 # (B, 128)

        return self.bbox_head(x), self.cls_head(x)


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

class MusicOCRLoss(nn.Module):
    """
    Combined bbox + classification loss.

    • SmoothL1 for bbox regression (only on non-none samples)
    • CrossEntropy for classification (all samples)
    """

    def __init__(self, bbox_weight=1.0, class_weight=1.0):
        super().__init__()
        self.bbox_weight = bbox_weight
        self.class_weight = class_weight
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.bbox_loss_fn = nn.SmoothL1Loss()

    def forward(self, bbox_pred, cls_pred, target_bbox, target_cls):
        """
        Args:
            bbox_pred:  (B, 4) predicted normalised bbox
            cls_pred:   (B, C) class logits
            target_bbox: (B, 4) target normalised bbox
            target_cls:  (B,) target class indices
        """
        class_loss = self.cls_loss_fn(cls_pred, target_cls)

        # Bbox loss only for non-none samples
        non_none = target_cls != NONE_CLASS_ID
        if non_none.any():
            bbox_loss = self.bbox_loss_fn(bbox_pred[non_none],
                                          target_bbox[non_none])
        else:
            bbox_loss = torch.tensor(0.0, device=bbox_pred.device)

        total = self.bbox_weight * bbox_loss + self.class_weight * class_loss
        return total, bbox_loss, class_loss


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class TeacherForcingDataset(Dataset):
    """
    Teacher-forcing dataset with previous-bbox conditioning.

    Each sample yields (retina_image, prev_bbox, target_bbox, target_cls).
    """

    def __init__(self, data_dir: str = None, num_images: int = 200,
                 retina_size: int = 1024, seed: int = 42,
                 parent_filter: str = None):
        self.retina_size = retina_size
        self.samples: List[dict] = []

        if data_dir and os.path.isfile(os.path.join(data_dir, 'annotations.json')):
            self._load_from_disk(data_dir)
        else:
            self._generate(num_images, seed)

        if parent_filter:
            total = len(self.samples)
            self.samples = [s for s in self.samples
                            if s.get('parent_class') == parent_filter]
            print(f"[parent_filter={parent_filter}] kept {len(self.samples)}"
                  f" / {total} samples")

    def _load_from_disk(self, data_dir):
        with open(os.path.join(data_dir, 'annotations.json')) as f:
            annotations = json.load(f)

        for ann in annotations:
            img_path = os.path.join(data_dir, 'images', ann['image'])
            img = Image.open(img_path).convert('RGB')
            tree = BBox.from_dict(ann['bbox_tree'])
            self.samples.extend(extract_training_samples(img, tree,
                                                         self.retina_size))

    def _generate(self, num_images, seed):
        gen = SheetMusicGenerator()
        random.seed(seed)
        for i in range(num_images):
            img, tree = gen.generate(seed=seed + i)
            self.samples.extend(extract_training_samples(img, tree,
                                                         self.retina_size))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        retina_img, _, _, _ = fit_to_retina(s['input_image'], self.retina_size)
        x = image_to_tensor(retina_img)                             # [3, R, R]
        bbox = torch.tensor(s['target_bbox'], dtype=torch.float32)  # [4]
        cls = torch.tensor(CLASS_TO_ID[s['target_class']], dtype=torch.long)

        # Previous bbox: (x1, y1, x2, y2, class_id)
        prev = s.get('prev_bbox', PREV_BBOX_NONE)
        prev_t = torch.tensor(prev, dtype=torch.float32)            # [5]

        return x, prev_t, bbox, cls


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    device = (
        torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu')
    )
    print(f'Device: {device}', flush=True)

    full_dataset = TeacherForcingDataset(
        data_dir=args.data_dir,
        num_images=args.num_images,
        retina_size=args.retina_size,
        seed=args.seed,
        parent_filter=getattr(args, 'parent_filter', None),
    )
    print(f'Dataset: {len(full_dataset)} total samples', flush=True)

    # Train / validation split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(args.seed))
    print(f'Train: {train_size}, Val: {val_size}', flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Model
    model = MusicOCRModel(num_classes=len(CLASSES)).to(device)
    if getattr(args, 'resume', None):
        print(f'Resuming from checkpoint: {args.resume}', flush=True)
        model.load_state_dict(
            torch.load(args.resume, map_location=device, weights_only=True))

    # Optimiser with differential learning rates
    backbone_params = list(model.backbone.parameters())
    head_params = (list(model.fc_shared.parameters())
                   + list(model.bbox_head.parameters())
                   + list(model.cls_head.parameters()))
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': head_params, 'lr': args.lr},
    ], weight_decay=1e-2)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    # Loss
    loss_fn = MusicOCRLoss(bbox_weight=1.0, class_weight=1.0).to(device)

    # Training
    num_batches = len(train_loader)
    log_interval = max(1, num_batches // 5)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = args.patience

    os.makedirs('checkpoints', exist_ok=True)
    pf = getattr(args, 'parent_filter', None)
    ckpt_path = (f'checkpoints/music_ocr_v2_finetune_{pf}.pt'
                 if pf else 'checkpoints/music_ocr_v2.pt')

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_total = train_bbox = train_cls = 0.0
        train_n = 0

        for batch_idx, (x, prev, bbox_target, cls_target) in enumerate(train_loader, 1):
            x = x.to(device)
            prev = prev.to(device)
            bbox_target = bbox_target.to(device)
            cls_target = cls_target.to(device)

            bbox_pred, cls_pred = model(x, prev)
            total, b_loss, c_loss = loss_fn(bbox_pred, cls_pred,
                                            bbox_target, cls_target)

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            bs = x.size(0)
            train_total += total.item() * bs
            train_bbox += b_loss.item() * bs
            train_cls += c_loss.item() * bs
            train_n += bs

            if batch_idx % log_interval == 0:
                print(f'  Epoch {epoch:3d}  batch {batch_idx}/{num_batches}  '
                      f'loss={total.item():.4f}', flush=True)

        train_total /= train_n
        train_bbox /= train_n
        train_cls /= train_n

        # ── Validate ──
        model.eval()
        val_total = val_bbox = val_cls = 0.0
        val_n = 0

        with torch.no_grad():
            for x, prev, bbox_target, cls_target in val_loader:
                x = x.to(device)
                prev = prev.to(device)
                bbox_target = bbox_target.to(device)
                cls_target = cls_target.to(device)

                bbox_pred, cls_pred = model(x, prev)
                total, b_loss, c_loss = loss_fn(bbox_pred, cls_pred,
                                                bbox_target, cls_target)

                bs = x.size(0)
                val_total += total.item() * bs
                val_bbox += b_loss.item() * bs
                val_cls += c_loss.item() * bs
                val_n += bs

        val_total /= val_n
        val_bbox /= val_n
        val_cls /= val_n

        scheduler.step(val_total)
        cur_lr = optimizer.param_groups[1]['lr']  # head LR

        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'train loss={train_total:.4f} (bbox={train_bbox:.4f} cls={train_cls:.4f}) | '
              f'val loss={val_total:.4f} (bbox={val_bbox:.4f} cls={val_cls:.4f}) | '
              f'lr={cur_lr:.2e}', flush=True)

        # Checkpoint best model
        if val_total < best_val_loss:
            best_val_loss = val_total
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f'  -> saved best model ({ckpt_path})', flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'  Early stopping: val loss not improved for '
                      f'{patience} epochs (best={best_val_loss:.4f})', flush=True)
                break

    print(f'Training complete. Best val loss: {best_val_loss:.4f}')


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model: MusicOCRModel, image: Image.Image,
            prev_bbox=None, retina_size: int = 1024, device=None):
    """Run one forward pass and return (bbox_norm, class_name, confidence)."""
    if device is None:
        device = next(model.parameters()).device
    if prev_bbox is None:
        prev_bbox = PREV_BBOX_NONE

    retina_img, scale, ox, oy = fit_to_retina(image, retina_size)
    x = image_to_tensor(retina_img).unsqueeze(0).to(device)
    prev_t = torch.tensor([prev_bbox], dtype=torch.float32).to(device)

    bbox_pred, cls_pred = model(x, prev_t)
    bbox = bbox_pred[0].cpu().tolist()
    probs = torch.softmax(cls_pred[0], dim=0)
    cls_id = probs.argmax().item()
    return bbox, CLASSES[cls_id], probs[cls_id].item()


@torch.no_grad()
def hierarchical_decode(model: MusicOCRModel, image: Image.Image,
                        retina_size: int = 1024, max_depth: int = 10,
                        max_detections: int = 500, device=None):
    """
    Recursively decode the bounding-box hierarchy using autoregressive
    previous-bbox conditioning.  The image is never modified — the model
    relies solely on prev_bbox to advance through siblings.

    Returns a BBox tree.
    """
    if device is None:
        device = next(model.parameters()).device

    total_detections = 0

    def _decode(img, depth):
        nonlocal total_detections
        if depth >= max_depth or img.width < 4 or img.height < 4:
            return []

        parent_area = img.width * img.height
        children = []
        prev_bbox = PREV_BBOX_NONE

        for _ in range(20):  # per-node safety cap
            if total_detections >= max_detections:
                break

            bbox_n, cls_name, conf = predict(
                model, img, prev_bbox=prev_bbox,
                retina_size=retina_size, device=device)

            if cls_name == 'none' or conf < 0.3:
                break

            x1 = int(bbox_n[0] * img.width)
            y1 = int(bbox_n[1] * img.height)
            x2 = int(bbox_n[2] * img.width)
            y2 = int(bbox_n[3] * img.height)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                break

            # Skip if child covers >95% of parent (model is confused)
            child_area = (x2 - x1) * (y2 - y1)
            if child_area > 0.95 * parent_area:
                break

            total_detections += 1

            child_crop = img.crop((x1, y1, x2, y2))
            sub_children = _decode(child_crop, depth + 1)

            # Offset sub-children from crop-local to parent coordinates
            offset_children = [
                _offset_bbox(sc, x1, y1) for sc in sub_children
            ]

            children.append(BBox(x1, y1, x2, y2, cls_name,
                                 children=offset_children))

            # Update prev_bbox for next iteration (normalised coords + class_id)
            prev_bbox = (bbox_n[0], bbox_n[1], bbox_n[2], bbox_n[3],
                         float(CLASS_TO_ID[cls_name]))

        return children

    subs = _decode(image, 0)
    return BBox(0, 0, image.width, image.height, 'page', children=subs)


# ──────────────────────────────────────────────────────────────────────────────
# Pixel analysis helpers (adapted from tiny-tessarachnid)
# ──────────────────────────────────────────────────────────────────────────────

def _offset_bbox(node, dx, dy):
    """Recursively offset a BBox tree by (dx, dy) to convert from
    crop-local coordinates to parent coordinates."""
    return BBox(
        node.x1 + dx, node.y1 + dy,
        node.x2 + dx, node.y2 + dy,
        node.label,
        children=[_offset_bbox(c, dx, dy) for c in node.children],
    )



def _find_content_regions(image, bg_color=(255, 255, 255),
                          gap_tolerance=5, scan_axis='rows',
                          mask_model=None, device=None):
    """Find distinct content regions separated by gaps.

    If mask_model is provided, uses the learned segmentation network to
    compute the content mask. Otherwise falls back to color comparison.

    scan_axis='rows': scan top-to-bottom, split by blank rows.
    scan_axis='cols': scan left-to-right, split by blank columns.

    Returns a list of (x1, y1, x2, y2) bounding boxes.
    """
    w, h = image.size

    if mask_model is not None and device is not None:
        mask_pil = mask_model.segment(image, device)
        content = np.array(mask_pil) > 127
    else:
        arr = np.array(image, dtype=np.int16)
        bg = np.array(bg_color, dtype=np.int16)
        content = np.abs(arr - bg).max(axis=2) > 0

    if scan_axis == 'rows':
        line_has_content = content.any(axis=1)
        length = h
    else:
        line_has_content = content.any(axis=0)
        length = w

    regions = []
    in_region = False
    region_start = 0
    gap_count = 0

    def _finish_region(region_end):
        if scan_axis == 'rows':
            band = content[region_start:region_end + 1, :]
            cross = band.any(axis=0)
            if cross.any():
                c1 = int(np.argmax(cross))
                c2 = int(len(cross) - 1 - np.argmax(cross[::-1]))
                regions.append((c1, region_start, c2 + 1, region_end + 1))
        else:
            band = content[:, region_start:region_end + 1]
            cross = band.any(axis=1)
            if cross.any():
                c1 = int(np.argmax(cross))
                c2 = int(len(cross) - 1 - np.argmax(cross[::-1]))
                regions.append((region_start, c1, region_end + 1, c2 + 1))

    for i in range(length):
        if line_has_content[i]:
            if not in_region:
                region_start = i
                in_region = True
            gap_count = 0
        else:
            if in_region:
                gap_count += 1
                if gap_count > gap_tolerance:
                    _finish_region(i - gap_count)
                    in_region = False

    if in_region:
        region_end = length - 1
        while region_end > region_start and not line_has_content[region_end]:
            region_end -= 1
        _finish_region(region_end)

    return regions


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid inference (pixel analysis + model classification)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def detect_level(model, source_img, device, retina_size=1024,
                 bg_color=(255, 255, 255), max_detections=50,
                 gap_tolerance=20, scan_axis='rows',
                 mask_model=None):
    """Detect items at one hierarchy level using hybrid approach.

    1. Pixel analysis finds content regions (bounding boxes).
    2. Model runs autoregressively (conditioned on prev_bbox) to count
       items and get class IDs.  The image is never modified.
    3. Regions are paired with class predictions.

    If mask_model is provided, uses learned segmentation for content detection.

    Returns list of ((x1,y1,x2,y2), class_name) tuples.
    """
    regions = _find_content_regions(source_img, bg_color,
                                    gap_tolerance=gap_tolerance,
                                    scan_axis=scan_axis,
                                    mask_model=mask_model,
                                    device=device)
    if not regions:
        return []

    # Run model autoregressively to count detections and get class names
    retina_img, scale, ox, oy = fit_to_retina(source_img, retina_size)
    x = image_to_tensor(retina_img).unsqueeze(0).to(device)
    class_names = []
    prev_bbox = PREV_BBOX_NONE

    for _ in range(max_detections):
        prev_t = torch.tensor([prev_bbox], dtype=torch.float32).to(device)

        bbox_pred, cls_pred = model(x, prev_t)
        probs = torch.softmax(cls_pred[0], dim=0)
        cls_id = probs.argmax().item()
        cls_name = CLASSES[cls_id]

        if cls_name == 'none' or probs[cls_id].item() < 0.3:
            break

        class_names.append(cls_name)

        if len(class_names) <= len(regions):
            # Build prev_bbox in normalised coords for next iteration
            region = regions[len(class_names) - 1]
            w, h = source_img.size
            prev_bbox = (
                region[0] / w, region[1] / h,
                region[2] / w, region[3] / h,
                float(cls_id),
            )
        else:
            break

    # Pair regions with class names
    n = min(len(class_names), len(regions))
    return [(regions[i], class_names[i]) for i in range(n)]


@torch.no_grad()
def hierarchical_decode_hybrid(model, image, retina_size=1024,
                               max_depth=10, device=None,
                               bg_color=(255, 255, 255),
                               mask_model=None):
    """
    Recursively decode the bbox hierarchy using the hybrid approach:
    pixel analysis for localisation, model for classification/counting.

    If mask_model is provided, uses learned segmentation for content detection
    instead of simple color comparison.

    Uses different scan parameters at each hierarchy level:
      - page → staff_systems:        row-scan, gap_tolerance=20
      - staff_system → children:     row-scan, gap_tolerance=3
      - melodic_line → notes:        col-scan, gap_tolerance=3
      - note → components:           col-scan, gap_tolerance=1

    Returns a BBox tree.
    """
    if device is None:
        device = next(model.parameters()).device

    # Scan parameters per parent label
    LEVEL_PARAMS = {
        'page':         {'scan_axis': 'rows', 'gap_tolerance': 20},
        'staff_system': {'scan_axis': 'rows', 'gap_tolerance': 3},
        'melodic_line': {'scan_axis': 'cols', 'gap_tolerance': 3},
        'note':         {'scan_axis': 'cols', 'gap_tolerance': 1},
    }

    def _decode(img, parent_label, depth):
        if depth >= max_depth or img.width < 4 or img.height < 4:
            return []

        params = LEVEL_PARAMS.get(parent_label,
                                  {'scan_axis': 'cols', 'gap_tolerance': 3})

        detections = detect_level(
            model, img, device, retina_size=retina_size,
            bg_color=bg_color, max_detections=50,
            gap_tolerance=params['gap_tolerance'],
            scan_axis=params['scan_axis'],
            mask_model=mask_model,
        )

        children = []
        for (x1, y1, x2, y2), cls_name in detections:
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            child_crop = img.crop((x1, y1, x2, y2))
            sub_children = _decode(child_crop, cls_name, depth + 1)

            # Offset sub-children from crop-local to parent coordinates
            offset_children = [
                _offset_bbox(sc, x1, y1) for sc in sub_children
            ]
            children.append(BBox(x1, y1, x2, y2, cls_name,
                                 children=offset_children))

        return children

    subs = _decode(image, 'page', 0)
    return BBox(0, 0, image.width, image.height, 'page', children=subs)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train the Music OCR model.')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Pre-generated data directory (with annotations.json).')
    parser.add_argument('--num_images', type=int, default=200,
                        help='Images to generate on-the-fly if no data_dir.')
    parser.add_argument('--retina_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Fraction of data used for validation.')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (epochs without improvement).')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to v2 checkpoint to resume training from.')
    parser.add_argument('--parent_filter', type=str, default=None,
                        help='Only train on samples from this parent class '
                             '(e.g. "note" for note-level pretraining).')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
