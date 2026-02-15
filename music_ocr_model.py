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

import cv2
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

    def __init__(self, bbox_weight=1.0, class_weight=1.0, cls_weights=None):
        super().__init__()
        self.bbox_weight = bbox_weight
        self.class_weight = class_weight
        self.cls_loss_fn = nn.CrossEntropyLoss(
            weight=cls_weights if cls_weights is not None else None)
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
                 parent_filter: str = None, noise_std: float = 0.03):
        self.retina_size = retina_size
        self.noise_std = noise_std
        self.training = True  # toggled off for validation
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
        # Preprocess: remove horizontal staff lines so the model learns
        # from line-free images (matching inference preprocessing).
        clean_img = _remove_horizontal_lines(s['input_image'])
        retina_img, _, _, _ = fit_to_retina(clean_img, self.retina_size)
        x = image_to_tensor(retina_img)                             # [3, R, R]
        bbox = torch.tensor(s['target_bbox'], dtype=torch.float32)  # [4]
        cls = torch.tensor(CLASS_TO_ID[s['target_class']], dtype=torch.long)

        # Previous bbox: (x1, y1, x2, y2, class_id)
        prev = s.get('prev_bbox', PREV_BBOX_NONE)
        prev_t = torch.tensor(prev, dtype=torch.float32)            # [5]

        # Noise injection: perturb prev_bbox coords during training
        # to reduce exposure bias (skip the zero sentinel)
        if self.training and self.noise_std > 0 and prev != PREV_BBOX_NONE:
            prev_t[:4] += torch.randn(4) * self.noise_std
            prev_t[:4].clamp_(0.0, 1.0)

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

    # Compute inverse-frequency class weights from training data
    cls_counts = torch.zeros(len(CLASSES))
    for idx in train_ds.indices:
        s = full_dataset.samples[idx]
        cls_counts[CLASS_TO_ID[s['target_class']]] += 1
    cls_counts.clamp_(min=1)
    inv_freq = cls_counts.sum() / (len(CLASSES) * cls_counts)
    inv_freq.clamp_(max=3.0)
    # Ensure 'none' gets at least 2.0x weight
    inv_freq[NONE_CLASS_ID] = max(inv_freq[NONE_CLASS_ID].item(), 2.0)
    print(f'Class weights: {dict(zip(CLASSES, inv_freq.tolist()))}', flush=True)

    # Loss
    loss_fn = MusicOCRLoss(bbox_weight=1.0, class_weight=1.0,
                           cls_weights=inv_freq.to(device)).to(device)

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
        full_dataset.training = True
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
        full_dataset.training = False
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
            prev_bbox=None, retina_size: int = 1024, device=None,
            parent_label: str = None):
    """Run one forward pass and return (bbox_norm, class_name, confidence).

    If parent_label is given, masks logits to only allow valid child classes
    (+ 'none') per the VALID_CHILDREN hierarchy.
    """
    if device is None:
        device = next(model.parameters()).device
    if prev_bbox is None:
        prev_bbox = PREV_BBOX_NONE

    retina_img, scale, ox, oy = fit_to_retina(image, retina_size)
    x = image_to_tensor(retina_img).unsqueeze(0).to(device)
    prev_t = torch.tensor([prev_bbox], dtype=torch.float32).to(device)

    bbox_pred, cls_pred = model(x, prev_t)
    logits = cls_pred[0]

    # Mask invalid child classes to -inf
    if parent_label is not None and parent_label in VALID_CHILDREN:
        valid = VALID_CHILDREN[parent_label] | {'none'}
        mask = torch.full_like(logits, float('-inf'))
        for cls_name in valid:
            mask[CLASS_TO_ID[cls_name]] = 0.0
        logits = logits + mask

    bbox = bbox_pred[0].cpu().tolist()
    probs = torch.softmax(logits, dim=0)
    cls_id = probs.argmax().item()
    return bbox, CLASSES[cls_id], probs[cls_id].item()


MAX_CHILDREN = {
    'page': 4,
    'staff_system': 4,
    'melodic_line': 20,
    'note': 3,
}

# Valid child classes for each parent (enforces hierarchy structure)
VALID_CHILDREN = {
    'page':         {'staff_system'},
    'staff_system': {'staff_lines', 'melodic_line'},
    'melodic_line': {'note', 'ledger_lines'},
    'note':         {'ledger_lines'},
    'staff_lines':  set(),       # leaf
    'ledger_lines': set(),       # leaf
}


def _remove_horizontal_lines(image: Image.Image) -> Image.Image:
    """Remove long horizontal lines (staff lines) from an image.

    Returns a new PIL image with horizontal lines erased (replaced by white).
    The original image is not modified.
    """
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Detect horizontal lines — open with a wide horizontal kernel
    h_len = max(image.width // 4, 15)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Dilate the detected lines slightly so we fully cover them
    h_lines = cv2.dilate(h_lines, np.ones((3, 1), np.uint8), iterations=1)

    # Erase the lines from the original grayscale by setting those pixels
    # to white, then convert back to RGB PIL image.
    result = gray.copy()
    result[h_lines > 0] = 255
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_GRAY2RGB))


def _has_content(crop: Image.Image, area_threshold: float = 0.005,
                 canny_low: int = 50, canny_high: int = 150) -> bool:
    """Check if a crop (already preprocessed with staff lines removed) has
    meaningful content using Canny edge detection and enclosed-region analysis.

    Steps:
      1. Run Canny edge detection on the (line-free) crop.
      2. Dilate edges to connect nearby fragments, then find contours.
      3. Compute the bounding-box area of each contour (the region
         enclosed by its edges).
      4. Return True if the total enclosed area fraction exceeds
         *area_threshold*.
    """
    gray = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)

    # Dilate to connect nearby edge fragments into regions
    dilate_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edges, dilate_kern, iterations=1)

    # Find contours and sum their bounding-rect areas (the region enclosed)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    enclosed = 0.0
    for c in contours:
        _, _, bw, bh = cv2.boundingRect(c)
        enclosed += bw * bh
    return enclosed / max(gray.size, 1) >= area_threshold


@torch.no_grad()
def hierarchical_decode(model: MusicOCRModel, image: Image.Image,
                        retina_size: int = 1024, max_depth: int = 5,
                        max_detections: int = 200, device=None):
    """
    Recursively decode the bounding-box hierarchy using autoregressive
    previous-bbox conditioning.  The image is never modified — the model
    relies solely on prev_bbox to advance through siblings.

    Uses VALID_CHILDREN to constrain which classes can appear at each level,
    preventing recursive class loops (e.g. melodic_line -> melodic_line).

    As a preprocessing step, long horizontal lines (staff lines) are removed
    from the image.  The cleaned version is used for Canny edge content
    checks so that empty regions (containing only staff lines) are reliably
    rejected.  The original image is still fed to the model for predictions
    (since it was trained with staff lines present).

    Returns a BBox tree.
    """
    if device is None:
        device = next(model.parameters()).device

    # Preprocessing: remove horizontal staff lines (matches training).
    # The cleaned image is used for BOTH model input and content checks.
    img = _remove_horizontal_lines(image)

    total_detections = 0

    def _decode(img, parent_label, depth):
        nonlocal total_detections
        if depth >= max_depth or img.width < 4 or img.height < 4:
            return []

        # Leaf nodes have no valid children
        if parent_label in VALID_CHILDREN and not VALID_CHILDREN[parent_label]:
            return []

        parent_area = img.width * img.height
        children = []
        prev_bbox = PREV_BBOX_NONE
        max_kids = MAX_CHILDREN.get(parent_label, 5)

        for _ in range(max_kids):
            if total_detections >= max_detections:
                break

            bbox_n, cls_name, conf = predict(
                model, img, prev_bbox=prev_bbox,
                retina_size=retina_size, device=device,
                parent_label=parent_label)

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

            # Canny edge content check (image is already line-free).
            # Line-type classes bypass this (they may have no edges left).
            child_crop = img.crop((x1, y1, x2, y2))
            if cls_name not in ('staff_lines', 'ledger_lines'):
                if not _has_content(child_crop):
                    prev_bbox = (bbox_n[0], bbox_n[1], bbox_n[2], bbox_n[3],
                                 float(CLASS_TO_ID[cls_name]))
                    continue

            total_detections += 1

            sub_children = _decode(child_crop, cls_name, depth + 1)

            # Offset sub-children from crop-local to parent coordinates
            offset_children = [
                _offset_bbox(sc, x1, y1) for sc in sub_children
            ]

            children.append(BBox(x1, y1, x2, y2, cls_name,
                                 children=offset_children))

            prev_bbox = (bbox_n[0], bbox_n[1], bbox_n[2], bbox_n[3],
                         float(CLASS_TO_ID[cls_name]))

        return children

    subs = _decode(img, 'page', 0)
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
# Bottom-up detection (adapted from tiny-tessarachnid)
# ──────────────────────────────────────────────────────────────────────────────

def _generate_candidate_boxes(region_crop, bg_color=(255, 255, 255),
                               gap_tolerance=1, scan_axis='cols'):
    """Generate candidate boxes from a region image.

    Uses content regions as natural candidate boxes.  For regions whose
    primary dimension exceeds 1.5× the expected size (touching symbols),
    generates overlapping sub-windows at stride = expected_size // 2.
    """
    regions = _find_content_regions(region_crop, bg_color,
                                    gap_tolerance=gap_tolerance,
                                    scan_axis=scan_axis)
    if not regions:
        return []

    # Estimate expected size from median region dimension along scan axis
    if scan_axis == 'cols':
        sizes = [r[2] - r[0] for r in regions]   # widths
    else:
        sizes = [r[3] - r[1] for r in regions]   # heights
    expected_size = int(np.median(sizes)) if sizes else 1
    expected_size = max(expected_size, 1)

    candidates = []
    for (x1, y1, x2, y2) in regions:
        if scan_axis == 'cols':
            dim = x2 - x1
            if dim <= expected_size * 1.5:
                candidates.append((x1, y1, x2, y2))
            else:
                stride = max(expected_size // 2, 1)
                for sx in range(x1, x2 - expected_size + 1, stride):
                    candidates.append((sx, y1, sx + expected_size, y2))
                if candidates and candidates[-1][2] < x2:
                    candidates.append((x2 - expected_size, y1, x2, y2))
        else:
            dim = y2 - y1
            if dim <= expected_size * 1.5:
                candidates.append((x1, y1, x2, y2))
            else:
                stride = max(expected_size // 2, 1)
                for sy in range(y1, y2 - expected_size + 1, stride):
                    candidates.append((x1, sy, x2, sy + expected_size))
                if candidates and candidates[-1][3] < y2:
                    candidates.append((x1, y2 - expected_size, x2, y2))

    return candidates


@torch.no_grad()
def _classify_candidates_batched(model, region_crop, bg_color, candidates,
                                  device, retina_size=1024, batch_size=32,
                                  parent_label=None):
    """Classify candidate boxes in batches.

    For each candidate: crop from region image, pad to match pretraining
    distribution (~1/5th occupancy), fit to retina, run model with
    prev_bbox = PREV_BBOX_NONE.

    If *parent_label* is given, logits are masked to only allow valid child
    classes (+ 'none') per VALID_CHILDREN.

    Returns list of (box, class_name, confidence) for non-'none' predictions.
    """
    results = []
    prev_t = torch.tensor([PREV_BBOX_NONE], dtype=torch.float32).to(device)

    # Build valid-child mask once if needed
    valid_mask = None
    if parent_label is not None and parent_label in VALID_CHILDREN:
        valid = VALID_CHILDREN[parent_label] | {'none'}
        valid_mask = torch.full((len(CLASSES),), float('-inf'), device=device)
        for cls_name in valid:
            valid_mask[CLASS_TO_ID[cls_name]] = 0.0

    for i in range(0, len(candidates), batch_size):
        batch_boxes = candidates[i:i + batch_size]
        imgs = []
        for (x1, y1, x2, y2) in batch_boxes:
            crop = region_crop.crop((x1, y1, x2, y2))
            cw, ch = crop.size
            if cw == 0 or ch == 0:
                continue
            # Pad to match pretraining distribution (symbol occupies ~1/5th)
            pad_x = cw * 2
            pad_y = ch * 2
            padded = Image.new('RGB', (cw + pad_x * 2, ch + pad_y * 2),
                               bg_color)
            padded.paste(crop, (pad_x, pad_y))
            retina_img, _, _, _ = fit_to_retina(padded, retina_size)
            img_t = image_to_tensor(retina_img)
            imgs.append((img_t, (x1, y1, x2, y2)))

        if not imgs:
            continue

        img_batch = torch.stack([t for t, _ in imgs]).to(device)
        prev_batch = prev_t.expand(img_batch.size(0), -1)

        _, cls_pred = model(img_batch, prev_batch)

        logits = cls_pred
        if valid_mask is not None:
            logits = logits + valid_mask.unsqueeze(0)

        probs = torch.softmax(logits, dim=1)
        cls_ids = probs.argmax(dim=1)
        confidences = probs.gather(1, cls_ids.unsqueeze(1)).squeeze(1)

        for j, (_, box) in enumerate(imgs):
            cid = cls_ids[j].item()
            cls_name = CLASSES[cid]
            conf = confidences[j].item()
            if cls_name != 'none':
                results.append((box, cls_name, conf))

    return results


def _iou(box_a, box_b):
    """Compute intersection-over-union of two (x1, y1, x2, y2) boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(detections, iou_threshold=0.3):
    """Non-maximum suppression on detections.

    detections: list of (box, class_name, confidence)
    Returns filtered list of (box, class_name) sorted left-to-right.
    """
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d[2], reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if _iou(best[0], d[0]) < iou_threshold]

    keep.sort(key=lambda d: d[0][0])
    return [(box, cls_name) for box, cls_name, _ in keep]


def _detect_bottom_up(model, region_crop, device, retina_size=1024,
                       bg_color=(255, 255, 255), gap_tolerance=1,
                       scan_axis='cols', iou_threshold=0.3,
                       confidence_threshold=0.1, batch_size=32,
                       parent_label=None):
    """Bottom-up detection: candidates → classify → NMS.

    Returns [(bbox, class_name), ...] in left-to-right order.
    """
    candidates = _generate_candidate_boxes(region_crop, bg_color,
                                            gap_tolerance=gap_tolerance,
                                            scan_axis=scan_axis)
    if not candidates:
        return []

    detections = _classify_candidates_batched(
        model, region_crop, bg_color, candidates, device,
        retina_size=retina_size, batch_size=batch_size,
        parent_label=parent_label)

    # Filter by confidence
    detections = [d for d in detections if d[2] >= confidence_threshold]

    return _nms(detections, iou_threshold=iou_threshold)


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
      - melodic_line → notes:        bottom-up (candidates → classify → NMS)
      - note → ledger_lines:         row-scan, gap_tolerance=3

    The melodic_line level uses bottom-up detection (adapted from
    tiny-tessarachnid): pixel analysis generates candidate boxes, each
    candidate is independently classified in batches, then NMS filters
    overlapping detections.  This avoids autoregressive error accumulation
    at the most granular detection level.

    Returns a BBox tree.
    """
    if device is None:
        device = next(model.parameters()).device

    # Scan parameters per parent label
    # staff_lines and ledger_lines are leaves — no children to decode
    LEVEL_PARAMS = {
        'page':         {'scan_axis': 'rows', 'gap_tolerance': 20},
        'staff_system': {'scan_axis': 'rows', 'gap_tolerance': 3},
        'melodic_line': {'scan_axis': 'cols', 'gap_tolerance': 3},
        'note':         {'scan_axis': 'rows', 'gap_tolerance': 3},
    }

    # Levels that use bottom-up detection (candidates → classify → NMS)
    # instead of autoregressive counting.  melodic_line → notes is the
    # leaf-like level analogous to line → characters in OCR.
    BOTTOM_UP_LEVELS = {'melodic_line'}

    def _decode(img, parent_label, depth):
        if depth >= max_depth or img.width < 4 or img.height < 4:
            return []

        params = LEVEL_PARAMS.get(parent_label,
                                  {'scan_axis': 'cols', 'gap_tolerance': 3})

        if parent_label in BOTTOM_UP_LEVELS:
            detections = _detect_bottom_up(
                model, img, device, retina_size=retina_size,
                bg_color=bg_color,
                gap_tolerance=params['gap_tolerance'],
                scan_axis=params['scan_axis'],
                parent_label=parent_label,
            )
        else:
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
