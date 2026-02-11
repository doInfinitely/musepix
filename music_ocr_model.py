"""
Music OCR model — single-retina bounding-box detector with teacher forcing.

Architecture
────────────
Input  : 1024 × 1024 RGB image (the retina)
Output : (x1, y1, x2, y2) normalised bbox  +  class_id

The model is a simple CNN backbone → two heads (bbox regression, classification).

Training with teacher forcing
─────────────────────────────
For each synthetic sheet-music image we build a tree of bounding boxes.
Walking the tree produces (input_crop, target_bbox, target_class) tuples
where each successive sibling has the prior siblings masked out of the
input — see generate_sheet_music.extract_training_samples().

Usage:
    python3.10 music_ocr_model.py --data_dir data/sheet_music --epochs 30
"""

import os
import json
import argparse
import random
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from retina import fit_to_retina, image_to_tensor
from generate_sheet_music import (
    CLASSES, CLASS_TO_ID, BBox, SheetMusicGenerator,
    extract_training_samples,
)

# ──────────────────────────────────────────────────────────────────────────────
# CNN backbone (works without torchvision)
# ──────────────────────────────────────────────────────────────────────────────

def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class SimpleBackbone(nn.Module):
    """Lightweight CNN: 1024 → 4×4 feature map via conv-pool stages."""

    def __init__(self):
        super().__init__()
        # 1024→512→256→128→64→32→16→8→4
        self.stages = nn.Sequential(
            _conv_block(3, 32),   nn.MaxPool2d(2),   # 512
            _conv_block(32, 64),  nn.MaxPool2d(2),   # 256
            _conv_block(64, 128), nn.MaxPool2d(2),   # 128
            _conv_block(128, 256), nn.MaxPool2d(2),  # 64
            _conv_block(256, 512), nn.MaxPool2d(2),  # 32
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # 4
        )
        self.feature_dim = 512 * 4 * 4  # 8192

    def forward(self, x):
        return self.stages(x).flatten(1)


# ──────────────────────────────────────────────────────────────────────────────
# OCR model
# ──────────────────────────────────────────────────────────────────────────────

class MusicOCRModel(nn.Module):
    """
    Single-retina detector.  For a given retina image the model predicts:
        • bbox — (x1, y1, x2, y2) in [0, 1]   (the first / most abstract box)
        • cls  — logits over CLASSES
    """

    def __init__(self, num_classes: int = len(CLASSES)):
        super().__init__()
        self.backbone = SimpleBackbone()
        dim = self.backbone.feature_dim

        self.bbox_head = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),   # output in [0, 1]
        )
        self.cls_head = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.bbox_head(features), self.cls_head(features)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset — on-the-fly generation or from pre-generated annotations
# ──────────────────────────────────────────────────────────────────────────────

class TeacherForcingDataset(Dataset):
    """
    Lazily generates sheet music and extracts teacher-forcing samples.

    Two modes:
        1. From annotations file  (data_dir with images/ and annotations.json)
        2. On-the-fly generation  (no data_dir — creates images in memory)
    """

    def __init__(self, data_dir: str = None, num_images: int = 200,
                 retina_size: int = 1024, seed: int = 42):
        self.retina_size = retina_size
        self.samples: List[dict] = []

        if data_dir and os.path.isfile(os.path.join(data_dir, 'annotations.json')):
            self._load_from_disk(data_dir)
        else:
            self._generate(num_images, seed)

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
        x = image_to_tensor(retina_img)                       # [3, R, R]
        bbox = torch.tensor(s['target_bbox'], dtype=torch.float32)  # [4]
        cls = torch.tensor(CLASS_TO_ID[s['target_class']], dtype=torch.long)
        return x, bbox, cls


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    device = (
        torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu')
    )
    print(f'Device: {device}')

    dataset = TeacherForcingDataset(
        data_dir=args.data_dir,
        num_images=args.num_images,
        retina_size=args.retina_size,
        seed=args.seed,
    )
    print(f'Dataset: {len(dataset)} training samples')

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)

    model = MusicOCRModel(num_classes=len(CLASSES)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    bbox_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_bbox_loss = 0.0
        total_cls_loss = 0.0
        n_batches = 0

        for x, bbox_target, cls_target in loader:
            x = x.to(device)
            bbox_target = bbox_target.to(device)
            cls_target = cls_target.to(device)

            bbox_pred, cls_pred = model(x)

            loss_bbox = bbox_criterion(bbox_pred, bbox_target)
            loss_cls = cls_criterion(cls_pred, cls_target)
            loss = loss_bbox + loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bbox_loss += loss_bbox.item()
            total_cls_loss += loss_cls.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        avg_bb = total_bbox_loss / max(n_batches, 1)
        avg_cl = total_cls_loss / max(n_batches, 1)
        print(f'Epoch {epoch:3d}/{args.epochs}  '
              f'loss={avg:.4f}  bbox={avg_bb:.4f}  cls={avg_cl:.4f}')

    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = 'checkpoints/music_ocr.pt'
    torch.save(model.state_dict(), ckpt_path)
    print(f'Saved checkpoint → {ckpt_path}')


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model: MusicOCRModel, image: Image.Image,
            retina_size: int = 1024, device=None):
    """Run one forward pass and return (bbox_norm, class_name, confidence)."""
    if device is None:
        device = next(model.parameters()).device

    retina_img, scale, ox, oy = fit_to_retina(image, retina_size)
    x = image_to_tensor(retina_img).unsqueeze(0).to(device)

    bbox_pred, cls_pred = model(x)
    bbox = bbox_pred[0].cpu().tolist()
    probs = torch.softmax(cls_pred[0], dim=0)
    cls_id = probs.argmax().item()
    return bbox, CLASSES[cls_id], probs[cls_id].item()


@torch.no_grad()
def hierarchical_decode(model: MusicOCRModel, image: Image.Image,
                        retina_size: int = 1024, max_depth: int = 10,
                        device=None):
    """
    Recursively decode the bounding-box hierarchy using teacher forcing at
    inference time (greedy).

    Returns a BBox tree.
    """
    if device is None:
        device = next(model.parameters()).device

    def _decode(img, depth):
        if depth > max_depth or img.width < 2 or img.height < 2:
            return []

        children = []
        working = img.copy()

        for _ in range(50):  # safety cap
            bbox_n, cls_name, conf = predict(model, working, retina_size, device)
            if cls_name == 'none' or conf < 0.3:
                break

            x1 = int(bbox_n[0] * img.width)
            y1 = int(bbox_n[1] * img.height)
            x2 = int(bbox_n[2] * img.width)
            y2 = int(bbox_n[3] * img.height)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                break

            child_crop = img.crop((x1, y1, x2, y2))
            sub_children = _decode(child_crop, depth + 1)

            children.append(BBox(x1, y1, x2, y2, cls_name,
                                 children=sub_children))

            # Mask out this child
            from PIL import ImageDraw
            ImageDraw.Draw(working).rectangle([x1, y1, x2, y2], fill='white')

        return children

    subs = _decode(image, 0)
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
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
