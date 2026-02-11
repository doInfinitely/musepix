"""
Dual-retina symbol similarity model.

Architecture
────────────
Two inputs are each scaled (preserving aspect ratio) and padded to fit
a square retina.  A shared CNN backbone extracts features from each retina.
The two feature vectors are concatenated and fed to a classifier that
outputs same-kind / different-kind.

Usage:
    # Build the pair dataset first:
    python3.10 symbol_similarity_dataset.py

    # Then train:
    python3.10 symbol_similarity_model.py \
        --pairs_json data/symbol_pairs/pairs.json \
        --symbols_dir sheet_music_symbols_noto_music_images \
        --epochs 30
"""

import os
import json
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from retina import fit_to_retina, image_to_tensor
from symbol_similarity_dataset import SymbolPairDataset


# ──────────────────────────────────────────────────────────────────────────────
# Shared backbone (lighter than the OCR backbone — symbol images are smaller)
# ──────────────────────────────────────────────────────────────────────────────

def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SymbolBackbone(nn.Module):
    """CNN backbone sized for 256×256 retinas (configurable)."""

    def __init__(self):
        super().__init__()
        # 256 → 128 → 64 → 32 → 16 → 8 → 4
        self.features = nn.Sequential(
            _conv_block(3, 32),  nn.MaxPool2d(2),   # 128
            _conv_block(32, 64), nn.MaxPool2d(2),   # 64
            _conv_block(64, 128), nn.MaxPool2d(2),  # 32
            _conv_block(128, 256), nn.MaxPool2d(2), # 16
            _conv_block(256, 512), nn.MaxPool2d(2), # 8
            nn.AdaptiveAvgPool2d(4),                # 4
        )
        self.feature_dim = 512 * 4 * 4  # 8192

    def forward(self, x):
        return self.features(x).flatten(1)


# ──────────────────────────────────────────────────────────────────────────────
# Dual-retina Siamese model
# ──────────────────────────────────────────────────────────────────────────────

class DualRetinaModel(nn.Module):
    """
    Two inputs → shared backbone → concatenated features → same / different.

    The backbone weights are shared (Siamese), so the model learns a
    representation that is invariant to the input slot.
    """

    def __init__(self):
        super().__init__()
        self.backbone = SymbolBackbone()
        dim = self.backbone.feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),   # 0 = different, 1 = same
        )

    def forward(self, x1, x2):
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        combined = torch.cat([f1, f2], dim=1)
        return self.classifier(combined)

    @torch.no_grad()
    def predict(self, img_a, img_b, retina_size=256, device=None):
        """
        Convenience method: takes two PIL Images, returns
        (same: bool, confidence: float).
        """
        if device is None:
            device = next(self.parameters()).device
        self.eval()

        ra, *_ = fit_to_retina(img_a, retina_size)
        rb, *_ = fit_to_retina(img_b, retina_size)
        ta = image_to_tensor(ra).unsqueeze(0).to(device)
        tb = image_to_tensor(rb).unsqueeze(0).to(device)

        logits = self(ta, tb)
        probs = torch.softmax(logits, dim=1)[0]
        pred = probs.argmax().item()
        return bool(pred), probs[pred].item()


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    device = (
        torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu')
    )
    print(f'Device: {device}')

    # Load pairs manifest
    with open(args.pairs_json) as f:
        pairs = json.load(f)
    print(f'Loaded {len(pairs)} pairs from {args.pairs_json}')

    dataset = SymbolPairDataset(pairs, retina_size=args.retina_size)

    # Train / val split
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    model = DualRetinaModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        total_loss = 0.0
        correct = total = 0
        for ta, tb, labels in train_loader:
            ta, tb, labels = ta.to(device), tb.to(device), labels.to(device)

            logits = model(ta, tb)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # ── validate ──
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for ta, tb, labels in val_loader:
                ta, tb, labels = ta.to(device), tb.to(device), labels.to(device)
                logits = model(ta, tb)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / max(val_total, 1)

        print(f'Epoch {epoch:3d}/{args.epochs}  '
              f'loss={train_loss:.4f}  '
              f'train_acc={train_acc:.3f}  val_acc={val_acc:.3f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/symbol_similarity.pt')

    print(f'\nBest validation accuracy: {best_val_acc:.3f}')
    print('Saved → checkpoints/symbol_similarity.pt')


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train the dual-retina symbol similarity model.')
    parser.add_argument('--pairs_json', type=str,
                        default='data/symbol_pairs/pairs.json',
                        help='Path to pairs manifest from symbol_similarity_dataset.py.')
    parser.add_argument('--symbols_dir', type=str,
                        default='sheet_music_symbols_noto_music_images')
    parser.add_argument('--retina_size', type=int, default=256,
                        help='Square retina size for each input.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
