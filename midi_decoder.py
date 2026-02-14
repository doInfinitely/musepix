"""
MIDI pitch decoder — GPT-2-style autoregressive model for melodic lines.

Given a cropped melodic-line image, predicts the sequence of MIDI pitches
(diatonic, C3–B5) using a shared ResNet-18 backbone (same as MusicOCRModel)
and a causal transformer decoder head.

Architecture
────────────
Visual encoder : ResNet-18 backbone → AdaptiveAvgPool → 512-d
                 → linear projection → num_visual_tokens (4) × d_model (256)
Token embedding: MIDI_VOCAB_SIZE (24) → 256-d
Positional emb : learnable, max_seq_len (30) positions
Decoder        : nn.TransformerEncoder with causal mask, 4 layers, 4 heads
Output         : 256 → 24 logits

Training uses teacher forcing; inference uses greedy autoregressive decoding.

Usage:
    python3 midi_decoder.py train [--ocr_checkpoint ...] [--epochs 50]
    python3 midi_decoder.py infer --checkpoint checkpoints/midi_decoder.pt --image img.png
"""

import os
import json
import math
import argparse
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

from retina import fit_to_retina, image_to_tensor
from generate_sheet_music import (
    BBox, SheetMusicGenerator,
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, PITCH_OFFSET,
    NUM_PITCHES, MIDI_VOCAB_SIZE,
    midi_to_token, token_to_midi,
    extract_midi_training_samples,
)

RETINA_SIZE = 1024


# ──────────────────────────────────────────────────────────────────────────────
# GPT-2-style decoder head
# ──────────────────────────────────────────────────────────────────────────────

class MidiDecoderHead(nn.Module):
    """Causal transformer that takes visual prefix tokens + MIDI tokens
    and predicts next-token logits over MIDI_VOCAB_SIZE."""

    def __init__(self, d_model: int = 256, nhead: int = 4, num_layers: int = 4,
                 num_visual_tokens: int = 4, max_seq_len: int = 30,
                 backbone_dim: int = 512):
        super().__init__()
        self.d_model = d_model
        self.num_visual_tokens = num_visual_tokens
        self.max_seq_len = max_seq_len

        # Project backbone features → visual prefix tokens
        self.visual_proj = nn.Linear(backbone_dim, num_visual_tokens * d_model)

        # Token embedding for MIDI vocabulary
        self.token_emb = nn.Embedding(MIDI_VOCAB_SIZE, d_model)

        # Learnable positional embeddings
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder used as decoder (with causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, MIDI_VOCAB_SIZE)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask (True = masked)."""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(self, visual_features: torch.Tensor,
                input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (B, backbone_dim) from shared backbone
            input_tokens:    (B, T) token indices (BOS, pitch tokens, ...)
        Returns:
            logits: (B, num_visual_tokens + T, MIDI_VOCAB_SIZE)
        """
        B, T = input_tokens.shape

        # Visual prefix: (B, backbone_dim) → (B, num_visual_tokens, d_model)
        vis = self.visual_proj(visual_features)
        vis = vis.view(B, self.num_visual_tokens, self.d_model)

        # Token embeddings: (B, T, d_model)
        tok = self.token_emb(input_tokens)

        # Concatenate: (B, num_visual_tokens + T, d_model)
        seq = torch.cat([vis, tok], dim=1)
        total_len = seq.size(1)

        # Add positional embeddings
        positions = torch.arange(total_len, device=seq.device)
        seq = seq + self.pos_emb(positions).unsqueeze(0)

        # Causal mask
        mask = self._causal_mask(total_len, seq.device)

        # Transformer
        out = self.transformer(seq, mask=mask)

        # Project to vocab logits
        return self.output_proj(out)

    @torch.no_grad()
    def generate(self, visual_features: torch.Tensor,
                 max_len: int = 25) -> List[int]:
        """Greedy autoregressive decoding for a single example.

        Args:
            visual_features: (1, backbone_dim) from shared backbone
        Returns:
            list of MIDI note numbers (excluding BOS/EOS)
        """
        device = visual_features.device
        tokens = [BOS_TOKEN]

        for _ in range(max_len):
            input_tokens = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = self.forward(visual_features, input_tokens)
            # Take logits at last position
            next_logit = logits[0, -1, :]  # (MIDI_VOCAB_SIZE,)
            next_token = next_logit.argmax().item()

            if next_token == EOS_TOKEN:
                break
            tokens.append(next_token)

        # Convert pitch tokens to MIDI numbers
        midi_notes = []
        for t in tokens[1:]:  # skip BOS
            if t >= PITCH_OFFSET and t < PITCH_OFFSET + NUM_PITCHES:
                midi_notes.append(token_to_midi(t))
        return midi_notes


# ──────────────────────────────────────────────────────────────────────────────
# Full model: shared backbone + decoder head
# ──────────────────────────────────────────────────────────────────────────────

class MidiOCRModel(nn.Module):
    """Shared ResNet-18 backbone + GPT-2 MIDI decoder head."""

    def __init__(self, d_model: int = 256, nhead: int = 4, num_layers: int = 4,
                 num_visual_tokens: int = 4, max_seq_len: int = 30):
        super().__init__()

        # Same ResNet-18 backbone as MusicOCRModel
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

        # Decoder head
        self.decoder = MidiDecoderHead(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            num_visual_tokens=num_visual_tokens, max_seq_len=max_seq_len,
            backbone_dim=512,
        )

    def extract_features(self, img: torch.Tensor) -> torch.Tensor:
        """Extract 512-d features from image tensor.

        Args:
            img: (B, 3, H, W) in [0, 1]
        Returns:
            (B, 512) feature vector
        """
        x = (img - self.img_mean) / self.img_std
        x = self.backbone(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)

    def forward(self, img: torch.Tensor,
                input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img:          (B, 3, H, W) in [0, 1]
            input_tokens: (B, T) token indices
        Returns:
            logits: (B, num_visual_tokens + T, MIDI_VOCAB_SIZE)
        """
        features = self.extract_features(img)
        return self.decoder(features, input_tokens)

    @torch.no_grad()
    def predict(self, img: torch.Tensor, max_len: int = 25) -> List[int]:
        """Run greedy decoding on a single image.

        Args:
            img: (1, 3, H, W) in [0, 1]
        Returns:
            list of MIDI note numbers
        """
        self.eval()
        features = self.extract_features(img)
        return self.decoder.generate(features, max_len=max_len)

    def load_backbone_from_ocr(self, checkpoint_path: str):
        """Load backbone weights from an existing MusicOCRModel checkpoint."""
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        backbone_keys = {k: v for k, v in state.items()
                         if k.startswith('backbone.')}
        pool_keys = {k: v for k, v in state.items()
                     if k.startswith('pool.')}
        norm_keys = {k: v for k, v in state.items()
                     if k.startswith('img_mean') or k.startswith('img_std')}
        all_keys = {**backbone_keys, **pool_keys, **norm_keys}
        if all_keys:
            missing, unexpected = self.load_state_dict(all_keys, strict=False)
            print(f'Loaded backbone from OCR checkpoint: '
                  f'{len(all_keys)} keys, '
                  f'{len(missing)} missing (decoder head), '
                  f'{len(unexpected)} unexpected')
        else:
            print('Warning: no backbone keys found in checkpoint')


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class MidiDataset(Dataset):
    """Dataset of melodic-line crops paired with MIDI token sequences.

    Either generates on-the-fly from SheetMusicGenerator, or loads from
    a pre-generated data directory with annotations.json.
    """

    def __init__(self, data_dir: str = None, num_images: int = 200,
                 retina_size: int = RETINA_SIZE, seed: int = 42,
                 max_seq_len: int = 25):
        self.retina_size = retina_size
        self.max_seq_len = max_seq_len
        self.samples: List[dict] = []

        if data_dir and os.path.isfile(os.path.join(data_dir, 'annotations.json')):
            self._load_from_disk(data_dir)
        else:
            self._generate(num_images, seed)

        print(f'MidiDataset: {len(self.samples)} melodic-line samples')

    def _load_from_disk(self, data_dir):
        with open(os.path.join(data_dir, 'annotations.json')) as f:
            annotations = json.load(f)

        for ann in annotations:
            img_path = os.path.join(data_dir, 'images', ann['image'])
            img = Image.open(img_path).convert('RGB')
            tree = BBox.from_dict(ann['bbox_tree'])
            self.samples.extend(extract_midi_training_samples(img, tree))

    def _generate(self, num_images, seed):
        gen = SheetMusicGenerator()
        random.seed(seed)
        for i in range(num_images):
            img, tree = gen.generate(seed=seed + i)
            self.samples.extend(extract_midi_training_samples(img, tree))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Prepare image
        retina_img, _, _, _ = fit_to_retina(
            s['melodic_line_image'], self.retina_size)
        img_tensor = image_to_tensor(retina_img)  # (3, R, R)

        # Token sequence: [BOS, tok1, ..., tokN, EOS]
        token_seq = s['token_sequence']

        # Input tokens: [BOS, tok1, ..., tokN] (teacher forcing input)
        # Target tokens: [tok1, ..., tokN, EOS] (shifted right)
        input_tokens = token_seq[:-1]
        target_tokens = token_seq[1:]

        # Pad to max_seq_len
        input_len = len(input_tokens)
        pad_len = self.max_seq_len - input_len
        if pad_len > 0:
            input_tokens = input_tokens + [PAD_TOKEN] * pad_len
            target_tokens = target_tokens + [PAD_TOKEN] * pad_len
        else:
            input_tokens = input_tokens[:self.max_seq_len]
            target_tokens = target_tokens[:self.max_seq_len]

        input_t = torch.tensor(input_tokens, dtype=torch.long)
        target_t = torch.tensor(target_tokens, dtype=torch.long)

        return img_tensor, input_t, target_t


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

class MidiLoss(nn.Module):
    """Cross-entropy loss on text-position logits only (skip visual prefix)."""

    def __init__(self, num_visual_tokens: int = 4):
        super().__init__()
        self.num_visual_tokens = num_visual_tokens
        self.ce = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    def forward(self, logits: torch.Tensor,
                target_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:        (B, num_visual_tokens + T, MIDI_VOCAB_SIZE)
            target_tokens: (B, T) target token indices
        """
        # Skip visual prefix positions, take only text positions
        text_logits = logits[:, self.num_visual_tokens:, :]
        B, T, V = text_logits.shape
        # Truncate to match target length
        T_target = target_tokens.size(1)
        T_min = min(T, T_target)
        text_logits = text_logits[:, :T_min, :]
        target = target_tokens[:, :T_min]
        return self.ce(text_logits.reshape(-1, V), target.reshape(-1))


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_midi(args):
    device = (
        torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu')
    )
    print(f'Device: {device}', flush=True)

    # Dataset
    full_dataset = MidiDataset(
        data_dir=args.data_dir,
        num_images=args.num_images,
        retina_size=args.retina_size,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
    )

    # Train / validation split
    val_size = int(len(full_dataset) * 0.15)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f'Train: {train_size}, Val: {val_size}', flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = MidiOCRModel(
        d_model=256, nhead=4, num_layers=4,
        num_visual_tokens=4, max_seq_len=args.max_seq_len + 4,
    ).to(device)

    # Optionally load backbone from OCR checkpoint
    if args.ocr_checkpoint:
        model.load_backbone_from_ocr(args.ocr_checkpoint)

    # Optionally freeze backbone
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print('Backbone frozen', flush=True)

    # Differential LR: backbone × 0.1, head × 1.0
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = (list(model.pool.parameters())
                   + list(model.decoder.parameters()))
    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': args.lr * 0.1})
    param_groups.append({'params': head_params, 'lr': args.lr})

    optimizer = optim.AdamW(param_groups, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    loss_fn = MidiLoss(num_visual_tokens=4).to(device)

    # Training loop
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = 'checkpoints/midi_decoder.pt'
    best_val_loss = float('inf')
    num_batches = len(train_loader)
    log_interval = max(1, num_batches // 5)

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total_tokens = 0
        train_n = 0

        for batch_idx, (img, input_tok, target_tok) in enumerate(train_loader, 1):
            img = img.to(device)
            input_tok = input_tok.to(device)
            target_tok = target_tok.to(device)

            logits = model(img, input_tok)
            loss = loss_fn(logits, target_tok)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = img.size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs

            # Token-level accuracy (ignoring PAD)
            text_logits = logits[:, 4:, :]
            T_min = min(text_logits.size(1), target_tok.size(1))
            preds = text_logits[:, :T_min, :].argmax(dim=-1)
            tgt = target_tok[:, :T_min]
            non_pad = tgt != PAD_TOKEN
            train_correct += (preds[non_pad] == tgt[non_pad]).sum().item()
            train_total_tokens += non_pad.sum().item()

            if batch_idx % log_interval == 0:
                print(f'  Epoch {epoch:3d}  batch {batch_idx}/{num_batches}  '
                      f'loss={loss.item():.4f}', flush=True)

        train_loss = train_loss_sum / max(train_n, 1)
        train_acc = train_correct / max(train_total_tokens, 1)

        # ── Validate ──
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total_tokens = 0
        val_n = 0

        with torch.no_grad():
            for img, input_tok, target_tok in val_loader:
                img = img.to(device)
                input_tok = input_tok.to(device)
                target_tok = target_tok.to(device)

                logits = model(img, input_tok)
                loss = loss_fn(logits, target_tok)

                bs = img.size(0)
                val_loss_sum += loss.item() * bs
                val_n += bs

                text_logits = logits[:, 4:, :]
                T_min = min(text_logits.size(1), target_tok.size(1))
                preds = text_logits[:, :T_min, :].argmax(dim=-1)
                tgt = target_tok[:, :T_min]
                non_pad = tgt != PAD_TOKEN
                val_correct += (preds[non_pad] == tgt[non_pad]).sum().item()
                val_total_tokens += non_pad.sum().item()

        val_loss = val_loss_sum / max(val_n, 1)
        val_acc = val_correct / max(val_total_tokens, 1)

        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[-1]['lr']

        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'train loss={train_loss:.4f} acc={train_acc:.3f} | '
              f'val loss={val_loss:.4f} acc={val_acc:.3f} | '
              f'lr={cur_lr:.2e}', flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f'  -> saved best model ({ckpt_path})', flush=True)

    print(f'Training complete. Best val loss: {best_val_loss:.4f}')


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_midi_for_tree(midi_model: MidiOCRModel, image: Image.Image,
                         bbox_tree: BBox, retina_size: int = RETINA_SIZE,
                         device=None):
    """Walk detected BBox tree, find melodic_line nodes, predict MIDI pitches,
    and attach them to note children.

    Args:
        midi_model: trained MidiOCRModel
        image: full page image
        bbox_tree: detected BBox tree (from MusicOCRModel)
        retina_size: retina size for image preprocessing
        device: torch device

    Returns:
        bbox_tree (modified in place with midi fields set on note nodes)
    """
    if device is None:
        device = next(midi_model.parameters()).device
    midi_model.eval()

    def _walk(node):
        if node.label == 'melodic_line' and node.children:
            # Crop melodic line
            x1 = max(0, int(node.x1))
            y1 = max(0, int(node.y1))
            x2 = min(image.width, int(node.x2))
            y2 = min(image.height, int(node.y2))
            if x2 - x1 > 0 and y2 - y1 > 0:
                crop = image.crop((x1, y1, x2, y2))
                retina_img, _, _, _ = fit_to_retina(crop, retina_size)
                img_tensor = image_to_tensor(retina_img).unsqueeze(0).to(device)

                midi_notes = midi_model.predict(img_tensor)

                # Attach to note children sorted left-to-right
                notes = sorted(
                    [c for c in node.children if c.label == 'note'],
                    key=lambda c: c.x1,
                )
                for i, note_node in enumerate(notes):
                    if i < len(midi_notes):
                        note_node.midi = midi_notes[i]

        for child in node.children:
            _walk(child)

    _walk(bbox_tree)
    return bbox_tree


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train or run inference with the MIDI pitch decoder.')
    sub = parser.add_subparsers(dest='command')

    # ── train ──
    tr = sub.add_parser('train', help='Train the MIDI decoder.')
    tr.add_argument('--data_dir', type=str, default=None,
                    help='Pre-generated data directory (with annotations.json).')
    tr.add_argument('--num_images', type=int, default=200,
                    help='Images to generate on-the-fly if no data_dir.')
    tr.add_argument('--ocr_checkpoint', type=str, default=None,
                    help='MusicOCRModel checkpoint to initialise backbone from.')
    tr.add_argument('--freeze_backbone', action='store_true',
                    help='Freeze the backbone during training.')
    tr.add_argument('--retina_size', type=int, default=RETINA_SIZE)
    tr.add_argument('--max_seq_len', type=int, default=25)
    tr.add_argument('--batch_size', type=int, default=8)
    tr.add_argument('--epochs', type=int, default=50)
    tr.add_argument('--lr', type=float, default=1e-4)
    tr.add_argument('--seed', type=int, default=42)

    # ── infer ──
    inf = sub.add_parser('infer', help='Run inference on a melodic line image.')
    inf.add_argument('--checkpoint', type=str,
                     default='checkpoints/midi_decoder.pt',
                     help='Model checkpoint to load.')
    inf.add_argument('--image', type=str, required=True,
                     help='Path to a melodic line image.')
    inf.add_argument('--retina_size', type=int, default=RETINA_SIZE)

    args = parser.parse_args()

    if args.command == 'train':
        train_midi(args)
    elif args.command == 'infer':
        device = (
            torch.device('mps') if torch.backends.mps.is_available()
            else torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu')
        )
        model = MidiOCRModel().to(device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        model.eval()

        img = Image.open(args.image).convert('RGB')
        retina_img, _, _, _ = fit_to_retina(img, args.retina_size)
        img_tensor = image_to_tensor(retina_img).unsqueeze(0).to(device)

        midi_notes = model.predict(img_tensor)
        note_names = []
        for m in midi_notes:
            octave = (m // 12) - 1
            semitone = m % 12
            name_map = {0: 'C', 2: 'D', 4: 'E', 5: 'F', 7: 'G', 9: 'A', 11: 'B'}
            name = name_map.get(semitone, f'?{semitone}')
            note_names.append(f'{name}{octave}')

        print(f'MIDI notes: {midi_notes}')
        print(f'Note names: {" ".join(note_names)}')
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
