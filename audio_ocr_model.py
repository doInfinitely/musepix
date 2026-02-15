"""
Audio OCR model — hierarchical sound detector with teacher forcing.

Architecture
────────────
Input  : mel-spectrogram image scaled to retina (1024 × 1024 RGB)
Output : start (scalar), end (scalar), freq_mask (n_mels), class logits

The model identifies the *first / most prominent* sound event in the
spectrogram by predicting its temporal bounds and a per-mel-bin frequency
mask.  Teacher forcing then masks that event out and the model detects
the next event, while also zooming into the detected region to recursively
decompose its structure.

Training
────────
From each synthetic song we build an AudioBBox tree.  Walking the tree
produces (spectrogram_view, target_start, target_end, target_mask,
target_class) tuples where siblings are masked from the working spec
before each prediction — exactly mirroring the sheet-music OCR approach.

Usage:
    python3.10 audio_ocr_model.py --data_dir data/audio --epochs 30
"""

import os
import json
import argparse
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from retina import fit_to_retina, image_to_tensor
from generate_audio import (
    AUDIO_CLASSES, AUDIO_CLASS_TO_ID, AudioBBox,
    SAMPLE_RATE, HOP_LENGTH, N_MELS, N_FFT, SPEC_FLOOR_DB,
    SimpleSynth, generate_random_song,
    compute_mel_spectrogram, spectrogram_to_image,
    build_annotation_tree, extract_audio_training_samples,
)

# ──────────────────────────────────────────────────────────────────────────────
# CNN backbone
# ──────────────────────────────────────────────────────────────────────────────

def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class AudioBackbone(nn.Module):
    """CNN backbone: retina image → flat feature vector."""

    def __init__(self):
        super().__init__()
        self.stages = nn.Sequential(
            _conv_block(3, 32),   nn.MaxPool2d(2),   # 512
            _conv_block(32, 64),  nn.MaxPool2d(2),   # 256
            _conv_block(64, 128), nn.MaxPool2d(2),   # 128
            _conv_block(128, 256), nn.MaxPool2d(2),  # 64
            _conv_block(256, 512), nn.MaxPool2d(2),  # 32
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 16
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 8
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),                  # 4
        )
        self.feature_dim = 512 * 4 * 4               # 8192

    def forward(self, x):
        return self.stages(x).flatten(1)


# ──────────────────────────────────────────────────────────────────────────────
# Audio OCR model
# ──────────────────────────────────────────────────────────────────────────────

class AudioOCRModel(nn.Module):
    """
    Single-retina audio detector.

    For a given spectrogram retina image the model predicts:
        • temporal — (start, end) in [0, 1]
        • freq_mask — per-mel-bin probabilities, (n_mels,) in [0, 1]
        • cls — logits over AUDIO_CLASSES
    """

    def __init__(self, n_mels: int = N_MELS,
                 num_classes: int = len(AUDIO_CLASSES)):
        super().__init__()
        self.backbone = AudioBackbone()
        dim = self.backbone.feature_dim

        self.temporal_head = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Sigmoid(),                             # (start, end) in [0,1]
        )

        self.freq_mask_head = nn.Sequential(
            nn.Linear(dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, n_mels),
            nn.Sigmoid(),                             # per-bin [0,1]
        )

        self.cls_head = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feat = self.backbone(x)
        temporal = self.temporal_head(feat)
        mask = self.freq_mask_head(feat)
        cls = self.cls_head(feat)
        return temporal, mask, cls


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class AudioTeacherForcingDataset(Dataset):
    """
    Generates / loads audio, builds AudioBBox trees, and flattens them
    into teacher-forcing training samples.
    """

    def __init__(self, data_dir: str = None, num_songs: int = 100,
                 retina_size: int = 1024, seed: int = 42):
        self.retina_size = retina_size
        self.samples: List[dict] = []

        if data_dir and os.path.isfile(os.path.join(data_dir, 'annotations.json')):
            self._load_from_disk(data_dir)
        else:
            self._generate(num_songs, seed)

    def _load_from_disk(self, data_dir):
        with open(os.path.join(data_dir, 'annotations.json')) as f:
            annotations = json.load(f)

        for ann in annotations:
            spec_path = os.path.join(data_dir, 'spec', ann['spec'])
            # Reconstruct spectrogram from image (approximate for training)
            spec_img = __import__('PIL').Image.open(spec_path).convert('RGB')
            tree = AudioBBox.from_dict(ann['bbox_tree'])

            # Regenerate the dB spectrogram from the image
            arr = np.array(spec_img)[:, :, 0].astype(np.float32) / 255.0
            arr = np.flipud(arr)       # undo the flip from spectrogram_to_image
            spec_db = arr * (-SPEC_FLOOR_DB) + SPEC_FLOOR_DB  # [floor, 0]

            self.samples.extend(
                extract_audio_training_samples(spec_db, tree, self.retina_size))

    def _generate(self, num_songs, seed):
        synth = SimpleSynth()
        random.seed(seed)
        np.random.seed(seed)

        for i in range(num_songs):
            events, dur = generate_random_song(seed=seed + i)
            mix = synth.render_song(events, dur)
            spec = compute_mel_spectrogram(mix)
            n_frames = spec.shape[1]
            tree = build_annotation_tree(events, synth, n_frames)
            self.samples.extend(
                extract_audio_training_samples(spec, tree, self.retina_size))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        retina_img, _, _, _ = fit_to_retina(s['spec_image'], self.retina_size)
        x = image_to_tensor(retina_img)                                  # [3,R,R]

        temporal = torch.tensor([s['target_start'], s['target_end']],
                                dtype=torch.float32)                     # [2]
        mask = torch.tensor(s['target_mask'], dtype=torch.float32)       # [n_mels]
        cls = torch.tensor(AUDIO_CLASS_TO_ID[s['target_class']],
                           dtype=torch.long)
        return x, temporal, mask, cls


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

    dataset = AudioTeacherForcingDataset(
        data_dir=args.data_dir,
        num_songs=args.num_songs,
        retina_size=args.retina_size,
        seed=args.seed,
    )
    print(f'Dataset: {len(dataset)} training samples')

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)

    model = AudioOCRModel(n_mels=N_MELS,
                          num_classes=len(AUDIO_CLASSES)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    temporal_crit = nn.MSELoss()
    mask_crit = nn.BCELoss()
    cls_crit = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss = tot_t = tot_m = tot_c = 0.0
        n = 0

        for x, t_target, m_target, c_target in loader:
            x = x.to(device)
            t_target = t_target.to(device)
            m_target = m_target.to(device)
            c_target = c_target.to(device)

            t_pred, m_pred, c_pred = model(x)

            loss_t = temporal_crit(t_pred, t_target)
            loss_m = mask_crit(m_pred, m_target)
            loss_c = cls_crit(c_pred, c_target)
            loss = loss_t + loss_m + loss_c

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
            tot_t += loss_t.item()
            tot_m += loss_m.item()
            tot_c += loss_c.item()
            n += 1

        d = max(n, 1)
        print(f'Epoch {epoch:3d}/{args.epochs}  '
              f'loss={tot_loss/d:.4f}  '
              f'temporal={tot_t/d:.4f}  '
              f'mask={tot_m/d:.4f}  '
              f'cls={tot_c/d:.4f}')

    os.makedirs('checkpoints', exist_ok=True)
    ckpt = 'checkpoints/audio_ocr.pt'
    torch.save(model.state_dict(), ckpt)
    print(f'Saved → {ckpt}')


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model: AudioOCRModel, spec_image,
            retina_size: int = 1024, device=None):
    """
    Run one forward pass.
    Returns (start, end, freq_mask_np, class_name, confidence).
    """
    from PIL import Image as PILImage
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    if isinstance(spec_image, np.ndarray):
        spec_image = spectrogram_to_image(spec_image)

    retina_img, _, _, _ = fit_to_retina(spec_image, retina_size)
    x = image_to_tensor(retina_img).unsqueeze(0).to(device)

    t_pred, m_pred, c_pred = model(x)
    start, end = t_pred[0].cpu().tolist()
    freq_mask = (m_pred[0].cpu().numpy() > 0.5)
    probs = torch.softmax(c_pred[0], dim=0)
    cls_id = probs.argmax().item()
    return start, end, freq_mask, AUDIO_CLASSES[cls_id], probs[cls_id].item()


@torch.no_grad()
def hierarchical_decode(model: AudioOCRModel,
                        spec: np.ndarray,
                        retina_size: int = 1024,
                        max_depth: int = 8,
                        device=None) -> AudioBBox:
    """
    Recursively decode the AudioBBox tree from a mel spectrogram.

    At each level:
      1. Predict (start, end, freq_mask, class).
      2. If class == 'none' → stop.
      3. Mask the predicted region → predict next sibling.
      4. Zoom into the predicted region → recurse for children.
    """
    if device is None:
        device = next(model.parameters()).device

    n_mels, n_frames = spec.shape

    def _decode(sub_spec, depth):
        if depth > max_depth or sub_spec.shape[1] < 2:
            return []

        children = []
        working = sub_spec.copy()
        n_f = working.shape[1]

        for _ in range(30):
            start, end, fmask, cls, conf = predict(
                model, working, retina_size, device)
            if cls == 'none' or conf < 0.3:
                break

            sf = int(start * n_f)
            ef = int(end * n_f)
            sf = max(0, min(n_f - 1, sf))
            ef = max(sf + 1, min(n_f, ef))

            fmask_list = fmask.tolist()

            # Zoom: crop + freq-mask for recursion
            zoom = sub_spec[:, sf:ef].copy()
            zoom[~fmask, :] = SPEC_FLOOR_DB

            sub_children = _decode(zoom, depth + 1)
            children.append(AudioBBox(sf, ef, fmask_list, cls,
                                      children=sub_children))

            # Mask out of working spectrogram
            working[fmask, sf:ef] = SPEC_FLOOR_DB

        return children

    subs = _decode(spec, 0)
    return AudioBBox(0, n_frames, [True] * n_mels, 'full_mix',
                     children=subs)


# ──────────────────────────────────────────────────────────────────────────────
# Bottom-up note detection (adapted from tiny-tessarachnid)
# ──────────────────────────────────────────────────────────────────────────────

def _find_spec_content_regions(spec_db, gap_tolerance=1,
                                energy_threshold=None):
    """Find temporal content regions in a spectrogram.

    Scans columns (time frames) for energy above *energy_threshold*
    (defaults to SPEC_FLOOR_DB + 10 dB).

    Returns a list of (start_frame, end_frame) tuples.
    """
    if energy_threshold is None:
        energy_threshold = SPEC_FLOOR_DB + 10.0

    col_has_content = (spec_db > energy_threshold).any(axis=0)
    n_frames = spec_db.shape[1]

    regions = []
    in_region = False
    region_start = 0
    gap_count = 0

    for i in range(n_frames):
        if col_has_content[i]:
            if not in_region:
                region_start = i
                in_region = True
            gap_count = 0
        else:
            if in_region:
                gap_count += 1
                if gap_count > gap_tolerance:
                    regions.append((region_start, i - gap_count))
                    in_region = False

    if in_region:
        region_end = n_frames - 1
        while region_end > region_start and not col_has_content[region_end]:
            region_end -= 1
        regions.append((region_start, region_end + 1))

    return regions


def _generate_temporal_candidates(spec_db, gap_tolerance=1,
                                   energy_threshold=None):
    """Generate candidate temporal windows from a spectrogram.

    Uses temporal content regions as natural windows.  For regions wider
    than 1.5× the expected note width, generates overlapping sub-windows
    at stride = expected_width // 2.
    """
    regions = _find_spec_content_regions(spec_db, gap_tolerance,
                                         energy_threshold)
    if not regions:
        return []

    widths = [r[1] - r[0] for r in regions]
    expected_w = int(np.median(widths)) if widths else 1
    expected_w = max(expected_w, 1)

    candidates = []
    for (sf, ef) in regions:
        w = ef - sf
        if w <= expected_w * 1.5:
            candidates.append((sf, ef))
        else:
            stride = max(expected_w // 2, 1)
            for sx in range(sf, ef - expected_w + 1, stride):
                candidates.append((sx, sx + expected_w))
            if candidates and candidates[-1][1] < ef:
                candidates.append((ef - expected_w, ef))

    return candidates


@torch.no_grad()
def _classify_audio_candidates_batched(model, spec_db, candidates, device,
                                        retina_size=1024, batch_size=32):
    """Classify temporal candidate windows in batches.

    For each candidate: crop the spectrogram to the time window, pad
    temporally to match training distribution (~1/5th occupancy),
    convert to image, fit to retina, run model.

    Returns list of ((start, end), freq_mask, class_name, confidence)
    for non-'none' predictions.
    """
    results = []

    for i in range(0, len(candidates), batch_size):
        batch_windows = candidates[i:i + batch_size]
        imgs = []
        for (sf, ef) in batch_windows:
            sub_spec = spec_db[:, sf:ef]
            if sub_spec.shape[1] < 2:
                continue
            # Pad temporally to match training distribution
            n_pad = sub_spec.shape[1] * 2
            padded = np.full(
                (spec_db.shape[0], sub_spec.shape[1] + n_pad * 2),
                SPEC_FLOOR_DB, dtype=np.float32)
            padded[:, n_pad:n_pad + sub_spec.shape[1]] = sub_spec
            spec_img = spectrogram_to_image(padded)
            retina_img, _, _, _ = fit_to_retina(spec_img, retina_size)
            img_t = image_to_tensor(retina_img)
            imgs.append((img_t, (sf, ef)))

        if not imgs:
            continue

        img_batch = torch.stack([t for t, _ in imgs]).to(device)
        _, m_pred, c_pred = model(img_batch)

        probs = torch.softmax(c_pred, dim=1)
        cls_ids = probs.argmax(dim=1)
        confidences = probs.gather(1, cls_ids.unsqueeze(1)).squeeze(1)
        freq_masks = (m_pred > 0.5)

        for j, (_, window) in enumerate(imgs):
            cid = cls_ids[j].item()
            cls_name = AUDIO_CLASSES[cid]
            conf = confidences[j].item()
            if cls_name != 'none':
                fmask = freq_masks[j].cpu().tolist()
                results.append((window, fmask, cls_name, conf))

    return results


def _temporal_iou(a, b):
    """Compute 1D IoU of two (start, end) intervals."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _temporal_nms(detections, iou_threshold=0.3):
    """Non-maximum suppression on temporal detections.

    detections: list of ((start, end), freq_mask, class_name, confidence)
    Returns filtered list sorted by start time.
    """
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d[3], reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if _temporal_iou(best[0], d[0]) < iou_threshold]

    keep.sort(key=lambda d: d[0][0])
    return keep


def _detect_audio_bottom_up(model, spec_db, device, retina_size=1024,
                              gap_tolerance=1, energy_threshold=None,
                              iou_threshold=0.3, confidence_threshold=0.1,
                              batch_size=32):
    """Bottom-up audio detection: temporal candidates → classify → NMS.

    Returns [AudioBBox, ...] sorted by start time.
    """
    candidates = _generate_temporal_candidates(spec_db, gap_tolerance,
                                                energy_threshold)
    if not candidates:
        return []

    detections = _classify_audio_candidates_batched(
        model, spec_db, candidates, device,
        retina_size=retina_size, batch_size=batch_size)

    # Filter by confidence
    detections = [d for d in detections if d[3] >= confidence_threshold]

    kept = _temporal_nms(detections, iou_threshold=iou_threshold)
    return [AudioBBox(sf, ef, fmask, cls_name, children=[])
            for (sf, ef), fmask, cls_name, _ in kept]


@torch.no_grad()
def hierarchical_decode_hybrid(model: AudioOCRModel,
                                spec: np.ndarray,
                                retina_size: int = 1024,
                                max_depth: int = 8,
                                device=None) -> AudioBBox:
    """
    Decode AudioBBox tree using bottom-up detection at the note level.

    At higher levels (full_mix → instrument_part → phrase), uses the
    original iterative masking approach.  At the phrase level (detecting
    notes), switches to bottom-up: temporal candidates → classify → NMS.
    This avoids autoregressive error accumulation at the most granular
    detection level.
    """
    if device is None:
        device = next(model.parameters()).device

    n_mels, n_frames = spec.shape

    # Parent classes whose children are detected bottom-up
    BOTTOM_UP_PARENTS = {'phrase'}

    def _decode(sub_spec, parent_cls, depth):
        if depth > max_depth or sub_spec.shape[1] < 2:
            return []

        # Bottom-up for note-level detection
        if parent_cls in BOTTOM_UP_PARENTS:
            return _detect_audio_bottom_up(
                model, sub_spec, device, retina_size=retina_size)

        # Standard iterative masking for higher levels
        children = []
        working = sub_spec.copy()
        n_f = working.shape[1]

        for _ in range(30):
            start, end, fmask, cls, conf = predict(
                model, working, retina_size, device)
            if cls == 'none' or conf < 0.3:
                break

            sf = int(start * n_f)
            ef = int(end * n_f)
            sf = max(0, min(n_f - 1, sf))
            ef = max(sf + 1, min(n_f, ef))

            fmask_list = fmask.tolist()

            # Zoom: crop + freq-mask for recursion
            zoom = sub_spec[:, sf:ef].copy()
            zoom[~fmask, :] = SPEC_FLOOR_DB

            sub_children = _decode(zoom, cls, depth + 1)
            children.append(AudioBBox(sf, ef, fmask_list, cls,
                                      children=sub_children))

            # Mask out of working spectrogram
            working[fmask, sf:ef] = SPEC_FLOOR_DB

        return children

    subs = _decode(spec, 'full_mix', 0)
    return AudioBBox(0, n_frames, [True] * n_mels, 'full_mix',
                     children=subs)


# ──────────────────────────────────────────────────────────────────────────────
# Inference on a WAV or spectrogram image
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(args):
    """Load a checkpoint and run hierarchical decoding on an audio file."""
    import wave as wave_mod
    from PIL import Image as PILImage
    from generate_audio import (
        save_wav, draw_audio_bbox_tree, AudioBBox,
        compute_mel_spectrogram as compute_spec,
        spectrogram_to_image as spec2img,
    )

    device = (
        torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu')
    )

    # Load model
    model = AudioOCRModel(n_mels=N_MELS,
                          num_classes=len(AUDIO_CLASSES)).to(device)
    print(f'Loading checkpoint: {args.checkpoint}')
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Load audio → spectrogram
    input_path = args.input
    if input_path.endswith('.wav'):
        with wave_mod.open(input_path, 'rb') as wf:
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32767.0
        if n_ch > 1:
            audio = audio.reshape(-1, n_ch).mean(axis=1)
        spec = compute_spec(audio, sr=sr)
        print(f'Loaded WAV: {input_path}  ({len(audio)/sr:.2f}s, {sr} Hz)')
    elif input_path.endswith('.png') or input_path.endswith('.jpg'):
        img = PILImage.open(input_path).convert('RGB')
        arr = np.array(img)[:, :, 0].astype(np.float32) / 255.0
        arr = np.flipud(arr)
        spec = arr * (-SPEC_FLOOR_DB) + SPEC_FLOOR_DB
        print(f'Loaded spectrogram image: {input_path}')
    else:
        raise ValueError(f'Unsupported input format: {input_path}')

    n_mels, n_frames = spec.shape
    print(f'Spectrogram: {n_mels} mel bins × {n_frames} frames')

    # Run hierarchical decoding
    print(f'Running hierarchical decode (max_depth={args.max_depth}) ...')
    tree = hierarchical_decode(model, spec, retina_size=args.retina_size,
                               max_depth=args.max_depth, device=device)

    # Print the decoded tree
    def show(node, indent=0):
        mask_count = sum(1 for m in node.freq_mask if m)
        print(f'{"  " * indent}{node.label}  '
              f'frames=[{node.start_frame}:{node.end_frame}]  '
              f'active_bins={mask_count}/{len(node.freq_mask)}  '
              f'children={len(node.children)}')
        for c in node.children:
            show(c, indent + 1)

    print('\n── Decoded tree ──')
    show(tree)

    # Save visualisation
    if args.output:
        spec_img = spec2img(spec)
        vis = draw_audio_bbox_tree(spec_img, tree, n_frames)
        vis.save(args.output)
        print(f'\nVisualisation saved → {args.output}')

    # Save tree as JSON
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(tree.to_dict(), f, indent=2)
        print(f'Tree JSON saved → {args.output_json}')


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train or run inference with the Audio OCR model.')
    sub = parser.add_subparsers(dest='command')

    # ── train ──
    tr = sub.add_parser('train', help='Train the model.')
    tr.add_argument('--data_dir', type=str, default=None,
                    help='Pre-generated data directory (with annotations.json).')
    tr.add_argument('--num_songs', type=int, default=100,
                    help='Songs to generate on-the-fly if no data_dir.')
    tr.add_argument('--retina_size', type=int, default=1024)
    tr.add_argument('--batch_size', type=int, default=4)
    tr.add_argument('--epochs', type=int, default=30)
    tr.add_argument('--lr', type=float, default=1e-4)
    tr.add_argument('--seed', type=int, default=42)

    # ── infer ──
    inf = sub.add_parser('infer', help='Run inference on a WAV or spectrogram.')
    inf.add_argument('input', type=str,
                     help='Path to a .wav file or spectrogram .png image.')
    inf.add_argument('--checkpoint', type=str,
                     default='checkpoints/audio_ocr.pt',
                     help='Model checkpoint to load.')
    inf.add_argument('--output', type=str, default=None,
                     help='Save visualisation to this image path.')
    inf.add_argument('--output_json', type=str, default=None,
                     help='Save decoded tree as JSON.')
    inf.add_argument('--retina_size', type=int, default=1024)
    inf.add_argument('--max_depth', type=int, default=8)

    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'infer':
        run_inference(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()