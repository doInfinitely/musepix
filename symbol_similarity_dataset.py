"""
Symbol Similarity dataset builder.

Builds same/different pairs from the existing musical-symbol PNGs and
(optionally) generates additional symbol images via the OpenAI DALL-E API.

Pair types
──────────
 • Positive (same kind):  two images of the same symbol
   – original + augmented copy  (rotation, scale, noise, blur)
   – original + OpenAI-generated variant
 • Negative (different kind):  two images of different symbols

Usage:
    python3.10 symbol_similarity_dataset.py \
        --symbols_dir sheet_music_symbols_noto_music_images \
        --out_dir data/symbol_pairs \
        --num_openai 20 \
        --pairs_per_symbol 5
"""

import os
import csv
import json
import random
import argparse
import io
import base64
from typing import List, Tuple

from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Load the symbol catalogue
# ──────────────────────────────────────────────────────────────────────────────

def load_symbol_catalogue(symbols_dir: str) -> List[dict]:
    """Read symbols.csv and return list of dicts with metadata + loaded image."""
    csv_path = os.path.join(symbols_dir, 'symbols.csv')
    catalogue = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(symbols_dir, row['image_file'])
            if not os.path.isfile(img_path):
                continue
            catalogue.append({
                'codepoint': row['codepoint'],
                'name': row['unicode_name'],
                'range': row['range'],
                'image_path': img_path,
            })
    return catalogue


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation — create visual variants of a symbol
# ──────────────────────────────────────────────────────────────────────────────

def augment_symbol(image: Image.Image) -> Image.Image:
    """Apply random augmentations to create a visually different but
    semantically identical variant of the symbol."""
    img = image.copy()

    # Random rotation (-15 to +15 degrees)
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False,
                     fillcolor=(255, 255, 255, 0) if img.mode == 'RGBA'
                     else (255, 255, 255))

    # Random scale (80% – 120%)
    factor = random.uniform(0.8, 1.2)
    new_size = (max(1, int(img.width * factor)),
                max(1, int(img.height * factor)))
    img = img.resize(new_size, Image.LANCZOS)

    # Re-center onto original canvas size
    canvas = Image.new(image.mode, image.size,
                       (255, 255, 255, 0) if image.mode == 'RGBA'
                       else (255, 255, 255))
    ox = (image.width - img.width) // 2
    oy = (image.height - img.height) // 2
    canvas.paste(img, (ox, oy))
    img = canvas

    # Random brightness / contrast
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(img.convert('RGB') if img.mode == 'RGBA' else img)
        img_rgb = enhancer.enhance(random.uniform(0.7, 1.3))
        if image.mode == 'RGBA':
            img.paste(img_rgb)
        else:
            img = img_rgb

    # Random Gaussian blur
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    # Random additive noise
    if random.random() < 0.3:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 8, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode=img.mode)

    return img


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI DALL-E symbol generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_symbol_with_openai(symbol_name: str, client) -> Image.Image:
    """Use DALL-E 3 to generate a new rendering of a musical symbol."""
    prompt = (
        f"A clean, black-on-white rendering of the musical notation symbol "
        f"'{symbol_name}'. Simple, high contrast, centered on a white "
        f"background, no additional decoration. Vector-art style."
    )

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
        response_format="b64_json",
    )

    image_data = base64.b64decode(response.data[0].b64_json)
    return Image.open(io.BytesIO(image_data)).convert('RGBA')


def batch_generate_openai(catalogue: List[dict], num_generate: int,
                          out_dir: str):
    """Generate new symbol images via OpenAI and save them."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('⚠  OPENAI_API_KEY not set — skipping OpenAI generation.')
        return {}

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    os.makedirs(out_dir, exist_ok=True)

    # Pick a random subset of symbols to generate variants for
    chosen = random.sample(catalogue, min(num_generate, len(catalogue)))
    generated = {}  # codepoint → list of paths

    for i, entry in enumerate(chosen):
        try:
            print(f'  OpenAI [{i+1}/{len(chosen)}] generating {entry["name"]}…')
            img = generate_symbol_with_openai(entry['name'], client)
            fname = f'openai_{entry["codepoint"]}_{i}.png'
            path = os.path.join(out_dir, fname)
            img.save(path)
            generated.setdefault(entry['codepoint'], []).append(path)
        except Exception as e:
            print(f'    ✗ {e}')

    return generated


# ──────────────────────────────────────────────────────────────────────────────
# Pair generation
# ──────────────────────────────────────────────────────────────────────────────

def build_pairs(catalogue: List[dict], openai_images: dict,
                pairs_per_symbol: int = 5,
                ) -> List[Tuple[str, str, int]]:
    """
    Build (path_a, path_b, label) triples.
        label = 1  → same kind of symbol
        label = 0  → different kind

    Strategy:
        • For each symbol, create `pairs_per_symbol` positive pairs
          using augmentation (saved to a temp dir) or OpenAI variants.
        • For each positive pair, create one negative pair by sampling
          a different symbol.
    """
    pairs = []

    # Group by codepoint
    by_code = {}
    for entry in catalogue:
        by_code.setdefault(entry['codepoint'], []).append(entry['image_path'])
    # Merge OpenAI images
    for cp, paths in openai_images.items():
        by_code.setdefault(cp, []).extend(paths)

    codepoints = list(by_code.keys())

    for cp in codepoints:
        images = by_code[cp]
        # Positive pairs: pair every available image with augmented versions
        for _ in range(pairs_per_symbol):
            anchor = random.choice(images)
            # We'll augment at training time, but store path + 'augment' flag
            pairs.append((anchor, anchor, 1))  # will augment img_b on load

        # Negative pairs (same count)
        other_cps = [c for c in codepoints if c != cp]
        for _ in range(pairs_per_symbol):
            anchor = random.choice(images)
            neg_cp = random.choice(other_cps)
            neg_img = random.choice(by_code[neg_cp])
            pairs.append((anchor, neg_img, 0))

    random.shuffle(pairs)
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

import torch
from torch.utils.data import Dataset as TorchDataset
from retina import fit_to_retina, image_to_tensor


class SymbolPairDataset(TorchDataset):
    """
    Yields (tensor_a, tensor_b, label) where both tensors are retina-fitted.
    For positive pairs the second image is augmented on-the-fly.
    """

    def __init__(self, pairs: List[Tuple[str, str, int]],
                 retina_size: int = 256):
        self.pairs = pairs
        self.retina_size = retina_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path_a, path_b, label = self.pairs[idx]

        img_a = Image.open(path_a)
        img_b = Image.open(path_b)

        # For positive pairs, augment the second image
        if label == 1:
            img_b = augment_symbol(img_b)

        retina_a, *_ = fit_to_retina(img_a, self.retina_size)
        retina_b, *_ = fit_to_retina(img_b, self.retina_size)

        return (image_to_tensor(retina_a),
                image_to_tensor(retina_b),
                torch.tensor(label, dtype=torch.long))


# ──────────────────────────────────────────────────────────────────────────────
# CLI — build and save the pair dataset
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Build symbol-similarity pair dataset.')
    parser.add_argument('--symbols_dir', type=str,
                        default='sheet_music_symbols_noto_music_images',
                        help='Directory with symbols.csv and images/.')
    parser.add_argument('--out_dir', type=str, default='data/symbol_pairs',
                        help='Where to save the pairs manifest.')
    parser.add_argument('--num_openai', type=int, default=0,
                        help='Number of symbols to generate via OpenAI DALL-E '
                             '(set >0 to use the API; costs money).')
    parser.add_argument('--pairs_per_symbol', type=int, default=5,
                        help='Positive + negative pairs per symbol.')
    parser.add_argument('--retina_size', type=int, default=256,
                        help='Retina size for the similarity model.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print('Loading symbol catalogue …')
    catalogue = load_symbol_catalogue(args.symbols_dir)
    print(f'  {len(catalogue)} symbols found.')

    # Optional OpenAI generation
    openai_images = {}
    if args.num_openai > 0:
        openai_dir = os.path.join(args.out_dir, 'openai_generated')
        openai_images = batch_generate_openai(catalogue, args.num_openai,
                                              openai_dir)
        print(f'  Generated {sum(len(v) for v in openai_images.values())} '
              f'OpenAI images across {len(openai_images)} symbols.')

    print('Building pairs …')
    pairs = build_pairs(catalogue, openai_images, args.pairs_per_symbol)
    print(f'  {len(pairs)} total pairs  '
          f'({sum(1 for _, _, l in pairs if l == 1)} positive, '
          f'{sum(1 for _, _, l in pairs if l == 0)} negative)')

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, 'pairs.json')
    with open(manifest_path, 'w') as f:
        json.dump(pairs, f)
    print(f'Saved → {manifest_path}')

    # Quick sanity check: load one pair
    ds = SymbolPairDataset(pairs, retina_size=args.retina_size)
    ta, tb, lbl = ds[0]
    print(f'Sample tensor shapes: {ta.shape}, {tb.shape}, label={lbl.item()}')


if __name__ == '__main__':
    main()
