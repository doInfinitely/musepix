"""
Synthetic sheet music generator with hierarchical bounding boxes.

Generates random melodies on a musical staff using PIL.  Every drawn element
(staff lines, noteheads, stems, ledger lines) gets a bounding box.  Notes
are grouped into melodic lines that split whenever the pitch range would
exceed an octave.

The bounding-box tree is then traversed to produce teacher-forcing training
samples: at each node the children are revealed one at a time (left→right),
conditioning on the previous detection's bbox + class.

Usage:
    python3.10 generate_sheet_music.py --num_images 100 --out_dir data/sheet_music
"""

import os
import json
import random
import argparse
from dataclasses import dataclass, field
from typing import List, Optional
from PIL import Image, ImageDraw

from retina import fit_to_retina, image_to_tensor

# ──────────────────────────────────────────────────────────────────────────────
# Music theory helpers
# ──────────────────────────────────────────────────────────────────────────────
NOTE_NAMES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
DIATONIC_SEMITONES = [0, 2, 4, 5, 7, 9, 11]  # C, D, E, F, G, A, B


def octave_note_to_midi(octave: int, note_index: int) -> int:
    """Convert (octave, diatonic note_index) to MIDI note number."""
    return 12 * (octave + 1) + DIATONIC_SEMITONES[note_index]


# Token vocabulary for MIDI decoder
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
PITCH_OFFSET = 3
NUM_PITCHES = 21          # C3..B5 = 3 octaves x 7 diatonic
MIDI_VOCAB_SIZE = 24      # PAD + BOS + EOS + 21 pitches

# MIDI range: C3 (48) to B5 (83)
_MIDI_C3 = octave_note_to_midi(3, 0)  # 48


def midi_to_token(midi_num: int) -> int:
    """Convert a MIDI note number (C3..B5) to a token (3..23)."""
    # Map C3=48 → token 3, D3=50 → token 4, ..., B5=83 → token 23
    # We use diatonic indices: find octave and note_index, then linear index
    for oct in range(3, 6):
        for ni in range(7):
            if octave_note_to_midi(oct, ni) == midi_num:
                return PITCH_OFFSET + (oct - 3) * 7 + ni
    raise ValueError(f'MIDI {midi_num} not in C3..B5 diatonic range')


def token_to_midi(token: int) -> int:
    """Convert a token (3..23) back to a MIDI note number."""
    if token < PITCH_OFFSET or token >= PITCH_OFFSET + NUM_PITCHES:
        raise ValueError(f'Token {token} not a pitch token')
    idx = token - PITCH_OFFSET
    octave = 3 + idx // 7
    note_index = idx % 7
    return octave_note_to_midi(octave, note_index)


# Element classes used by the OCR model
CLASSES = [
    'none',           # 0  — no more elements to detect
    'page',           # 1
    'staff_system',   # 2
    'staff_lines',    # 3
    'staff_line',     # 4
    'melodic_line',   # 5
    'note',           # 6
    'notehead',       # 7
    'stem',           # 8
    'ledger_line',    # 9
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


def pitch_to_staff_position(octave: int, note_index: int) -> int:
    """
    Convert a diatonic pitch to a staff position where 0 = bottom line (E4).
    Each diatonic step is one half-space on the staff.
    """
    e4_degree = 4 * 7 + 2  # E in octave 4
    pitch_degree = octave * 7 + note_index
    return pitch_degree - e4_degree


def staff_position_to_y(staff_pos: int, bottom_line_y: float,
                        line_spacing: float) -> float:
    """Staff position → pixel y (positive-down coordinate system)."""
    return bottom_line_y - staff_pos * (line_spacing / 2.0)


# ──────────────────────────────────────────────────────────────────────────────
# Bounding-box tree node
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    children: List['BBox'] = field(default_factory=list)
    midi: Optional[int] = None

    def as_tuple(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self):
        d = {
            'bbox': [self.x1, self.y1, self.x2, self.y2],
            'label': self.label,
            'children': [c.to_dict() for c in self.children],
        }
        if self.midi is not None:
            d['midi'] = self.midi
        return d

    @staticmethod
    def from_dict(d):
        return BBox(
            x1=d['bbox'][0], y1=d['bbox'][1],
            x2=d['bbox'][2], y2=d['bbox'][3],
            label=d['label'],
            children=[BBox.from_dict(c) for c in d.get('children', [])],
            midi=d.get('midi', None),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────────────
class SheetMusicGenerator:
    """Renders a single-staff piece of random sheet music and returns
    the image together with a hierarchical bounding-box annotation."""

    def __init__(self, *,
                 image_width: int = 800,
                 image_height: int = 400,
                 line_spacing: int = 14,
                 staff_margin_left: int = 80,
                 staff_margin_right: int = 40,
                 staff_margin_top: int = 120,
                 note_spacing: int = 50,
                 notehead_width_ratio: float = 1.3,
                 stem_length_ratio: float = 3.5):
        self.image_width = image_width
        self.image_height = image_height
        self.line_spacing = line_spacing
        self.staff_margin_left = staff_margin_left
        self.staff_margin_right = staff_margin_right
        self.staff_margin_top = staff_margin_top
        self.note_spacing = note_spacing
        self.notehead_w = int(line_spacing * notehead_width_ratio)
        self.notehead_h = int(line_spacing * 0.85)
        self.stem_length = int(line_spacing * stem_length_ratio)

        # Staff geometry
        self.staff_top_y = staff_margin_top
        self.bottom_line_y = self.staff_top_y + 4 * line_spacing
        self.staff_width = image_width - staff_margin_left - staff_margin_right

    # ── public API ────────────────────────────────────────────────────────
    def generate(self, num_notes: Optional[int] = None, seed=None):
        """Return (PIL.Image, BBox tree root)."""
        if seed is not None:
            random.seed(seed)

        if num_notes is None:
            num_notes = random.randint(8, 20)

        max_notes = max(1, (self.staff_width - 20) // self.note_spacing)
        num_notes = min(num_notes, max_notes)

        notes = self._generate_melody(num_notes)
        melodic_lines = self._split_into_melodic_lines(notes)

        img = Image.new('RGB', (self.image_width, self.image_height), 'white')
        draw = ImageDraw.Draw(img)
        bbox_tree = self._draw_and_annotate(draw, notes, melodic_lines)

        return img, bbox_tree

    # ── melody generation ─────────────────────────────────────────────────
    def _generate_melody(self, num_notes):
        """Random walk on the diatonic scale → list[(octave, note_index, x)]."""
        octave = 4
        note_idx = random.randint(0, 6)
        x = self.staff_margin_left + 30

        notes = []
        for _ in range(num_notes):
            notes.append((octave, note_idx, x))
            x += self.note_spacing

            step = random.choice([-3, -2, -1, -1, 0, 1, 1, 2, 3])
            note_idx += step
            while note_idx > 6:
                note_idx -= 7
                octave += 1
            while note_idx < 0:
                note_idx += 7
                octave -= 1

            # clamp to C3–B5
            if octave < 3:
                octave, note_idx = 3, max(note_idx, 0)
            elif octave > 5:
                octave, note_idx = 5, min(note_idx, 6)

        return notes

    # ── melodic-line splitting ────────────────────────────────────────────
    @staticmethod
    def _split_into_melodic_lines(notes):
        """Group consecutive note indices so that no group spans more
        than an octave (7 diatonic steps) in staff-position range."""
        if not notes:
            return []

        lines: List[List[int]] = []
        current_line = [0]
        sp0 = pitch_to_staff_position(notes[0][0], notes[0][1])
        lo = hi = sp0

        for i in range(1, len(notes)):
            sp = pitch_to_staff_position(notes[i][0], notes[i][1])
            new_lo, new_hi = min(lo, sp), max(hi, sp)

            if new_hi - new_lo > 7:          # octave exceeded → new line
                lines.append(current_line)
                current_line = [i]
                lo = hi = sp
            else:
                current_line.append(i)
                lo, hi = new_lo, new_hi

        if current_line:
            lines.append(current_line)
        return lines

    # ── drawing + annotation ──────────────────────────────────────────────
    def _draw_and_annotate(self, draw, notes, melodic_lines):
        ls = self.line_spacing

        # ── staff lines ──
        staff_line_bboxes = []
        sl_x1 = self.staff_margin_left
        sl_x2 = self.image_width - self.staff_margin_right
        for i in range(5):
            y = self.staff_top_y + i * ls
            draw.line([(sl_x1, y), (sl_x2, y)], fill='black', width=1)
            staff_line_bboxes.append(
                BBox(sl_x1, y - 1, sl_x2, y + 1, 'staff_line'))

        staff_lines_bbox = BBox(
            sl_x1, self.staff_top_y - 1,
            sl_x2, self.bottom_line_y + 1,
            'staff_lines',
            children=staff_line_bboxes,
        )

        # ── notes ──
        note_bboxes: List[BBox] = []
        for octave, note_idx, x in notes:
            children: List[BBox] = []
            sp = pitch_to_staff_position(octave, note_idx)
            cy = staff_position_to_y(sp, self.bottom_line_y, ls)

            # notehead (filled ellipse)
            nh_x1 = x - self.notehead_w // 2
            nh_y1 = cy - self.notehead_h // 2
            nh_x2 = x + self.notehead_w // 2
            nh_y2 = cy + self.notehead_h // 2
            draw.ellipse([nh_x1, nh_y1, nh_x2, nh_y2], fill='black')
            children.append(BBox(nh_x1, nh_y1, nh_x2, nh_y2, 'notehead'))

            # stem
            stem_up = sp < 4  # below middle line ⇒ stem up (right side)
            if stem_up:
                sx = nh_x2
                sy1, sy2 = cy - self.stem_length, cy
            else:
                sx = nh_x1
                sy1, sy2 = cy, cy + self.stem_length
            draw.line([(sx, sy1), (sx, sy2)], fill='black', width=2)
            children.append(
                BBox(sx - 1, min(sy1, sy2), sx + 1, max(sy1, sy2), 'stem'))

            # ledger lines (below staff)
            if sp <= -2:
                lowest = sp if sp % 2 == 0 else sp + 1  # round toward 0
                for lp in range(-2, lowest - 1, -2):
                    ly = staff_position_to_y(lp, self.bottom_line_y, ls)
                    ll_x1 = x - self.notehead_w // 2 - 4
                    ll_x2 = x + self.notehead_w // 2 + 4
                    draw.line([(ll_x1, ly), (ll_x2, ly)],
                              fill='black', width=1)
                    children.append(
                        BBox(ll_x1, ly - 1, ll_x2, ly + 1, 'ledger_line'))

            # ledger lines (above staff)
            if sp >= 10:
                highest = sp if sp % 2 == 0 else sp - 1
                for lp in range(10, highest + 1, 2):
                    ly = staff_position_to_y(lp, self.bottom_line_y, ls)
                    ll_x1 = x - self.notehead_w // 2 - 4
                    ll_x2 = x + self.notehead_w // 2 + 4
                    draw.line([(ll_x1, ly), (ll_x2, ly)],
                              fill='black', width=1)
                    children.append(
                        BBox(ll_x1, ly - 1, ll_x2, ly + 1, 'ledger_line'))

            # encompassing note bbox
            all_x = [c.x1 for c in children] + [c.x2 for c in children]
            all_y = [c.y1 for c in children] + [c.y2 for c in children]
            midi_num = octave_note_to_midi(octave, note_idx)
            note_bboxes.append(
                BBox(min(all_x), min(all_y), max(all_x), max(all_y),
                     'note', children=children, midi=midi_num))

        # ── melodic line bboxes ──
        ml_bboxes = []
        for line_idx, indices in enumerate(melodic_lines):
            line_notes = [note_bboxes[i] for i in indices]
            ml_bboxes.append(BBox(
                min(n.x1 for n in line_notes) - 2,
                min(n.y1 for n in line_notes) - 2,
                max(n.x2 for n in line_notes) + 2,
                max(n.y2 for n in line_notes) + 2,
                'melodic_line',
                children=line_notes,
            ))

        # ── staff system ──
        all_elems = [staff_lines_bbox] + ml_bboxes
        staff_sys = BBox(
            min(e.x1 for e in all_elems) - 5,
            min(e.y1 for e in all_elems) - 5,
            max(e.x2 for e in all_elems) + 5,
            max(e.y2 for e in all_elems) + 5,
            'staff_system',
            children=[staff_lines_bbox] + ml_bboxes,
        )

        # ── page (root) ──
        page = BBox(0, 0, self.image_width, self.image_height, 'page',
                    children=[staff_sys])

        return page


# ──────────────────────────────────────────────────────────────────────────────
# Teacher-forcing training sample extraction
# ──────────────────────────────────────────────────────────────────────────────
def extract_training_samples(full_img: Image.Image, root: BBox,
                             retina_size: int = 1024):
    """
    Walk the bounding-box tree and produce teacher-forcing training tuples.

    For every node with children (sorted left → right):
        1.  Present the node's cropped region as input (unmodified).
        2.  Target = first child bbox (normalised to node crop) + class.
        3.  Condition on the previous detection via prev_bbox.
        4.  Target = second child bbox + class …
        5.  After all children: target = (0,0,0,0), class = 'none'.
        6.  Recurse into each child.

    Each sample dict:
        input_image   — PIL Image (the node crop, always unmodified)
        target_bbox   — (x1, y1, x2, y2) normalised to [0, 1] within the crop
        target_class  — string label
        child_image   — PIL Image of the child crop (used as input for recursion),
                        or None for the 'none' sentinel.
    """
    samples = []
    _traverse(full_img, root, samples, parent_label=root.label)
    return samples


def _traverse(full_img: Image.Image, node: BBox, samples: list,
              parent_label: str = ''):
    if not node.children:
        return

    # Crop to node region (clamp to image bounds)
    nx1 = max(0, int(node.x1))
    ny1 = max(0, int(node.y1))
    nx2 = min(full_img.width, int(node.x2))
    ny2 = min(full_img.height, int(node.y2))
    node_w, node_h = nx2 - nx1, ny2 - ny1
    if node_w <= 0 or node_h <= 0:
        return

    node_crop = full_img.crop((nx1, ny1, nx2, ny2)).copy()

    # Sort children left → right
    children = sorted(node.children, key=lambda c: c.x1)

    prev_bbox = (0.0, 0.0, 0.0, 0.0, 0.0)  # none for first child

    for child in children:
        # Normalised bbox relative to node crop
        rel = (
            max(0.0, min(1.0, (child.x1 - nx1) / node_w)),
            max(0.0, min(1.0, (child.y1 - ny1) / node_h)),
            max(0.0, min(1.0, (child.x2 - nx1) / node_w)),
            max(0.0, min(1.0, (child.y2 - ny1) / node_h)),
        )

        # Child crop from the original image
        cx1 = max(0, int(child.x1))
        cy1 = max(0, int(child.y1))
        cx2 = min(full_img.width, int(child.x2))
        cy2 = min(full_img.height, int(child.y2))
        child_crop = full_img.crop((cx1, cy1, cx2, cy2)).copy()

        samples.append({
            'input_image': node_crop,
            'target_bbox': rel,
            'target_class': child.label,
            'child_image': child_crop,
            'prev_bbox': prev_bbox,
            'parent_class': parent_label,
        })

        # Update prev_bbox: normalised coords + class_id
        prev_bbox = (rel[0], rel[1], rel[2], rel[3],
                     float(CLASS_TO_ID[child.label]))

    # Sentinel: no more children (prev_bbox is the last child detected)
    samples.append({
        'input_image': node_crop,
        'target_bbox': (0.0, 0.0, 0.0, 0.0),
        'target_class': 'none',
        'child_image': None,
        'prev_bbox': prev_bbox,
        'parent_class': parent_label,
    })

    # Recurse into each child
    for child in children:
        _traverse(full_img, child, samples, parent_label=child.label)


# ──────────────────────────────────────────────────────────────────────────────
# MIDI training sample extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_midi_training_samples(full_img: Image.Image, root: BBox):
    """Walk tree, find melodic_line nodes, extract image crop + MIDI token sequence.

    Returns a list of dicts:
        melodic_line_image — PIL Image crop of the melodic line
        midi_sequence      — list of MIDI note numbers (sorted left-to-right)
        token_sequence     — list of tokens: [BOS, tok1, ..., tokN, EOS]
    """
    samples = []
    _collect_melodic_lines(full_img, root, samples)
    return samples


def _collect_melodic_lines(full_img: Image.Image, node: BBox, samples: list):
    if node.label == 'melodic_line' and node.children:
        # Crop to melodic line bbox
        x1 = max(0, int(node.x1))
        y1 = max(0, int(node.y1))
        x2 = min(full_img.width, int(node.x2))
        y2 = min(full_img.height, int(node.y2))
        if x2 - x1 > 0 and y2 - y1 > 0:
            crop = full_img.crop((x1, y1, x2, y2)).copy()

            # Collect notes sorted left-to-right
            notes = sorted(
                [c for c in node.children if c.label == 'note' and c.midi is not None],
                key=lambda c: c.x1,
            )
            if notes:
                midi_seq = [n.midi for n in notes]
                token_seq = ([BOS_TOKEN]
                             + [midi_to_token(m) for m in midi_seq]
                             + [EOS_TOKEN])
                samples.append({
                    'melodic_line_image': crop,
                    'midi_sequence': midi_seq,
                    'token_sequence': token_seq,
                })

    for child in node.children:
        _collect_melodic_lines(full_img, child, samples)


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helper
# ──────────────────────────────────────────────────────────────────────────────
PALETTE = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe',
]


def draw_bbox_tree(img: Image.Image, node: BBox, depth: int = 0,
                   max_depth: int = 99) -> Image.Image:
    """Overlay bounding boxes coloured by depth."""
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    _draw_recursive(draw, node, depth, max_depth)
    return overlay


def _draw_recursive(draw, node, depth, max_depth):
    if depth > max_depth:
        return
    colour = PALETTE[depth % len(PALETTE)]
    draw.rectangle([node.x1, node.y1, node.x2, node.y2],
                   outline=colour, width=2)
    for child in node.children:
        _draw_recursive(draw, child, depth + 1, max_depth)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic sheet music with bounding-box annotations.')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images to generate.')
    parser.add_argument('--out_dir', type=str, default='data/sheet_music',
                        help='Output directory.')
    parser.add_argument('--retina_size', type=int, default=1024,
                        help='Retina size for training sample extraction.')
    parser.add_argument('--visualise', action='store_true',
                        help='Also save images with overlaid bounding boxes.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    args = parser.parse_args()

    random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'images'), exist_ok=True)
    if args.visualise:
        os.makedirs(os.path.join(args.out_dir, 'vis'), exist_ok=True)

    gen = SheetMusicGenerator()

    annotations = []
    total_samples = 0

    for i in range(args.num_images):
        img, tree = gen.generate(seed=args.seed + i)

        # Save raw image
        img_name = f'sheet_{i:05d}.png'
        img_path = os.path.join(args.out_dir, 'images', img_name)
        img.save(img_path)

        # Save annotation
        ann = {
            'image': img_name,
            'width': img.width,
            'height': img.height,
            'bbox_tree': tree.to_dict(),
        }
        annotations.append(ann)

        # Extract training samples (count only; saving all images would be
        # very large — the training loop generates them on-the-fly instead)
        samples = extract_training_samples(img, tree, args.retina_size)
        total_samples += len(samples)

        if args.visualise:
            vis = draw_bbox_tree(img, tree)
            vis.save(os.path.join(args.out_dir, 'vis', img_name))

        if (i + 1) % 10 == 0 or i == 0:
            print(f'  [{i+1}/{args.num_images}]  '
                  f'samples so far: {total_samples}')

    # Save all annotations
    ann_path = os.path.join(args.out_dir, 'annotations.json')
    with open(ann_path, 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f'\nDone. {args.num_images} images → {total_samples} training samples.')
    print(f'Annotations: {ann_path}')


if __name__ == '__main__':
    main()
