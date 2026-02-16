"""
Annotate real sheet music images using OpenAI vision + programmatic nesting.

Pipeline:
  1. Send image to OpenAI vision → flat list of {label, bbox} dicts
  2. Nest flat bboxes into a BBox tree using spatial containment rules

The output matches annotations.json format for direct use with TeacherForcingDataset.

Hierarchy:
  page > staff_system > staff_lines (leaf)
                      > melodic_line > note > ledger_lines (optional bundle)

Usage:
  python3 openai_annotator.py annotate --image path/to/image.png
  python3 openai_annotator.py batch --image_dir data/real_sheets --out annotations.json
"""

import os
import json
import base64
import argparse
from PIL import Image

from generate_sheet_music import BBox, CLASSES

# Valid parent → children relationships
CHILDREN_OF = {
    'page': ['staff_system'],
    'staff_system': ['staff_lines', 'melodic_line'],
    'melodic_line': ['note'],
    'note': ['ledger_lines'],
}

# Reverse: child → valid parent label
PARENT_OF = {}
for parent, kids in CHILDREN_OF.items():
    for kid in kids:
        PARENT_OF[kid] = parent

# Labels that are leaves (no children)
LEAF_LABELS = {'staff_lines', 'ledger_lines'}

# Processing order: parents before children
LEVEL_ORDER = [
    'page', 'staff_system', 'staff_lines', 'melodic_line',
    'note', 'ledger_lines',
]


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI client
# ──────────────────────────────────────────────────────────────────────────────

def _get_client():
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY not set in environment.')

    from openai import OpenAI
    return OpenAI(api_key=api_key)


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Flat detection via OpenAI vision
# ──────────────────────────────────────────────────────────────────────────────

DETECT_PROMPT = """\
You are a sheet music element detector. Analyze this sheet music image and detect \
every visible element listed below. Return pixel-coordinate bounding boxes.

The image is {width}x{height} pixels. All coordinates must be within these bounds.

Element types to detect:
- staff_system: the full region containing a set of staff lines plus any notes
- staff_lines: the rectangular region spanning all 5 staff lines (just the lines, no notes)
- melodic_line: a horizontal group of consecutive notes on the staff
- note: the entire note including notehead, stem, and any ledger lines (one bbox per note)
- ledger_lines: the bundle of short horizontal lines above or below the staff for a note \
(one bbox encompassing all ledger lines for that note, not individual lines)

Return a JSON object with a single key "elements" containing a list. Each element is:
{{"label": "<element_type>", "bbox": [x1, y1, x2, y2]}}

where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner in pixels.

Detect ALL instances of each element type. Be precise with bounding boxes — they should \
tightly enclose each element. Do NOT create thin bounding boxes for individual lines or stems.
"""


def detect_flat_bboxes(image_path, model="gpt-4o-mini"):
    """Send image to OpenAI vision and get a flat list of detected elements."""
    client = _get_client()

    img = Image.open(image_path).convert('RGB')
    width, height = img.size

    with open(image_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')

    ext = os.path.splitext(image_path)[1].lower()
    mime = 'image/png' if ext == '.png' else 'image/jpeg'

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": DETECT_PROMPT.format(width=width, height=height)},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{b64}",
                    "detail": "high",
                }},
            ],
        }],
        max_tokens=4096,
    )

    raw = json.loads(response.choices[0].message.content)
    elements = raw.get('elements', raw.get('detections', []))

    # Validate
    valid_labels = set(CLASSES) - {'none', 'page'}
    validated = []
    for el in elements:
        label = el.get('label', '')
        bbox = el.get('bbox', [])
        if label not in valid_labels:
            continue
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        # Clamp to image bounds
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        validated.append({'label': label, 'bbox': [x1, y1, x2, y2]})

    return validated


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Programmatic nesting via spatial containment
# ──────────────────────────────────────────────────────────────────────────────

def _overlap_fraction(child_bbox, parent_bbox):
    """Fraction of child's area that overlaps with parent."""
    cx1, cy1, cx2, cy2 = child_bbox
    px1, py1, px2, py2 = parent_bbox

    ix1 = max(cx1, px1)
    iy1 = max(cy1, py1)
    ix2 = min(cx2, px2)
    iy2 = min(cy2, py2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    child_area = (cx2 - cx1) * (cy2 - cy1)
    if child_area <= 0:
        return 0.0
    return inter / child_area


def _bbox_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def nest_flat_bboxes(flat_bboxes, image_width, image_height):
    """Build a nested BBox tree from flat detections using spatial containment.

    Uses the known hierarchy to assign each element to its correct parent
    based on bounding box overlap.

    Returns a BBox root node (label='page').
    """
    # Group by label
    by_label = {}
    for el in flat_bboxes:
        by_label.setdefault(el['label'], []).append(el['bbox'])

    # Create BBox nodes for each detection, keyed by (label, index)
    nodes = {}  # (label, idx) → BBox
    for label in LEVEL_ORDER:
        if label == 'page':
            continue
        for idx, bbox in enumerate(by_label.get(label, [])):
            nodes[(label, idx)] = BBox(
                x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                label=label, children=[],
            )

    # Build the page root
    page = BBox(0, 0, image_width, image_height, 'page', children=[])

    # Assign children to parents level by level
    for parent_label in LEVEL_ORDER:
        child_labels = CHILDREN_OF.get(parent_label, [])
        if not child_labels:
            continue

        # Collect parent nodes
        if parent_label == 'page':
            parents = [('page', 0, page)]
        else:
            parents = [
                (parent_label, idx, nodes[(parent_label, idx)])
                for idx in range(len(by_label.get(parent_label, [])))
            ]

        if not parents:
            continue

        # For each child label, assign each child to the best-overlapping parent
        for child_label in child_labels:
            for cidx in range(len(by_label.get(child_label, []))):
                child_node = nodes[(child_label, cidx)]
                child_bbox = [child_node.x1, child_node.y1,
                              child_node.x2, child_node.y2]

                best_parent = None
                best_overlap = 0.3  # minimum threshold

                for _, _, pnode in parents:
                    parent_bbox = [pnode.x1, pnode.y1, pnode.x2, pnode.y2]
                    overlap = _overlap_fraction(child_bbox, parent_bbox)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_parent = pnode

                if best_parent is not None:
                    best_parent.children.append(child_node)

    # Sort children left-to-right at each level
    _sort_children(page)

    return page


def _sort_children(node):
    """Recursively sort children by x1 (left to right)."""
    node.children.sort(key=lambda c: c.x1)
    for child in node.children:
        _sort_children(child)


# ──────────────────────────────────────────────────────────────────────────────
# Combined pipeline
# ──────────────────────────────────────────────────────────────────────────────

def annotate_image(image_path, model="gpt-4o-mini"):
    """Full pipeline: detect flat bboxes → nest into tree → return annotation dict."""
    img = Image.open(image_path).convert('RGB')
    width, height = img.size

    print(f'  Detecting elements in {image_path}...')
    flat = detect_flat_bboxes(image_path, model=model)
    print(f'  Found {len(flat)} flat elements')

    tree = nest_flat_bboxes(flat, width, height)

    return {
        'image': os.path.basename(image_path),
        'width': width,
        'height': height,
        'bbox_tree': tree.to_dict(),
    }


def batch_annotate(image_dir, out_path, model="gpt-4o-mini"):
    """Process all .png images in a directory and write annotations.json."""
    images = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith('.png')
    )
    print(f'Found {len(images)} images in {image_dir}')

    annotations = []
    for i, fname in enumerate(images):
        image_path = os.path.join(image_dir, fname)
        try:
            ann = annotate_image(image_path, model=model)
            annotations.append(ann)
            print(f'  [{i+1}/{len(images)}] {fname} — OK')
        except Exception as e:
            print(f'  [{i+1}/{len(images)}] {fname} — FAILED: {e}')

    with open(out_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f'Wrote {len(annotations)} annotations to {out_path}')


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Annotate sheet music images using OpenAI vision.')
    sub = parser.add_subparsers(dest='command')

    # annotate single image
    p_ann = sub.add_parser('annotate', help='Annotate a single image.')
    p_ann.add_argument('--image', required=True, help='Path to image.')
    p_ann.add_argument('--model', default='gpt-4o-mini')

    # batch annotate
    p_batch = sub.add_parser('batch', help='Batch annotate a directory.')
    p_batch.add_argument('--image_dir', required=True, help='Directory of .png images.')
    p_batch.add_argument('--out', default='annotations.json', help='Output JSON path.')
    p_batch.add_argument('--model', default='gpt-4o-mini')

    args = parser.parse_args()

    if args.command == 'annotate':
        result = annotate_image(args.image, model=args.model)
        # Verify round-trip
        tree = BBox.from_dict(result['bbox_tree'])
        print(json.dumps(result, indent=2))
        print(f'\nTree root: {tree.label}, children: {len(tree.children)}')

    elif args.command == 'batch':
        batch_annotate(args.image_dir, args.out, model=args.model)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
