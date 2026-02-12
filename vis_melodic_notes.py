"""
Visualise note bounding boxes that are children of melodic_line nodes
from an inference result tree.

Child bboxes in the JSON are relative to their parent's crop, so we
accumulate offsets while walking the tree.
"""

import json
import sys
from PIL import Image, ImageDraw, ImageFont


def find_melodic_note_boxes(node, parent_abs_x=0, parent_abs_y=0):
    """
    Walk the tree and collect absolute bounding boxes for:
      - melodic_line nodes (drawn in blue)
      - note children of melodic_line nodes (drawn in red)

    Child bbox coords are relative to the parent's crop region,
    so we accumulate offsets.
    """
    abs_x1 = parent_abs_x + node["bbox"][0]
    abs_y1 = parent_abs_y + node["bbox"][1]
    abs_x2 = parent_abs_x + node["bbox"][2]
    abs_y2 = parent_abs_y + node["bbox"][3]

    melodic_boxes = []
    note_boxes = []

    if node["label"] == "melodic_line":
        melodic_boxes.append((abs_x1, abs_y1, abs_x2, abs_y2))
        # Collect note children
        for child in node.get("children", []):
            if child["label"] == "note":
                cx1 = abs_x1 + child["bbox"][0]
                cy1 = abs_y1 + child["bbox"][1]
                cx2 = abs_x1 + child["bbox"][2]
                cy2 = abs_y1 + child["bbox"][3]
                note_boxes.append((cx1, cy1, cx2, cy2))

    # Recurse into all children
    for child in node.get("children", []):
        m, n = find_melodic_note_boxes(child, abs_x1, abs_y1)
        melodic_boxes.extend(m)
        note_boxes.extend(n)

    return melodic_boxes, note_boxes


def main():
    # Load tree
    with open("test_results/inference_result.json") as f:
        tree = json.load(f)

    # Find the source image — try to determine from the tree's page bbox
    page_w = tree["bbox"][2]
    page_h = tree["bbox"][3]

    # Use a source image (try sheet_00000.png as fallback)
    source_img = sys.argv[1] if len(sys.argv) > 1 else "data/sheet_music/images/sheet_00000.png"
    img = Image.open(source_img).convert("RGB")

    melodic_boxes, note_boxes = find_melodic_note_boxes(tree)

    draw = ImageDraw.Draw(img)

    # Draw melodic line boxes (blue, dashed-style with thinner line)
    for (x1, y1, x2, y2) in melodic_boxes:
        draw.rectangle([x1, y1, x2, y2], outline="#4363d8", width=2)
        # Label
        draw.text((x1 + 2, y1 - 12), "melodic_line", fill="#4363d8")

    # Draw note boxes (red, thicker)
    for i, (x1, y1, x2, y2) in enumerate(note_boxes):
        draw.rectangle([x1, y1, x2, y2], outline="#e6194b", width=3)
        draw.text((x1, y1 - 12), f"note", fill="#e6194b")

    out_path = "test_results/melodic_note_boxes.png"
    img.save(out_path)
    print(f"Saved to {out_path}")
    print(f"Found {len(melodic_boxes)} melodic_line(s) with {len(note_boxes)} note children total")


if __name__ == "__main__":
    main()
