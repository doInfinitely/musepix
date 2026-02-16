"""
Test script for the sheet music OCR model.

Generates synthetic sheet music (with known ground-truth bounding-box trees),
runs hierarchical decoding, compares predictions to ground truth, and saves
visualisations.

Usage:
    # Quick sanity check with a fresh (untrained) model:
    python3 test_music_ocr.py

    # Test a trained checkpoint:
    python3 test_music_ocr.py --checkpoint checkpoints/music_ocr.pt

    # Test on a specific image (no ground truth comparison):
    python3 test_music_ocr.py --checkpoint checkpoints/music_ocr.pt --image my_sheet.png

    # Train first, then test:
    python3 test_music_ocr.py --train --epochs 30 --num_train_images 200

    # Full pipeline: train + test + save outputs:
    python3 test_music_ocr.py --train --epochs 30 --out_dir test_results
"""

import os
import sys
import json
import random
import argparse
from collections import Counter

import torch
from PIL import Image, ImageDraw, ImageFont

from retina import fit_to_retina, image_to_tensor
from generate_sheet_music import (
    CLASSES, CLASS_TO_ID, BBox, SheetMusicGenerator,
    extract_training_samples, draw_bbox_tree, PALETTE,
)
from music_ocr_model import (
    MusicOCRModel, MusicOCRLoss, TeacherForcingDataset,
    predict, hierarchical_decode, hierarchical_decode_hybrid,
)
from mask_network import MaskNetwork


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def flatten_tree(node, depth=0):
    """Flatten a BBox tree into a list of (bbox_tuple, label, depth)."""
    items = [(node.as_tuple(), node.label, depth)]
    for child in node.children:
        items.extend(flatten_tree(child, depth + 1))
    return items


def tree_stats(node):
    """Return (total_nodes, max_depth, class_counts)."""
    flat = flatten_tree(node)
    counts = Counter(label for _, label, _ in flat)
    max_d = max(d for _, _, d in flat)
    return len(flat), max_d, counts


def bbox_iou(a, b):
    """Intersection-over-union for two (x1, y1, x2, y2) bboxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compare_trees(gt_tree, pred_tree, img_w, img_h):
    """
    Compare ground-truth and predicted trees.
    Returns a dict of metrics.
    """
    gt_flat = flatten_tree(gt_tree)
    pred_flat = flatten_tree(pred_tree)

    gt_nodes, gt_depth, gt_counts = tree_stats(gt_tree)
    pred_nodes, pred_depth, pred_counts = tree_stats(pred_tree)

    # Class-level comparison
    all_classes = set(gt_counts.keys()) | set(pred_counts.keys())
    class_comparison = {}
    for cls in sorted(all_classes):
        class_comparison[cls] = {
            'ground_truth': gt_counts.get(cls, 0),
            'predicted': pred_counts.get(cls, 0),
        }

    # Greedy match: for each GT node, find closest predicted node of same class
    matched_ious = []
    for gt_bbox, gt_label, _ in gt_flat:
        if gt_label in ('page', 'none'):
            continue
        best_iou = 0.0
        for pred_bbox, pred_label, _ in pred_flat:
            if pred_label == gt_label:
                iou = bbox_iou(gt_bbox, pred_bbox)
                best_iou = max(best_iou, iou)
        matched_ious.append((gt_label, best_iou))

    mean_iou = (sum(iou for _, iou in matched_ious) / len(matched_ious)
                if matched_ious else 0.0)

    return {
        'gt_nodes': gt_nodes,
        'pred_nodes': pred_nodes,
        'gt_depth': gt_depth,
        'pred_depth': pred_depth,
        'class_comparison': class_comparison,
        'mean_matched_iou': mean_iou,
        'matched_details': matched_ious,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def draw_comparison(img, gt_tree, pred_tree):
    """
    Create a side-by-side image: ground truth (left) vs prediction (right).
    """
    gt_vis = draw_bbox_tree(img, gt_tree)
    pred_vis = draw_bbox_tree(img, pred_tree)

    # Add labels
    for vis, label in [(gt_vis, "Ground Truth"), (pred_vis, "Predicted")]:
        draw = ImageDraw.Draw(vis)
        draw.rectangle([0, 0, 140, 20], fill='white')
        draw.text((5, 3), label, fill='black')

    combined = Image.new('RGB', (img.width * 2 + 10, img.height), 'gray')
    combined.paste(gt_vis, (0, 0))
    combined.paste(pred_vis, (img.width + 10, 0))
    return combined


def print_tree(node, indent=0, max_depth=6):
    """Pretty-print a BBox tree."""
    prefix = "  " * indent
    bbox = f"[{node.x1:.0f}, {node.y1:.0f}, {node.x2:.0f}, {node.y2:.0f}]"
    n_children = len(node.children)
    print(f"{prefix}{node.label}  {bbox}  children={n_children}")
    if indent < max_depth:
        for child in node.children:
            print_tree(child, indent + 1, max_depth)
    elif node.children:
        print(f"{prefix}  ... ({n_children} children omitted)")


# ──────────────────────────────────────────────────────────────────────────────
# Training helper
# ──────────────────────────────────────────────────────────────────────────────

def train_model(args, device):
    """Train the model and return it."""
    from torch.utils.data import DataLoader
    import torch.optim as optim

    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")

    dataset = TeacherForcingDataset(
        data_dir=args.data_dir,
        num_images=args.num_train_images,
        retina_size=args.retina_size,
        seed=args.seed,
    )
    print(f"Dataset: {len(dataset)} training samples")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)

    model = MusicOCRModel(num_classes=len(CLASSES)).to(device)

    # Differential learning rates (backbone slower, heads faster)
    backbone_params = list(model.backbone.parameters())
    head_params = (list(model.fc_shared.parameters())
                   + list(model.bbox_head.parameters())
                   + list(model.cls_head.parameters()))
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': head_params, 'lr': args.lr},
    ], weight_decay=1e-2)

    loss_fn = MusicOCRLoss().to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_bbox_loss = 0.0
        total_cls_loss = 0.0
        n_batches = 0

        for x, prev, bbox_target, cls_target in loader:
            x = x.to(device)
            prev = prev.to(device)
            bbox_target = bbox_target.to(device)
            cls_target = cls_target.to(device)

            bbox_pred, cls_pred = model(x, prev)
            loss, loss_bbox, loss_cls = loss_fn(bbox_pred, cls_pred,
                                                bbox_target, cls_target)

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
        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"loss={avg:.4f}  bbox={avg_bb:.4f}  cls={avg_cl:.4f}")

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/music_ocr_v2.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Saved checkpoint -> {ckpt_path}")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Main test routine
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test the sheet music OCR model.")

    # Model / checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a trained checkpoint (.pt file).")
    parser.add_argument("--mask_checkpoint", type=str, default=None,
                        help="Path to mask segmentation checkpoint (.pt file).")
    parser.add_argument("--retina_size", type=int, default=1024)
    parser.add_argument("--max_depth", type=int, default=10,
                        help="Max recursion depth for hierarchical decoding.")

    # Training options
    parser.add_argument("--train", action="store_true",
                        help="Train the model before testing.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Training data directory.")
    parser.add_argument("--num_train_images", type=int, default=200,
                        help="Number of images for on-the-fly training.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Test options
    parser.add_argument("--image", type=str, default=None,
                        help="Run inference on a single image (no GT comparison).")
    parser.add_argument("--num_test_images", type=int, default=5,
                        help="Number of synthetic test images to generate.")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Random seed (different from training default).")
    parser.add_argument("--out_dir", type=str, default="test_results",
                        help="Directory to save test outputs.")

    args = parser.parse_args()

    # ── Device ──
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    # ── Get model ──
    model = None

    if args.train:
        model = train_model(args, device)
    elif args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
        # Detect num_classes from checkpoint (cls_head.2.bias shape)
        ckpt_num_classes = state_dict['cls_head.2.bias'].shape[0]
        if ckpt_num_classes != len(CLASSES):
            print(f"  Note: checkpoint has {ckpt_num_classes} classes, "
                  f"current CLASSES has {len(CLASSES)} — using checkpoint's count")
        model = MusicOCRModel(num_classes=ckpt_num_classes).to(device)
        model.load_state_dict(state_dict)
    else:
        print("No checkpoint provided and --train not set.")
        print("Using randomly initialised model (sanity check mode).")
        model = MusicOCRModel(num_classes=len(CLASSES)).to(device)

    model.eval()

    # ── Mask model ──
    mask_model = None
    if args.mask_checkpoint:
        print(f"Loading mask checkpoint: {args.mask_checkpoint}")
        mask_model = MaskNetwork().to(device)
        mask_model.load_state_dict(
            torch.load(args.mask_checkpoint, map_location=device,
                       weights_only=True))
        mask_model.eval()

    # ── Output directory ──
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Single image mode ──
    if args.image:
        print(f"\n{'='*60}")
        print(f"INFERENCE: {args.image}")
        print(f"{'='*60}")

        img = Image.open(args.image).convert("RGB")
        print(f"Image size: {img.width} x {img.height}")

        if mask_model is not None:
            pred_tree = hierarchical_decode_hybrid(
                model, img, retina_size=args.retina_size,
                max_depth=args.max_depth, device=device,
                mask_model=mask_model)
        else:
            pred_tree = hierarchical_decode(
                model, img, retina_size=args.retina_size,
                max_depth=args.max_depth, device=device)

        print("\n-- Predicted tree --")
        print_tree(pred_tree)

        pred_nodes, pred_depth, pred_counts = tree_stats(pred_tree)
        print(f"\nTotal nodes: {pred_nodes}")
        print(f"Max depth:   {pred_depth}")
        print(f"Classes:     {dict(pred_counts)}")

        # Save visualisation
        vis = draw_bbox_tree(img, pred_tree)
        vis_path = os.path.join(args.out_dir, "inference_result.png")
        vis.save(vis_path)
        print(f"\nVisualisation saved -> {vis_path}")

        # Save tree JSON
        json_path = os.path.join(args.out_dir, "inference_result.json")
        with open(json_path, "w") as f:
            json.dump(pred_tree.to_dict(), f, indent=2)
        print(f"Tree JSON saved    -> {json_path}")
        return

    # ── Synthetic test set with ground truth ──
    print(f"\n{'='*60}")
    print(f"TESTING on {args.num_test_images} synthetic images")
    print(f"{'='*60}")

    gen = SheetMusicGenerator()
    random.seed(args.seed)

    all_metrics = []

    for i in range(args.num_test_images):
        img, gt_tree = gen.generate(seed=args.seed + i)

        print(f"\n--- Test image {i+1}/{args.num_test_images} ---")

        # Run hierarchical decode
        if mask_model is not None:
            pred_tree = hierarchical_decode_hybrid(
                model, img, retina_size=args.retina_size,
                max_depth=args.max_depth, device=device,
                mask_model=mask_model)
        else:
            pred_tree = hierarchical_decode(
                model, img, retina_size=args.retina_size,
                max_depth=args.max_depth, device=device)

        # Compare
        metrics = compare_trees(gt_tree, pred_tree, img.width, img.height)
        all_metrics.append(metrics)

        print(f"  GT nodes:   {metrics['gt_nodes']:4d}   "
              f"Pred nodes: {metrics['pred_nodes']:4d}")
        print(f"  GT depth:   {metrics['gt_depth']:4d}   "
              f"Pred depth: {metrics['pred_depth']:4d}")
        print(f"  Mean IoU:   {metrics['mean_matched_iou']:.3f}")

        print(f"  Class counts (GT -> Pred):")
        for cls, vals in metrics['class_comparison'].items():
            gt_c = vals['ground_truth']
            pr_c = vals['predicted']
            marker = " *" if gt_c != pr_c else ""
            print(f"    {cls:15s}  {gt_c:3d} -> {pr_c:3d}{marker}")

        # Save visualisations
        comparison = draw_comparison(img, gt_tree, pred_tree)
        comp_path = os.path.join(args.out_dir, f"comparison_{i:03d}.png")
        comparison.save(comp_path)

        pred_vis = draw_bbox_tree(img, pred_tree)
        pred_path = os.path.join(args.out_dir, f"predicted_{i:03d}.png")
        pred_vis.save(pred_path)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    mean_ious = [m['mean_matched_iou'] for m in all_metrics]
    gt_node_counts = [m['gt_nodes'] for m in all_metrics]
    pred_node_counts = [m['pred_nodes'] for m in all_metrics]

    print(f"  Images tested:    {len(all_metrics)}")
    print(f"  Mean IoU:         {sum(mean_ious)/len(mean_ious):.3f}")
    print(f"  Avg GT nodes:     {sum(gt_node_counts)/len(gt_node_counts):.1f}")
    print(f"  Avg Pred nodes:   {sum(pred_node_counts)/len(pred_node_counts):.1f}")

    # Aggregate class counts
    agg_gt = Counter()
    agg_pred = Counter()
    for m in all_metrics:
        for cls, vals in m['class_comparison'].items():
            agg_gt[cls] += vals['ground_truth']
            agg_pred[cls] += vals['predicted']

    print(f"\n  Aggregate class counts:")
    for cls in sorted(set(agg_gt.keys()) | set(agg_pred.keys())):
        print(f"    {cls:15s}  GT={agg_gt[cls]:4d}  Pred={agg_pred[cls]:4d}")

    # Save summary JSON
    summary = {
        'num_images': len(all_metrics),
        'mean_iou': sum(mean_ious) / len(mean_ious),
        'per_image': [
            {
                'gt_nodes': m['gt_nodes'],
                'pred_nodes': m['pred_nodes'],
                'gt_depth': m['gt_depth'],
                'pred_depth': m['pred_depth'],
                'mean_iou': m['mean_matched_iou'],
                'class_comparison': m['class_comparison'],
            }
            for m in all_metrics
        ],
    }
    summary_path = os.path.join(args.out_dir, "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to:  {args.out_dir}/")
    print(f"  Summary JSON:      {summary_path}")
    print(f"  Comparison images: {args.out_dir}/comparison_*.png")
    print(f"  Prediction images: {args.out_dir}/predicted_*.png")


if __name__ == "__main__":
    main()
