"""
Test script for the audio OCR model.

Loads audio files with known ground-truth AudioBBox trees, runs hierarchical
decoding, compares predictions to ground truth, and saves visualisations.

Usage:
    # Test a trained checkpoint on the arturia dataset:
    python test_audio_ocr.py --checkpoint checkpoints/audio_ocr.pt \
        --data_dir data/audio_arturia

    # Test on a single WAV file (no GT comparison):
    python test_audio_ocr.py --checkpoint checkpoints/audio_ocr.pt \
        --wav my_audio.wav

    # Quick sanity check with untrained model:
    python test_audio_ocr.py
"""

import os
import sys
import json
import argparse
from collections import Counter

import numpy as np
import torch

from generate_audio import (
    AUDIO_CLASSES, AUDIO_CLASS_TO_ID, AudioBBox,
    SAMPLE_RATE, HOP_LENGTH, N_MELS, N_FFT, SPEC_FLOOR_DB,
    compute_mel_spectrogram, spectrogram_to_image, draw_audio_bbox_tree,
)
from audio_ocr_model import (
    AudioOCRModel, hierarchical_decode,
)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def flatten_tree(node, depth=0):
    """Flatten an AudioBBox tree into a list of (start, end, label, depth)."""
    items = [(node.start_frame, node.end_frame, node.label, depth)]
    for child in node.children:
        items.extend(flatten_tree(child, depth + 1))
    return items


def tree_stats(node):
    """Return (total_nodes, max_depth, class_counts)."""
    flat = flatten_tree(node)
    counts = Counter(label for _, _, label, _ in flat)
    max_d = max(d for _, _, _, d in flat)
    return len(flat), max_d, counts


def temporal_iou(a_start, a_end, b_start, b_end):
    """Intersection-over-union for two temporal intervals."""
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    inter = max(0, inter_end - inter_start)
    len_a = max(0, a_end - a_start)
    len_b = max(0, b_end - b_start)
    union = len_a + len_b - inter
    return inter / union if union > 0 else 0.0


def compare_trees(gt_tree, pred_tree):
    """
    Compare ground-truth and predicted AudioBBox trees.
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
    for gt_sf, gt_ef, gt_label, _ in gt_flat:
        if gt_label in ('full_mix', 'none'):
            continue
        best_iou = 0.0
        for pr_sf, pr_ef, pr_label, _ in pred_flat:
            if pr_label == gt_label:
                iou = temporal_iou(gt_sf, gt_ef, pr_sf, pr_ef)
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

def draw_comparison(spec_img, gt_tree, pred_tree, n_frames):
    """
    Create a side-by-side image: ground truth (top) vs prediction (bottom).
    """
    from PIL import ImageDraw

    gt_vis = draw_audio_bbox_tree(spec_img, gt_tree, n_frames)
    pred_vis = draw_audio_bbox_tree(spec_img, pred_tree, n_frames)

    # Add labels
    for vis, label in [(gt_vis, "Ground Truth"), (pred_vis, "Predicted")]:
        draw = ImageDraw.Draw(vis)
        draw.rectangle([0, 0, 120, 16], fill='white')
        draw.text((4, 2), label, fill='black')

    gap = 4
    combined = gt_vis.copy().convert('RGB')
    w, h = combined.size
    from PIL import Image as PILImage
    out = PILImage.new('RGB', (w, h * 2 + gap), 'gray')
    out.paste(combined, (0, 0))
    out.paste(pred_vis.convert('RGB'), (0, h + gap))
    return out


def print_tree(node, indent=0, max_depth=6):
    """Pretty-print an AudioBBox tree."""
    prefix = "  " * indent
    mask_count = sum(1 for m in node.freq_mask if m)
    n_ch = len(node.children)
    print(f"{prefix}{node.label}  "
          f"frames=[{node.start_frame}:{node.end_frame}]  "
          f"bins={mask_count}/{len(node.freq_mask)}  "
          f"children={n_ch}")
    if indent < max_depth:
        for child in node.children:
            print_tree(child, indent + 1, max_depth)
    elif node.children:
        print(f"{prefix}  ... ({n_ch} children omitted)")


# ──────────────────────────────────────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    """Load an AudioOCRModel, handling both old and new checkpoint formats."""
    model = AudioOCRModel(n_mels=N_MELS,
                          num_classes=len(AUDIO_CLASSES)).to(device)
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            epoch = ckpt.get('epoch', '?')
            print(f"  Loaded model from epoch {epoch}")
        else:
            model.load_state_dict(ckpt)
            print(f"  Loaded model weights (legacy format)")
    else:
        print("No checkpoint — using randomly initialised model (sanity check).")
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Main test routine
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test the audio OCR model.")

    # Model / checkpoint
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/audio_ocr.pt",
                        help="Path to a trained checkpoint (.pt file).")
    parser.add_argument("--retina_size", type=int, default=1024)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--max_siblings", type=int, default=15)
    parser.add_argument("--max_total_calls", type=int, default=500)

    # Data
    parser.add_argument("--data_dir", type=str, default="data/audio_arturia",
                        help="Data directory with annotations.json, wav/, spec/, vis/.")
    parser.add_argument("--wav", type=str, default=None,
                        help="Run inference on a single WAV file (no GT comparison).")
    parser.add_argument("--num_test", type=int, default=5,
                        help="Number of test songs from annotations (0 = all).")
    parser.add_argument("--out_dir", type=str, default="test_results_audio",
                        help="Directory to save test outputs.")

    args = parser.parse_args()

    # ── Device ──
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    # ── Load model ──
    ckpt_path = args.checkpoint if os.path.isfile(args.checkpoint) else None
    model = load_model(ckpt_path, device)

    # ── Output directory ──
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Single WAV mode ──
    if args.wav:
        import wave as wave_mod

        print(f"\n{'='*60}")
        print(f"INFERENCE: {args.wav}")
        print(f"{'='*60}")

        with wave_mod.open(args.wav, 'rb') as wf:
            n_ch = wf.getnchannels()
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32767.0
        if n_ch > 1:
            audio = audio.reshape(-1, n_ch).mean(axis=1)

        spec = compute_mel_spectrogram(audio, sr=sr)
        n_mels, n_frames = spec.shape
        print(f"WAV: {len(audio)/sr:.2f}s, {sr} Hz -> {n_mels} mels x {n_frames} frames")

        pred_tree = hierarchical_decode(
            model, spec, retina_size=args.retina_size,
            max_depth=args.max_depth, max_siblings=args.max_siblings,
            max_total_calls=args.max_total_calls, device=device)

        print("\n-- Predicted tree --")
        print_tree(pred_tree)

        pred_nodes, pred_depth, pred_counts = tree_stats(pred_tree)
        print(f"\nTotal nodes: {pred_nodes}")
        print(f"Max depth:   {pred_depth}")
        print(f"Classes:     {dict(pred_counts)}")

        # Save visualisation
        spec_img = spectrogram_to_image(spec)
        vis = draw_audio_bbox_tree(spec_img, pred_tree, n_frames)
        vis_path = os.path.join(args.out_dir, "inference_result.png")
        vis.save(vis_path)
        print(f"\nVisualisation saved -> {vis_path}")

        # Save tree JSON
        json_path = os.path.join(args.out_dir, "inference_result.json")
        with open(json_path, "w") as f:
            json.dump(pred_tree.to_dict(), f, indent=2)
        print(f"Tree JSON saved    -> {json_path}")
        return

    # ── Test against ground truth from annotations ──
    ann_path = os.path.join(args.data_dir, "annotations.json")
    if not os.path.isfile(ann_path):
        print(f"ERROR: annotations not found at {ann_path}")
        sys.exit(1)

    with open(ann_path) as f:
        annotations = json.load(f)

    num_test = args.num_test if args.num_test > 0 else len(annotations)
    num_test = min(num_test, len(annotations))

    print(f"\n{'='*60}")
    print(f"TESTING on {num_test} songs from {args.data_dir}")
    print(f"{'='*60}")

    all_metrics = []

    for i in range(num_test):
        ann = annotations[i]
        wav_path = os.path.join(args.data_dir, "wav", ann["wav"])
        gt_tree = AudioBBox.from_dict(ann["bbox_tree"])
        n_frames = ann["n_frames"]

        print(f"\n--- Song {i+1}/{num_test}: {ann['wav']} ---")

        # Load audio -> spectrogram
        import wave as wave_mod
        with wave_mod.open(wav_path, 'rb') as wf:
            n_ch = wf.getnchannels()
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32767.0
        if n_ch > 1:
            audio = audio.reshape(-1, n_ch).mean(axis=1)
        spec = compute_mel_spectrogram(audio, sr=sr)
        n_mels_actual, n_frames_actual = spec.shape
        print(f"  Spec: {n_mels_actual} mels x {n_frames_actual} frames")

        # Run hierarchical decode
        pred_tree = hierarchical_decode(
            model, spec, retina_size=args.retina_size,
            max_depth=args.max_depth, max_siblings=args.max_siblings,
            max_total_calls=args.max_total_calls, device=device)

        # Compare
        metrics = compare_trees(gt_tree, pred_tree)
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
            print(f"    {cls:20s}  {gt_c:3d} -> {pr_c:3d}{marker}")

        # Save visualisations
        spec_img = spectrogram_to_image(spec)
        comparison = draw_comparison(spec_img, gt_tree, pred_tree, n_frames_actual)
        comp_path = os.path.join(args.out_dir, f"comparison_{i:03d}.png")
        comparison.save(comp_path)

        # GT + pred side by side trees
        print(f"\n  -- Ground truth --")
        print_tree(gt_tree, indent=2, max_depth=4)
        print(f"  -- Predicted --")
        print_tree(pred_tree, indent=2, max_depth=4)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    mean_ious = [m['mean_matched_iou'] for m in all_metrics]
    gt_node_counts = [m['gt_nodes'] for m in all_metrics]
    pred_node_counts = [m['pred_nodes'] for m in all_metrics]

    print(f"  Songs tested:     {len(all_metrics)}")
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
        print(f"    {cls:20s}  GT={agg_gt[cls]:4d}  Pred={agg_pred[cls]:4d}")

    # Save summary JSON
    summary = {
        'num_songs': len(all_metrics),
        'mean_iou': sum(mean_ious) / len(mean_ious),
        'per_song': [
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


if __name__ == "__main__":
    main()
