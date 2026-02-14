"""Compare inference result.json against ground truth from annotations.json."""
import json
import sys


def flatten_tree(node, depth=0):
    """Flatten a bbox tree into a list of (depth, label, start, end, active_bins)."""
    mask = node.get("freq_mask", [])
    active = sum(1 for m in mask if m)
    items = [(depth, node["label"], node["start_frame"], node["end_frame"], active)]
    for c in node.get("children", []):
        items.extend(flatten_tree(c, depth + 1))
    return items


def print_tree(node, indent=0):
    mask = node.get("freq_mask", [])
    active = sum(1 for m in mask if m)
    n_ch = len(node.get("children", []))
    print(f"{'  ' * indent}{node['label']}  "
          f"frames=[{node['start_frame']}:{node['end_frame']}]  "
          f"bins={active}/{len(mask)}  children={n_ch}")
    for c in node.get("children", []):
        print_tree(c, indent + 1)


def count_nodes(node):
    return 1 + sum(count_nodes(c) for c in node.get("children", []))


def max_depth(node, d=0):
    if not node.get("children"):
        return d
    return max(max_depth(c, d + 1) for c in node["children"])


def main():
    ann_path = sys.argv[1] if len(sys.argv) > 1 else "data/audio_arturia/annotations.json"
    pred_path = sys.argv[2] if len(sys.argv) > 2 else "result.json"
    song_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    with open(ann_path) as f:
        gt_tree = json.load(f)[song_idx]["bbox_tree"]
    with open(pred_path) as f:
        pred_tree = json.load(f)

    print("=" * 60)
    print("GROUND TRUTH")
    print("=" * 60)
    print_tree(gt_tree)
    print(f"\nTotal nodes: {count_nodes(gt_tree)}  |  Max depth: {max_depth(gt_tree)}")

    print()
    print("=" * 60)
    print("PREDICTION")
    print("=" * 60)
    print_tree(pred_tree)
    print(f"\nTotal nodes: {count_nodes(pred_tree)}  |  Max depth: {max_depth(pred_tree)}")

    # --- Leaf-level comparison ---
    def get_leaves(node):
        if not node.get("children"):
            return [(node["label"], node["start_frame"], node["end_frame"])]
        leaves = []
        for c in node.get("children", []):
            leaves.extend(get_leaves(c))
        return leaves

    gt_leaves = get_leaves(gt_tree)
    pred_leaves = get_leaves(pred_tree)

    print()
    print("=" * 60)
    print("LEAF COMPARISON")
    print("=" * 60)
    print(f"Ground truth leaves: {len(gt_leaves)}")
    for label, s, e in gt_leaves:
        print(f"  {label}  [{s}:{e}]  (span={e - s})")

    print(f"\nPredicted leaves: {len(pred_leaves)}")
    for label, s, e in pred_leaves[:20]:
        print(f"  {label}  [{s}:{e}]  (span={e - s})")
    if len(pred_leaves) > 20:
        print(f"  ... and {len(pred_leaves) - 20} more")


if __name__ == "__main__":
    main()
