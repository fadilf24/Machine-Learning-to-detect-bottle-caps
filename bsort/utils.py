# bsort/utils.py
"""Utility helper functions for bsort.

Contains lightweight helpers for counting labels and simple visualization.
"""

from collections import Counter
from typing import Dict
import os


def count_labels_in_folder(label_dir: str) -> Dict[int, int]:
    """Count label occurrences per class in a label folder.

    Args:
        label_dir: Path to folder containing YOLO .txt label files.

    Returns:
        A dictionary mapping class index to number of bounding boxes.
    """
    counts: Counter = Counter()
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(label_dir, fname)
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls = int(parts[0])
                except ValueError:
                    # skip malformed lines
                    continue
                counts[cls] += 1

    return dict(counts)


def remap_classes_in_file(label_path: str, mapping: Dict[int, int]) -> None:
    """Remap class indices in a single YOLO label file according to mapping.

    Example: mapping = {1: 0, 2: 1} will change class 1 -> 0 and 2 -> 1.

    Args:
        label_path: Path to single label .txt file.
        mapping: Dictionary mapping old class -> new class.

    Raises:
        FileNotFoundError: If label_path does not exist.
    """
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    lines = []
    with open(label_path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls = int(parts[0])
            except ValueError:
                # keep original line if class not integer
                lines.append(line)
                continue
            new_cls = mapping.get(cls, cls)
            lines.append(" ".join([str(new_cls)] + parts[1:]) + "\n")

    with open(label_path, "w", encoding="utf8") as f:
        f.writelines(lines)
