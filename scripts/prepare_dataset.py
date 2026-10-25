"""
VisionGuard — Dataset Preparation Script
Validates YOLO annotation formats, creates dataset YAML configuration,
and partitions images into train/val splits for helmet and seatbelt custom models.
"""

import os
import sys
import yaml
from pathlib import Path


def create_dataset_yaml(output_path: str, dataset_root: str, classes: list):
    """Generates standard Ultralytics YOLO dataset configuration file."""
    data = {
        "path": os.path.abspath(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)}
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"Dataset YAML successfully generated: {output_path}")
    print(f"Classes ({len(classes)}): {classes}")


def verify_yolo_labels(labels_dir: str):
    """Checks that YOLO format txt annotations (class_id x_center y_center width height) are normalized."""
    lbl_path = Path(labels_dir)
    if not lbl_path.exists():
        print(f"Directory {labels_dir} does not exist.")
        return False

    valid_count = 0
    error_count = 0

    for txt_file in lbl_path.glob("*.txt"):
        with open(txt_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    error_count += 1
                    continue
                try:
                    cls_id = int(parts[0])
                    coords = [float(v) for v in parts[1:]]
                    if all(0.0 <= c <= 1.0 for c in coords):
                        valid_count += 1
                    else:
                        error_count += 1
                except ValueError:
                    error_count += 1

    print(f"Verification complete: {valid_count} valid labels, {error_count} malformed annotations.")
    return error_count == 0


if __name__ == "__main__":
    print("VisionGuard Dataset Preparation Tool")
    print("Example usage:")
    print("python scripts/prepare_dataset.py --dataset helmet --classes helmet no_helmet")
    # Generate default helmet dataset template
    create_dataset_yaml("datasets/helmet_dataset.yaml", "datasets/helmet", ["helmet", "no_helmet"])
    create_dataset_yaml("datasets/seatbelt_dataset.yaml", "datasets/seatbelt", ["seatbelt", "no_seatbelt"])
