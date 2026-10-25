"""
VisionGuard — Model Evaluation Script
Computes precision, recall, mAP@0.5, and mAP@0.5:0.95 across validation test sets.
"""

import os
import argparse
from ultralytics import YOLO


def evaluate(weights_path: str, data_yaml: str):
    """Evaluates validation metrics for a trained YOLO weights file."""
    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' does not exist.")
        return

    if not os.path.exists(data_yaml):
        print(f"Error: Dataset YAML '{data_yaml}' not found.")
        return

    print(f"Loading model for evaluation: {weights_path}")
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml, split="val", verbose=True)

    print("\n================ EVALUATION SUMMARY ================")
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")
    print("====================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VisionGuard Model")
    parser.add_argument("--weights", type=str, default="models/yolov8n.pt", help="Model weights path")
    parser.add_argument("--data", type=str, default="datasets/helmet_dataset.yaml", help="Dataset yaml path")
    args = parser.parse_args()

    evaluate(args.weights, args.data)
