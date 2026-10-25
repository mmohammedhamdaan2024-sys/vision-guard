"""
VisionGuard — Model Training Script
Fine-tunes Ultralytics YOLOv8 architecture for specialized traffic safety classes
(e.g., helmet/no_helmet or seatbelt/no_seatbelt).
"""

import os
import argparse
from ultralytics import YOLO


def train(data_yaml: str, model_weights: str = "yolov8n.pt", epochs: int = 50,
          imgsz: int = 640, batch: int = 16, project: str = "models/train_runs"):
    """
    Executes deep learning training pipeline with PyTorch and Ultralytics YOLO.
    """
    if not os.path.exists(data_yaml):
        print(f"Error: Dataset configuration '{data_yaml}' not found.")
        print("Please run scripts/prepare_dataset.py first or point to a valid dataset YAML.")
        return

    print(f"Initializing YOLOv8 fine-tuning from base weights: {model_weights}")
    print(f"Dataset config: {data_yaml} | Epochs: {epochs} | Image size: {imgsz}")

    model = YOLO(model_weights)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name="visionguard_custom",
        save=True,
        verbose=True
    )

    print("Training finished!")
    best_weights = os.path.join(project, "visionguard_custom", "weights", "best.pt")
    if os.path.exists(best_weights):
        print(f"Optimized model weights saved to: {best_weights}")
        print("Copy these weights to 'models/helmet_yolo.pt' or 'models/seatbelt_yolo.pt' to activate in VisionGuard.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionGuard Model Training Script")
    parser.add_argument("--data", type=str, default="datasets/helmet_dataset.yaml", help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Initial weights")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    train(data_yaml=args.data, model_weights=args.weights, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
