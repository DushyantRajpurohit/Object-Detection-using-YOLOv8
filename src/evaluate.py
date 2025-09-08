"""
evaluate.py
Evaluate a YOLOv8 model on the validation dataset.
Saves metrics to JSON and displays evaluation plots (PR curve, F1 curve, confusion matrix).
"""

# Importing Libraries
from ultralytics import YOLO
import os
import json
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

MODEL_PATH = cfg["model_path"]
DATA_YAML = cfg["data_yaml"]


def evaluate():
    """Run evaluation on validation dataset and display results."""
    print("Evaluating YOLOv8 model on validation set...")

    # Load Model
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Run evaluation
    metrics = model.val(
        data=DATA_YAML,
        save=True,
        save_txt=False,
        save_conf=True,
        exist_ok=True
    )

    # Print MEAN metrics
    print("\nEvaluation Metrics (averaged):")
    print(f" Precision (P):     {np.mean(metrics.box.p):.4f}")
    print(f" Recall (R):        {np.mean(metrics.box.r):.4f}")
    print(f" F1 Score:          {np.mean(metrics.box.f1):.4f}")
    print(f" mAP@0.5:           {np.mean(metrics.box.map50):.4f}")
    print(f" mAP@0.5:0.95:      {np.mean(metrics.box.map):.4f}")
    print("=" * 100)

    # Results directory
    results_dir = metrics.save_dir
    print(f"Results saved to: {results_dir}")

    # Save FULL metrics (JSON)
    metrics_dict = {
        "mean": {
            "precision": float(np.mean(metrics.box.p)),
            "recall": float(np.mean(metrics.box.r)),
            "f1": float(np.mean(metrics.box.f1)),
            "map50": float(np.mean(metrics.box.map50)),
            "map": float(np.mean(metrics.box.map)),
        },
        "per_class": {
            "precision": metrics.box.p.tolist(),
            "recall": metrics.box.r.tolist(),
            "f1": metrics.box.f1.tolist(),
            "map50": metrics.box.map50.tolist(),
            "map": metrics.box.map.tolist(),
        }
    }

    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metrics JSON saved at: {metrics_path}")

    # Display Evaluation Plots
    plot_paths = [
        os.path.join(results_dir, "BoxF1_curve.png"),
        os.path.join(results_dir, "BoxPR_curve.png"),
        os.path.join(results_dir, "confusion_matrix.png")
    ]
    titles = ["F1 Score Curve", "Precision-Recall Curve", "Confusion Matrix"]

    plt.figure(figsize=(20, 8))
    for i, (path, title) in enumerate(zip(plot_paths, titles)):
        if os.path.exists(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(1, 3, i + 1)
            plt.imshow(img)
            plt.title(title)
            plt.axis("off")
        else:
            print(f"Plot not found: {path}")

    plt.tight_layout()
    plt.show()

    print("Evaluation complete!")

if __name__ == "__main__":
    evaluate()