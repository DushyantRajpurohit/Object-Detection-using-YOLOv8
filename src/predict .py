"""
predict.py
Run YOLOv8 inference on a folder of images and display random predictions.
"""

# Importing Libraries
import os
import cv2
import random
import matplotlib.pyplot as plt
from ultralytics import YOLO
import yaml


def show_random_predictions(save_dir, n=6):
    """
    Display random sample predictions from YOLO output.

    Args:
        save_dir (str): Path to YOLO save directory (e.g., runs/detect/predict).
        n (int): Number of random images to display (default: 6).
    """
    # Collect predicted images
    predicted_images = [
        os.path.join(save_dir, f)
        for f in os.listdir(save_dir)
        if f.lower().endswith(('.jpg', '.png'))
    ]

    if not predicted_images:
        print(f"No images found in {save_dir}")
        return

    # Pick a random subset
    num_samples = min(n, len(predicted_images))
    random_images = random.sample(predicted_images, num_samples)

    # Plot images
    plt.figure(figsize=(15, 10))
    for i, img_path in enumerate(random_images):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load {img_path}, skipping...")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(f"Prediction {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def run_inference(model_path, source, conf=0.25, device="cpu", n=6):
    """
    Run YOLOv8 inference and show predictions.

    Args:
        model_path (str): Path to YOLOv8 model .pt file.
        source (str): Path to image folder or single image.
        conf (float): Confidence threshold (default: 0.25).
        device (str/int): Device to run on (default: "cpu").
        n (int): Number of random predictions to display.
    """
    # Validate paths
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source path not found: {source}")

    # Load model
    model = YOLO(model_path)

    # Run prediction
    results = model.predict(
        source=source,
        conf=conf,
        save=True,
        device=device
    )

    # Get save directory
    save_dir = results[0].save_dir
    print("Predictions saved in:", save_dir)

    # Show random predictions
    show_random_predictions(save_dir, n=n)


if __name__ == "__main__":
    # Load defaults from config.yaml
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    run_inference(
        model_path=cfg["model_path"],
        source=cfg["test_images"],
        conf=cfg["conf_threshold"],
        device=cfg["device"],
        n=cfg["max_predictions"]
    )