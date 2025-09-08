"""
train.py - YOLOv8 Training Script

Trains a YOLOv8 model using settings from config.yaml.
"""

# Importing Libraries
import os
import torch
import yaml
import random
import numpy as np
from ultralytics import YOLO
import shutil


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device_and_batch(default_batch: int):
    """
    Auto-detect device (GPU or CPU) and pick batch size based on GPU memory.
    Falls back to CPU if no GPU is available.
    """
    if torch.cuda.is_available():
        device = 0  # GPU index
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_mem_gb = gpu_props.total_memory / (1024 ** 3)
        print(f"Using GPU: {gpu_props.name}, VRAM: {gpu_mem_gb:.1f} GB")

        # Auto batch size selection
        if gpu_mem_gb < 8:
            batch_size = 8
        elif gpu_mem_gb < 16:
            batch_size = 16
        else:
            batch_size = 32
    else:
        device = "cpu"
        batch_size = 4  # small batch for CPU
        print("No GPU found, running on CPU (much slower).")

    # Respect config’s default if it is smaller
    batch_size = min(batch_size, default_batch)

    print(f"Final batch size set to: {batch_size}")
    print("-" * 50)
    return device, batch_size


def train():
    """Main training loop: loads config, sets device/seed, trains YOLO model."""
    # Load config.yaml
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # Set random seed
    set_seed(cfg.get("seed", 42))

    # Device handling: use config if set to CPU, else auto-detect
    if cfg.get("device", None) == "cpu":
        device, batch_size = "cpu", min(4, cfg["batch_size"])
        print("CPU mode as per config.")
    else:
        device, batch_size = get_device_and_batch(cfg["batch_size"])

    # Load YOLO model
    print(f"Loading model from {cfg['model_path']} ...")
    model = YOLO(cfg["model_path"])

    # Start training
    print("Starting training...")
    results = model.train(
        data=cfg["data_yaml"],       # dataset YAML (train/val/test)
        epochs=cfg["epochs"],        # max epochs
        batch=batch_size,            # auto batch size
        imgsz=cfg["img_size"],       # image size
        patience=10,                 # early stop if no improvement
        workers=2,                   # reduce RAM usage in Colab/low RAM systems
        device=device,               # GPU or CPU
        exist_ok=True                # allow overwrite
    )

    # Training summary
    print("\nTraining complete!")
    save_dir = results.save_dir  # YOLO stores final run directory here
    print(f"Results saved to: {save_dir}")

    # Path to best model weights
    best_weights = os.path.join(save_dir, "weights", "best.pt")

    # Ensure root /models directory exists
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Copy best.pt → /models/best.pt
    if os.path.exists(best_weights):
        final_path = os.path.join(models_dir, "best.pt")
        shutil.copy(best_weights, final_path)
        print(f"Best model copied to: {final_path}")
    else:
        print("Could not find best.pt in results directory.")

if __name__ == "__main__":
    train()
