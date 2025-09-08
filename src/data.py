"""
data.py

This script downloads the License Plate Recognition dataset from Roboflow,
renames the dataset folder to 'data', and generates a YOLOv8-compatible
data.yaml file inside the 'data' folder.

Workflow:
1. Connect to Roboflow using your API key.
2. Access the specified workspace, project, and dataset version.
3. Download and extract the dataset (Roboflow SDK handles extraction automatically).
4. Rename the downloaded folder to 'data', removing any existing folder.
5. Update the 'data.yaml' file to reflect the correct paths for YOLOv8.

After running this script, the folder structure will be:

data/
├─ train/
├─ valid/
├─ test/
└─ data.yaml

This setup is ready for YOLOv8 training.
"""

from roboflow import Roboflow
import os
import shutil
import yaml

def download_dataset():
    """
    Downloads the License Plate Recognition dataset from Roboflow, renames the
    folder to 'data', and updates the YOLOv8-compatible data.yaml file.

    Steps:
    1. Connect to Roboflow using API key.
    2. Access the specified workspace, project, and dataset version.
    3. Download and extract the dataset (SDK handles extraction automatically).
    4. Rename the downloaded folder to 'data', removing any existing folder.
    5. Update the 'data.yaml' file with correct paths for YOLOv8.
    """

    # Initialize Roboflow with your API key
    rf = Roboflow(api_key="8QR8NiJscJEAfxBnLiIF")

    # Access the project workspace and specific project
    project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
    version = project.version(4)  # specify dataset version

    # Download dataset using SDK
    dataset = version.download("yolov8")
    original_folder = dataset.location  # folder created by SDK

    # Prepare destination folder named 'data'
    ROOT_DIR = os.getcwd()
    DATA_DIR = os.path.join(ROOT_DIR, "data")

    # Remove existing 'data' folder if it exists
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)

    # Rename downloaded folder to 'data'
    os.rename(original_folder, DATA_DIR)

    # Prepare YOLOv8-compatible data.yaml paths
    data_yaml_path = os.path.join(DATA_DIR, "data.yaml")
    data_yaml = {
        "train": os.path.join(DATA_DIR, "train/images"),
        "val": os.path.join(DATA_DIR, "valid/images"),
        "test": os.path.join(DATA_DIR, "test/images"),
        "nc": 1,  # number of classes
        "names": ["License_Plate"]  # class names
    }

    # Write data.yaml file
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    # Confirmation messages
    print(f"Dataset prepared in folder: {DATA_DIR}")
    print(f"data.yaml updated at: {data_yaml_path}")


if __name__ == "__main__":
    download_dataset()