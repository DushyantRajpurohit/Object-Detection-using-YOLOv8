"""
main.py
Automatic Number Plate Recognition (ANPR) pipeline.
"""

# Importing Libraries
from ultralytics import YOLO
import cv2
import numpy as np
from util import get_car, read_license_plate, write_csv
import sys
import os
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sort.sort import Sort

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# Dictionary to store all results per frame
results = {}

# Initialize SORT tracker for multi-object tracking
mot_tracker = Sort()

# Load detection models
coco_model = YOLO(cfg['coco_model'])  # Pretrained on COCO dataset (for vehicles)
license_plate_detector = YOLO(cfg['model_path'])  # Custom plate detector

# Load video
video_path = os.path.abspath(cfg["input_video"])
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")
cap = cv2.VideoCapture(video_path)

# Vehicle class IDs in COCO dataset (2=car, 3=motorbike, 5=bus, 7=truck)
vehicles = [2, 3, 5, 7]

# Frame counter
frame_nmr = -1
ret = True

# Process video frames
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break

    results[frame_nmr] = {}

    # Detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # Track vehicles across frames
    track_ids = mot_tracker.update(np.asarray(detections_))

    # Detect license plates
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Match plate to a vehicle
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1:
            # Crop license plate region
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            # Preprocess crop for OCR
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(
                license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
            )

            # OCR: read license plate number
            license_plate_text, license_plate_text_score = read_license_plate(
                license_plate_crop_thresh
            )

            # Save results if OCR is successful
            if license_plate_text is not None:
                results[frame_nmr][car_id] = {
                    "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                    "license_plate": {
                        "bbox": [x1, y1, x2, y2],
                        "bbox_score": score,
                        "text": license_plate_text,
                        "text_score": license_plate_text_score,
                    },
                }

# Save all results directly to 'result' folder in project root ---
SRC_DIR = os.path.dirname(__file__)  # directory containing main.py
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))  # parent of src
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# Destination CSV path in the result folder
csv_filename = os.path.basename(cfg["raw_csv"])  # e.g., test.csv
output_csv = os.path.join(RESULT_DIR, csv_filename)

# Save all results to CSV
write_csv(results, output_csv)
print("Processing complete. Results saved to test.csv")