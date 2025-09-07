"""
Streamlit Application for Object and License Plate Detection using YOLOv8 + EasyOCR.

Features:
- Detect cars and license plates in uploaded images.
- Extract license plate text using EasyOCR.
- Save cropped license plate images into `csv_detections/crops/`.
- Save detection results (bounding boxes, text, scores) into `csv_detections/detection_results.csv`.
- Display detection results (annotated image, cropped plates, extracted text).
- Show progress in both terminal and Streamlit UI.
- Allow CSV download from the app.
"""

import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import easyocr
import pandas as pd
import uuid
import os
import sys
import warnings

# Suppress PyTorch pin_memory warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")


# Setup Directories and Paths
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(SRC_DIR)
from util import set_background, write_csv  # Custom utilities for background and CSV writing

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "csv_detections")
CROPS_DIR = os.path.join(RESULTS_DIR, "crops")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)

# Set app background
set_background(os.path.join(os.path.dirname(__file__), "background.png"))

# Model Paths
LICENSE_MODEL_DETECTION_DIR = os.path.join("models", "best.pt")  # YOLO license plate model
COCO_MODEL_DIR = os.path.join("models", "yolov8n.pt")  # YOLO COCO model for cars

# Initialize EasyOCR
print("Initializing EasyOCR reader...")
reader = easyocr.Reader(["en"], gpu=False)
print("EasyOCR initialized.")

# YOLO COCO class ID for cars
vehicles = [2]

# Load YOLO models
print("Loading YOLO models...")
coco_model = YOLO(COCO_MODEL_DIR)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)
print("YOLO models loaded successfully.")


# Helper Functions
def read_license_plate(license_plate_crop, img):
    """
    Run OCR on a cropped license plate image to extract text.

    Args:
        license_plate_crop (np.ndarray): Grayscale image of the license plate.
        img (np.ndarray): Original full image (for context, optional).

    Returns:
        tuple: (recognized_text (str) or None, average_confidence (float))
    """
    scores = 0
    detections = reader.readtext(license_plate_crop)

    if detections == []:
        return None, None

    rectangle_size = license_plate_crop.shape[0] * license_plate_crop.shape[1]
    plate = []

    # Filter out small OCR detections
    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_size > 0.17:
            _, text, score = result
            text = text.upper()
            scores += score
            plate.append(text)

    if len(plate) != 0:
        return " ".join(plate), scores / len(plate)
    else:
        return None, 0


def model_prediction(img, st_progress=None):
    """
    Perform YOLOv8 detection for cars and license plates, run OCR on detected plates,
    and optionally update a Streamlit progress bar.

    Args:
        img (np.ndarray): Uploaded image in RGB format.
        st_progress (st.progress, optional): Streamlit progress bar object.

    Returns:
        list: [annotated_image (np.ndarray), license_texts (list), license_plate_crop_paths (list)]
    """

    license_numbers = 0
    results = {}
    licenses_texts = []
    license_plate_crops_total = []

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Car Detection
    if st_progress: st_progress.progress(5)
    print("Starting car detection...")
    object_detections = coco_model.predict(img_bgr, imgsz=320, device="cpu")[0]
    print(f"Detected {len(object_detections.boxes)} objects.")
    if st_progress: st_progress.progress(20)

    # Draw bounding boxes for cars
    if len(object_detections.boxes.cls.tolist()) != 0:
        for detection in object_detections.boxes.data.tolist():
            xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection
            if int(class_id) in vehicles:
                cv2.rectangle(img_bgr, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)
    else:
        xcar1, ycar1, xcar2, ycar2, car_score = 0, 0, 0, 0, 0

    # License Plate Detection
    print("Starting license plate detection...")
    if st_progress: st_progress_text.text("Detecting license plates...")
    license_detections = license_plate_detector.predict(img_bgr, imgsz=320, device="cpu")[0]
    num_plates = len(license_detections.boxes)
    print(f"Detected {num_plates} license plates.")
    if st_progress: st_progress.progress(40)

    # Avoid division by zero
    per_plate_progress = 50 / num_plates if num_plates > 0 else 50

    # Process license plates
    for idx, license_plate in enumerate(license_detections.boxes.data.tolist()):
        x1, y1, x2, y2, score, class_id = license_plate
        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

        license_plate_crop = img_bgr[int(y1):int(y2), int(x1):int(x2), :]
        img_name = f"{uuid.uuid4()}.jpg"
        crop_path = os.path.join(CROPS_DIR, img_name)
        cv2.imwrite(crop_path, license_plate_crop)

        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        if st_progress: st_progress_text.text(f"Processing license plate {idx+1}/{num_plates}...")
        print(f"Processing license plate {idx+1}...")

        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img_bgr)
        print(f"Recognized text: {license_plate_text} (score: {license_plate_text_score})")

        licenses_texts.append(license_plate_text)

        if license_plate_text is not None:
            license_plate_crops_total.append(crop_path)
            results[license_numbers] = {
                license_numbers: {
                    "car": {"bbox": [xcar1, ycar1, xcar2, ycar2], "car_score": car_score},
                    "license_plate": {
                        "bbox": [x1, y1, x2, y2],
                        "text": license_plate_text,
                        "bbox_score": score,
                        "text_score": license_plate_text_score,
                    },
                }
            }
            license_numbers += 1

        # Update progress per license plate
        if st_progress: st_progress.progress(40 + int(per_plate_progress*(idx+1)))

    # Save Detection Results
    write_csv(results, os.path.join(RESULTS_DIR, "detection_results.csv"))
    print(f"Detection results saved to {os.path.join(RESULTS_DIR, 'detection_results.csv')}")
    if st_progress:
        st_progress.progress(100)
        st_progress_text.text("Detection completed!")

    img_wth_box = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return [img_wth_box, licenses_texts, license_plate_crops_total]



# Streamlit UI
header = st.container()
body = st.container()

with header:
    # Title
    _, col1, _ = st.columns([0.2, 1, 0.1])
    col1.title("Object Detection using YOLOv8")

    # Show sample images
    _, col0, _ = st.columns([0.15, 1, 0.1])
    app_dir = os.path.dirname(__file__)
    col0.image(os.path.join(app_dir, "images", "test_background.jpg"), width=500)

    _, col4, _ = st.columns([0.1, 1, 0.2])
    col4.subheader("License Plate Detection")

    _, col, _ = st.columns([0.3, 1, 0.1])
    col.image(os.path.join(app_dir, "images", "plate_test.jpg"))

    st.write(
        "This app detects cars and license plates using YOLOv8, "
        "extracts license text with EasyOCR, saves cropped license plates, "
        "and generates a CSV with detection results."
    )

with body:
    _, col1, _ = st.columns([0.1, 1, 0.2])
    col1.subheader("Try it out: Upload a Car Image!")

    # File uploader
    img = st.file_uploader("Upload a Car Image: ", type=["png", "jpg", "jpeg"])
    _, col2, _ = st.columns([0.3, 1, 0.2])
    _, col5, _ = st.columns([0.8, 1, 0.2])

    if img is not None:
        image = np.array(Image.open(img))
        col2.image(image, width=400, caption="Uploaded Image")

        if col5.button("Apply Detection"):
            # Initialize Streamlit progress bar and status text
            st_progress = st.progress(0)
            global st_progress_text
            st_progress_text = st.empty()

            # Run detection
            results = model_prediction(image, st_progress)
            prediction, texts, license_plate_crops = results
            texts = [t for t in texts if t is not None]

            # Show annotated detection image
            _, col3, _ = st.columns([0.4, 1, 0.2])
            col3.header("Detection Results:")
            _, col4, _ = st.columns([0.1, 1, 0.1])
            col4.image(prediction, caption="Detected Objects")

            # Show license plate crops and recognized text
            if len(license_plate_crops) > 0:
                _, col9, _ = st.columns([0.4, 1, 0.2])
                col9.header("License Crops:")

                col10, col11 = st.columns([1, 1])
                for i, crop_path in enumerate(license_plate_crops):
                    col10.image(crop_path, width=300, caption=f"Crop {i+1}")
                    col11.success(f"License Number {i+1}: {texts[i]}")

                # Show CSV and download button
                csv_path = os.path.join(RESULTS_DIR, "detection_results.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    st.dataframe(df)
                    st.download_button(
                        label="Download Results CSV",
                        data=open(csv_path, "rb").read(),
                        file_name="detection_results.csv",
                        mime="text/csv",
                    )