# 🚗 Automatic Number Plate Recognition (ANPR) Pipeline

This project implements a full **Automatic Number Plate Recognition (ANPR)** system using **YOLOv8** for detection, **SORT** for tracking, and **EasyOCR** for license plate recognition.  
It supports training, evaluation, video inference, result interpolation, and visualization.

---

## 📂 Project Structure
```
.
├── config.yaml              # Central config for all scripts
├── requirements.txt         # Dependencies
├── train.py                 # Train YOLOv8 model
├── evaluate.py              # Evaluate trained model
├── predict.py               # Run inference on test images
├── main.py                  # Detect & track cars and license plates in video
├── interpolate.py           # Fill missing detections across frames
├── visualize.py             # Overlay results on video
├── util.py                  # Helper functions (OCR, CSV, bbox utils)
└── sort/                    # SORT tracker (Kalman filter + Hungarian matching)
```

---

## ⚙️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/anpr-yolov8.git
   cd anpr-yolov8
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install PyTorch separately (with CUDA if available):  
   👉 [Official installation guide](https://pytorch.org/get-started/locally/)

---

## 📑 Configuration

All settings (paths, hyperparameters, thresholds) are defined in **`config.yaml`**:

```yaml
# Model & Training
model_path: "models/best.pt"
data_yaml: "data/data.yaml"
epochs: 50
batch_size: 16
img_size: 640
seed: 42
device: "cuda"   # "cpu" or "cuda"

# Inference / Prediction
test_images: "data/test_images"
conf_threshold: 0.25
max_predictions: 6

# Video Processing
input_video: "../sample.mp4"
output_video: "../out.mp4"

# CSV Files
raw_csv: "test.csv"
interpolated_csv: "test_interpolated.csv"
```

---

## 🚀 Usage

### 1️⃣ Train a model
```bash
python train.py
```

### 2️⃣ Evaluate model performance
```bash
python evaluate.py
```

### 3️⃣ Run inference on test images
```bash
python predict.py
```

### 4️⃣ Process video (detect, track, OCR, save CSV)
```bash
python main.py
```

### 5️⃣ Interpolate missing detections
```bash
python interpolate.py
```

### 6️⃣ Visualize results on video
```bash
python visualize.py
```

---

## 📊 Output Files
- **`test.csv`** → raw detections per frame (from `main.py`)  
- **`test_interpolated.csv`** → filled-in tracking results (from `interpolate.py`)  
- **`out.mp4`** → final annotated video with bounding boxes, plate crops & numbers (from `visualize.py`)  

---

## 🔮 Future Improvements
- Add **multi-video batch processing**
- Replace EasyOCR with **PaddleOCR** for better accuracy
- Store bounding boxes in JSON format instead of strings
- Add a **FastAPI web service** for real-time inference

---

## 👨‍💻 Authors
Developed as a modular pipeline for **YOLOv8-based ANPR** research and deployment.
