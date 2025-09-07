# Automatic Number Plate Recognition (ANPR) with OCR

This project implements an **Automatic Number Plate Recognition (ANPR)** system using **YOLOv8** for object detection and **EasyOCR** for text recognition.  
It supports both **images** and **videos** for detecting license plates, extracting them, and reading plate numbers.

The project comes with:
- Vehicle & License Plate Detection (YOLOv8)
- OCR for reading license numbers (EasyOCR)
- Training, Evaluation & Visualization pipelines
- A simple **Streamlit app** to upload images/videos and get results instantly

---

## Project Structure

```
project/
│── streamlit_app.py      # Streamlit web app (run this for demo)
│── requirements.txt      # Dependencies
│── models/               # Trained models (YOLO weights will be stored here)
│── results/              # Output CSVs, annotated videos, metrics
│
└── src/
    ├── config.yaml       # Configuration file
    ├── data.py           # Dataset download/setup from Roboflow
    ├── train.py          # Train YOLOv8 model
    ├── evaluate.py       # Evaluate trained model
    ├── predict.py        # Run inference on test images
    ├── main.py           # Full ANPR pipeline (vehicle + plate + OCR)
    ├── interpolate.py    # Interpolates missing bounding boxes across frames
    ├── visualize.py      # Creates annotated video output
    ├── util.py           # Helper functions (OCR, CSV writing, matching)
```

---

## Installation

1. **Clone this repo**  
```bash
git clone https://github.com/DushyantRajpurohit/Object-Detection-using-YOLOv8.git
cd Object-Detection-using-YOLOv8
```

2. **Install dependencies**  
```bash
pip install -r requirements.txt
```

3. **(Optional) Install GPU PyTorch**  
Check [PyTorch installation guide](https://pytorch.org/get-started/locally/) for CUDA support.

---

## 🚀 Usage

### 1️⃣ Run the Streamlit App
```bash
streamlit run streamlit_app.py
```
- Upload an **image** → get bounding boxes + OCR results.  
- Upload a **video** → frame-by-frame recognition with live preview.  

---

### 2️⃣ Training a Model
```bash
python src/train.py
```
- Uses `config.yaml` for dataset and model paths.  
- Trains YOLOv8 on license plate dataset from Roboflow.  
- Best weights saved in `models/best.pt`.

---

### 3️⃣ Evaluating Model
```bash
python src/evaluate.py
```
- Runs YOLO evaluation (Precision, Recall, mAP).  
- Saves metrics + plots in `results/`.  

---

### 4️⃣ Running Full Pipeline (Video ANPR)
```bash
python src/main.py
```
- Detects vehicles & plates → runs OCR → saves results to `results/test.csv`.  

---

### 5️⃣ Visualizing Results
```bash
python src/visualize.py
```
- Reads CSV from pipeline.  
- Creates **annotated video** with bounding boxes + OCR text overlay.  

---

## 🛠️ Configuration

All settings (model paths, dataset, thresholds, input video, etc.) are stored in:

```yaml
# src/config.yaml
model_path: "models/best.pt"
coco_model: "yolov8n.pt"
data_yaml: "data/data.yaml"
input_video: "data/sample.mp4"
raw_csv: "test.csv"
interpolated_csv: "test_interpolated.csv"
output_video: "output.mp4"
conf_threshold: 0.25
device: "cpu"
batch_size: 16
epochs: 50
img_size: 640
max_predictions: 6
seed: 42
```

Modify this file to adjust dataset paths, training parameters, or input/output filenames.

---

## 📊 Results

- Detection & OCR results are saved as CSV in `results/`
- Example CSV:
```
frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score
0,1,[100 200 400 500],[120 220 180 260],0.92,ABC1234,0.87
```

- Annotated video saved in `results/output.mp4`

---

## 🧰 Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) – Object Detection  
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) – Optical Character Recognition  
- [OpenCV](https://opencv.org/) – Image & Video Processing  
- [Streamlit](https://streamlit.io/) – Web Interface  
- [Roboflow](https://roboflow.com/) – Dataset management  

---

## 🙌 Acknowledgements
- YOLOv8 by [Ultralytics](https://github.com/ultralytics/ultralytics)  
- EasyOCR by [JaidedAI](https://github.com/JaidedAI/EasyOCR)  
- Dataset via [Roboflow](https://roboflow.com/)  

---

## 📜 License
This project is licensed under the MIT License.
