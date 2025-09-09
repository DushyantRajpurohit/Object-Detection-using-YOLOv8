# Automatic Number Plate Recognition (ANPR) using YOLOv8 & EasyOCR

This project implements an **Automatic Number Plate Recognition (ANPR)** system using **YOLOv8** for license plate detection and **EasyOCR** for text recognition. It includes training, evaluation, video processing, and a user-friendly **Streamlit dashboard**.

---

## Problem Statement
Automatic Number Plate Recognition (ANPR) is a critical technology for **traffic monitoring, toll collection, smart parking, law enforcement, and logistics tracking**. Traditional systems often struggle with challenges like:
- Low-light or poor weather conditions
- Motion blur from moving vehicles
- Non-standard or occluded license plates
- Varying plate sizes and fonts

This project addresses these challenges using **YOLOv8 for robust plate detection** and **EasyOCR for efficient text extraction**, providing a modular and deployable solution.

---

## Features
- **License Plate Detection** using YOLOv8
- **Text Extraction (OCR)** with EasyOCR
- **Video Inference** support
- **Evaluation Metrics & Visualization** (mAP, precision, recall, loss curves)
- **Streamlit Dashboard** for interactive demo
- Modular structure (`train.py`, `predict.py`, `evaluate.py`, etc.)

---

## Pipeline Diagram

```mermaid
graph TD
    A["Input<br/>Images / Videos"] --> B["Vehicle Detection<br/>(YOLOv8 - COCO)"]
    B --> C["License Plate Detection<br/>(YOLOv8 - Custom)"]
    C --> D["Plate Cropping<br/>& Preprocessing"]
    D --> E["OCR<br/>(EasyOCR)"]
    E --> F["Extracted<br/>Plate Number"]
    F --> G["Save Results<br/>(CSV & Crops)"]
    G --> H["Interpolation<br/>(Missing Frames)"]
    H --> I["Video<br/>Visualization"]
    I --> J["Streamlit<br/>Dashboard"]
```

---

## Project Structure
```
Object-Detection-using-YOLOv8/
├── app/
│   ├── Interface/
│   ├── images/
│   ├── images/
│   ├── background.png
│   └── streamlit_app.py
├── csv_detections/
│   ├──crops/
│   └── detection_results.csv
├── models/
│   ├── best.pt
│   └── yolo8n.pt
├── notebooks/
│   ├── YOLOv8_Object_Detection_Video_Inference.ipynb
│   ├── YOLOv8_Object_Detection_Training.ipynb
│   ├── Result_Insights.ipynb
│   ├── Data_Exploration.ipynb
│   ├── Model_Experiments.ipynb
│   └── Model_Evaluation.ipynb
├── results/
│   ├── test.csv
│   ├── test_interpolated.csv
│   ├── license_detection.gif
│   └── out-sample.m4v
├── runs/detect/
│   ├── predict
│   ├── train
│   └── val
├── sort/
├── src/             
│   ├── train.py        
│   ├── predict.py     
│   ├── evaluate.py     
│   ├── util.py        
│   ├── config.yaml
│   ├── main.py
│   ├── data.py
│   ├── interpolate.py
│   └── visualize.py
├── README.md     
└── requirements.txt       
```

---

## Dataset

The dataset used in this project is publicly available on **Roboflow Universe**.  
You can download it directly [from here](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4).  

The dataset contains images of vehicles and their corresponding annotated license plates, which were used for training and evaluation of the YOLOv8 + OCR pipeline.  

---

---

## Sample Video  

A sample video used for testing the detection pipeline can be downloaded [here](https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/).  

This video showcases highway traffic flow and was used to demonstrate **real-time license plate detection and recognition** with the trained YOLOv8 + OCR model.  

---

## Getting Started

### Clone the Repository
```bash
git clone https://github.com/DushyantRajpurohit/Object-Detection-using-YOLOv8.git
cd Object-Detection-using-YOLOv8
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download the Dataset
```bash
python src/data.py
```
- Enter your own API key.

### Train the Model
```bash
python src/train.py
```
- Trained weights will be saved in `runs/detect/train/weights/`

### Evaluate the Model
```bash
python src/evaluate.py
```

### Run Inference
```bash
python src/predict.py
```

### Detect & Recognize Plates
```bash
python src/main.py
```

### Interpolate Missing Frames
```bash
python src/interpolate.py
```

### Video Inference
```bash
python src/visualize.py
```

### Streamlit Dashboard
```bash
streamlit run app.py
```

---

## Results

Training results are logged in `runs/detect/train/` and include:
- **Loss curves**: box_loss, cls_loss, dfl_loss
- **Metrics**: precision, recall, mAP@0.5, mAP@0.5:0.95

Example training curves:

![results](runs/detect/train/results.png)

Prediction Results:

![results](runs/detect/predict/0002a5b67e5f0909_jpg.rf.07ca41e79eb878b14032f650f34d0967.jpg)

Video Inference Results:

![results](results/license_detection.gif)


---

## Insights
- The **box_loss and cls_loss decrease steadily**, indicating the model is learning effective spatial and classification features.
- **Precision and recall are both high (>95%)**, showing balanced detection without significant false positives/negatives.
- The **mAP@0.5 (~98%)** confirms strong detection accuracy, while **mAP@0.5:0.95 (~70%)** suggests opportunities for improving small-scale/occluded plate recognition.
- OCR accuracy depends heavily on plate clarity; preprocessing (denoising, contrast adjustment) improves performance.

---

## Business Insights
- **Traffic Management**: Automates vehicle monitoring, enabling toll collection, congestion tracking, and real-time analytics.
- **Security & Surveillance**: Enhances law enforcement by detecting and logging suspicious or blacklisted vehicles.
- **Parking Solutions**: Enables smart parking by automating entry/exit systems, reducing manual intervention.
- **Logistics & Fleet Tracking**: Helps companies monitor fleet movement, improve accountability, and optimize delivery routes.
- **Revenue Generation**: Supports governments and businesses in toll automation, parking fees, and traffic fines collection.

---

## Future Improvements
- Improve OCR accuracy with preprocessing (denoising, contrast adjustment)
- Add **DeepSORT tracking** for multi-frame plate tracking
- Experiment with advanced backbones (BiFPN, FasterNet)
- Hyperparameter tuning for higher mAP
- Support for edge-device deployment

---

## License
This project is released under the **MIT License**. Note: **YOLOv8** is licensed under **AGPL-3.0**, which may affect commercial usage.

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

---

## Author
**Dushyant Rajpurohit**  
📧 Contact: [dushyantrajpurohit5412@gmail.com]  
🔗 GitHub: [DushyantRajpurohit](https://github.com/DushyantRajpurohit)

