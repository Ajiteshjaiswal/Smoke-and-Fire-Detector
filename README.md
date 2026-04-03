# 🔥 Smoke and Fire Detection System (YOLO11)

A real-time deep learning application designed to detect smoke and fire in images, videos, and live webcam feeds. This project leverages the **YOLO11 (2026)** architecture for high-speed, high-accuracy inference, replacing traditional classification models with precise object detection and segmentation.

## 🚀 Features
- **Real-time Detection:** Optimized for video at 30+ FPS using YOLO11 Nano.
- **Automatic Inference:** Gradio UI triggers detection immediately upon file upload.
- **Segmentation Masks:** Provides pixel-level outlines of fire and smoke plumes.
- **Webcam Support:** Live streaming mode for real-time safety monitoring.
- **Hardware Optimized:** Configured to utilize NVIDIA CUDA (e.g., GeForce MX350).

---

## 📂 Project Structure
To ensure the training and application scripts function correctly, please maintain the following directory layout:

```text
E:/Smoke-and-Fire-Detector/
├── datasets/
│   └── data/
│       ├── train/
│       │   ├── images/          # Training images (.jpg, .png)
│       │   └── labels/          # YOLO format labels (.txt)
│       ├── val/
│       │   ├── images/          # Validation images
│       │   └── labels/          # Validation labels
│       └── data.yaml            # Dataset configuration file
├── smoke_fire_detection/        # Output folder for trained models
│   └── weights/
│       └── best.pt              # The final trained model
├── app.py                       # Gradio Web Interface
├── train.py                     # YOLO11 Training Script
└── requirements.txt             # Project Dependencies
```

## 🛠️ Installation & Setup

### 1. Create a Virtual Environment
It is recommended to use a virtual environment to avoid dependency conflicts.
```bash
# Create environment
python -m venv asenv

# Activate environment (Windows)
asenv\Scripts\activate

# Activate environment (Mac/Linux)
source asenv/bin/activate
```

### 2. Install Dependencies
```bash
pip install ultralytics gradio opencv-python torch torchvision
```

### 3. Configure Dataset (data.yaml)
Ensure your datasets/smoke_fire/data.yaml contains absolute paths to avoid loading errors:
```yaml
path: E:/Fire-and-Smoke-Classifier-main/datasets/smoke_fire
train: train/images
val: val/images
nc: 2
names: ['smoke', 'fire']
```

## 🏃 How to Run
The entire pipeline is consolidated into one file.

```bash
python main.py
```

### What happens when you run it?
- **Auto-Train:** The script checks if runs/segment/train/weights/best.pt exists. If not, it automatically starts training on your dataset.
- **Auto-Launch:** Once the model is ready, it launches the Gradio Web Interface.
- **URL:** Open http://127.0.0.1:7860 in your browser.

## 📊 Model Details
- **Model Architecture:** YOLO11n-seg (Nano Segmentation)
- **Input Size:** 640x640
- **Optimization:** vid_stride=2 used for video processing to maintain real-time performance.
- **Inference Speed:** ~20-30ms per frame on NVIDIA MX350.

---

## 🗂️ Dataset

**Small Fire Smoke OD (700 - 1,500 images):** If you just want to see it working today, there are several "starter" sets on Roboflow. These are tiny and will train in minutes.

**Download:** [Roboflow Small Fire Smoke](https://universe.roboflow.com/fire-smoke-od/small-fire-smoke-od-dataset)

---

## 📂 Project Structure
```text
.
├── datasets/
│   └── smoke_fire/
│       ├── train/
│       ├── val/
│       └── data.yaml     # Dataset configuration
├── main.py               # Main logic (Train + Gradio)
├── asenv/                # Virtual environment
└── README.md
```

## ⚠️ Important Notes for Windows Users
- **Pathing:** Always use forward slashes (/) in your data.yaml file to avoid Python Unicode escape errors.
- **GPU Memory:** If you encounter "Out of Memory" (OOM) errors on the MX350, go to main.py and reduce the batch_size to 8 or 4.
- **Cuda Support:** Ensure you have CUDA Toolkit installed if you wish to use the MX350 GPU for training.

## 📜 License
This project is for educational purposes. YOLO11 is provided by Ultralytics.
