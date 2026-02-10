### 🧠 ASL Sign Language Translator

Real-time American Sign Language (ASL) alphabet recognition using deep learning and webcam input

---

### 📌 Project Description

This project implements a real-time ASL alphabet recognition system using TensorFlow / Keras with a MobileNetV2 backbone and OpenCV for webcam capture. The model is trained on an ASL alphabet dataset and achieves 96% test accuracy, allowing users to make hand signs in front of a webcam and see predictions live.

---

### 🚀 Features

✔️ Real-time ASL letter prediction from webcam

✔️ Transfer learning with pretrained MobileNetV2

✔️ Data augmentation for better generalization

✔️ Test accuracy: 96.10%

✔️ Clean and simple Python scripts for training and inference

---

### 🧠 How It Works

✔️ Data Preparation:
Images are loaded from directory structure and augmented (rescaling, rotation, shifts, zoom, flip).

✔️ Model Architecture:
MobileNetV2 (pretrained on ImageNet) is used as backbone with a custom classification head that predicts ASL letters.

✔️ Training:
The model is trained with early stopping and learning rate adjustments for improved validation performance.

✔️ Inference:
A webcam feed is captured, a region of interest (ROI) is cropped and preprocessed, and the model predicts the sign in real time.

---

### 📊 Results

| Metric                  | Score      |
| ----------------------- | ---------- |
| **Training Accuracy**   | **~97.50** |
| **Validation Accuracy** | **~94.72** |
| **Test Accuracy**       | **96.10%** |

---

### 📂 Dataset

- **ASL Alphabet Dataset** from Kaggle
- Dataset is **not included** in this repository due to size limitations, download and place it in the data/ folder.

🔗 Dataset link:  
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

---

### 🚀 How to Run

```bash
#Clone the repository
git clone https://github.com/<your-username>/asl-sign-language-translator.git
cd asl-sign-language-translator

# Install dependencies
pip install -r requirements.txt

# Train the model(optional)
python src/train.py
# ---- Save trained model in /models folder or load the saved model in the /models folder.

# Run Real-Time Inference
python src/webcam_predict.py

```
