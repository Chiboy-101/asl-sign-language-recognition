# ASL Sign Language Recognition using Deep Learning

This project implements an American Sign Language (ASL) alphabet recognition system using a Convolutional Neural Network with MobileNetV2.

---

## 📌 Features
- Image classification of ASL letters (A–Z)
- Transfer learning with MobileNetV2
- Data augmentation to improve generalization
- ~93% validation accuracy
- Single image prediction with confidence score

---

## 🧠 Model Architecture
- MobileNetV2 (pretrained on ImageNet)
- Global Average Pooling
- Dense layer (256 units)
- Softmax output layer

---

## 📂 Dataset
- **ASL Alphabet Dataset** from Kaggle  
- Dataset is **not included** in this repository due to size limitations, download and place it in the data/ folder.

🔗 Dataset link:  
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py
# ---- Save trained model in /models folder.

# Predict a single image
python src/predict_image.py --image path/to/image.jpg
