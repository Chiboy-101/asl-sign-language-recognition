import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

# Load the trained model
model = tf.keras.models.load_model("models/asl_mobilenet_model.keras")

IMG_SIZE = 160
BATCH_SIZE = 32

# Create a data generator
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

train_data = datagen.flow_from_directory(
    "data/asl_alphabet_train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
)
class_labels = list(train_data.class_indices.keys())


# Function to predict ASL sign from an image
def predict_asl_image(image_path, model, class_labels, IMG_SIZE=64):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found at", image_path)
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred_probs = model.predict(img)
    class_index = np.argmax(pred_probs)
    confidence = pred_probs[0][class_index] * 100

    predicted_letter = class_labels[class_index]
    print(f"Predicted sign: {predicted_letter} ({confidence:.2f}% confidence)")

    return predicted_letter
