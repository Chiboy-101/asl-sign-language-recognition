import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

# Load trained model
model = tf.keras.models.load_model("models/asl_mobilenet_model.keras")

IMG_SIZE = 160
BATCH_SIZE = 32

# Load class labels (must be same order as training)
datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = datagen.flow_from_directory(
    "data/asl_alphabet_train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

class_labels = list(train_data.class_indices.keys())

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

print("Webcam started!. Press ctrl + c to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect (optional)
    frame = cv2.flip(frame, 1)

    # Define Region of Interest (ROI)
    x1, y1, x2, y2 = 80, 80, 320, 320
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi_normalized = roi_resized / 255.0
    roi_input = np.expand_dims(roi_normalized, axis=0)

    # Predict
    preds = model.predict(roi_input, verbose=0)
    class_index = np.argmax(preds)
    confidence = preds[0][class_index] * 100
    predicted_label = class_labels[class_index]

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display text
    text = f"{predicted_label} ({confidence:.2f}%)"
    cv2.putText(
        frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )

    cv2.imshow("ASL Sign Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
