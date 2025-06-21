import os
import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
MODEL_PATH = "model/sign_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
CLASSES = sorted(os.listdir("dataset/asl_alphabet_train"))

# Webcam setup
cap = cv2.VideoCapture(0)
print("📷 Webcam started. Press 'q' to quit.")

IMG_SIZE = 64
x, y, w, h = 150, 150, 224, 224  # Tighter, more consistent region of interest (ROI)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw and extract the ROI
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi = frame[y:y + h, x:x + w]
    
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = CLASSES[predicted_index]
    confidence = prediction[predicted_index] * 100

    # Display prediction and confidence
    label = f"{predicted_class} ({confidence:.2f}%)"
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()