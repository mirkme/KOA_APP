import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys
import os

# Load the trained model
model = tf.keras.models.load_model('C:\\Users\\Asus\\Desktop\\trainknee2\\knee_xray_classifier.h5')  # or use full path if needed

# Check command-line argument
if len(sys.argv) != 2:
    print("Usage: python test.py path_to_image")
    sys.exit()

img_path = sys.argv[1]

# Validate path
if not os.path.exists(img_path):
    print("Error: Image path not found.")
    sys.exit()

# Load and preprocess image as per training
img = image.load_img(img_path, target_size=(150, 150))  # your training size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # shape becomes (1, 150, 150, 3)
img_array = img_array / 255.0  # normalize like training

# Make prediction
pred = model.predict(img_array)[0][0]

# Output result
if pred >= 0.5:
    print(f"Prediction: NOT a Knee X-ray (Confidence: {pred:.2f})")
else:
    print(f"Prediction: Knee X-ray (Confidence: {1 - pred:.2f})")
