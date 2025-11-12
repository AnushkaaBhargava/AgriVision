# backend/test_model.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing import image
import numpy as np

# ---- Configuration ----
IMAGE_SIZE = 224
NUM_CLASSES = 3
labels = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# ---- Rebuild the model exactly like your training setup ----
resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    tf.keras.layers.Rescaling(1.0 / 255)
])

base_model = MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = Sequential([
    resize_and_rescale,
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

# ✅ Build model before loading weights
model.build((None, IMAGE_SIZE, IMAGE_SIZE, 3))

# ✅ Now load weights
model.load_weights("models/disease_model.weights.h5")
print("✅ Model weights loaded successfully!")

# ---- Test with an image ----
img_path = "early_blight.JPG"  # make sure file exists
img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

preds = model.predict(x)
pred_label = labels[np.argmax(preds)]
confidence = np.max(preds)

print(f"\nPrediction: {pred_label}")
print(f"Confidence: {confidence:.2f}")
