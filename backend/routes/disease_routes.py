from fastapi import APIRouter, UploadFile, File
import numpy as np
from PIL import Image, ImageOps
import io
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input

router = APIRouter()

IMAGE_SIZE = 224
NUM_CLASSES = 3
labels = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# --- ✅ Build same model architecture as training ---
base_model = MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs, outputs)

# --- ✅ Load fine-tuned weights instead of old .keras model ---
model.load_weights("models/disease_model.weights.h5")
print("✅ Disease model weights loaded successfully!")

@router.get("/ping")
async def ping():
    return {"message": "Disease detection model is active!"}

@router.post("/detect_disease")
async def detect_disease(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ✅ Ensure consistent orientation and size
    img = ImageOps.exif_transpose(img)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

    # ✅ Use MobileNetV2 preprocessing (IMPORTANT!)
    arr = np.array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)
    idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))

    return {
        "disease": labels[idx],
        "confidence": round(confidence * 100, 2)
    }
