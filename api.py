from fastapi import FastAPI, UploadFile, File
from PIL import Image
import tensorflow as tf
import numpy as np
import json
import io

app = FastAPI(
    title="Leaf Disease Detection API",
    description="API for detecting plant diseases from leaf images using a CNN model.",
    version="1.0.0"
)

# Load model
model = tf.keras.models.load_model("leaf_disease_model.h5")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)


@app.get("/")
def home():
    return {
        "message": "Leaf Disease Detection API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Resize image
    image = image.resize((224, 224))

    # Convert to numpy
    image_array = np.array(image)

    # Normalize
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array)

    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]) * 100)

    predicted_class = class_names[predicted_index]

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 2)
    }