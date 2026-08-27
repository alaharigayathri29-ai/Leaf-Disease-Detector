import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
import datetime
import uuid
import threading
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------------------------------
# 1. FastAPI REST API Setup
# ----------------------------------------------------
api_app = FastAPI(
    title="Leaf Disease Detection API",
    description="API for detecting plant diseases from leaf images using a CNN model.",
    version="1.0.0"
)

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared in-memory store for predictions
predictions_store = []

# Global Model & Class Names Lazy Loading for API & Streamlit
@st.cache_resource
def get_model_and_classes():
    m = tf.keras.models.load_model("leaf_disease_model.h5")
    with open("class_names.json", "r") as f:
        c = json.load(f)
    return m, c

model, class_names = get_model_and_classes()


@api_app.get("/")
def home():
    return {
        "message": "Leaf Disease Detection API is running",
        "endpoints": {
            "POST /predict": "Upload a leaf image (Input)",
            "GET /result": "Retrieve latest prediction result (Output)",
            "GET /results": "Retrieve all stored prediction history",
            "GET /classes": "Retrieve all supported plant disease classes",
            "GET /health": "API health status"
        }
    }


@api_app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded",
        "total_predictions": len(predictions_store)
    }


@api_app.get("/classes")
def get_classes():
    return {
        "total_classes": len(class_names),
        "classes": class_names
    }


# POST Endpoint: Submit Input Image & Get Output Prediction
@api_app.post("/predict")
async def predict_input(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]) * 100)
    predicted_class = class_names[predicted_index]

    record = {
        "id": str(uuid.uuid4()),
        "filename": file.filename,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "prediction": predicted_class,
        "confidence": round(confidence, 2)
    }

    predictions_store.append(record)

    return {
        "message": "Input received and processed successfully",
        "output": record
    }


# GET Endpoint: Fetch Latest Prediction Output
@api_app.get("/result")
def get_latest_output():
    if not predictions_store:
        return {
            "message": "No predictions have been made yet.",
            "output": None
        }
    return {
        "status": "success",
        "output": predictions_store[-1]
    }


# GET Endpoint: Fetch All Prediction Outputs History
@api_app.get("/results")
def get_all_outputs():
    return {
        "status": "success",
        "count": len(predictions_store),
        "outputs": predictions_store
    }


# Function to run FastAPI server in background thread if needed
def start_fastapi_server():
    uvicorn.run(api_app, host="0.0.0.0", port=8000, log_level="error")


@st.cache_resource
def launch_api_thread():
    t = threading.Thread(target=start_fastapi_server, daemon=True)
    t.start()
    return True

# Auto-start API server on port 8000 when Streamlit loads
launch_api_thread()


# ----------------------------------------------------
# 2. Streamlit User Interface
# ----------------------------------------------------
st.set_page_config(
    page_title="Leaf Disease Detector & API",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Leaf Disease Detector & API")
st.caption("Streamlit UI combined with FastAPI backend (GET / POST endpoints active at http://localhost:8000)")

tabs = st.tabs(["🖼️ Upload & Predict (UI)", "🔌 API Documentation & Test Output"])

with tabs[0]:
    st.write("Upload a leaf image to detect the disease.")

    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

        if st.button("Predict Disease"):
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]
            confidence = float(np.max(prediction)) * 100

            record = {
                "id": str(uuid.uuid4()),
                "filename": uploaded_file.name,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "prediction": predicted_class,
                "confidence": round(confidence, 2)
            }
            predictions_store.append(record)

            st.success(f"**Prediction:** {predicted_class}")
            st.info(f"**Confidence:** {confidence:.2f}%")

with tabs[1]:
    st.markdown("### API Endpoints Summary")
    st.markdown("""
    - **`POST /predict`**: Input endpoint (Upload leaf image to receive disease prediction)
    - **`GET /result`**: Output endpoint (Fetch latest processed prediction result)
    - **`GET /results`**: Output endpoint (Fetch all historical prediction records)
    - **`GET /classes`**: GET endpoint (List supported disease categories)
    - **`GET /health`**: Health check
    """)

    st.markdown("---")
    st.subheader("Current Stored Output Results (`GET /results`)")
    if predictions_store:
        st.json(predictions_store)
    else:
        st.write("No predictions stored yet. Upload an image above or send a `POST` request to `http://localhost:8000/predict`.")