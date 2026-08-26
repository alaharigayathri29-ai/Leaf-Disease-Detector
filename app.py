import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(
    page_title="Leaf Disease Detector",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Leaf Disease Detector")
st.write("Upload a leaf image to detect the disease.")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("leaf_disease_model.h5")

model = load_model()

# Load class names
with open("class_names.json") as f:
    class_names = json.load(f)

# Upload image
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
    image,
    caption="Uploaded Leaf Image"
    )

    if st.button("Predict Disease"):

        # Resize image
        img = image.resize((224, 224))

        # Convert to array and normalize
        img_array = np.array(img) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)

        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]

        confidence = float(np.max(prediction)) * 100

        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")