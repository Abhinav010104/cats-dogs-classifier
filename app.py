import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import os

# Auto-load model
model_path = None
if os.path.exists("models/best_model.keras"):
    model_path = "models/best_model.keras"
elif os.path.exists("models/best_model.h5"):
    model_path = "models/best_model.h5"
else:
    st.error("❌ No model file found! Train first.")
    st.stop()

model = keras.models.load_model(model_path)

# Get input shape from the model dynamically
model_input_shape = model.input_shape[1:3]  # (height, width)
st.write(f"Model expects images of size: {model_input_shape}")

# Streamlit UI
st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image and let the model predict!")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize(model_input_shape)                 # Resize to model input
    img_array = keras.utils.img_to_array(img)           # Convert to array
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize & batch

    # Predict
    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        st.success(f"🐶 It's a Dog! (Confidence: {prediction:.2f})")
    else:
        st.success(f"🐱 It's a Cat! (Confidence: {1 - prediction:.2f})")
    st.write(f"Raw model output (sigmoid): {prediction:.4f}")