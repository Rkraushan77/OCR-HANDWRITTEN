import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pickle

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model("handwritten_model.keras")

# -----------------------------
# Load Label Encoder
# -----------------------------
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# -----------------------------
# Page Title
# -----------------------------
st.set_page_config(page_title="Handwritten OCR")

st.title("📝 Handwritten Text Recognition")

st.write("Upload a handwritten image")

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize
    image = cv2.resize(image, (128, 64))

    # Normalize
    image = image / 255.0

    image = image.reshape(1, 64, 128, 1)

    prediction = model.predict(image)

    index = np.argmax(prediction)

    word = encoder.inverse_transform([index])[0]

    confidence = float(np.max(prediction) * 100)

    st.success(f"Predicted Word : {word}")

    st.info(f"Confidence : {confidence:.2f}%")
