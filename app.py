import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch

from preprocess import correct_orientation, deskew, denoise_and_binarize
from dataset import NUM_CLASSES
from model import CRNN
from ctc_decoder import greedy_decode
from baselines import run_tesseract, run_easyocr

st.set_page_config(page_title="Handwritten OCR", layout="wide")
st.title("Handwritten Text Recognition")
st.caption("Upload an image of handwriting to see preprocessing steps and predictions.")

CHECKPOINT_PATH = "checkpoints/best.pth"


@st.cache_resource
def load_crnn_model():
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    return model, device


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    tmp_path = "temp_upload.png"
    cv2.imwrite(tmp_path, original)

    st.subheader("Preprocessing steps")
    cols = st.columns(4)

    cols[0].image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original")

    oriented = correct_orientation(original)
    cols[1].image(cv2.cvtColor(oriented, cv2.COLOR_BGR2RGB), caption="Orientation corrected")

    gray = cv2.cvtColor(oriented, cv2.COLOR_BGR2GRAY)
    deskewed = deskew(gray)
    cols[2].image(deskewed, caption="Deskewed", clamp=True)

    clean = denoise_and_binarize(deskewed)
    cols[3].image(clean, caption="Denoised / binarized", clamp=True)

    st.subheader("Predictions")

    with st.spinner("Running Tesseract..."):
        tesseract_result = run_tesseract(tmp_path)
    with st.spinner("Running EasyOCR..."):
        easyocr_result = run_easyocr(tmp_path)

    result_cols = st.columns(3)
    result_cols[0].metric("Tesseract", "")
    result_cols[0].write(tesseract_result or "*(no text detected)*")

    result_cols[1].metric("EasyOCR", "")
    result_cols[1].write(easyocr_result or "*(no text detected)*")

    loaded = load_crnn_model()
    if loaded:
        model, device = loaded
        from infer import predict
        crnn_result = predict(model, tmp_path, device)
        result_cols[2].metric("Your CRNN", "")
        result_cols[2].write(crnn_result or "*(no text detected)*")
    else:
        result_cols[2].metric("Your CRNN", "")
        result_cols[2].warning("No checkpoint found at checkpoints/best.pth — train the model first.")

    os.remove(tmp_path)
else:
    st.info("Upload an image to get started.")
