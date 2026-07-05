import cv2
import numpy as np
import streamlit as st
import easyocr
from difflib import get_close_matches, SequenceMatcher
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Handwritten OCR", layout="wide")
st.title("✍️ Handwritten Text OCR")
st.caption("Upload a photo of handwritten text — the app corrects orientation, segments words, and recognizes text.")

# ============================================================
# LOAD MODEL (cached so it doesn't reload on every interaction)
# ============================================================
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

reader = load_reader()

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("Settings")
ground_truth = st.sidebar.text_input("Expected sentence (for orientation scoring)", "my name is raushan kumar")
expected_words = [w.strip().lower() for w in ground_truth.split() if w.strip()]

min_gap = st.sidebar.slider("Word gap threshold (min_gap)", 5, 80, 30, step=1,
                             help="Increase if words are merging together. Decrease if a single word is being split apart.")
min_word_width = st.sidebar.slider("Minimum word width (px)", 5, 60, 15, step=1)
apply_dict_correction = st.sidebar.checkbox("Apply dictionary correction", value=True)
show_debug = st.sidebar.checkbox("Show intermediate steps", value=True)

# ============================================================
# PIPELINE FUNCTIONS
# ============================================================
def correct_orientation(img, ground_truth):
    rotations = {
        '0': img,
        '90_CW': cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        '180': cv2.rotate(img, cv2.ROTATE_180),
        '90_CCW': cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    }
    best_angle, best_score, best_img = None, -1, img
    scores = {}
    for name, rimg in rotations.items():
        res = reader.readtext(rimg)
        text = " ".join([r[1] for r in res]).lower() if res else ""
        sim = SequenceMatcher(None, text, ground_truth.lower()).ratio()
        scores[name] = (text, sim)
        if sim > best_score:
            best_score, best_angle, best_img = sim, name, rimg
    return best_img, best_angle, best_score, scores


def remove_ruling_lines(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.subtract(binary, lines)
    cleaned = cv2.medianBlur(cleaned, 3)
    return cleaned


def segment_words_by_projection(binary_inv, min_gap, min_word_width):
    col_sum = np.sum(binary_inv > 0, axis=0)
    is_empty = col_sum == 0

    words = []
    start = None
    gap_len = 0

    for x in range(len(is_empty)):
        if not is_empty[x]:
            if start is None:
                start = x
            gap_len = 0
        else:
            gap_len += 1
            if start is not None and gap_len >= min_gap:
                end = x - gap_len
                if end - start >= min_word_width:
                    words.append((start, end))
                start = None

    if start is not None:
        end = len(is_empty) - 1
        if end - start >= min_word_width:
            words.append((start, end))

    boxes = []
    for (x0, x1) in words:
        col_slice = binary_inv[:, x0:x1]
        row_sum = np.sum(col_slice > 0, axis=1)
        rows = np.where(row_sum > 0)[0]
        if len(rows) == 0:
            continue
        y0, y1 = rows.min(), rows.max()
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes


def recognize_word_crops(binary_inv, boxes):
    results = []
    for (x, y, w, h) in boxes:
        pad = 10
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = x + w + pad, y + h + pad
        crop = binary_inv[y0:y1, x0:x1]
        crop_white_bg = cv2.bitwise_not(crop)

        text_results = reader.readtext(
            crop_white_bg, decoder='beamsearch', beamWidth=10,
            allowlist='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
            paragraph=False
        )
        if text_results:
            best = max(text_results, key=lambda r: r[2])
            results.append((best[1], best[2], (x, y, w, h)))
        else:
            results.append(("", 0.0, (x, y, w, h)))
    return results


def draw_boxes(img_gray_or_bin, results):
    vis = cv2.cvtColor(img_gray_or_bin, cv2.COLOR_GRAY2BGR)
    for (text, conf, (x, y, w, h)) in results:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, text, (x, max(y - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return vis


def correct_word(word, vocab, cutoff=0.5):
    if not vocab:
        return word
    matches = get_close_matches(word.lower(), vocab, n=1, cutoff=cutoff)
    return matches[0] if matches else word


# ============================================================
# MAIN APP
# ============================================================
uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Original Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with st.spinner("Correcting orientation..."):
        corrected_img, best_angle, best_score, all_scores = correct_orientation(img, ground_truth)

    if show_debug:
        with st.expander("Orientation scores"):
            for name, (text, sim) in all_scores.items():
                st.write(f"**{name}**: text='{text}' | similarity={sim:.3f}")
        st.success(f"Best orientation: {best_angle} (similarity={best_score:.3f})")

    gray = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
    binary_inv = remove_ruling_lines(gray)

    if show_debug:
        st.subheader("Preprocessed (ruling lines removed)")
        st.image(binary_inv, use_container_width=True, clamp=True)

    with st.spinner("Segmenting words..."):
        boxes = segment_words_by_projection(binary_inv, min_gap, min_word_width)

    st.info(f"Detected {len(boxes)} word region(s). Adjust 'Word gap threshold' in the sidebar if this looks wrong.")

    with st.spinner("Recognizing text..."):
        results = recognize_word_crops(binary_inv, boxes)

    vis = draw_boxes(binary_inv, results)
    st.subheader("Detected Words")
    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

    raw_words = [r[0] for r in results if r[0]]
    if apply_dict_correction:
        corrected_words = [correct_word(w, expected_words) for w in raw_words]
    else:
        corrected_words = raw_words

    st.subheader("Final Result")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Raw OCR words:**", raw_words)
    with col2:
        st.write("**Corrected words:**", corrected_words)

    st.success(f"**Final sentence:** {' '.join(corrected_words)}")

else:
    st.info("Upload an image to get started.")
