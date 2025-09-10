"""
app.py
-------
Streamlit UI for real-time (webcam) and image-based detection.
"""
import os
import time
import tempfile
from typing import Optional, List
import numpy as np
import cv2
import streamlit as st
from PIL import Image

from main import (
    load_model,
    detect_image,
    process_results,
    draw_boxes,
    save_image,
    pil_to_cv2,
    cv2_to_pil,
    DEFAULT_MODEL,
)

# Ensure output directories exist
OUTPUT_DIR = "output_data"
TEMP_DIR = "temp_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

st.set_page_config(page_title="YOLO Detection App", layout="wide")

# Sidebar controls
st.sidebar.title("Settings")

st.sidebar.subheader("Model")
use_default = st.sidebar.checkbox("Use default model (yolov8n.pt)", value=True)
custom_model_path = None
if not use_default:
    custom_model_path = st.sidebar.text_input("Custom model path (.pt)", value="")

conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.01)
iou = st.sidebar.slider("IoU threshold (NMS)", 0.1, 0.95, 0.45, 0.01)
max_det = st.sidebar.slider("Max detections", 10, 1000, 300, 10)
agnostic = st.sidebar.checkbox("Agnostic NMS", value=False)
device = st.sidebar.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Run Mode")
mode = st.sidebar.radio("Choose Mode", options=["Image Upload", "Webcam (Realtime)"], index=0)

# Load model once and cache
@st.cache_resource(show_spinner=True)
def _load_model_cached(model_path: str):
    return load_model(model_path)

model_path = DEFAULT_MODEL if use_default or not custom_model_path else custom_model_path.strip()
model = _load_model_cached(model_path)

st.title("🔍 YOLO Object Detection")
st.caption("Supports image upload and simple Webcam (realtime-ish) preview. Processed images are saved under `output_data/`.")

if mode == "Image Upload":
    st.subheader("Image-based Detection")
    uploaded_files = st.file_uploader(
        "Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    col1, col2 = st.columns([1,1])
    with col1:
        run_btn = st.button("Run Detection on Uploaded Images", use_container_width=True, type="primary")
    with col2:
        clear_btn = st.button("Clear Temp Images", use_container_width=True)

    if clear_btn:
        for f in os.listdir(TEMP_DIR):
            try:
                os.remove(os.path.join(TEMP_DIR, f))
            except Exception:
                pass
        st.success("Cleared temporary images.")

    if uploaded_files and run_btn:
        for up in uploaded_files:
            # Save to temp
            tmp_path = os.path.join(TEMP_DIR, f"upload_{int(time.time()*1000)}_{up.name}")
            with open(tmp_path, "wb") as f:
                f.write(up.getbuffer())

            pil_img = Image.open(tmp_path).convert("RGB")
            bgr = pil_to_cv2(pil_img)

            results = detect_image(
                model,
                bgr,
                conf=conf,
                iou=iou,
                classes=None,
                max_det=max_det,
                agnostic_nms=agnostic,
                device=None if device == "auto" else device,
            )
            dets = process_results(results)
            drawn = draw_boxes(bgr, dets)

            save_path = save_image(drawn, OUTPUT_DIR, prefix="image")
            st.image(cv2_to_pil(drawn), caption=f"Detections saved: {save_path}", use_column_width=True)

            with st.expander("Detections JSON"):
                st.json(dets)

else:
    st.subheader("Webcam Detection")
    st.caption("Click **Start** to begin. Click **Stop** to end. Frames are sampled to keep CPU usage manageable.")
    start = st.button("Start", type="primary")
    stop = st.button("Stop")

    # Session state flags
    if "running" not in st.session_state:
        st.session_state.running = False

    if start:
        st.session_state.running = True
    if stop:
        st.session_state.running = False

    frame_placeholder = st.empty()
    info_placeholder = st.empty()

    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam.")
            st.session_state.running = False
        else:
            try:
                frame_count = 0
                while st.session_state.running:
                    ok, frame_bgr = cap.read()
                    if not ok:
                        st.warning("Failed to read frame from webcam.")
                        break

                    # Optionally downscale for speed
                    h, w = frame_bgr.shape[:2]
                    scale = 640.0 / max(h, w)
                    if scale < 1.0:
                        frame_bgr = cv2.resize(frame_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

                    # Run detection every N frames to reduce load
                    do_infer = (frame_count % 2 == 0)
                    if do_infer:
                        results = detect_image(
                            model,
                            frame_bgr,
                            conf=conf,
                            iou=iou,
                            classes=None,
                            max_det=max_det,
                            agnostic_nms=agnostic,
                            device=None if device == "auto" else device,
                        )
                        dets = process_results(results)
                        frame_drawn = draw_boxes(frame_bgr, dets)
                    else:
                        frame_drawn = frame_bgr

                    frame_placeholder.image(cv2_to_pil(frame_drawn), use_column_width=True)
                    info_placeholder.write(f"Resolution: {frame_drawn.shape[1]}x{frame_drawn.shape[0]} | Running: {st.session_state.running}")
                    frame_count += 1

                    # Let Streamlit breathe
                    time.sleep(0.02)
            finally:
                cap.release()
                st.info("Webcam released.")

st.markdown("---")
st.caption("Tip: For custom models, set the full path to your .pt in the sidebar and uncheck 'Use default model'.")
