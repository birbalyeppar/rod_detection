# YOLO + Streamlit Detection App

A minimal end-to-end project that loads a YOLO model (Ultralytics), performs detections on images or webcam frames, and saves annotated outputs to `output_data/`.

## Project Structure

```text
yolo_streamlit_project/
├── app.py               # Streamlit UI
├── main.py              # Core detection utilities
├── __init__.py          # Optional package exports
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── output_data/         # Saved annotated images
└── temp_images/         # Temporarily uploaded images
```

## Quick Start

1. **Create & activate a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

   The first run will download `yolov8n.pt`. To use a custom model, uncheck the "Use default model" checkbox in the sidebar and specify the path to your `.pt` file.

## Notes

- Outputs are saved in `output_data/` with timestamped filenames.
- Webcam mode is a simple loop for demo purposes; adjust sampling or resize for performance.
- If CUDA is available, choose `cuda:0` in the sidebar to speed up inference.
