# __init__.py
# Make core functions importable from the package, if needed.
from .main import (
    load_model,
    detect_image,
    process_results,
    draw_boxes,
    save_image,
    pil_to_cv2,
    cv2_to_pil,
    DEFAULT_MODEL,
)
