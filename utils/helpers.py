import cv2
import numpy as np
from pathlib import Path

def save_img(path: Path, image: np.ndarray):
    """Guarda una imagen OpenCV en la ruta especificada."""
    ext = path.suffix.lower() if path.suffix else ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise IOError(f"No se pudo guardar la imagen: {path}")
    encoded.tofile(str(path))
