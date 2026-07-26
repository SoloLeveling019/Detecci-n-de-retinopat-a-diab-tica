import cv2
import numpy as np
from pathlib import Path

try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

from .preprocess import ensure_bgr

def load_image_any_format(image_path: str) -> np.ndarray:
    """Carga JPG, PNG, TIFF, WEBP, capturas y rutas con tildes/espacios."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    # Intentar cargar con OpenCV primero, usando fromfile para soportar rutas con caracteres especiales (tildes)
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    if img is not None:
        return ensure_bgr(img)

    # Fallback a Pillow si OpenCV no pudo cargarla (ej. HEIC en algunos sistemas)
    if PIL_AVAILABLE:
        pil_img = Image.open(path)
        pil_img = ImageOps.exif_transpose(pil_img) # Corregir rotación EXIF

        if pil_img.mode in ("RGBA", "LA"):
            fondo = Image.new("RGBA", pil_img.size, (0, 0, 0, 255))
            pil_img = Image.alpha_composite(fondo, pil_img.convert("RGBA")).convert("RGB")
        else:
            pil_img = pil_img.convert("RGB")

        rgb = np.array(pil_img)
        # Convertir RGB de Pillow a BGR de OpenCV
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    raise ValueError("No se pudo cargar la imagen. Instala Pillow: pip install pillow")
