import cv2
import numpy as np
from typing import Tuple

def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convierte imágenes de 16 bits/float a uint8."""
    if img is None:
        raise ValueError("Imagen vacía.")

    if img.dtype == np.uint8:
        return img

    img_float = img.astype(np.float32)
    min_val = float(np.nanmin(img_float))
    max_val = float(np.nanmax(img_float))

    if max_val <= min_val:
        return np.zeros(img.shape, dtype=np.uint8)

    img_norm = (img_float - min_val) / (max_val - min_val)
    return np.clip(img_norm * 255.0, 0, 255).astype(np.uint8)

def ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Asegura imagen BGR de 3 canales."""
    img = normalize_to_uint8(img)

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 4:
        # Mezclar alfa con fondo negro
        bgr = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        fondo = np.zeros_like(bgr, dtype=np.float32)
        compuesto = bgr * alpha + fondo * (1.0 - alpha)
        return compuesto.astype(np.uint8)

    if img.ndim == 3 and img.shape[2] >= 3:
        return img[:, :, :3].copy()

    raise ValueError(f"Formato no compatible. Shape recibido: {img.shape}")

def resize_for_processing(img: np.ndarray, max_side: int = 1200) -> Tuple[np.ndarray, float]:
    """Redimensiona imágenes grandes para un análisis más rápido y consistente."""
    h, w = img.shape[:2]
    mayor = max(h, w)
    if mayor <= max_side:
        return img.copy(), 1.0

    scale = max_side / float(mayor)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

def resize_to_standard(img: np.ndarray, target_size: Tuple[int, int] = (800, 400)) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
    """
    Estandariza un recorte manual al tamaño objetivo sin distorsionarlo (rellena con negro).
    Retorna: canvas, content_mask, scale, (x_offset, y_offset)
    """
    target_w, target_h = target_size
    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    content_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    content_mask[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = 255

    return canvas, content_mask, scale, (x_offset, y_offset)
