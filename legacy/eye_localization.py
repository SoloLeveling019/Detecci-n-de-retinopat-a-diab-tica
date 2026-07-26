import numpy as np
from typing import Tuple

def detect_eye_region(img_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Encuentra una región delimitadora aproximada del ojo.
    Para esta fase, asumimos que el ojo es prominente en la imagen,
    por lo que retornamos los límites completos. En futuras fases,
    podría usar Haar Cascades o mediapipe.
    Retorna: (x, y, ancho, alto)
    """
    h, w = img_bgr.shape[:2]
    return (0, 0, w, h)
