import cv2
import numpy as np
from .iris_pupil_detection import detect_pupil_center, estimate_iris_region

def check_image_quality_sclera(img_bgr: np.ndarray, content_mask: np.ndarray = None) -> tuple[bool, bool, str]:
    """
    Verifica calidad para el módulo de Esclerótica/Conjuntivitis.
    Retorna: (calidad_ok, requiere_iris_manual, mensaje)
    """
    h, w = img_bgr.shape[:2]

    # Validación 1: Desenfoque
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < 20.0:
        return False, False, "La imagen está demasiado borrosa."

    # Validación 2: Oscuridad extrema
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, _, v_channel = cv2.split(hsv)
    avg_brightness = np.mean(v_channel)
    if avg_brightness < 20:
        return False, False, "La imagen está demasiado oscura."

    # Validación 3: Presencia y tamaño del iris/pupila
    pupil_x, pupil_y, pupil_r, found_pupil, _ = detect_pupil_center(img_bgr, content_mask)
    
    if not found_pupil:
        # Falló el detector automático, sugerir marcado manual en lugar de cancelar
        return False, True, "No se detectó el iris/pupila automáticamente."
        
    iris_r = estimate_iris_region(pupil_x, pupil_y, pupil_r, (h, w))
    
    # Validar tamaño
    min_iris_ratio = 0.04
    if iris_r / w < min_iris_ratio:
        return False, False, "El ojo ocupa muy poca área de la imagen. Acerque la cámara o recorte manualmente."

    return True, False, ""

def check_image_quality_cataract(img_bgr: np.ndarray) -> tuple[bool, str]:
    """
    Verifica calidad para Cataratas. NO exige pupila oscura.
    Retorna: (calidad_ok, mensaje)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < 15.0:
        return False, "La imagen está demasiado borrosa para análisis de catarata."

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, _, v_channel = cv2.split(hsv)
    if np.mean(v_channel) < 15:
        return False, "La imagen está demasiado oscura."

    return True, ""
