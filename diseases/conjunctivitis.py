import numpy as np
import cv2
from typing import Dict, Any

def analyze_conjunctivitis(sclera_roi_mask: np.ndarray, red_mask: np.ndarray, iris_info: tuple) -> Dict[str, Any]:
    """
    Analiza el porcentaje de enrojecimiento y su distribución (localizada vs difusa).
    Divide la esclerótica en 4 cuadrantes respecto al centro de la pupila.
    """
    sclera_area = int(np.count_nonzero(sclera_roi_mask))
    if sclera_area <= 0:
        return {
            "status": "warning",
            "condition": "conjuntivitis",
            "level": "No evaluado",
            "redness_percent": 0.0,
            "diffuse_score": 0.0,
            "affected_sectors": 0,
            "affected_side": "N/A",
            "distribution": "N/A",
            "confidence": 0,
            "message": "No se detectó esclerótica para analizar."
        }

    red_area = int(np.count_nonzero(red_mask))
    redness_percent = min((red_area / float(sclera_area)) * 100.0, 100.0)

    # Coordenadas del centro de la pupila
    try:
        pupil_x, pupil_y, _ = iris_info
        pupil_x, pupil_y = int(pupil_x), int(pupil_y)
    except:
        h, w = sclera_roi_mask.shape
        pupil_x, pupil_y = w // 2, h // 2

    h, w = sclera_roi_mask.shape

    # Crear máscaras por sector
    left_mask = np.zeros_like(sclera_roi_mask)
    left_mask[:, :pupil_x] = 255
    right_mask = np.zeros_like(sclera_roi_mask)
    right_mask[:, pupil_x:] = 255

    top_mask = np.zeros_like(sclera_roi_mask)
    top_mask[:pupil_y, :] = 255
    bottom_mask = np.zeros_like(sclera_roi_mask)
    bottom_mask[pupil_y:, :] = 255

    sectors = {
        "Izquierdo Superior": cv2.bitwise_and(sclera_roi_mask, cv2.bitwise_and(left_mask, top_mask)),
        "Izquierdo Inferior": cv2.bitwise_and(sclera_roi_mask, cv2.bitwise_and(left_mask, bottom_mask)),
        "Derecho Superior": cv2.bitwise_and(sclera_roi_mask, cv2.bitwise_and(right_mask, top_mask)),
        "Derecho Inferior": cv2.bitwise_and(sclera_roi_mask, cv2.bitwise_and(right_mask, bottom_mask))
    }

    affected_sectors = 0
    valid_sectors = 0
    left_red = 0
    right_red = 0

    for name, mask in sectors.items():
        sector_area = np.count_nonzero(mask)
        if sector_area > (sclera_area * 0.05):  # Si el sector es representativo
            valid_sectors += 1
            sector_red_mask = cv2.bitwise_and(red_mask, mask)
            sector_red = np.count_nonzero(sector_red_mask)
            sector_red_percent = (sector_red / sector_area) * 100
            
            if "Izquierdo" in name:
                left_red += sector_red
            else:
                right_red += sector_red

            if sector_red_percent > 3.0: # Umbral para considerar que hay rojo en el sector
                affected_sectors += 1

    diffuse_score = (affected_sectors / max(valid_sectors, 1)) * 100

    # Determinar lado
    affected_side = "Ninguno"
    if redness_percent > 1.0:
        if left_red > (right_red * 3):
            affected_side = "Izquierdo"
        elif right_red > (left_red * 3):
            affected_side = "Derecho"
        else:
            affected_side = "Ambos"

    # Lógica de clasificación
    if redness_percent < 0.60:
        base_level = "No detectado"
        distribution = "N/A"
    else:
        if diffuse_score >= 50.0 and redness_percent > 2.0:
            distribution = "Difusa"
            if redness_percent > 8.0:
                base_level = "Alto"
            elif redness_percent > 4.0:
                base_level = "Moderado"
            else:
                base_level = "Leve"
        else:
            distribution = "Localizada"
            # Si está localizado, penalizamos el nivel
            if redness_percent > 15.0:
                base_level = "Moderado"
            else:
                base_level = "Leve"

    if distribution == "Localizada":
        level_str = "Enrojecimiento localizado"
    elif distribution == "Difusa":
        level_str = f"Posible conjuntivitis / irritación difusa - Nivel {base_level}"
    else:
        level_str = "No detectado"

    # Confidence: alta si hay buena sclera y los colores son claros. Simple aproximación.
    confidence = min(80 + int(sclera_area / (h*w) * 100), 98)

    return {
        "status": "ok",
        "condition": "conjuntivitis",
        "level": level_str,
        "redness_percent": redness_percent,
        "diffuse_score": diffuse_score,
        "affected_sectors": affected_sectors,
        "valid_sectors": valid_sectors,
        "affected_side": affected_side,
        "distribution": distribution,
        "confidence": confidence,
        "message": "Análisis de enrojecimiento completado."
    }
