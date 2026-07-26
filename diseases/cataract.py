import cv2
import numpy as np

def analyze_cataract_visible(img_bgr, iris_info, iris_pupil_mask=None, pterygium_mask=None):
    """
    Analiza signos visibles compatibles con catarata en la zona pupilar.
    Busca opacidad blanca/grisácea o amarillenta en el centro del ojo.
    Anula el análisis si hay superposición significativa con carnosidad.
    """
    try:
        cx, cy, r = iris_info
        cx, cy, r = int(cx), int(cy), int(r)
    except:
        return {
            "status": "error",
            "condition": "catarata_visible",
            "level": "Error",
            "opacity_percent": 0.0,
            "message": "Datos de iris inválidos",
            "pupil_zone_mask": np.zeros(img_bgr.shape[:2], dtype=np.uint8),
            "opacity_mask": np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        }

    h, w = img_bgr.shape[:2]
    
    # 1. Definir zona pupilar
    pupil_zone_mask = np.zeros((h, w), dtype=np.uint8)
    pupil_radius = max(5, int(r * 0.45))
    cv2.circle(pupil_zone_mask, (cx, cy), pupil_radius, 255, -1)
    
    if iris_pupil_mask is not None:
        pupil_zone_mask = cv2.bitwise_and(pupil_zone_mask, iris_pupil_mask)
        
    pupil_area = np.count_nonzero(pupil_zone_mask)
    if pupil_area == 0:
        return {
            "status": "warning",
            "condition": "catarata_visible",
            "level": "No evaluado",
            "opacity_percent": 0.0,
            "message": "No se pudo delimitar la pupila",
            "pupil_zone_mask": pupil_zone_mask,
            "opacity_mask": np.zeros((h, w), dtype=np.uint8)
        }

    # 2. Control de solapamiento con pterigión
    if pterygium_mask is not None:
        overlap = cv2.bitwise_and(pterygium_mask, pupil_zone_mask)
        overlap_area = np.count_nonzero(overlap)
        if overlap_area / pupil_area > 0.25:
            return {
                "status": "warning",
                "condition": "catarata_visible",
                "level": "No evaluado / resultado no confiable",
                "opacity_percent": 0.0,
                "message": "Motivo: la zona pupilar está obstruida por posible carnosidad u opacidad corneal.",
                "pupil_zone_mask": pupil_zone_mask,
                "opacity_mask": np.zeros((h, w), dtype=np.uint8)
            }

    # 3. Filtros de opacidad
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    
    # Opacidad blanca/gris:
    # V > 95, S < 70, V < 240
    white_gray_opacity = ((v_channel > 95) & (s_channel < 70) & (v_channel < 240))
    
    # Opacidad amarillenta:
    # H entre 15 y 45, S entre 25 y 140, V > 90, V < 240
    yellow_opacity = ((h_channel >= 15) & (h_channel <= 45) & 
                      (s_channel >= 25) & (s_channel <= 140) & 
                      (v_channel > 90) & (v_channel < 240))
    
    opacity_candidate = (white_gray_opacity | yellow_opacity)
    
    # Exclusión de reflejos especulares
    specular_mask = ((v_channel > 245) & (s_channel < 40))
    
    opacity_candidate = (opacity_candidate & (~specular_mask))
    opacity_mask_raw = opacity_candidate.astype(np.uint8) * 255
    
    # Restringir a la zona pupilar
    opacity_mask = cv2.bitwise_and(opacity_mask_raw, pupil_zone_mask)
    
    # Excluir adicionalmente cualquier zona que caiga sobre pterigión
    if pterygium_mask is not None:
        opacity_mask = cv2.bitwise_and(opacity_mask, cv2.bitwise_not(pterygium_mask))
    
    # Limpieza
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opacity_mask = cv2.morphologyEx(opacity_mask, cv2.MORPH_OPEN, kernel)
    
    opacity_area = np.count_nonzero(opacity_mask)
    opacity_percent = (opacity_area / pupil_area) * 100.0
    
    # 4. Clasificación
    level = "Normal"
    if opacity_percent > 40:
        level = "Alto"
    elif opacity_percent > 15:
        level = "Moderado"
    elif opacity_percent > 5:
        level = "Leve"
        
    return {
        "status": "ok",
        "condition": "catarata_visible",
        "level": level,
        "opacity_percent": opacity_percent,
        "message": f"Se detecta opacidad pupilar al {opacity_percent:.1f}%",
        "pupil_zone_mask": pupil_zone_mask,
        "opacity_mask": opacity_mask
    }
