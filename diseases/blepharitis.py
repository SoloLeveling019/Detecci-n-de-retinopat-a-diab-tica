import cv2
import numpy as np

def analyze_blepharitis(img_bgr, upper_eyelid_band_mask, lower_eyelid_band_mask):
    """
    Analiza signos visibles compatibles con blefaritis en los bordes palpebrales superior e inferior.
    """
    if upper_eyelid_band_mask is None or lower_eyelid_band_mask is None:
        h, w = img_bgr.shape[:2]
        empty = np.zeros((h, w), dtype=np.uint8)
        return {
            "status": "warning",
            "condition": "blefaritis",
            "level": "No evaluado",
            "redness_percent": 0.0,
            "scale_crust_percent": 0.0,
            "message": "No se pudo delimitar correctamente la banda palpebral.",
            "blepharitis_mask": empty
        }
        
    eyelid_band_mask = cv2.bitwise_or(upper_eyelid_band_mask, lower_eyelid_band_mask)
    valid_pixels = np.count_nonzero(eyelid_band_mask)
    
    if valid_pixels < 10:
        return {
            "status": "warning",
            "condition": "blefaritis",
            "level": "N/A",
            "redness_percent": 0.0,
            "scale_crust_percent": 0.0,
            "message": "Banda palpebral muy pequeña o no delimitada.",
            "blepharitis_mask": np.zeros_like(eyelid_band_mask)
        }

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Enrojecimiento palpebral
    # Rojos (hue cerca de 0 o 180), S > 40, V > 50
    red_mask = (
        ((h <= 15) | (h >= 165)) &
        (s > 40) & (s < 200) &
        (v > 50) & (v < 240)
    ).astype(np.uint8) * 255
    red_band = cv2.bitwise_and(red_mask, eyelid_band_mask)
    
    # Costras o escamas (puntos amarillentos o blanquecinos)
    # Colores claros/amarillos: Hue [15, 45], o zonas blancas muy brillantes (V > 200, S baja)
    crust_mask = (
        (((h >= 15) & (h <= 45)) & (s > 40) & (v > 120)) |
        ((v > 200) & (s < 50))
    ).astype(np.uint8) * 255
    crust_band = cv2.bitwise_and(crust_mask, eyelid_band_mask)
    
    # Combinar características
    blepharitis_mask = cv2.bitwise_or(red_band, crust_band)
    
    red_pixels = np.count_nonzero(red_band)
    crust_pixels = np.count_nonzero(crust_band)
    
    redness_percent = (red_pixels / valid_pixels) * 100
    crust_percent = (crust_pixels / valid_pixels) * 100
    
    # Reglas heurísticas
    score = redness_percent + (crust_percent * 2.0) # Las costras tienen mayor peso
    
    level = "No detectado"
    if crust_percent < 2 and redness_percent > 0:
        if redness_percent > 10:
            level = "Leve"
        else:
            level = "No detectado"
    else:
        if score > 25:
            level = "Alto"
        elif score > 12:
            level = "Moderado"
        elif score > 5:
            level = "Leve"
        
    return {
        "status": "ok",
        "condition": "blefaritis",
        "level": level,
        "redness_percent": redness_percent,
        "scale_crust_percent": crust_percent,
        "message": "Posibles signos compatibles con blefaritis. Resultado orientativo.",
        "blepharitis_mask": blepharitis_mask
    }
