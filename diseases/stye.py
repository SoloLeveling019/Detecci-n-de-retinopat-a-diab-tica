import cv2
import numpy as np

def analyze_stye(img_bgr, upper_eyelid_band_mask, lower_eyelid_band_mask):
    """
    Busca posibles bultos inflamados compatibles con orzuelo.
    """
    if upper_eyelid_band_mask is None or lower_eyelid_band_mask is None:
        h, w = img_bgr.shape[:2]
        empty = np.zeros((h, w), dtype=np.uint8)
        return {
            "status": "warning",
            "condition": "orzuelo",
            "level": "No evaluado",
            "candidate_count": 0,
            "message": "No se pudo delimitar correctamente la banda palpebral.",
            "stye_mask": empty
        }

    try:
        from skimage.measure import label, regionprops
    except ImportError:
        return {
            "status": "warning",
            "condition": "orzuelo",
            "level": "N/A",
            "candidate_count": 0,
            "message": "Requiere instalar scikit-image para analizar orzuelo.",
            "stye_mask": np.zeros_like(upper_eyelid_band_mask)
        }

    eyelid_band_mask = cv2.bitwise_or(upper_eyelid_band_mask, lower_eyelid_band_mask)
    h, w = img_bgr.shape[:2]
    
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    
    # Rango de rojo/rosado inflamado (Hue [0-20] o [160-180], saturación moderada/alta)
    red_mask = (
        ((h_channel <= 20) | (h_channel >= 160)) &
        (s_channel > 50) &
        (v_channel > 50)
    ).astype(np.uint8) * 255
    
    # También incluimos posibles centros amarillentos (Hue [20-40])
    yellowish_mask = (
        (h_channel > 20) & (h_channel <= 40) &
        (s_channel > 40) &
        (v_channel > 120)
    ).astype(np.uint8) * 255
    
    combined_mask = cv2.bitwise_or(red_mask, yellowish_mask)
    candidate_band = cv2.bitwise_and(combined_mask, eyelid_band_mask)
    
    # Operación morfológica para agrupar píxeles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate_band = cv2.morphologyEx(candidate_band, cv2.MORPH_CLOSE, kernel)
    
    labels = label(candidate_band > 0)
    regions = regionprops(labels)
    
    stye_mask = np.zeros_like(candidate_band)
    candidates_found = 0
    max_score = 0
    
    for region in regions:
        # 1. Área suficiente
        if region.area < int(0.001 * h * w):
            continue
            
        # 2. Circularidad y solidez (orzuelo es un bulto redondo/elíptico)
        # solidity = area / convex_area
        if region.solidity < 0.6:
            continue
            
        # 3. Excentricidad (no debe ser una línea extremadamente larga y fina)
        if region.eccentricity > 0.95:
            continue
            
        # Pintar el candidato válido
        for coords in region.coords:
            stye_mask[coords[0], coords[1]] = 255
        
        candidates_found += 1
        
        # Evaluar gravedad (basado en área, solidez)
        score = (region.area / (h*w)) * 100 * region.solidity
        if score > max_score:
            max_score = score
            
    if candidates_found == 0:
        level = "No detectado"
    elif max_score > 1.5:
        level = "Sospecha alta"
    elif max_score > 0.5:
        level = "Sospecha moderada"
    elif max_score > 0.1:
        level = "Sospecha leve"
    else:
        level = "No detectado"
        
    return {
        "status": "ok",
        "condition": "orzuelo",
        "level": level,
        "candidate_count": candidates_found,
        "message": "Posible lesión compatible con orzuelo. Resultado orientativo." if candidates_found > 0 else "No detectado.",
        "stye_mask": stye_mask
    }
