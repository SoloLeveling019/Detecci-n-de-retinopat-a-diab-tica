import cv2
import numpy as np

def analyze_pterygium(img_bgr, eye_opening_mask, iris_info):
    """
    Analiza posibles regiones compatibles con pterigión/carnosidad.
    Utiliza 3 zonas: esclerótica, limbo y córnea para evaluar si el tejido 
    invade desde el lateral hacia el centro del ojo.
    """
    try:
        from skimage.measure import label, regionprops
    except ImportError:
        return {
            "status": "warning",
            "condition": "pterigion",
            "level": "N/A",
            "candidate_area": 0,
            "pterygium_percent": 0.0,
            "invasion_linear_percent": 0.0,
            "invasion_area_percent": 0.0,
            "confidence": 0,
            "affected_side": "N/A",
            "message": "Requiere instalar scikit-image para analizar pterigión.",
            "pterygium_mask": np.zeros_like(eye_opening_mask),
            "limbus_zone": np.zeros_like(eye_opening_mask),
            "invasion_zone": np.zeros_like(eye_opening_mask),
            "seed_mask": np.zeros_like(eye_opening_mask),
            "growth_mask": np.zeros_like(eye_opening_mask)
        }

    h, w = img_bgr.shape[:2]
    try:
        cx, cy, iris_r = iris_info
        cx, cy = int(cx), int(cy)
        iris_r = max(1, int(iris_r))
    except:
        return {
            "status": "error",
            "condition": "pterigion",
            "level": "No evaluado",
            "candidate_area": 0,
            "pterygium_percent": 0.0,
            "invasion_linear_percent": 0.0,
            "invasion_area_percent": 0.0,
            "confidence": 0,
            "affected_side": "N/A",
            "message": "Datos de iris inválidos. No se pudo evaluar pterigión.",
            "pterygium_mask": np.zeros_like(eye_opening_mask),
            "limbus_zone": np.zeros_like(eye_opening_mask),
            "invasion_zone": np.zeros_like(eye_opening_mask),
            "seed_mask": np.zeros_like(eye_opening_mask),
            "growth_mask": np.zeros_like(eye_opening_mask)
        }

    # --- 1. Crear zonas anatómicas ---
    y_indices, x_indices = np.indices((h, w))
    dist_to_center = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
    
    cornea_zone = (dist_to_center <= 1.15 * iris_r).astype(np.uint8) * 255
    cornea_zone = cv2.bitwise_and(cornea_zone, eye_opening_mask)
    cornea_area = np.count_nonzero(cornea_zone) + 1
    
    limbus_zone = ((dist_to_center >= 0.95 * iris_r) & (dist_to_center <= 1.25 * iris_r)).astype(np.uint8) * 255
    limbus_zone = cv2.bitwise_and(limbus_zone, eye_opening_mask)
    
    combined_cornea_limbus = cv2.bitwise_or(cornea_zone, limbus_zone)
    sclera_zone = cv2.bitwise_and(eye_opening_mask, cv2.bitwise_not(combined_cornea_limbus))

    # --- 2. Crear máscaras (Semilla y Crecimiento) ---
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    
    pink_red_hue = ((h_channel <= 25) | (h_channel >= 155))
    
    # seed_mask: estricta en sclera_zone (rosado/rojizo más fuerte)
    seed_condition = (pink_red_hue & (s_channel > 30) & (v_channel > 50))
    seed_mask_raw = seed_condition.astype(np.uint8) * 255
    seed_mask = cv2.bitwise_and(seed_mask_raw, sclera_zone)
    
    # growth_mask: más amplia, rosado/rojo/blanquecino en toda la abertura
    growth_condition = (
        (pink_red_hue & (s_channel > 15) & (v_channel > 60)) | 
        ((s_channel < 40) & (v_channel > 120))  # blanquecino/opaco
    )
    growth_mask_raw = growth_condition.astype(np.uint8) * 255
    growth_mask = cv2.bitwise_and(growth_mask_raw, eye_opening_mask)
    
    # Limpieza morfológica para agrupar tejido en la máscara de crecimiento
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    growth_mask = cv2.morphologyEx(growth_mask, cv2.MORPH_CLOSE, kernel)
    growth_mask = cv2.morphologyEx(growth_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    
    # --- 3. Análisis de componentes conectados ---
    labels_mask = label(growth_mask > 0)
    regions = regionprops(labels_mask)
    
    best_candidate_mask = np.zeros_like(eye_opening_mask)
    max_score = 0
    best_area = 0
    best_region = None
    
    min_area = int(0.003 * h * w) 
    eye_opening_area = np.count_nonzero(eye_opening_mask) + 1
    
    for region in regions:
        if region.area < min_area:
            continue
            
        region_mask = (labels_mask == region.label).astype(np.uint8) * 255
        
        # Validar topología: el candidato de crecimiento debe intersecar con la semilla y con el limbo
        touches_seed = np.any(cv2.bitwise_and(region_mask, seed_mask))
        touches_limbus = np.any(cv2.bitwise_and(region_mask, limbus_zone))
        
        if not (touches_seed and touches_limbus):
            continue 
            
        y0, x0 = region.centroid
        d_to_center = np.sqrt((x0 - cx)**2 + (y0 - cy)**2)
        
        score = region.area * region.eccentricity
        if d_to_center < iris_r * 2.0:
            score *= 1.5
            
        if score > max_score:
            max_score = score
            best_area = region.area
            best_region = region
            best_candidate_mask = region_mask

    # --- 4. Cálculo de métricas sobre el mejor candidato ---
    pterygium_percent = 0.0
    invasion_linear_percent = 0.0
    invasion_area_percent = 0.0
    confidence = 0
    affected_side = "N/A"
    level = "No detectado"
    message = "No detectado."

    if best_region is not None:
        pterygium_percent = (best_area / eye_opening_area) * 100.0
        
        candidate_in_cornea = cv2.bitwise_and(best_candidate_mask, cornea_zone)
        area_in_cornea = np.count_nonzero(candidate_in_cornea)
        invasion_area_percent = (area_in_cornea / cornea_area) * 100.0
        
        if area_in_cornea > 0:
            y_coords, x_coords = np.where(candidate_in_cornea > 0)
            distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            min_dist = np.min(distances)
            invasion_linear_percent = max(0.0, (iris_r - min_dist) / iris_r) * 100.0
        
        y0, x0 = best_region.centroid
        if x0 < cx:
            affected_side = "Izquierdo"
        else:
            affected_side = "Derecho"
            
        conf = 40 + (best_region.eccentricity * 30) + (invasion_linear_percent * 0.5) + (pterygium_percent * 2)
        confidence = min(int(conf), 98)
        
        if invasion_linear_percent > 30.0 or invasion_area_percent > 15.0:
            level = "Sospecha alta"
        elif invasion_linear_percent > 5.0 or invasion_area_percent > 2.0:
            level = "Sospecha moderada"
        else:
            level = "Sospecha leve"
            
        message = "Posible pterigión / carnosidad."
        if pterygium_percent > 45.0:
            level = "No confiable / posible sobresegmentación"
            message = "La máscara sospechosa es demasiado grande para considerarse pterigión. Revise la selección o ajuste la segmentación."
            best_candidate_mask = np.zeros_like(eye_opening_mask)
            pterygium_percent = 0.0
            invasion_linear_percent = 0.0
            invasion_area_percent = 0.0
            confidence = 0
            
    return {
        "status": "ok",
        "condition": "pterigion",
        "level": level,
        "candidate_area": best_area,
        "pterygium_percent": pterygium_percent,
        "invasion_linear_percent": invasion_linear_percent,
        "invasion_area_percent": invasion_area_percent,
        "confidence": confidence,
        "affected_side": affected_side,
        "message": message,
        "pterygium_mask": best_candidate_mask,
        "limbus_zone": limbus_zone,
        "invasion_zone": cornea_zone,
        "seed_mask": seed_mask,
        "growth_mask": growth_mask
    }
