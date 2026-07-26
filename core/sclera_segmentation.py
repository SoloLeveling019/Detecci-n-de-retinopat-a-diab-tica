import cv2
import numpy as np
from typing import Dict, Any

from .iris_pupil_detection import detect_pupil_center, estimate_iris_region

def build_sclera_masks(img_bgr: np.ndarray, content_mask: np.ndarray = None, manual_iris_info: tuple = None, manual_eye_opening_mask: np.ndarray = None) -> Dict[str, Any]:
    """
    Nuevo flujo anatómico:
    1. Detectar pupila e iris (automático o manual).
    2. Crear máscara de abertura ocular visible.
    3. Excluir iris y pupila.
    4. Quedarse solo con lados del iris.
    5. Filtrar piel/párpados.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    b, g, r = cv2.split(img_bgr)
    
    min_rgb = np.minimum(np.minimum(r, g), b)
    max_rgb = np.maximum(np.maximum(r, g), b)
    
    # 1. Iris y pupila
    iris_detection_mode = "automática"
    if manual_iris_info is not None:
        pupil_x, pupil_y, iris_r = manual_iris_info
        iris_detection_mode = "manual"
    else:
        pupil_x, pupil_y, pupil_r, found_pupil, debug_imgs = detect_pupil_center(img_bgr, content_mask)
        
        if not found_pupil:
            return {
                "success": False,
                "warning": "No se pudo detectar la pupila/iris.",
                "iris_info": None,
                "eye_opening_mask": np.zeros((h, w), dtype=np.uint8),
                "iris_pupil_mask": np.zeros((h, w), dtype=np.uint8),
                "sclera_roi_mask": np.zeros((h, w), dtype=np.uint8),
                "white_sclera_mask": np.zeros((h, w), dtype=np.uint8),
                "iris_detection_mode": "fallida"
            }
        iris_r = estimate_iris_region(pupil_x, pupil_y, pupil_r, (h, w))
    
    # 2. Máscara de abertura ocular (eye_opening_mask) - Base geométrica o Manual o Auto
    if manual_eye_opening_mask is not None:
        fallback_eye_mask = manual_eye_opening_mask.copy()
        eye_opening_mask = manual_eye_opening_mask.copy()
    else:
        # Elipse anatómica inicial generosa (más ancha que alta para cubrir lagrimal y canto externo)
        fallback_eye_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            fallback_eye_mask,
            (int(pupil_x), int(pupil_y)),
            (int(4.0 * iris_r), int(2.5 * iris_r)),
            0, 0, 360, 255, -1
        )
        
        # 2.1 Refinamiento usando Otsu y Canny (Flujo Automático Completo)
        from core.classical_methods import detect_edges_canny, otsu_threshold_roi, clean_binary_mask
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = detect_edges_canny(gray)
        
        # Otsu en V-channel para separar esclerótica (brillante) de piel/sombras
        v_otsu_mask, _ = otsu_threshold_roi(v_channel, mask=fallback_eye_mask)
        
        # Filtro de color estricto para piel
        skin_like = (
            (s_channel > 60) &
            (v_channel > 60) &
            (h_channel > 0) & (h_channel < 35)
        ).astype(np.uint8) * 255
        
        # Restar bordes y piel a la máscara de Otsu
        refined = cv2.bitwise_and(v_otsu_mask, cv2.bitwise_not(edges))
        refined = cv2.bitwise_and(refined, cv2.bitwise_not(skin_like))
        
        # Limpieza morfológica
        refined = clean_binary_mask(refined, min_area=int(iris_r * iris_r * 0.5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_close)
        
        # Rellenar la pupila/iris en la máscara de abertura, ya que Otsu/piel pudo haber cortado el centro
        iris_pupil_temp = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(iris_pupil_temp, (int(pupil_x), int(pupil_y)), int(iris_r * 1.1), 255, -1)
        refined = cv2.bitwise_or(refined, iris_pupil_temp)
        
        # Aplicar elipse inicial como límite máximo absoluto
        eye_opening_mask = cv2.bitwise_and(refined, fallback_eye_mask)
        
        # Extraer el componente más grande que contiene a la pupila
        from skimage.measure import label
        labels = label(eye_opening_mask > 0)
        pupil_label = labels[int(pupil_y), int(pupil_x)]
        if pupil_label > 0:
            eye_opening_mask = (labels == pupil_label).astype(np.uint8) * 255
        else:
            eye_opening_mask = fallback_eye_mask.copy()
            
        # Suavizar bordes
        eye_opening_mask = cv2.morphologyEx(eye_opening_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # 3. Excluir iris y pupila
    iris_pupil_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(iris_pupil_mask, (int(pupil_x), int(pupil_y)), int(1.05 * iris_r), 255, -1)
    
    # 4. Zonas laterales
    side_mask = np.zeros((h, w), dtype=np.uint8)
    l_x = int(pupil_x - 0.7 * iris_r)
    r_x = int(pupil_x + 0.7 * iris_r)
    if l_x > 0:
        side_mask[:, :l_x] = 255
    if r_x < w:
        side_mask[:, r_x:] = 255
        
    if content_mask is not None:
        fallback_eye_mask = cv2.bitwise_and(fallback_eye_mask, content_mask)
        side_mask = cv2.bitwise_and(side_mask, content_mask)
        
    sclera_roi_mask = cv2.bitwise_and(eye_opening_mask, cv2.bitwise_not(iris_pupil_mask))
    sclera_roi_mask = cv2.bitwise_and(sclera_roi_mask, side_mask)
    
    # 5. Refinamiento por color dentro de la ROI
    if manual_eye_opening_mask is not None:
        # Modo guiado: Omitir filtro de color agresivo (piel), solo proteger contra pestañas/oscuridad (V > 35)
        eye_color_candidate = (v_channel > 35).astype(np.uint8) * 255
    else:
        eye_color_candidate = (
            (v_channel > 50) &
            (s_channel < 200) & # Relajado para rojo
            (min_rgb > 30)
        ).astype(np.uint8) * 255
        
        # Filtro suave de piel
        skin_like = (
            (s_channel > 60) &
            (v_channel > 70) &
            (h_channel > 0) & (h_channel < 35)
        ).astype(np.uint8) * 255
        
        # Zona de protección cercana al iris (aquí no se elimina piel, por si es inflamación roja)
        protected_zone = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            protected_zone,
            (int(pupil_x), int(pupil_y)),
            (int(2.0 * iris_r), int(0.9 * iris_r)),
            0, 0, 360, 255, -1
        )
        
        skin_to_remove = cv2.bitwise_and(skin_like, cv2.bitwise_not(protected_zone))
        eye_color_candidate = cv2.bitwise_and(eye_color_candidate, cv2.bitwise_not(skin_to_remove))
        
    sclera_roi_mask = cv2.bitwise_and(sclera_roi_mask, eye_color_candidate)
    eye_opening_mask = cv2.bitwise_and(fallback_eye_mask, eye_color_candidate) # Para depuración
    
    # Limpieza morfológica para unir la abertura
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    sclera_roi_mask = cv2.morphologyEx(sclera_roi_mask, cv2.MORPH_OPEN, kernel_open)
    sclera_roi_mask = cv2.morphologyEx(sclera_roi_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Conservar el componente conectado más cercano a la pupila en cada lado
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sclera_roi_mask, connectivity=8)
    clean_sclera_roi = np.zeros_like(sclera_roi_mask)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]
        dist = np.sqrt((cx - pupil_x)**2 + (cy - pupil_y)**2)
        
        # Filtro de distancia al iris
        if dist < 2.5 * iris_r and area > int(0.001 * h * w):
            clean_sclera_roi[labels == i] = 255

    sclera_roi_mask_raw = clean_sclera_roi.copy()
    
    # Rellenar huecos con scipy
    try:
        from utils.advanced_analysis import clean_mask_holes
        sclera_roi_mask = clean_mask_holes(sclera_roi_mask_raw)
        
        if content_mask is not None:
            sclera_roi_mask = cv2.bitwise_and(sclera_roi_mask, content_mask)
        sclera_roi_mask = cv2.bitwise_and(sclera_roi_mask, cv2.bitwise_not(iris_pupil_mask))
        sclera_roi_mask = cv2.bitwise_and(sclera_roi_mask, side_mask)
    except Exception:
        sclera_roi_mask = sclera_roi_mask_raw
    
    if content_mask is not None:
        sclera_roi_mask_raw = cv2.bitwise_and(sclera_roi_mask_raw, content_mask)
        
    # Validación de área mínima
    min_roi_area = int(0.002 * h * w)
    sclera_roi_area = np.count_nonzero(sclera_roi_mask)
    
    success = True
    warning = ""
    if sclera_roi_area < min_roi_area:
        success = False
        warning = "No se pudo detectar una región anatómica válida de esclerótica."

    # 6. Crear white_sclera_mask para validación y UI
    white_core = (
        (v_channel > 90) &       
        (s_channel < 110) &        
        (min_rgb > 60) &         
        ((max_rgb.astype(np.int16) - min_rgb.astype(np.int16)) < 80) 
    ).astype(np.uint8) * 255
    
    white_sclera_mask_raw = cv2.bitwise_and(white_core, sclera_roi_mask)
    white_sclera_mask_raw = cv2.morphologyEx(white_sclera_mask_raw, cv2.MORPH_OPEN, kernel_open)
    
    try:
        white_sclera_mask = clean_mask_holes(white_sclera_mask_raw)
        
        if content_mask is not None:
            white_sclera_mask = cv2.bitwise_and(white_sclera_mask, content_mask)
        white_sclera_mask = cv2.bitwise_and(white_sclera_mask, cv2.bitwise_not(iris_pupil_mask))
        white_sclera_mask = cv2.bitwise_and(white_sclera_mask, side_mask)
    except Exception:
        white_sclera_mask = white_sclera_mask_raw
    
    white_area = np.count_nonzero(white_sclera_mask)
    if success and white_area < int(0.05 * sclera_roi_area):
        warning = "La esclerótica blanca es pequeña; posible enrojecimiento fuerte o mala iluminación."
    
    # 6.5 Crear bandas del párpado (eyelid_band_mask) para blefaritis y orzuelo
    band_thickness = int(max(12, min(35, iris_r * 0.18)))
    kernel_band = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_thickness, band_thickness))
    dilated_eye = cv2.dilate(eye_opening_mask, kernel_band)
    eyelid_band_mask = cv2.bitwise_and(dilated_eye, cv2.bitwise_not(eye_opening_mask))
    if content_mask is not None:
        eyelid_band_mask = cv2.bitwise_and(eyelid_band_mask, content_mask)
        
    upper_eyelid_band_mask = np.zeros_like(eyelid_band_mask)
    lower_eyelid_band_mask = np.zeros_like(eyelid_band_mask)
    upper_eyelid_band_mask[:int(pupil_y), :] = eyelid_band_mask[:int(pupil_y), :]
    lower_eyelid_band_mask[int(pupil_y):, :] = eyelid_band_mask[int(pupil_y):, :]

    # 7. Retornar diccionario
    result = {
        "success": success,
        "warning": warning,
        "iris_info": (pupil_x, pupil_y, iris_r),
        "eye_opening_mask": eye_opening_mask,
        "iris_pupil_mask": iris_pupil_mask,
        "sclera_roi_mask_raw": sclera_roi_mask_raw,
        "sclera_roi_mask": sclera_roi_mask,
        "white_sclera_mask": white_sclera_mask,
        "eyelid_band_mask": eyelid_band_mask,
        "upper_eyelid_band_mask": upper_eyelid_band_mask,
        "lower_eyelid_band_mask": lower_eyelid_band_mask,
        "iris_detection_mode": iris_detection_mode
    }
    
    if 'debug_imgs' in locals():
        result['debug_imgs'] = debug_imgs
        
    return result

def build_red_vessel_mask(img_bgr: np.ndarray, sclera_roi_mask: np.ndarray) -> np.ndarray:
    """Detecta zonas rojizas estrictamente dentro de la esclerótica anatómica."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(img_bgr)

    red_hsv = (
        cv2.inRange(hsv, np.array([0, 55, 60]), np.array([12, 255, 255])) |
        cv2.inRange(hsv, np.array([168, 55, 60]), np.array([180, 255, 255]))
    )

    r16 = r.astype(np.int16)
    g16 = g.astype(np.int16)
    b16 = b.astype(np.int16)

    red_dominance = (
        (r16 > g16 + 25) &
        (r16 > b16 + 25) &
        (r16 > 90)
    ).astype(np.uint8) * 255

    red_mask = cv2.bitwise_and(red_hsv, red_dominance)
    red_mask = cv2.bitwise_and(red_mask, red_mask, mask=sclera_roi_mask)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    return red_mask
