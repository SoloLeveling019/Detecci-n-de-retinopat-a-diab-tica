import cv2
import numpy as np
from typing import Tuple
from core.classical_methods import detect_pupil_hough

def detect_pupil_center(img_bgr: np.ndarray, content_mask: np.ndarray = None) -> Tuple[int, int, float, bool, dict]:
    """
    Detecta de forma aproximada el centro de la pupila usando zonas oscuras o Hough.
    Retorna: (cx, cy, radio, encontrado, debug_imgs_dict)
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, s_channel, v_channel = cv2.split(hsv)

    # Buscar regiones muy oscuras
    dark = ((v_channel < 45) & (s_channel < 170)).astype(np.uint8) * 255
    if content_mask is not None:
        dark = cv2.bitwise_and(dark, content_mask)
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Preparamos la llamada al fallback de Hough por si acaso
    hough_result = None
    debug_imgs = {}
    
    if not contours:
        hough_result, debug_imgs = detect_pupil_hough(img_bgr, content_mask)
        if hough_result is not None:
            x, y, r = hough_result
            return x, y, float(r), True, debug_imgs
            
        # 3. Búsqueda automática global por regiones oscuras circulares (3er pase)
        dark_loose = ((v_channel < 80)).astype(np.uint8) * 255
        if content_mask is not None:
            dark_loose = cv2.bitwise_and(dark_loose, content_mask)
        contours_loose, _ = cv2.findContours(dark_loose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_c = None
        best_c_score = -1
        for cnt in contours_loose:
            area = cv2.contourArea(cnt)
            if area < 0.001 * h * w:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
            if circularity > 0.6:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dist_center = np.hypot(cx - w / 2, cy - h / 2)
                    score = area * circularity - 0.1 * dist_center
                    if score > best_c_score:
                        best_c_score = score
                        best_c = (cx, cy, np.sqrt(area/np.pi))
        
        if best_c is not None:
            return best_c[0], best_c[1], float(best_c[2]), True, debug_imgs
            
        return 0, 0, 0.0, False, debug_imgs

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.001 * h * w:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Penalizar la distancia al centro para evitar pestañas oscuras en los bordes
        dist_center = np.hypot(cx - w / 2, cy - h / 2)
        score = area - 0.15 * dist_center
        candidates.append((score, area, cx, cy))

    if not candidates:
        if hough_result is None:
            hough_result, debug_imgs = detect_pupil_hough(img_bgr, content_mask)
        if hough_result is not None:
            x, y, r = hough_result
            return x, y, float(r), True, debug_imgs
            
        # 3. Búsqueda automática global por regiones oscuras circulares (3er pase)
        dark_loose = ((v_channel < 80)).astype(np.uint8) * 255
        if content_mask is not None:
            dark_loose = cv2.bitwise_and(dark_loose, content_mask)
        contours_loose, _ = cv2.findContours(dark_loose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_c = None
        best_c_score = -1
        for cnt in contours_loose:
            area = cv2.contourArea(cnt)
            if area < 0.001 * h * w:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
            if circularity > 0.6:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dist_center = np.hypot(cx - w / 2, cy - h / 2)
                    score = area * circularity - 0.1 * dist_center
                    if score > best_c_score:
                        best_c_score = score
                        best_c = (cx, cy, np.sqrt(area/np.pi))
        
        if best_c is not None:
            return best_c[0], best_c[1], float(best_c[2]), True, debug_imgs
            
        return 0, 0, 0.0, False, debug_imgs

    _, area, cx, cy = max(candidates, key=lambda x: x[0])
    radius = float(np.sqrt(area / np.pi))
    return cx, cy, radius, True, debug_imgs

def estimate_iris_region(pupil_cx: int, pupil_cy: int, pupil_r: float, img_shape: Tuple[int, int]) -> int:
    """
    Estima el radio del iris basándose en el radio de la pupila.
    """
    h, w = img_shape
    iris_radius = int(max(pupil_r * 3.7, min(h, w) * 0.22))
    # Límite superior para evitar que se desborde si detecta algo oscuro gigante
    iris_radius = min(iris_radius, int(min(h, w) * 0.4))
    return iris_radius
