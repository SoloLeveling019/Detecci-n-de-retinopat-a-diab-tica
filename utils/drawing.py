import cv2
import numpy as np

def draw_conjunctivitis_overlay_clean(img_bgr: np.ndarray, sclera_roi_mask: np.ndarray, white_sclera_mask: np.ndarray, red_mask: np.ndarray) -> np.ndarray:
    overlay = img_bgr.copy()
    red_layer = np.zeros_like(img_bgr)
    
    # Colorear la capa roja donde la mascara es 255
    red_layer[:, :, 2] = red_mask
    
    # Combinar con un alpha para transparencia
    overlay = cv2.addWeighted(overlay, 1.0, red_layer, 0.65, 0)

    # Dibujar contorno del ROI en celeste
    contours_roi, _ = cv2.findContours(sclera_roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_roi, -1, (255, 255, 0), 2)  # BGR para celeste

    # Dibujar contorno de la zona blanca en amarillo
    contours_white, _ = cv2.findContours(white_sclera_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_white, -1, (0, 255, 255), 2)  # BGR para amarillo

    return overlay

def draw_cataract_overlay_clean(img_bgr: np.ndarray, pupil_zone_mask: np.ndarray, opacity_mask: np.ndarray) -> np.ndarray:
    overlay = img_bgr.copy()
    
    if pupil_zone_mask is not None:
        # Dibujar contorno de la zona pupilar (azul)
        contours_pupil, _ = cv2.findContours(pupil_zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_pupil, -1, (255, 0, 0), 2)
        
    if opacity_mask is not None:
        # Colorear la mascara de opacidad (amarillo semi transparente)
        yellow_layer = np.zeros_like(img_bgr)
        yellow_layer[:, :, 1] = opacity_mask
        yellow_layer[:, :, 2] = opacity_mask
        overlay = cv2.addWeighted(overlay, 1.0, yellow_layer, 0.65, 0)
    
    return overlay

def draw_pterygium_overlay_clean(img_bgr: np.ndarray, sclera_roi_mask: np.ndarray, pterygium_mask: np.ndarray) -> np.ndarray:
    overlay = img_bgr.copy()
    
    if sclera_roi_mask is not None:
        # Dibujar contorno de la ROI en celeste
        contours_roi, _ = cv2.findContours(sclera_roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_roi, -1, (255, 255, 0), 2)
        
    if pterygium_mask is not None:
        # Dibujar contorno del pterigion en magenta
        contours_pterygium, _ = cv2.findContours(pterygium_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_pterygium, -1, (255, 0, 255), 2)
        
        # Colorear relleno en magenta semi transparente
        magenta_layer = np.zeros_like(img_bgr)
        magenta_layer[:, :, 0] = pterygium_mask
        magenta_layer[:, :, 2] = pterygium_mask
        overlay = cv2.addWeighted(overlay, 1.0, magenta_layer, 0.5, 0)
    
    return overlay

def draw_blepharitis_overlay_clean(img_bgr: np.ndarray, blepharitis_mask: np.ndarray) -> np.ndarray:
    overlay = img_bgr.copy()
    
    if blepharitis_mask is not None:
        # Dibujar contorno de la blefaritis en naranja
        contours, _ = cv2.findContours(blepharitis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 165, 255), 2)  # Naranja en BGR
    
    return overlay

def draw_stye_overlay_clean(img_bgr: np.ndarray, stye_mask: np.ndarray) -> np.ndarray:
    overlay = img_bgr.copy()
    
    if stye_mask is not None:
        # Dibujar contorno del orzuelo en cyan
        contours, _ = cv2.findContours(stye_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)  # Cyan en BGR
        
        cyan_layer = np.zeros_like(img_bgr)
        cyan_layer[:, :, 0] = stye_mask
        cyan_layer[:, :, 1] = stye_mask
        overlay = cv2.addWeighted(overlay, 1.0, cyan_layer, 0.5, 0)
    
    return overlay

def draw_retinopathy_overlay_clean(img_bgr: np.ndarray, candidates_mask: np.ndarray, optic_disc_mask: np.ndarray) -> np.ndarray:
    overlay = img_bgr.copy()
    
    # Dibujar contorno del disco optico en amarillo
    contours_od, _ = cv2.findContours(optic_disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_od, -1, (0, 255, 255), 2)
    
    # Resaltar microaneurismas en verde brillante
    green_layer = np.zeros_like(img_bgr)
    green_layer[:, :, 1] = candidates_mask
    overlay = cv2.addWeighted(overlay, 1.0, green_layer, 0.8, 0)
    
    # Contorno de los microaneurismas
    contours_cand, _ = cv2.findContours(candidates_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_cand:
        # Dibujar un circulito alrededor de cada candidato
        (x,y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = max(5, int(radius) + 2)
        cv2.circle(overlay, center, radius, (0, 255, 0), 2)
    
    return overlay

def draw_general_clean_overlay(img_bgr: np.ndarray, sclera_roi_mask: np.ndarray, red_mask: np.ndarray, 
                               cataract_pupil_mask: np.ndarray = None, cataract_opacity_mask: np.ndarray = None,
                               pterygium_mask: np.ndarray = None, blepharitis_mask: np.ndarray = None, stye_mask: np.ndarray = None) -> np.ndarray:
    """
    Dibuja un overlay general limpio, sin textos ni cuadros, solo los contornos y capas de las enfermedades detectadas.
    """
    overlay = img_bgr.copy()
    
    # 1. Enrojecimiento (Rojo semitransparente)
    if red_mask is not None and np.count_nonzero(red_mask) > 0:
        red_layer = np.zeros_like(img_bgr)
        red_layer[:, :, 2] = red_mask
        overlay = cv2.addWeighted(overlay, 1.0, red_layer, 0.65, 0)
    
    # 2. Esclerótica (Contorno Celeste)
    if sclera_roi_mask is not None:
        contours_roi, _ = cv2.findContours(sclera_roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_roi, -1, (255, 255, 0), 2)
        
    # 3. Catarata (Contorno Azul y relleno amarillo)
    if cataract_pupil_mask is not None and cataract_opacity_mask is not None:
        contours_pupil, _ = cv2.findContours(cataract_pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_pupil, -1, (255, 0, 0), 2)
        
        yellow_layer = np.zeros_like(img_bgr)
        yellow_layer[:, :, 1] = cataract_opacity_mask
        yellow_layer[:, :, 2] = cataract_opacity_mask
        overlay = cv2.addWeighted(overlay, 1.0, yellow_layer, 0.65, 0)
        
    # 4. Pterigión (Contorno y relleno Magenta)
    if pterygium_mask is not None and np.count_nonzero(pterygium_mask) > 0:
        contours_pterygium, _ = cv2.findContours(pterygium_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_pterygium, -1, (255, 0, 255), 2)
        
        magenta_layer = np.zeros_like(img_bgr)
        magenta_layer[:, :, 0] = pterygium_mask
        magenta_layer[:, :, 2] = pterygium_mask
        overlay = cv2.addWeighted(overlay, 1.0, magenta_layer, 0.5, 0)
        
    # 5. Blefaritis (Contorno Naranja)
    if blepharitis_mask is not None and np.count_nonzero(blepharitis_mask) > 0:
        contours_bleph, _ = cv2.findContours(blepharitis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_bleph, -1, (0, 165, 255), 2)
        
    # 6. Orzuelo (Contorno y relleno Cyan)
    if stye_mask is not None and np.count_nonzero(stye_mask) > 0:
        contours_stye, _ = cv2.findContours(stye_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_stye, -1, (255, 255, 0), 2)
        
        cyan_layer = np.zeros_like(img_bgr)
        cyan_layer[:, :, 0] = stye_mask
        cyan_layer[:, :, 1] = stye_mask
        overlay = cv2.addWeighted(overlay, 1.0, cyan_layer, 0.5, 0)

    return overlay

