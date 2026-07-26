import cv2
import numpy as np

from core.classical_methods import (
    otsu_threshold_roi, 
    detect_fundus_circle_hough, 
    clean_binary_mask, 
    detect_mser_candidates
)

def analyze_retina_fundus(img_bgr):
    """
    Análisis de retinografía / fondo de ojo para detectar posibles microaneurismas.
    """
    h, w = img_bgr.shape[:2]
    
    # 1. Extraer canal verde (mejor contraste para vasos y lesiones)
    b, g, r = cv2.split(img_bgr)
    
    # 2. Máscara del campo de visión (FOV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Umbral de Otsu inverso (o directo según la intensidad, pero FOV es más brillante que el negro del fondo)
    _, fov_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY) 
    
    # Refinar FOV con Otsu
    fov_otsu, _ = otsu_threshold_roi(gray)
    # Como Otsu puede agarrar demasiado, lo combinamos con un threshold bajo que da el contorno inicial grueso
    fov_mask = cv2.bitwise_and(fov_mask, fov_otsu)
    
    # Limpiar FOV
    kernel_fov = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_OPEN, kernel_fov)
    fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_CLOSE, kernel_fov)
    
    # Intentar refinar FOV con Hough
    cf = detect_fundus_circle_hough(gray)
    if cf is not None:
        hough_fov_mask = np.zeros_like(gray)
        cv2.circle(hough_fov_mask, (cf[0], cf[1]), cf[2], 255, -1)
        # Intersectar con el threshold para mayor robustez
        combined = cv2.bitwise_and(fov_mask, hough_fov_mask)
        # Solo usar si la intersección no destruyó la máscara
        if cv2.countNonZero(combined) > (h * w * 0.05):
            fov_mask = combined
            
    if cv2.countNonZero(fov_mask) < (h * w * 0.05):
        return {
            "status": "error",
            "message": "Fallo de segmentación: No se detectó un campo de visión (FOV) confiable.",
        }

    # 3. Ecualización adaptativa (CLAHE) en el canal verde
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_clahe = clahe.apply(g)
    
    # 4. Detectar y excluir el Disco Óptico (la zona más brillante y grande)
    # Suavizar para evitar ruido
    g_blur = cv2.GaussianBlur(g_clahe, (25, 25), 0)
    _, bright_mask = cv2.threshold(g_blur, 200, 255, cv2.THRESH_BINARY)
    bright_mask = cv2.bitwise_and(bright_mask, fov_mask)
    
    optic_disc_mask = np.zeros_like(fov_mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
    
    # Buscar el componente más grande (excluyendo el fondo)
    max_area = 0
    od_label = -1
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            od_label = i
            
    if od_label != -1:
        optic_disc_mask[labels == od_label] = 255
    else:
        # Fallback de Disco Óptico con Hough
        circles_od = cv2.HoughCircles(
            g_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h,w)*0.2,
            param1=50, param2=20, minRadius=int(min(h,w)*0.05), maxRadius=int(min(h,w)*0.15)
        )
        if circles_od is not None:
            # Buscar el círculo más brillante
            circles_od = np.round(circles_od[0, :]).astype("int")
            best_od = None
            best_bri = -1
            for cx, cy, r in circles_od:
                if cx < 0 or cy < 0 or cx >= w or cy >= h:
                    continue
                temp_mask = np.zeros_like(gray)
                cv2.circle(temp_mask, (cx, cy), r, 255, -1)
                temp_mask = cv2.bitwise_and(temp_mask, fov_mask)
                mean_bri = cv2.mean(g_clahe, mask=temp_mask)[0]
                if mean_bri > best_bri:
                    best_bri = mean_bri
                    best_od = (cx, cy, r)
            if best_od is not None:
                cv2.circle(optic_disc_mask, (best_od[0], best_od[1]), best_od[2], 255, -1)

    if np.any(optic_disc_mask):
        # Dilatar un poco para asegurar que se excluya por completo
        kernel_od = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        optic_disc_mask = cv2.dilate(optic_disc_mask, kernel_od)
        
    analysis_mask = cv2.bitwise_and(fov_mask, cv2.bitwise_not(optic_disc_mask))
    
    # 5. Detección de microaneurismas usando Bottom-Hat / Black-Hat
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(g_clahe, cv2.MORPH_BLACKHAT, kernel_bh)
    
    # Umbralizar candidatos con Otsu dentro del FOV
    candidates, _ = otsu_threshold_roi(blackhat, mask=analysis_mask)
    
    # 6. Detección MSER (Auxiliar)
    mser_mask_raw = detect_mser_candidates(g_clahe, mask=analysis_mask)
    
    # 7. Limpieza morfológica y filtrado con regionprops
    try:
        from skimage.measure import label, regionprops
        labels_bh = label(candidates > 0)
        regions_bh = regionprops(labels_bh)
        
        labels_mser = label(mser_mask_raw > 0)
        regions_mser = regionprops(labels_mser)
        
        microaneurysm_mask = np.zeros_like(candidates)
        mser_only_mask = np.zeros_like(candidates)
        count = 0
        mser_count = 0
        
        # Filtrar candidatos principales (BlackHat + Otsu)
        for region in regions_bh:
            if 5 < region.area < 150 and region.eccentricity < 0.9:
                for coords in region.coords:
                    microaneurysm_mask[coords[0], coords[1]] = 255
                count += 1
                
        # Preparar fov erosionado para checar bordes
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fov_eroded = cv2.erode(fov_mask, kernel_erode)
        
        # Filtrar candidatos auxiliares (MSER)
        for region in regions_mser:
            perimeter = region.perimeter
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * region.area) / (perimeter ** 2)
            
            if 5 < region.area < 120 and region.eccentricity < 0.85 and circularity > 0.4:
                # Extraer intensidades medias (deben ser oscuras)
                mean_intensity = np.mean([g_clahe[r, c] for r, c in region.coords])
                if mean_intensity < 140:
                    # Verificar que no toque el borde del FOV
                    touches_border = False
                    for r, c in region.coords:
                        if fov_eroded[r, c] == 0:
                            touches_border = True
                            break
                    if touches_border:
                        continue
                        
                    # Verificar si ya fue detectado por la vía principal
                    is_new = True
                    for coords in region.coords:
                        if microaneurysm_mask[coords[0], coords[1]] > 0:
                            is_new = False
                            break
                    if is_new:
                        for coords in region.coords:
                            mser_only_mask[coords[0], coords[1]] = 255
                        mser_count += 1
                    
        # Combinar máscaras para el overlay general
        final_candidates = cv2.bitwise_or(microaneurysm_mask, mser_only_mask)
        total_count = count + mser_count
        
        msg = f"Candidatos principales: {count}"
        if mser_count > 0:
            msg += f"\nCandidatos auxiliares (MSER): {mser_count}"
            
    except ImportError:
        # Fallback sin scikit-image
        microaneurysm_mask = clean_binary_mask(candidates, min_area=5)
        mser_only_mask = clean_binary_mask(mser_mask_raw, min_area=5)
        
        # Eliminar solapamiento
        mser_only_mask = cv2.bitwise_and(mser_only_mask, cv2.bitwise_not(microaneurysm_mask))
        final_candidates = cv2.bitwise_or(microaneurysm_mask, mser_only_mask)
        
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(final_candidates)
        total_count = max(0, num_labels - 1)
        msg = f"Posibles candidatos totales (incluyendo MSER si hubiese): {total_count}"
        
    return {
        "status": "ok",
        "condition": "retinopatia",
        "candidate_count": total_count,
        "message": msg + "\nResultado orientativo.",
        "fundus_preprocessed": g_clahe,
        "fundus_fov_mask": fov_mask,
        "optic_disc_mask": optic_disc_mask,
        "microaneurysm_candidates": final_candidates,
        "mser_only_mask": mser_only_mask,
        "otsu_mask_debug": candidates,
        "mser_candidates_debug": mser_mask_raw,
        "hough_fundus_circle": hough_fov_mask if 'hough_fov_mask' in locals() else np.zeros_like(fov_mask)
    }
