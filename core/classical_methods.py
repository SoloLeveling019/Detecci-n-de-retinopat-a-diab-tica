import cv2
import numpy as np

def detect_edges_canny(img_gray, sigma=0.33, lower_thresh=None, upper_thresh=None):
    if lower_thresh is None or upper_thresh is None:
        v = np.median(img_gray)
        lower_thresh = int(max(0, (1.0 - sigma) * v))
        upper_thresh = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img_gray, lower_thresh, upper_thresh)

def otsu_threshold_roi(img_gray, mask=None, inverse=False):
    if mask is None:
        mode = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU if inverse else cv2.THRESH_BINARY + cv2.THRESH_OTSU
        ret, thresh = cv2.threshold(img_gray, 0, 255, mode)
        return thresh, ret
    
    masked_pixels = img_gray[mask > 0]
    if len(masked_pixels) == 0:
        return np.zeros_like(img_gray), 0
        
    ret, _ = cv2.threshold(masked_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mode = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(img_gray, ret, 255, mode)
    thresh = cv2.bitwise_and(thresh, mask)
    return thresh, ret

def clean_binary_mask(mask, min_area=50):
    try:
        from skimage.morphology import remove_small_objects
        bool_mask = mask > 0
        clean_bool = remove_small_objects(bool_mask, min_size=min_area)
        return (clean_bool * 255).astype(np.uint8)
    except ImportError:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        clean_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean_mask[labels == i] = 255
        return clean_mask

def detect_pupil_hough(img_bgr, content_mask=None, use_canny=True):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if content_mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=content_mask)

    gray_blur = cv2.medianBlur(gray, 5)
    
    debug_imgs = {}
    
    if use_canny:
        edges = detect_edges_canny(gray_blur)
        input_img = edges
        debug_imgs["canny_edges"] = edges
    else:
        input_img = gray_blur

    h, w = gray.shape[:2]
    min_r = int(min(h, w) * 0.04)
    max_r = int(min(h, w) * 0.18)

    circles = cv2.HoughCircles(
        input_img,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) * 0.2,
        param1=80,
        param2=25,
        minRadius=min_r,
        maxRadius=max_r
    )

    if circles is None:
        return None, debug_imgs

    circles = np.round(circles[0, :]).astype("int")

    best_circle = None
    best_score = -1

    for x, y, r in circles:
        if x < 0 or y < 0 or x >= w or y >= h:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)

        if content_mask is not None:
            mask = cv2.bitwise_and(mask, content_mask)

        inside_mean = cv2.mean(gray, mask=mask)[0]
        dist_center = np.hypot(x - w / 2, y - h / 2)

        score = (255 - inside_mean) - 0.08 * dist_center

        if score > best_score:
            best_score = score
            best_circle = (x, y, r)

    if best_circle is not None:
        debug_img = img_bgr.copy()
        cv2.circle(debug_img, (best_circle[0], best_circle[1]), best_circle[2], (0, 255, 0), 2)
        debug_imgs["hough_pupil_debug"] = debug_img

    return best_circle, debug_imgs

def detect_fundus_circle_hough(img_gray):
    h, w = img_gray.shape[:2]
    circles_fov = cv2.HoughCircles(
        cv2.medianBlur(img_gray, 5), cv2.HOUGH_GRADIENT, dp=1.2, minDist=h/2,
        param1=50, param2=30, minRadius=int(min(h,w)*0.3), maxRadius=int(max(h,w)*0.6)
    )
    
    if circles_fov is not None:
        cf = np.round(circles_fov[0, 0]).astype("int")
        return (cf[0], cf[1], cf[2])
    return None

def detect_mser_candidates(img_gray, mask=None):
    mser = cv2.MSER_create(delta=5, min_area=5, max_area=150, max_variation=0.25)
    regions, _ = mser.detectRegions(img_gray)
    
    mser_mask = np.zeros_like(img_gray)
    for p in regions:
        for (x, y) in p:
            if mask is None or mask[y, x] > 0:
                mser_mask[y, x] = 255
    return mser_mask
