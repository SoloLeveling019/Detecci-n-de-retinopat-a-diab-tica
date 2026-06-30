"""
Analizador de ojo externo: detección de enrojecimiento/venas rojas en la esclerótica.

IMPORTANTE:
- Este programa NO diagnostica diabetes.
- Analiza una fotografía externa del ojo y calcula un índice de enrojecimiento superficial.
- Para retinopatía diabética se necesitan imágenes de fondo de ojo/retinografía, no una foto externa.

Uso:
    python analizador_ojo_externo_interfaz.py

Instalación:
    pip install opencv-python numpy pillow

Opcional para HEIC/HEIF:
    pip install pillow-heif
"""

import os
import csv
import platform
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageOps, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    # pyrefly: ignore [missing-import]
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


SUPPORTED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
    ".webp", ".jp2", ".j2k", ".heic", ".heif"
)


# ============================================================
# 1. CARGA ROBUSTA DE IMÁGENES
# ============================================================

def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convierte imágenes de 16 bits/float a uint8."""
    if img is None:
        raise ValueError("Imagen vacía.")

    if img.dtype == np.uint8:
        return img

    img_float = img.astype(np.float32)
    min_val = float(np.nanmin(img_float))
    max_val = float(np.nanmax(img_float))

    if max_val <= min_val:
        return np.zeros(img.shape, dtype=np.uint8)

    img_norm = (img_float - min_val) / (max_val - min_val)
    return np.clip(img_norm * 255.0, 0, 255).astype(np.uint8)


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Asegura imagen BGR de 3 canales."""
    img = normalize_to_uint8(img)

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        fondo = np.zeros_like(bgr, dtype=np.float32)
        compuesto = bgr * alpha + fondo * (1.0 - alpha)
        return compuesto.astype(np.uint8)

    if img.ndim == 3 and img.shape[2] >= 3:
        return img[:, :, :3].copy()

    raise ValueError(f"Formato no compatible. Shape recibido: {img.shape}")


def load_image_any_format(image_path: str) -> np.ndarray:
    """Carga JPG, PNG, TIFF, WEBP, capturas y rutas con tildes/espacios."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    if img is not None:
        return ensure_bgr(img)

    if PIL_AVAILABLE:
        pil_img = Image.open(path)
        pil_img = ImageOps.exif_transpose(pil_img)

        if pil_img.mode in ("RGBA", "LA"):
            fondo = Image.new("RGBA", pil_img.size, (0, 0, 0, 255))
            pil_img = Image.alpha_composite(fondo, pil_img.convert("RGBA")).convert("RGB")
        else:
            pil_img = pil_img.convert("RGB")

        rgb = np.array(pil_img)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    raise ValueError("No se pudo cargar la imagen. Instala Pillow: pip install pillow")


# ============================================================
# 2. ANÁLISIS DE OJO EXTERNO / ESCLERÓTICA
# ============================================================

def resize_for_processing(img: np.ndarray, max_side: int = 1200) -> Tuple[np.ndarray, float]:
    """Redimensiona imágenes grandes para análisis más rápido."""
    h, w = img.shape[:2]
    mayor = max(h, w)
    if mayor <= max_side:
        return img.copy(), 1.0

    scale = max_side / float(mayor)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def detect_pupil_center(img_bgr: np.ndarray) -> Tuple[int, int, float, bool]:
    """Detecta de forma aproximada el centro de la pupila usando zonas oscuras."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, s_channel, v_channel = cv2.split(hsv)

    dark = ((v_channel < 45) & (s_channel < 170)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return w // 2, h // 2, min(h, w) * 0.07, False

    # La pupila suele ser el componente oscuro grande más cercano al centro.
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
        dist_center = np.hypot(cx - w / 2, cy - h / 2)
        score = area - 0.15 * dist_center
        candidates.append((score, area, cx, cy))

    if not candidates:
        return w // 2, h // 2, min(h, w) * 0.07, False

    _, area, cx, cy = max(candidates, key=lambda x: x[0])
    radius = float(np.sqrt(area / np.pi))
    return cx, cy, radius, True


def build_sclera_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Segmenta de forma aproximada la esclerótica.

    Esta versión usa dos ideas:
    1) Detectar la pupila para excluir iris/pupila.
    2) Buscar regiones claras de baja saturación a los lados del iris.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    b, g, r = cv2.split(img_bgr)

    min_rgb = np.minimum(np.minimum(r, g), b)
    max_rgb = np.maximum(np.maximum(r, g), b)

    # Núcleo blanco de la esclerótica. Más estricto para no tomar piel/párpados.
    white_core = (
        (v_channel > 145) &
        (s_channel < 75) &
        (min_rgb > 115) &
        ((max_rgb.astype(np.int16) - min_rgb.astype(np.int16)) < 85)
    ).astype(np.uint8) * 255

    pupil_x, pupil_y, pupil_r, found_pupil = detect_pupil_center(img_bgr)

    if found_pupil:
        # Excluir el iris alrededor de la pupila.
        iris_radius = int(max(pupil_r * 3.7, min(h, w) * 0.22))
        outside_iris = np.ones((h, w), dtype=np.uint8) * 255
        cv2.circle(outside_iris, (pupil_x, pupil_y), iris_radius, 0, -1)

        # La esclerótica visible suele estar en una banda horizontal alrededor de la pupila.
        band = np.zeros((h, w), dtype=np.uint8)
        y1 = max(0, int(pupil_y - h * 0.18))
        y2 = min(h, int(pupil_y + h * 0.20))
        band[y1:y2, :] = 255

        mask = cv2.bitwise_and(white_core, white_core, mask=outside_iris)
        mask = cv2.bitwise_and(mask, mask, mask=band)
    else:
        mask = white_core

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Mantener componentes grandes compatibles con regiones de esclerótica.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)

    min_area = int(0.003 * h * w)
    kept = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx_comp, cy_comp = centroids[i]
        if area < min_area:
            continue
        if found_pupil and abs(cy_comp - pupil_y) > 0.22 * h:
            continue
        clean[labels == i] = 255
        kept += 1

    if kept == 0:
        clean = mask

    clean = cv2.dilate(clean, kernel, iterations=1)
    return clean

def build_red_vessel_mask(img_bgr: np.ndarray, sclera_mask: np.ndarray) -> np.ndarray:
    """Detecta zonas rojizas dentro o cerca de la esclerótica."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(img_bgr)

    # Rojos en HSV: cerca de 0° o 180°.
    lower_red1 = np.array([0, 35, 40], dtype=np.uint8)
    upper_red1 = np.array([15, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([165, 35, 40], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    red_hsv = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # Dominancia del canal rojo sobre verde y azul.
    r16 = r.astype(np.int16)
    g16 = g.astype(np.int16)
    b16 = b.astype(np.int16)
    red_dominance = ((r16 > g16 + 12) & (r16 > b16 + 12) & (r16 > 70)).astype(np.uint8) * 255

    red_mask = cv2.bitwise_and(red_hsv, red_dominance)

    # Ampliar un poco la máscara de esclerótica para incluir venitas finas en el borde.
    kernel_sclera = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    sclera_near = cv2.dilate(sclera_mask, kernel_sclera, iterations=1)
    red_mask = cv2.bitwise_and(red_mask, red_mask, mask=sclera_near)

    # Eliminar ruido.
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    return red_mask


def redness_level(redness_percent: float) -> str:
    """Clasificación heurística, no clínica."""
    if redness_percent < 0.60:
        return "Bajo"
    if redness_percent < 2.00:
        return "Leve"
    if redness_percent < 5.00:
        return "Moderado"
    return "Alto"


def recommendation_for_level(level: str) -> str:
    if level == "Bajo":
        return "No se aprecia enrojecimiento marcado en la zona clara detectada."
    if level == "Leve":
        return "Se aprecia enrojecimiento leve. Puede relacionarse con cansancio, sequedad o irritación, pero no confirma enfermedad."
    if level == "Moderado":
        return "Se aprecia enrojecimiento moderado. Conviene revisar síntomas asociados: dolor, secreción, picazón o visión borrosa."
    return "Se aprecia enrojecimiento alto. Si hay dolor, secreción, sensibilidad a la luz o pérdida de visión, se recomienda evaluación médica."


def analyze_external_eye(image_path: str, output_dir: str = "resultados_ojo_externo") -> Dict:
    """Analiza enrojecimiento/venas rojas en una foto externa del ojo."""
    original_full = load_image_any_format(image_path)
    img, scale = resize_for_processing(original_full, max_side=1200)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    sclera_mask = build_sclera_mask(img)
    red_mask = build_red_vessel_mask(img, sclera_mask)

    sclera_area = int(np.count_nonzero(sclera_mask))
    red_area = int(np.count_nonzero(red_mask))
    total_area = img.shape[0] * img.shape[1]

    if sclera_area > 0:
        redness_percent = (red_area / float(sclera_area)) * 100.0
    else:
        redness_percent = 0.0

    level = redness_level(redness_percent)
    recommendation = recommendation_for_level(level)

    # Overlay visual.
    overlay = img.copy()
    red_layer = np.zeros_like(img)
    red_layer[:, :, 2] = red_mask
    overlay = cv2.addWeighted(overlay, 1.0, red_layer, 0.65, 0)

    # Dibujar contorno de la esclerótica detectada.
    contours, _ = cv2.findContours(sclera_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

    text1 = f"Enrojecimiento: {redness_percent:.2f}%"
    text2 = f"Nivel: {level}"
    cv2.rectangle(overlay, (10, 10), (430, 82), (0, 0, 0), -1)
    cv2.putText(overlay, text1, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, text2, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    def save_img(path: Path, image: np.ndarray):
        ext = path.suffix.lower() if path.suffix else ".png"
        ok, encoded = cv2.imencode(ext, image)
        if not ok:
            raise IOError(f"No se pudo guardar: {path}")
        encoded.tofile(str(path))

    output_path = output_dir_path / f"{stem}_analisis_ojo_externo.png"
    sclera_path = output_dir_path / f"{stem}_mascara_esclerotica.png"
    red_path = output_dir_path / f"{stem}_mascara_rojo.png"
    csv_path = output_dir_path / "resumen_ojo_externo.csv"

    save_img(output_path, overlay)
    save_img(sclera_path, sclera_mask)
    save_img(red_path, red_mask)

    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "imagen", "salida", "area_esclerotica_px", "area_roja_px",
                "porcentaje_enrojecimiento", "nivel", "nota"
            ])
        writer.writerow([
            str(image_path), str(output_path), sclera_area, red_area,
            f"{redness_percent:.4f}", level,
            "No diagnostica diabetes; solo cuantifica enrojecimiento superficial."
        ])

    return {
        "image_path": str(image_path),
        "output_path": str(output_path),
        "sclera_mask_path": str(sclera_path),
        "red_mask_path": str(red_path),
        "csv_path": str(csv_path),
        "original_bgr": img,
        "overlay_bgr": overlay,
        "sclera_area": sclera_area,
        "red_area": red_area,
        "total_area": total_area,
        "redness_percent": redness_percent,
        "level": level,
        "recommendation": recommendation,
        "scale": scale,
    }


# ============================================================
# 3. INTERFAZ GRÁFICA
# ============================================================

class ExternalEyeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Analizador de ojo externo - Esclerótica y venas rojas")
        self.root.geometry("1180x760")
        self.root.minsize(1050, 680)

        self.selected_path = tk.StringVar(value="Ninguna imagen seleccionada")
        self.status = tk.StringVar(value="Selecciona una foto externa del ojo.")
        self.output_dir = Path.cwd() / "resultados_ojo_externo"

        self.current_original_tk = None
        self.current_result_tk = None
        self.last_result = None

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(main, text="Controles", padding=12)
        controls.pack(side="left", fill="y", padx=(0, 10))

        ttk.Button(controls, text="Seleccionar foto del ojo", command=self.select_image).pack(fill="x", pady=(0, 8))
        ttk.Button(controls, text="Analizar esclerótica/venas rojas", command=self.analyze).pack(fill="x", pady=(0, 8))
        ttk.Button(controls, text="Abrir carpeta de resultados", command=self.open_results_folder).pack(fill="x", pady=(0, 15))

        ttk.Label(controls, text="Imagen seleccionada:").pack(anchor="w")
        ttk.Label(controls, textvariable=self.selected_path, wraplength=260).pack(fill="x", pady=(0, 15))

        ttk.Separator(controls).pack(fill="x", pady=8)
        ttk.Label(
            controls,
            text=(
                "Este modo analiza una foto externa del ojo.\n\n"
                "Calcula un índice de enrojecimiento en la zona blanca/esclerótica.\n\n"
                "No diagnostica diabetes. Para retinopatía diabética se necesita retinografía/fondo de ojo."
            ),
            wraplength=260,
            justify="left"
        ).pack(anchor="w")

        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        image_frame = ttk.Frame(right)
        image_frame.pack(fill="both", expand=True)

        original_box = ttk.LabelFrame(image_frame, text="Imagen original", padding=8)
        original_box.pack(side="left", fill="both", expand=True, padx=(0, 5))

        result_box = ttk.LabelFrame(image_frame, text="Resultado", padding=8)
        result_box.pack(side="left", fill="both", expand=True, padx=(5, 0))

        self.original_label = ttk.Label(original_box, text="Aquí aparecerá la imagen original", anchor="center")
        self.original_label.pack(fill="both", expand=True)

        self.result_label = ttk.Label(result_box, text="Aquí aparecerá el análisis", anchor="center")
        self.result_label.pack(fill="both", expand=True)

        analysis_box = ttk.LabelFrame(right, text="Análisis", padding=8)
        analysis_box.pack(fill="x", pady=(10, 0))

        self.analysis_text = tk.Text(analysis_box, height=12, wrap="word")
        self.analysis_text.pack(fill="x", expand=False)
        self.write_analysis(
            "Selecciona una foto del ojo y presiona 'Analizar esclerótica/venas rojas'.\n\n"
            "La zona amarilla marca la esclerótica aproximada y la zona roja marca venas/enrojecimiento detectado."
        )

        status_bar = ttk.Label(self.root, textvariable=self.status, anchor="w", padding=6)
        status_bar.pack(side="bottom", fill="x")

    def select_image(self):
        filetypes = [
            ("Imágenes soportadas", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.jp2 *.j2k *.heic *.heif"),
            ("Todos los archivos", "*.*"),
        ]
        selected = filedialog.askopenfilename(title="Seleccionar foto externa del ojo", filetypes=filetypes)
        if not selected:
            return

        self.selected_path.set(selected)
        try:
            img = load_image_any_format(selected)
            img, _ = resize_for_processing(img, max_side=1200)
            self.set_image(self.original_label, img, "original")
            self.result_label.config(image="", text="Presiona analizar para ver el resultado")
            self.current_result_tk = None
            self.write_analysis("Imagen cargada correctamente. Presiona 'Analizar esclerótica/venas rojas'.")
            self.status.set("Imagen cargada.")
        except Exception as exc:
            messagebox.showerror("Error al cargar imagen", str(exc))

    def analyze(self):
        image_path = self.selected_path.get()
        if not image_path or image_path == "Ninguna imagen seleccionada":
            messagebox.showwarning("Falta imagen", "Primero selecciona una foto del ojo.")
            return

        try:
            result = analyze_external_eye(image_path, output_dir=str(self.output_dir))
            self.last_result = result
            self.set_image(self.original_label, result["original_bgr"], "original")
            self.set_image(self.result_label, result["overlay_bgr"], "result")
            self.write_analysis(self.format_analysis(result))
            self.status.set(f"Análisis completado. Resultado guardado en: {result['output_path']}")
        except Exception as exc:
            messagebox.showerror("Error en análisis", str(exc))
            self.status.set("Error en el análisis.")

    def set_image(self, label: ttk.Label, img_bgr: np.ndarray, target: str):
        if not PIL_AVAILABLE:
            label.config(text="Instala Pillow para visualizar imágenes: pip install pillow")
            return

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((460, 430), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        label.config(image=tk_img, text="")

        if target == "original":
            self.current_original_tk = tk_img
        else:
            self.current_result_tk = tk_img

    def write_analysis(self, text: str):
        self.analysis_text.config(state="normal")
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", text)
        self.analysis_text.config(state="disabled")

    def format_analysis(self, result: Dict) -> str:
        return (
            "ANÁLISIS DE OJO EXTERNO\n"
            "------------------------\n"
            f"Imagen: {result['image_path']}\n"
            f"Área aproximada de esclerótica detectada: {result['sclera_area']} px\n"
            f"Área rojiza/venas detectadas: {result['red_area']} px\n"
            f"Índice de enrojecimiento: {result['redness_percent']:.2f}%\n"
            f"Nivel orientativo: {result['level']}\n\n"
            f"Interpretación: {result['recommendation']}\n\n"
            "ADVERTENCIA IMPORTANTE:\n"
            "Este resultado NO confirma diabetes. Solo mide enrojecimiento superficial visible en la foto.\n"
            "La diabetes se confirma con pruebas de glucosa/HbA1c y la retinopatía diabética se evalúa con fondo de ojo/retinografía.\n\n"
            f"Resultado guardado en:\n{result['output_path']}\n\n"
            f"Máscara de esclerótica:\n{result['sclera_mask_path']}\n"
            f"Máscara roja:\n{result['red_mask_path']}\n"
            f"CSV resumen:\n{result['csv_path']}\n"
        )

    def open_results_folder(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        folder = str(self.output_dir.resolve())
        try:
            system = platform.system().lower()
            if system == "windows":
                os.startfile(folder)  # type: ignore[attr-defined]
            elif system == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as exc:
            messagebox.showinfo("Carpeta de resultados", f"Carpeta: {folder}\n\nNo se pudo abrir automáticamente: {exc}")


def main():
    if not PIL_AVAILABLE:
        print("Advertencia: Pillow no está instalado. Ejecuta: pip install pillow")

    root = tk.Tk()
    ExternalEyeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
