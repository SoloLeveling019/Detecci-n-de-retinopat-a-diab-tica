import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import platform
import os
import subprocess
import cv2
import numpy as np

from config import BASE_OUTPUT_DIR, SUPPORTED_EXTENSIONS, MAX_IMAGE_SIDE, APP_TITLE, APP_MIN_WIDTH, APP_MIN_HEIGHT
from core.image_io import load_image_any_format, PIL_AVAILABLE
from core.preprocess import resize_for_processing
from core.sclera_segmentation import build_sclera_masks, build_red_vessel_mask
from diseases.conjunctivitis import analyze_conjunctivitis
from utils.helpers import save_img

if PIL_AVAILABLE:
    from PIL import Image, ImageTk

class ModularEyeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(f"{APP_MIN_WIDTH}x{APP_MIN_HEIGHT}")
        self.root.minsize(APP_MIN_WIDTH - 100, APP_MIN_HEIGHT - 100)

        self.selected_path = tk.StringVar(value="Ninguna imagen seleccionada")
        self.status = tk.StringVar(value="Selecciona una imagen y un modo de análisis.")
        self.output_dir = BASE_OUTPUT_DIR

        self.current_original_tk = None
        self.current_result_tk = None

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        # Panel izquierdo (Controles)
        controls = ttk.LabelFrame(main, text="Configuración y Controles", padding=12)
        controls.pack(side="left", fill="y", padx=(0, 10))

        ttk.Button(controls, text="Seleccionar Imagen", command=self.select_image).pack(fill="x", pady=(0, 10))
        
        ttk.Label(controls, text="Imagen seleccionada:").pack(anchor="w")
        ttk.Label(controls, textvariable=self.selected_path, wraplength=260).pack(fill="x", pady=(0, 15))

        ttk.Separator(controls).pack(fill="x", pady=15)
        
        # Botones de Análisis Generales
        ttk.Label(controls, text="Módulos de Análisis:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        
        # Botón Guiado (Principal Recomendado)
        btn_guided = ttk.Button(
            controls, 
            text="Modo guiado / corrección manual (Recomendado)", 
            command=lambda: self.run_flow("general_guiado")
        )
        btn_guided.pack(fill="x", pady=5)
        
        # Botón Automático Completo
        btn_auto = ttk.Button(
            controls, 
            text="Análisis automático completo", 
            command=lambda: self.run_auto_general_analysis()
        )
        btn_auto.pack(fill="x", pady=5)
        
        # Botón Retina
        ttk.Separator(controls).pack(fill="x", pady=10)
        ttk.Label(controls, text="Módulo Retina:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        btn_retina = ttk.Button(controls, text="Análisis de fondo de ojo / retinografía", command=lambda: self.run_retina_flow())
        btn_retina.pack(fill="x", pady=(0, 15))
        
        ttk.Button(controls, text="Abrir carpeta de resultados", command=self.open_results_folder).pack(fill="x", pady=(0, 15))

        # Panel derecho (Imágenes y Resultados)
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        image_frame = ttk.Frame(right)
        image_frame.pack(fill="both", expand=True)

        original_box = ttk.LabelFrame(image_frame, text="Imagen Original", padding=8)
        original_box.pack(side="left", fill="both", expand=True, padx=(0, 5))

        result_box = ttk.LabelFrame(image_frame, text="Resultado (Overlay)", padding=8)
        result_box.pack(side="left", fill="both", expand=True, padx=(5, 0))

        self.original_label = ttk.Label(original_box, text="Carga una imagen", anchor="center")
        self.original_label.pack(fill="both", expand=True)

        self.result_label = ttk.Label(result_box, text="Esperando análisis...", anchor="center")
        self.result_label.pack(fill="both", expand=True)

        analysis_box = ttk.LabelFrame(right, text="Reporte de Análisis", padding=8)
        analysis_box.pack(fill="x", pady=(10, 0))

        self.analysis_text = tk.Text(analysis_box, height=14, wrap="word")
        self.analysis_text.pack(fill="x", expand=False)
        self.write_analysis("Selecciona una foto, elige el módulo y presiona el botón de analizar.")

        status_bar = ttk.Label(self.root, textvariable=self.status, anchor="w", padding=6)
        status_bar.pack(side="bottom", fill="x")

    def select_image(self):
        filetypes = [
            ("Imágenes soportadas", " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS)),
            ("Todos los archivos", "*.*"),
        ]
        selected = filedialog.askopenfilename(title="Seleccionar imagen", filetypes=filetypes)
        if not selected:
            return

        self.selected_path.set(selected)
        try:
            img = load_image_any_format(selected)
            img, _ = resize_for_processing(img, max_side=MAX_IMAGE_SIDE)
            self.set_image(self.original_label, img, "original")
            self.result_label.config(image="", text="Presiona el botón de análisis")
            self.current_result_tk = None
            self.write_analysis("Imagen cargada correctamente. Listo para analizar.")
            self.status.set("Imagen cargada.")
        except Exception as exc:
            messagebox.showerror("Error al cargar imagen", str(exc))
            
    def manual_iris_selection_tool(self, img_bgr: np.ndarray):
        clone = img_bgr.copy()
        pts = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(pts) < 2:
                    pts.append((x, y))
                    cv2.circle(clone, (x, y), 3, (0, 255, 0), -1)
                    if len(pts) == 2:
                        cx, cy = pts[0]
                        bx, by = pts[1]
                        r = int(np.hypot(bx - cx, by - cy))
                        cv2.circle(clone, (cx, cy), r, (255, 0, 0), 2)
                    cv2.imshow("Marcar Iris (1: Centro, 2: Borde) - Enter para confirmar, R para repetir", clone)
                    
        cv2.namedWindow("Marcar Iris (1: Centro, 2: Borde) - Enter para confirmar, R para repetir")
        cv2.setMouseCallback("Marcar Iris (1: Centro, 2: Borde) - Enter para confirmar, R para repetir", mouse_callback)
        
        while True:
            cv2.imshow("Marcar Iris (1: Centro, 2: Borde) - Enter para confirmar, R para repetir", clone)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13 or key == 32:  # Enter or Space
                if len(pts) == 2:
                    break
                else:
                    messagebox.showwarning("Atención", "Debe marcar 2 puntos (centro y borde).")
            elif key == ord('r') or key == ord('R'):
                clone = img_bgr.copy()
                pts.clear()
            elif key == 27: # Esc
                cv2.destroyWindow("Marcar Iris (1: Centro, 2: Borde) - Enter para confirmar, R para repetir")
                return None
                
            if cv2.getWindowProperty("Marcar Iris (1: Centro, 2: Borde) - Enter para confirmar, R para repetir", cv2.WND_PROP_VISIBLE) < 1:
                if len(pts) == 2:
                    break
                return None

        cv2.destroyWindow("Marcar Iris (1: Centro, 2: Borde) - Enter para confirmar, R para repetir")
        if len(pts) == 2:
            cx, cy = pts[0]
            bx, by = pts[1]
            r = int(np.hypot(bx - cx, by - cy))
            
            h, w = img_bgr.shape[:2]
            if r < 10 or r > min(h, w):
                messagebox.showwarning("Atención", "El radio del iris seleccionado es inválido (muy pequeño o muy grande).")
                return None
            return (cx, cy, r)
        return None

    def manual_eye_opening_selection(self, img_bgr: np.ndarray):
        clone = img_bgr.copy()
        pts = []
        window_name = "Abertura Ocular"
        
        def draw_preview():
            preview = clone.copy()
            # Dibujar puntos y líneas
            for i, p in enumerate(pts):
                cv2.circle(preview, p, 3, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(preview, pts[i-1], p, (0, 200, 0), 1)
            # Dibujar polígono cerrado si hay más de 2 puntos
            if len(pts) > 2:
                cv2.line(preview, pts[-1], pts[0], (0, 200, 0), 1)
            
            # Dibujar overlay celeste si hay al menos 6 puntos
            if len(pts) >= 6:
                poly = np.array(pts, dtype=np.int32)
                overlay = preview.copy()
                cv2.fillPoly(overlay, [poly], (255, 255, 0))
                cv2.addWeighted(overlay, 0.4, preview, 0.6, 0, preview)
                
            # Textos de instrucciones
            cv2.putText(preview, "Marca el borde de la abertura ocular en sentido horario.", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(preview, "No marques piel ni parpado.", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(preview, f"Puntos: {len(pts)}/12 (Minimo 6). Click izq: agregar, Click der: deshacer.", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            if len(pts) >= 6:
                cv2.putText(preview, "Enter/Espacio: Confirmar  |  R: Reiniciar  |  ESC: Cancelar", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return preview

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(pts) < 12:
                    pts.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(pts) > 0:
                    pts.pop()
                    
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while True:
            preview = draw_preview()
            cv2.imshow(window_name, preview)
            
            key = cv2.waitKey(20) & 0xFF
            if key == 13 or key == 32:  # Enter or Space
                if len(pts) >= 6:
                    break
                else:
                    messagebox.showwarning("Atención", "Debe marcar mínimo 6 puntos.")
            elif key == ord('r') or key == ord('R'):
                pts.clear()
            elif key == 27: # Esc
                cv2.destroyWindow(window_name)
                return None
                
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                if len(pts) >= 6:
                    break
                return None

        cv2.destroyWindow(window_name)
        
        if len(pts) >= 6:
            mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
            poly = np.array(pts, dtype=np.int32)
            cv2.fillPoly(mask, [poly], 255)
            return mask
        return None


    def run_auto_general_analysis(self):
        image_path = self.selected_path.get()
        if not image_path or image_path == "Ninguna imagen seleccionada":
            messagebox.showwarning("Falta imagen", "Primero selecciona una foto.")
            return
            
        try:
            from core.preprocess import resize_to_standard
            from core.iris_pupil_detection import detect_pupil_center
            
            original_full = load_image_any_format(image_path)
            img_screen, scale = resize_for_processing(original_full, max_side=MAX_IMAGE_SIDE)
            
            self.status.set("Detectando región ocular...")
            self.root.update_idletasks()
            
            # Detectar pupila inicial en imagen completa para anclar el ROI
            cx, cy, r, found, _ = detect_pupil_center(img_screen)
            
            if not found:
                messagebox.showwarning(
                    "Fallo de detección", 
                    "No evaluable automáticamente. No se detectó pupila/iris con suficiente confianza.\nUse una imagen más clara o active el 'Modo guiado / corrección manual'."
                )
                self.status.set("Fallo detección automática.")
                return
                
            # Auto ROI crop: ancho ~5.5r, alto ~3.2r
            crop_w = int(r * 5.5)
            crop_h = int(r * 3.2)
            
            x_min = max(0, int(cx - crop_w / 2))
            x_max = min(img_screen.shape[1], int(cx + crop_w / 2))
            y_min = max(0, int(cy - crop_h / 2))
            y_max = min(img_screen.shape[0], int(cy + crop_h / 2))
            
            # Proyectar a escala original
            ox, oy = int(x_min / scale), int(y_min / scale)
            ow, oh = int((x_max - x_min) / scale), int((y_max - y_min) / scale)
            
            crop_img = original_full[oy:oy+oh, ox:ox+ow]
            
            # 2. Normalizar a 800x400
            std_img, content_mask, std_scale, offsets = resize_to_standard(crop_img)
            
            self.status.set("Analizando...")
            self.root.update_idletasks()
            
            # Guardamos el ROI original antes de analizar
            self.auto_eye_roi = crop_img.copy()

            self.run_analysis_pipeline(std_img, image_path, content_mask, manual_iris_info=None, manual_eye_opening_mask=None, module_name="automatico")
            
        except Exception as exc:
            cv2.destroyAllWindows()
            messagebox.showerror("Error", f"Ocurrió un error: {exc}")

    def run_retina_flow(self):
        image_path = self.selected_path.get()
        if not image_path or image_path == "Ninguna imagen seleccionada":
            messagebox.showwarning("Falta imagen", "Primero selecciona una foto.")
            return
            
        try:
            original_full = load_image_any_format(image_path)
            img_screen, _ = resize_for_processing(original_full, max_side=MAX_IMAGE_SIDE)
            
            self.status.set("Preparando retinografía...")
            self.root.update_idletasks()
            
            self.run_analysis_pipeline(img_screen, image_path, content_mask=None, manual_iris_info=None, manual_eye_opening_mask=None, module_name="retina")
            
        except Exception as exc:
            cv2.destroyAllWindows()
            messagebox.showerror("Error", f"Ocurrió un error: {exc}")

    def run_flow(self, module_name: str):
        image_path = self.selected_path.get()
        if not image_path or image_path == "Ninguna imagen seleccionada":
            messagebox.showwarning("Falta imagen", "Primero selecciona una foto.")
            return
            
        try:
            from core.preprocess import resize_to_standard
            
            original_full = load_image_any_format(image_path)
            img_screen, scale = resize_for_processing(original_full, max_side=MAX_IMAGE_SIDE)
            
            # 1. Recorte manual de la región del ojo
            x, y, w, h = cv2.selectROI("Seleccionar ROI", img_screen, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Seleccionar ROI")
            
            if w == 0 or h == 0:
                messagebox.showwarning("ROI inválida", "No se seleccionó una región válida del ojo.")
                return
                
            ox, oy = int(x / scale), int(y / scale)
            ow, oh = int(w / scale), int(h / scale)
            
            if ow < 50 or oh < 25:
                messagebox.showwarning("Recorte inválido", "El recorte es demasiado pequeño.")
                return
                
            crop_img = original_full[oy:oy+oh, ox:ox+ow]
            
            # 2. Normalizar a 800x400
            std_img, content_mask, std_scale, offsets = resize_to_standard(crop_img)
            
            # 3. Control de Calidad según el módulo
            manual_iris_info = None
            manual_eye_opening_mask = None
            
            if module_name == "general_guiado":
                # MODO GUIADO: Sin chequeos automáticos, el usuario asume control total
                messagebox.showinfo("Paso 1: Iris", "Por favor, marque el centro y el borde del iris.")
                manual_iris_info = self.manual_iris_selection_tool(std_img)
                if manual_iris_info is None:
                    messagebox.showwarning("Cancelado", "Selección de iris cancelada.")
                    return
                
                messagebox.showinfo("Paso 2: Abertura ocular", "Ahora marque la abertura del ojo (entre 6 y 12 clics alrededor del borde visible).")
                manual_eye_opening_mask = self.manual_eye_opening_selection(std_img)
                if manual_eye_opening_mask is None:
                    messagebox.showwarning("Cancelado", "Selección de abertura cancelada.")
                    return

            self.status.set("Analizando...")
            self.root.update_idletasks()

            self.run_analysis_pipeline(std_img, image_path, content_mask, manual_iris_info, manual_eye_opening_mask, module_name)
            
        except Exception as exc:
            cv2.destroyAllWindows()
            messagebox.showerror("Error", f"Ocurrió un error: {exc}")

    def set_image(self, label: ttk.Label, img_bgr: np.ndarray, target: str):
        if not PIL_AVAILABLE:
            label.config(text="Instala Pillow para visualizar imágenes: pip install pillow")
            return

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        max_w, max_h = 460, 380 
        pil_img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
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

    def open_results_folder(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        folder = str(self.output_dir.resolve())
        try:
            system = platform.system().lower()
            if system == "windows":
                os.startfile(folder)
            elif system == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as exc:
            messagebox.showinfo("Carpeta de resultados", f"Carpeta: {folder}\n\nNo se pudo abrir automáticamente: {exc}")
            
    def run_analysis_pipeline(self, img: np.ndarray, image_path: str, content_mask: np.ndarray = None, manual_iris_info: tuple = None, manual_eye_opening_mask: np.ndarray = None, module_name: str = "general_guiado"):
        try:
            stem = Path(image_path).stem
            import time
            out_dir = self.output_dir / f"{stem}_{int(time.time())}"
            out_dir.mkdir(parents=True, exist_ok=True)

            if module_name == "retina":
                from diseases.diabetic_retinopathy_fundus import analyze_retina_fundus
                from utils.drawing import draw_retinopathy_overlay_clean
                
                # Ask user if they want to crop the retina
                do_crop = messagebox.askyesno("Recorte de Retinografía", "¿Desea recortar la retinografía antes de analizar?\n\n(Recomendado si hay más de un ojo, bordes grandes o texto)")
                if do_crop:
                    # Mostrar ventana y esperar a que el usuario recorte
                    roi = cv2.selectROI("Seleccione la retina (ENTER/SPACE para confirmar, C para cancelar)", img, fromCenter=False, showCrosshair=True)
                    cv2.destroyWindow("Seleccione la retina (ENTER/SPACE para confirmar, C para cancelar)")
                    
                    x, y, w, h = roi
                    if w > 0 and h > 0:
                        img = img[y:y+h, x:x+w]
                        # Rescale max side if needed, just in case
                        img, _ = resize_for_processing(img, MAX_IMAGE_SIDE)
                
                res = analyze_retina_fundus(img)
                
                if res["status"] == "ok":
                    count = res["candidate_count"]
                    overlay = draw_retinopathy_overlay_clean(img, res["microaneurysm_candidates"], res["optic_disc_mask"])
                    
                    save_img(out_dir / "fundus_preprocessed.png", res["fundus_preprocessed"])
                    save_img(out_dir / "fundus_fov_mask.png", res["fundus_fov_mask"])
                    save_img(out_dir / "optic_disc_mask.png", res["optic_disc_mask"])
                    save_img(out_dir / "microaneurysm_candidates.png", res["microaneurysm_candidates"])
                    
                    # Imágenes de depuración de métodos clásicos
                    if "otsu_mask_debug" in res:
                        save_img(out_dir / "otsu_mask_debug.png", res["otsu_mask_debug"])
                    if "mser_candidates_debug" in res:
                        save_img(out_dir / "mser_candidates_debug.png", res["mser_candidates_debug"])
                    if "hough_fundus_circle" in res:
                        save_img(out_dir / "hough_fundus_circle.png", res["hough_fundus_circle"])
                        
                    save_img(out_dir / "overlay_retinopathy.png", overlay)
                    
                    result_data = {
                        "image_path": image_path,
                        "output_path": str(out_dir / "overlay_retinopathy.png"),
                        "microaneurysm_candidates": count
                    }
                    csv_path = out_dir / "resumen_retinopatia.csv"
                    import csv
                    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=result_data.keys())
                        writer.writeheader()
                        writer.writerow(result_data)
                        
                    reporte_texto = (
                        "REPORTE DE FONDO DE OJO / RETINOGRAFÍA\n"
                        "----------------------------------------\n\n"
                        f"{res['message']}\n\n"
                        "Advertencia: Este sistema no reemplaza el diagnóstico médico. Consulte a un especialista."
                    )
                    self.set_image(self.original_label, img, "original")
                    self.set_image(self.result_label, overlay, "result")
                    self.write_analysis(reporte_texto)
                    self.status.set(f"Análisis de retina completado. Resultados en: {out_dir.name}")
                    return
                else:
                    self.write_analysis(f"Error en análisis de retina:\n{res.get('message', 'Fallo desconocido.')}")
                    self.status.set("Error en análisis de retina.")
                    messagebox.showerror("Error de Análisis", res.get("message", "Fallo desconocido."))
                    return

            # 1. Segmentación Base (Esclerótica, Iris, Pupila)
            sclera_dict = build_sclera_masks(img, content_mask, manual_iris_info, manual_eye_opening_mask)
            
            sclera_roi_mask = sclera_dict.get("sclera_roi_mask")
            white_sclera_mask = sclera_dict.get("white_sclera_mask")
            iris_info = manual_iris_info if manual_iris_info else sclera_dict.get("iris_info")
            iris_pupil_mask = sclera_dict.get("iris_pupil_mask")
            upper_eyelid_band_mask = sclera_dict.get("upper_eyelid_band_mask")
            lower_eyelid_band_mask = sclera_dict.get("lower_eyelid_band_mask")
            
            if not sclera_dict.get("success", False) or sclera_roi_mask is None:
                messagebox.showwarning("Fallo de Detección", "No se pudo detectar correctamente la esclerótica. Intente usar el modo guiado.")
                self.write_analysis("Análisis abortado: Fallo de segmentación.")
                self.status.set("Fallo de segmentación.")
                return

            # Guardado Base
            save_img(out_dir / "normalized_eye_crop.png", img)
            
            # Si estamos en automático, prefijar nombres
            prefix = "auto_" if module_name == "automatico" else ""
            
            if hasattr(self, "auto_eye_roi") and self.auto_eye_roi is not None and module_name == "automatico":
                save_img(out_dir / "auto_eye_roi.png", self.auto_eye_roi)
                
            if manual_eye_opening_mask is not None:
                save_img(out_dir / f"{prefix}eye_opening_mask.png", manual_eye_opening_mask)
            elif module_name == "automatico":
                save_img(out_dir / "auto_eye_opening_mask.png", sclera_dict.get("eye_opening_mask", np.zeros_like(img[:,:,0])))

            save_img(out_dir / f"{prefix}iris_pupil_mask.png", iris_pupil_mask if iris_pupil_mask is not None else np.zeros_like(img[:,:,0]))
            save_img(out_dir / f"{prefix}sclera_roi_mask.png", sclera_roi_mask)
            save_img(out_dir / "white_sclera_mask.png", white_sclera_mask)
            
            if upper_eyelid_band_mask is not None and lower_eyelid_band_mask is not None:
                combined_band = cv2.bitwise_or(upper_eyelid_band_mask, lower_eyelid_band_mask)
                save_img(out_dir / f"{prefix}eyelid_band_mask.png", combined_band)
            
            if "debug_imgs" in sclera_dict:
                for k, v in sclera_dict["debug_imgs"].items():
                    save_img(out_dir / f"{k}.png", v)

            # --- VARIABLES POR DEFECTO ---
            pterygium_level = "No evaluado en modo automático. Use el modo guiado recomendado."
            pterygium_percent = 0.0
            p_inv_linear = 0.0
            p_inv_area = 0.0
            pterygium_confidence = 0
            p_side = "N/A"
            pterygium_message = "No detectado."
            
            cataract_level = "No evaluado"
            cataract_opacity_percent = 0.0
            
            blepharitis_level = "No evaluado en modo automático."
            stye_level = "No evaluado en modo automático."
            
            p_mask = None
            b_mask = None
            s_mask = None
            res_cataract = {"pupil_zone_mask": None, "opacity_mask": None}
            
            # --- VALIDACIÓN DE PÁRPADOS ---
            valid_eyelids = False
            if module_name == "general_guiado" and manual_eye_opening_mask is not None:
                eye_opening_area = int(np.count_nonzero(manual_eye_opening_mask))
                band_area = 0
                if upper_eyelid_band_mask is not None and lower_eyelid_band_mask is not None:
                    band_area = int(np.count_nonzero(upper_eyelid_band_mask)) + int(np.count_nonzero(lower_eyelid_band_mask))
                if eye_opening_area > 0 and (band_area / eye_opening_area) >= 0.15:
                    valid_eyelids = True

            # ==========================================
            # ORDEN LÓGICO DEL PIPELINE
            # ==========================================
            
            # --- 1. PTERIGIÓN (Modo Guiado y Automático) ---
            from diseases.pterygium import analyze_pterygium
            from utils.drawing import draw_pterygium_overlay_clean
            
            # Determinar confianza global
            global_confidence = "Alta" if module_name == "general_guiado" else "Media"
            h, w = img.shape[:2]
            if np.count_nonzero(sclera_roi_mask) < 0.05 * h * w:
                global_confidence = "Baja"
                
            res_pterygium = analyze_pterygium(img, sclera_dict.get("eye_opening_mask", manual_eye_opening_mask), iris_info)
            pterygium_level = res_pterygium["level"]
            
            if global_confidence == "Baja" and pterygium_level in ["Sospecha alta", "Sospecha moderada"]:
                pterygium_level = "No evaluable (Baja Confianza)"
                pterygium_message = "Detección de ojo pobre, no se puede clasificar carnosidad con seguridad."
                p_mask = None
            else:
                pterygium_percent = res_pterygium.get("pterygium_percent", 0.0)
                pterygium_confidence = res_pterygium.get("confidence", 0)
                p_inv_linear = res_pterygium.get("invasion_linear_percent", 0.0)
                p_inv_area = res_pterygium.get("invasion_area_percent", 0.0)
                p_side = res_pterygium.get("affected_side", "N/A")
                pterygium_message = res_pterygium.get("message", "")
                
                pterygium_valid = (pterygium_level in ["Sospecha leve", "Sospecha moderada", "Sospecha alta"])
                p_mask = res_pterygium["pterygium_mask"] if pterygium_valid else None
                
                if p_mask is not None:
                    save_img(out_dir / "pterygium_mask.png", p_mask)
                    
            overlay_pterygium = draw_pterygium_overlay_clean(img, sclera_roi_mask, p_mask)
            save_img(out_dir / "overlay_pterygium.png", overlay_pterygium)

            # --- 2. CATARATA VISIBLE ---
            if iris_info:
                from diseases.cataract import analyze_cataract_visible
                from utils.drawing import draw_cataract_overlay_clean
                res_cataract = analyze_cataract_visible(img, iris_info, iris_pupil_mask, p_mask)
                cataract_level = res_cataract["level"]
                cataract_opacity_percent = res_cataract["opacity_percent"]
                cataract_message = res_cataract.get("message", "")
                
                if res_cataract["pupil_zone_mask"] is not None:
                    save_img(out_dir / "cataract_pupil_mask.png", res_cataract["pupil_zone_mask"])
                if res_cataract["opacity_mask"] is not None:
                    save_img(out_dir / "cataract_opacity_mask.png", res_cataract["opacity_mask"])
                    
                overlay_catarata = draw_cataract_overlay_clean(img, res_cataract.get("pupil_zone_mask"), res_cataract.get("opacity_mask"))
                save_img(out_dir / "overlay_catarata.png", overlay_catarata)
            else:
                cataract_message = "No se pudo delimitar la pupila."

            # --- 3. CONJUNTIVITIS ---
            sclera_area = int(np.count_nonzero(sclera_roi_mask))
            red_mask_raw = build_red_vessel_mask(img, sclera_roi_mask)
            
            # Restar área de pterigión para no ensuciar conjuntivitis
            if p_mask is not None:
                red_mask = cv2.bitwise_and(red_mask_raw, cv2.bitwise_not(p_mask))
            else:
                red_mask = red_mask_raw
                
            center_info = iris_info if iris_info else (img.shape[1]//2, img.shape[0]//2, 0)
            res_conj = analyze_conjunctivitis(sclera_roi_mask, red_mask, center_info)
            
            # Si se excluyó mucho rojo por la carnosidad, modificar el resultado
            red_area_raw = int(np.count_nonzero(red_mask_raw))
            red_area_clean = int(np.count_nonzero(red_mask))
            
            if red_area_raw > 0 and red_area_clean < (red_area_raw * 0.3) and p_mask is not None and pterygium_percent > 2.0:
                conjuntivitis_level = "Enrojecimiento asociado a posible carnosidad."
            else:
                conjuntivitis_level = res_conj["level"]
                
            redness_percent = res_conj["redness_percent"]
            
            save_img(out_dir / "red_mask.png", red_mask)
            from utils.drawing import draw_conjunctivitis_overlay_clean
            overlay_conjuntivitis = draw_conjunctivitis_overlay_clean(img, sclera_roi_mask, white_sclera_mask, red_mask)
            save_img(out_dir / "overlay_conjuntivitis.png", overlay_conjuntivitis)
            
            try:
                from utils.advanced_analysis import plot_area_comparison
                plot_area_comparison(sclera_area, red_area_clean, str(out_dir / "grafico_areas.png"))
            except Exception as e:
                print(f"Error al generar gráfico: {e}")

            # --- 4. BLEFARITIS Y ORZUELO (Guiado y Auto) ---
            if valid_eyelids or module_name == "automatico":
                from diseases.blepharitis import analyze_blepharitis
                from utils.drawing import draw_blepharitis_overlay_clean
                res_bleph = analyze_blepharitis(img, upper_eyelid_band_mask, lower_eyelid_band_mask)
                blepharitis_level = res_bleph["level"]
                
                if global_confidence == "Baja" and blepharitis_level in ["Sospecha alta", "Sospecha moderada"]:
                    blepharitis_level = "No evaluable (Baja Confianza)"
                    b_mask = None
                else:
                    b_mask = res_bleph["blepharitis_mask"]
                    save_img(out_dir / "blepharitis_mask.png", b_mask)
                    overlay_blepharitis = draw_blepharitis_overlay_clean(img, b_mask)
                    save_img(out_dir / "overlay_blepharitis.png", overlay_blepharitis)
            else:
                blepharitis_level = "No evaluado. Para analizar blefaritis debe seleccionarse el ojo completo incluyendo bordes de párpados y pestañas."
                
            if valid_eyelids or module_name == "automatico":
                from diseases.stye import analyze_stye
                from utils.drawing import draw_stye_overlay_clean
                res_stye = analyze_stye(img, upper_eyelid_band_mask, lower_eyelid_band_mask)
                stye_level = res_stye["level"]
                
                if global_confidence == "Baja" and stye_level in ["Sospecha alta", "Sospecha moderada"]:
                    stye_level = "No evaluable (Baja Confianza)"
                    s_mask = None
                else:
                    s_mask = res_stye["stye_mask"]
                    save_img(out_dir / "stye_mask.png", s_mask)
                    overlay_stye = draw_stye_overlay_clean(img, s_mask)
                    save_img(out_dir / "overlay_stye.png", overlay_stye)
            else:
                stye_level = "No evaluado. Para analizar orzuelo debe seleccionarse el ojo completo incluyendo párpados."

            # --- OVERLAY LIMPIO (Muestra Principal) ---
            from utils.drawing import draw_general_clean_overlay
            overlay_limpio = draw_general_clean_overlay(
                img, sclera_roi_mask, red_mask, 
                res_cataract.get("pupil_zone_mask"), 
                res_cataract.get("opacity_mask"),
                p_mask, b_mask, s_mask
            )
            save_img(out_dir / "overlay_general_limpio.png", overlay_limpio)
            
            # Reemplazar la vista mostrada en pantalla para que use overlay_limpio
            self.set_image(self.result_label, overlay_limpio, "result")

            # --- REPORTE Y CSV ---
            result_data = {
                "image_path": image_path,
                "output_path": str(out_dir / "overlay_general_limpio.png"),
                "modo_usado": "Automático completo" if module_name == "automatico" else "Asistido / Manual",
                "confianza_anatomica": global_confidence,
                "conjuntivitis_nivel": conjuntivitis_level,
                "conjuntivitis_porcentaje_rojo": f"{redness_percent:.2f}",
                "conjuntivitis_distribucion": res_conj.get("distribution", "N/A"),
                "conjuntivitis_sectores_afectados": res_conj.get("affected_sectors", 0),
                "conjuntivitis_lado_afectado": res_conj.get("affected_side", "N/A"),
                "catarata_nivel": cataract_level,
                "catarata_opacidad_porcentaje": f"{cataract_opacity_percent:.2f}",
                "pterigion_nivel": pterygium_level,
                "pterigion_porcentaje": f"{pterygium_percent:.2f}",
                "pterigion_invasion_lineal": f"{p_inv_linear:.2f}",
                "pterigion_invasion_area": f"{p_inv_area:.2f}",
                "pterigion_lado": p_side,
                "pterigion_confianza": pterygium_confidence,
                "blefaritis_nivel": blepharitis_level,
                "orzuelo_nivel": stye_level
            }
            
            csv_path = out_dir / "resumen_general.csv"
            import csv
            with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=result_data.keys())
                writer.writeheader()
                writer.writerow(result_data)

            # --- TEXTO FINAL ---
            dist_str = res_conj.get('distribution', 'N/A')
            sect_str = f"{res_conj.get('affected_sectors', 0)} de {res_conj.get('valid_sectors', 0)}"
            lado_str = res_conj.get('affected_side', 'N/A')
            
            reporte_texto = (
                "REPORTE GENERAL DE ANÁLISIS OCULAR\n"
                "----------------------------------\n\n"
                f"Modo usado: {'Automático completo' if module_name == 'automatico' else 'Asistido / Manual'}\n"
                f"Confianza de detección anatómica: {global_confidence}\n\n"
                "Conjuntivitis / enrojecimiento:\n"
                f"Resultado orientativo: {conjuntivitis_level}\n"
                f"Porcentaje de enrojecimiento: {redness_percent:.1f}%\n"
                f"Distribución: {dist_str}\n"
                f"Sectores afectados: {sect_str}\n"
                f"Lado afectado: {lado_str}\n\n"
                "Catarata visible:\n"
                f"Resultado orientativo: {cataract_level}\n"
                f"{cataract_message}\n"
                f"Opacidad pupilar: {cataract_opacity_percent:.1f}%\n\n"
                "Pterigión / carnosidad:\n"
                f"Resultado orientativo: {pterygium_level}\n"
                f"{pterygium_message}\n"
                f"Área sospechosa sobre zona ocular: {pterygium_percent:.1f}%\n"
                f"Invasión lineal hacia córnea: {p_inv_linear:.1f}%\n"
                f"Área corneal comprometida: {p_inv_area:.1f}%\n"
                f"Confianza geométrica: {pterygium_confidence}%\n"
                f"Lado afectado: {p_side}\n\n"
                "Blefaritis:\n"
                f"{blepharitis_level}\n\n"
                "Orzuelo:\n"
                f"{stye_level}\n\n"
                "ADVERTENCIA:\n"
                "Este sistema es un prototipo académico y no reemplaza el diagnóstico médico."
            )

            # Actualizar UI
            self.set_image(self.original_label, img, "original")
            self.set_image(self.result_label, overlay_limpio, "result")
            self.write_analysis(reporte_texto)
            self.status.set(f"Análisis general completado. Resultados en: {out_dir.name}")

        except Exception as exc:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error en análisis", f"Ocurrió un error inesperado:\n{str(exc)}")
            self.status.set("Error en el análisis.")
