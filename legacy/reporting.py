import csv
from pathlib import Path
from typing import Dict

def save_to_csv(csv_path: Path, result_data: Dict):
    """Guarda una fila en el archivo CSV."""
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "imagen", "salida", "area_esclerotica_px", "area_roja_px",
                "porcentaje_enrojecimiento", "nivel", "deteccion_iris", "nota"
            ])
        writer.writerow([
            result_data["image_path"], 
            result_data["output_path"], 
            result_data["sclera_area"], 
            result_data["red_area"],
            f"{result_data['redness_percent']:.4f}", 
            result_data["level"],
            result_data.get("iris_detection_mode", "N/A"),
            "Resultado orientativo."
        ])

def format_analysis_text(result_data: Dict) -> str:
    """Formatea el texto que se mostrará en la interfaz."""
    base_text = (
        "ANÁLISIS DE OJO EXTERNO\n"
        "------------------------\n"
        f"Imagen: {result_data['image_path']}\n"
        f"Detección del iris: {result_data.get('iris_detection_mode', 'N/A')}\n"
        f"Área aproximada de esclerótica detectada: {result_data['sclera_area']} px\n"
        f"Área rojiza/venas detectadas: {result_data['red_area']} px\n"
        f"Índice de enrojecimiento: {result_data['redness_percent']:.2f}%\n"
        f"Resultado orientativo: Posible conjuntivitis - Nivel {result_data['level']}\n\n"
        f"Interpretación: {result_data['recommendation']}\n\n"
    )
    
    if result_data.get("warning"):
        base_text += f"ADVERTENCIA DE PROCESAMIENTO:\n{result_data['warning']}\n\n"
        
    # Añadimos los placeholders para mostrar que el resto de los módulos no detectaron
    base_text += (
        "Posible pterigión: no implementado\n"
        "Posible catarata visible: no implementado\n"
        "Blefaritis / Orzuelo: no implementado\n\n"
    )
    
    base_text += (
        "ADVERTENCIA IMPORTANTE:\n"
        "Este sistema es un prototipo académico y no reemplaza el diagnóstico médico.\n"
        "No confirma diabetes ni otras patologías de forma clínica.\n\n"
    )
    
    base_text += (
        f"Resultado guardado en:\n{result_data['output_path']}\n"
    )
    return base_text
