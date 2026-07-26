from pathlib import Path

# Tamaño máximo para procesamiento de imágenes para evitar demoras
MAX_IMAGE_SIDE = 1200

# Directorio base para resultados
BASE_OUTPUT_DIR = Path.cwd() / "resultados_ojo_externo"

# Tipos de imágenes soportadas
SUPPORTED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
    ".webp", ".jp2", ".j2k", ".heic", ".heif"
)

# Constantes de configuración de UI
APP_TITLE = "Analizador Ocular Modular"
APP_MIN_WIDTH = 1100
APP_MIN_HEIGHT = 700
