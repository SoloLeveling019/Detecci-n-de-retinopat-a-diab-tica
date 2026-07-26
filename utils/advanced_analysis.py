import numpy as np

try:
    from skimage.measure import label, regionprops
    import scipy.ndimage as ndi
    import matplotlib.pyplot as plt
    HAS_ADVANCED_LIBS = True
except ImportError:
    HAS_ADVANCED_LIBS = False

def clean_mask_holes(binary_mask: np.ndarray) -> np.ndarray:
    """
    Usa scipy para rellenar huecos en una máscara binaria.
    """
    if not HAS_ADVANCED_LIBS:
        return binary_mask
    
    # Asegurar que la máscara sea booleana
    boolean_mask = binary_mask > 0
    filled = ndi.binary_fill_holes(boolean_mask)
    return (filled * 255).astype(np.uint8)

def filter_regions_by_geometry(binary_mask: np.ndarray, min_area: int = 0, min_solidity: float = 0.0) -> np.ndarray:
    """
    Usa skimage para filtrar regiones conectadas basándose en sus propiedades geométricas (sin IA).
    Ideal para descartar falsos positivos de conjuntivitis o pterigión.
    """
    if not HAS_ADVANCED_LIBS:
        return binary_mask
        
    labeled_mask = label(binary_mask > 0)
    regions = regionprops(labeled_mask)
    
    clean_mask = np.zeros_like(binary_mask)
    
    for region in regions:
        if region.area >= min_area and region.solidity >= min_solidity:
            # Mantener esta región
            coords = region.coords
            clean_mask[coords[:, 0], coords[:, 1]] = 255
            
    return clean_mask

def plot_area_comparison(sclera_area: int, red_area: int, output_path: str):
    """
    Genera un gráfico simple de barras comparando áreas usando matplotlib.
    """
    if not HAS_ADVANCED_LIBS:
        return
        
    non_red_area = max(sclera_area - red_area, 0)
        
    plt.figure(figsize=(7, 5))
    labels = ["Esclerótica total", "Área roja detectada", "Área no roja"]
    values = [sclera_area, red_area, non_red_area]
    colors = ['#cccccc', '#ff4c4c', '#a8d5ba']
    
    plt.bar(labels, values, color=colors)
    plt.title("Comparación de Áreas Detectadas (Pixeles)")
    plt.ylabel("Cantidad de Pixeles")
    
    # Añadir los valores sobre las barras
    for i, v in enumerate(values):
        plt.text(i, v + (sclera_area*0.02), str(v), ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
