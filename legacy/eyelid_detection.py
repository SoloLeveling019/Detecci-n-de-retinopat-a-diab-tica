from typing import Tuple

def estimate_palpebral_fissure(pupil_cy: int, img_h: int) -> Tuple[int, int]:
    """
    Estima los límites superior e inferior de la hendidura palpebral
    (la franja vertical donde es visible el ojo).
    Retorna: (y_superior, y_inferior)
    """
    # En una foto típica del ojo, la esclerótica visible suele estar
    # en una banda de +/- 20% a 25% del alto total alrededor de la pupila.
    y_top = max(0, int(pupil_cy - img_h * 0.22))
    y_bottom = min(img_h, int(pupil_cy + img_h * 0.22))
    return y_top, y_bottom
