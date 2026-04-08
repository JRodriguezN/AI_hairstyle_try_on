import cv2
import numpy as np

def analyze_hair(mask: np.ndarray, face_mask: np.ndarray):
    """
    Analiza el cabello:
    - Detecta si es calvo
    - Detecta longitud
    - Detecta textura
    """

    hair_pixels = np.sum(mask > 0)

    if hair_pixels < 1000:
        return {
            "is_bald": True,
            "length": "none",
            "texture": "none",
            "type": "bald"
        }

    ys, xs = np.where(mask > 0)

    hair_height = np.max(ys) - np.min(ys)
    hair_width = np.max(xs) - np.min(xs)
    print(hair_height, hair_width)

    ys_face, xs_face = np.where(face_mask > 0)

    if len(ys_face) == 0:
        return {"type": "unknown"}

    face_height = np.max(ys_face) - np.min(ys_face)
    print(face_height)
    

    length_ratio = hair_height / (face_height + 1)

    density = hair_pixels / (hair_height * (hair_width + 1))
    print("Length ratio: ", length_ratio)
    if density < 0.15:
        return {
            "is_bald": True,
            "length": "none",
            "texture": "none",
            "type": "bald"
        }

    # LONGITUD
    if length_ratio < 0.9:
        length = "short"
    elif length_ratio < 1.5:
        length = "medium"
    else:
        length = "long"

    #TEXTURA
    edges = cv2.Canny(mask.astype(np.uint8), 50, 150)
    edge_density = np.sum(edges > 0) / hair_pixels

    if edge_density < 0.05:
        texture = "straight"
    elif edge_density < 0.12:
        texture = "wavy"
    else:
        texture = "curly"

    return {
        "is_bald": False,
        "length": length,
        "texture": texture,
        "type": f"{length}_{texture}",
        "density": density,
        "edge_density": edge_density
    }