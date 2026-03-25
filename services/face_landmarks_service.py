"""
Servicio de landmarks faciales para protección precisa del rostro.
Usa MediaPipe FaceLandmarker con el óvalo facial (contorno real, no bounding box).
Permite también generar máscara de scalp para personas calvas.
"""
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

_model_dir = Path(__file__).resolve().parent.parent
_landmarker_path = _model_dir / "Models" / "face_landmarker.task"
_landmarker = None

# Índices del óvalo facial (orden cerrado para formar polígono)
# Extraído de FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL
_FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]


def _get_landmarker():
    """Carga el FaceLandmarker de forma diferida."""
    global _landmarker
    if _landmarker is None:
        if not _landmarker_path.exists():
            try:
                from services.model_utils import ensure_face_landmarker
                ensure_face_landmarker()
            except Exception:
                pass
        if _landmarker_path.exists():
            from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
            from mediapipe.tasks.python import BaseOptions
            with open(_landmarker_path, "rb") as f:
                model_bytes = f.read()
            opts = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=model_bytes),
                num_faces=1,
            )
            _landmarker = FaceLandmarker.create_from_options(opts)
    return _landmarker


def get_face_oval_mask(
    image_bytes: bytes,
    padding: float = 0.08,
) -> Optional[np.ndarray]:
    """
    Detecta el rostro y devuelve una máscara 255 donde está el óvalo facial.
    Usa landmarks para precisión (contorno real, no rectángulo).
    padding: margen extra alrededor del óvalo (0.08 = 8%)
    """
    landmarker = _get_landmarker()
    if landmarker is None:
        return None

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp = __import__("mediapipe", fromlist=["Image", "ImageFormat"])
    mp_image = mp.Image(mp.ImageFormat.SRGB, rgb.copy())

    try:
        result = landmarker.detect(mp_image)
    except Exception:
        return None

    if not result or not result.face_landmarks:
        return None

    face_mask = np.zeros((h, w), dtype=np.uint8)

    for landmarks in result.face_landmarks:
        points = []
        for idx in _FACE_OVAL_INDICES:
            if idx < len(landmarks):
                lm = landmarks[idx]
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append([x, y])

        if len(points) < 3:
            continue

        pts = np.array(points, dtype=np.int32)

        if padding > 0:
            centroid = np.mean(pts, axis=0)
            pts_centered = pts - centroid
            pts_scaled = pts_centered * (1 + padding) + centroid
            pts = pts_scaled.astype(np.int32)

        cv2.fillPoly(face_mask, [pts], 255)

    return face_mask if np.sum(face_mask) > 0 else None


def get_scalp_mask_for_bald(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Para personas calvas: crea máscara amplia de la región del scalp (cabeza sin rostro).
    Cubre frente, coronilla, laterales y parte trasera simulada. Excluye el óvalo facial.
    """
    landmarker = _get_landmarker()
    if landmarker is None:
        return None

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp = __import__("mediapipe", fromlist=["Image", "ImageFormat"])
    mp_image = mp.Image(mp.ImageFormat.SRGB, rgb.copy())

    try:
        result = landmarker.detect(mp_image)
    except Exception:
        return None

    if not result or not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]
    pts_oval = []
    for idx in _FACE_OVAL_INDICES:
        if idx < len(landmarks):
            lm = landmarks[idx]
            pts_oval.append([lm.x * w, lm.y * h])

    if len(pts_oval) < 10:
        return None

    pts = np.array(pts_oval, dtype=np.float32)
    y_min = np.min(pts[:, 1])
    y_max = np.max(pts[:, 1])
    x_min = np.min(pts[:, 0])
    x_max = np.max(pts[:, 0])
    face_width = x_max - x_min
    face_height = y_max - y_min
    center_x = (x_min + x_max) / 2

    face_mask = np.zeros((h, w), dtype=np.uint8)
    pts_int = np.array(pts, dtype=np.int32)
    cv2.fillPoly(face_mask, [pts_int], 255)

    head_width = face_width * 1.85
    scalp_top = max(0, y_min - face_height * 0.8)
    scalp_bottom = y_min + face_height * 0.75
    x1 = int(max(0, center_x - head_width / 2))
    x2 = int(min(w, center_x + head_width / 2))
    y1_scalp = int(max(0, scalp_top - 5))
    y2_scalp = int(min(h, scalp_bottom + 15))

    scalp_mask = np.zeros((h, w), dtype=np.uint8)
    scalp_mask[y1_scalp:y2_scalp, x1:x2] = 255

    scalp_mask = np.where(face_mask > 0, 0, scalp_mask).astype(np.uint8)

    kernel = np.ones((21, 21), np.uint8)
    scalp_mask = cv2.dilate(scalp_mask, kernel, iterations=2)
    scalp_mask = np.where(face_mask > 0, 0, scalp_mask).astype(np.uint8)
    scalp_mask = cv2.GaussianBlur(scalp_mask, (25, 25), 0)
    scalp_mask = np.where(scalp_mask > 40, 255, 0).astype(np.uint8)

    if np.sum(scalp_mask) < 2000:
        return None
    return scalp_mask
