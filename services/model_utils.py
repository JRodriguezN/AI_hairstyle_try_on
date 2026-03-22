"""
Utilidades para gestión de modelos .tflite.
Incluye descarga automática cuando faltan.
"""
from pathlib import Path
import urllib.request

_MODEL_DIR = Path(__file__).resolve().parent.parent / "Models"

HAIR_SEGMENTER_URL = "https://huggingface.co/yolain/selfie_multiclass_256x256/resolve/main/hair_segmenter.tflite"
HAIR_SEGMENTER_PATH = _MODEL_DIR / "hair_segmenter.tflite"

FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
FACE_LANDMARKER_PATH = _MODEL_DIR / "face_landmarker.task"


def ensure_hair_segmenter() -> bool:
    """
    Asegura que hair_segmenter.tflite existe y tiene contenido. Si no, intenta descargarlo.
    Returns: True si el modelo existe (o se descargó), False si falló la descarga.
    """
    if HAIR_SEGMENTER_PATH.exists() and HAIR_SEGMENTER_PATH.stat().st_size > 100_000:
        return True
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(HAIR_SEGMENTER_URL, HAIR_SEGMENTER_PATH)
        return HAIR_SEGMENTER_PATH.exists() and HAIR_SEGMENTER_PATH.stat().st_size > 100_000
    except Exception:
        return False


def ensure_face_landmarker() -> bool:
    """
    Asegura que face_landmarker.task existe. Si no, intenta descargarlo.
    Returns: True si existe, False si falló. (Protección facial con landmarks)
    """
    if FACE_LANDMARKER_PATH.exists() and FACE_LANDMARKER_PATH.stat().st_size > 1_000_000:
        return True
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH)
        return FACE_LANDMARKER_PATH.exists() and FACE_LANDMARKER_PATH.stat().st_size > 1_000_000
    except Exception:
        return False
