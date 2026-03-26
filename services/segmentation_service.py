import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision.core.image import Image
import cv2
import numpy as np
from pathlib import Path

# Ruta del modelo relativa al directorio del proyecto (independiente del cwd)
_model_dir = Path(__file__).resolve().parent.parent
_model_path = _model_dir / "Models" / "selfie_multiclass_256x256.tflite"

_segmenter = None


def get_segmenter():
    """Carga el segmentador de cabello de forma diferida (solo cuando se necesita)."""
    global _segmenter
    if _segmenter is None:
        if not _model_path.exists():
            from services.model_utils import ensure_hair_segmenter
            if not ensure_hair_segmenter():
                raise FileNotFoundError(
                    "No se pudo descargar el modelo automáticamente. "
                    "Ejecuta en la terminal: python scripts/download_model.py "
                    "y vuelve a intentar."
                )
        # Cargar modelo como bytes para evitar errores con rutas con caracteres especiales (ej. Estadía en Windows)
        try:
            with open(_model_path, "rb") as f:
                model_bytes = f.read()
        except OSError as e:
            raise FileNotFoundError(
                f"No se pudo leer el modelo en {_model_path}. "
                "Ejecuta: python scripts/download_model.py"
            ) from e
        if len(model_bytes) < 100_000:
            raise FileNotFoundError("El archivo del modelo está corrupto o incompleto. Elimínalo y ejecuta: python scripts/download_model.py")
        options = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True
        )
        _segmenter = vision.ImageSegmenter.create_from_options(options)
    return _segmenter


def process_image(image_byte: bytes):
    nparr = np.frombuffer(image_byte, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen. Verifica que el archivo sea una imagen válida (jpeg, png, webp).")
    
    #Convert color to RGB and to mediaPipe object
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_rgb = mp.Image(mp.ImageFormat.SRGB, image_rgb)

    return image_rgb

def segmenter_hair(image: Image, segmenter=None):
    if segmenter is None:
        segmenter = get_segmenter()
    segmenter_result = segmenter.segment(image)
    
    category_mask =  segmenter_result.category_mask
    category_mask_np = category_mask.numpy_view()

    ##hair_color = (255,255,255)
    #print(category_mask_np.shape)
    ##category_mask_rgb = cv2.cvtColor(category_mask_np,cv2.COLOR_GRAY2RGB)

    #print(category_mask_rgb.shape)
    ##category_mask_rgb[np.where(category_mask_np.squeeze() == 1)] = hair_color
    #mask_bin = np.where(category_mask_np.squeeze() == 1, 0, 255).astype(np.uint8)
    
    return category_mask_np
    
    