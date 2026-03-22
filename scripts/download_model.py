#!/usr/bin/env python3
"""
Descarga los modelos .tflite necesarios:
- hair_segmenter.tflite: segmentación de cabello
- face_landmarker.task: landmarks faciales (protección precisa del rostro)
"""
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "Models"

MODELS = [
    ("hair_segmenter.tflite", "https://huggingface.co/yolain/selfie_multiclass_256x256/resolve/main/hair_segmenter.tflite"),
    ("face_landmarker.task", "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"),
]


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS:
        path = MODELS_DIR / name
        if path.exists():
            print(f"El modelo ya existe: {path}")
            continue
        print(f"Descargando {name}...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"  Descargado correctamente.")
        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Descarga manual desde: {url}")


if __name__ == "__main__":
    main()
