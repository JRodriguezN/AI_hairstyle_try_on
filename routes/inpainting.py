from datetime import datetime
from pathlib import Path
import base64

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from services.inpainting_service import generate_new_style
import cv2
import numpy as np

hair_try__in_route = APIRouter()

def resize_for_model(image_bytes, max_size=(1024, 1024)):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if cv_image is None:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen enviada.")

    h, w = cv_image.shape[:2]
    scale = min(max_size[0] / w, max_size[1] / h, 1)
    new_w, new_h = int(w*scale), int(h*scale)

    resized = cv2.resize(cv_image, (new_w, new_h))
    # Convertir de nuevo a bytes para pasarlo a generate_new_style
    _, buffer = cv2.imencode(".png", resized)
    return buffer.tobytes()



@hair_try__in_route.post("/hair/change")
async def change_hair_style(image: UploadFile = File ( ... ), prompt: str = Form(...)):
    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Debes enviar una imagen.")

    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise HTTPException(status_code=400, detail="Debes escribir un prompt con el cambio deseado.")

    resized_image = resize_for_model(image_bytes)
    try:
        result = generate_new_style(resized_image, normalized_prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"Error del servicio de IA: {e}")

    success, buffer = cv2.imencode(".png", result)
    if not success:
        raise HTTPException(status_code=500, detail="No se pudo codificar la imagen generada.")

    return {
        "status_code": 200,
        "message": "Imagen generada correctamente",
        "image_base64": base64.b64encode(buffer).decode("utf-8"),
        "image_mime_type": "image/png",
    }





