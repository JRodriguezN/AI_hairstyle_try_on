from services.segmentation_service import process_image, segmenter_hair
from services.hair_analysis_service import analyze_hair
import boto3
import base64, cv2, json, random
import numpy as np
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from mediapipe.tasks.python.vision.core.image import Image


load_dotenv()

#Bedrock client
client = boto3.client('bedrock-runtime', region_name='us-east-1')

#model_id = "amazon.titan-image-generator-v2:0"
model_id = "us.stability.stable-image-inpaint-v1:0"

#Model for LLM
ll_model_id= "us.meta.llama3-1-70b-instruct-v1:0"

# Face landmarks para protección precisa del rostro
from services.face_landmarks_service import get_face_oval_mask, get_scalp_mask_for_bald


def _apply_face_protection(mask: np.ndarray, image: Image,  hair_info: str = "default") -> np.ndarray:
    """
    Elimina de la máscara cualquier píxel que cubra el rostro.
    Usa FaceLandmarker (óvalo facial preciso) en lugar de bounding box.
    """
    result = segmenter_hair(image)
    category_mask = result.squeeze()

    face_class = 3
    skin_class = 2
    face_mask = ((category_mask == face_class) | (category_mask == skin_class)).astype(np.uint8) * 255

    """if hair_info == "bald":
        kernel = np.ones((4, 4), np.uint8)  # menos agresivo
        iterations = 0
    else:"""
    kernel = np.ones((10, 10), np.uint8)
    iterations = 2
    
    face_mask_dilated = cv2.dilate(face_mask, kernel, iterations=iterations)
    if hair_info["is_bald"]:
        ys, xs = np.where(face_mask_dilated > 0)

        if len(ys) > 0:
            top_y = np.min(ys)        # donde empieza la cara
            bottom_y = np.max(ys)

            face_height = bottom_y - top_y

            # 🔥 cortar SOLO la frente (10%–20% funciona bien)
            forehead_cut = int(face_height * 0.18)

            face_mask_dilated[top_y:top_y + forehead_cut, :] = 0
    else:
        ys, xs = np.where(face_mask_dilated > 0)

        if len(ys) > 0:
            top_y = np.min(ys)        # donde empieza la cara
            bottom_y = np.max(ys)

            face_height = bottom_y - top_y

            # 🔥 cortar SOLO la frente (10%–20% funciona bien)
            forehead_cut = int(face_height * 0.18)

            face_mask_dilated[top_y:top_y + forehead_cut, :] = 0
            
    # eliminar completamente esa zona
    mask = np.where(face_mask_dilated> 0, 0, mask).astype(np.uint8)


    return mask


def get_technical_instructions(user_prompt):
    prompt_system ="""
    You are a technical image editing assistant specialized in photorealistic hair transformations.
    Analyze the user's request carefully and output ONLY a valid JSON object with these keys:
    
    - "style": (choose one: "long", "short", "bald", "color")
    
        * Use "long" when the user wants to extend short hair to shoulders, chest, or longer.
        * Use "short" when the user wants to reduce long hair to bob, pixie, or similar.
        * Use "bald" when the user wants to remove all hair and show a clean shaved scalp.
        * Use "color" when the user wants to change hair color or style without altering length.


    - "refined_prompt": (English description for an inpainting model. 
        MANDATORY start: "same face unchanged, photorealistic," then hair description.
        Extract attributes like color, texture, or length detail from the user request.
        * For LONG hair: "same face unchanged, photorealistic, natural long flowing hair extending to chest, realistic hair strands and texture, seamless integration with scalp, natural lighting, like a real photograph".
        * For SHORT hair: "same face unchanged, photorealistic, short natural haircut, realistic strands and texture, seamless integration with scalp, natural lighting".
        * For BALD: "same face unchanged, photorealistic, completely bald head, fully shaved scalp, no hair at all, no stubble, no hair strands, smooth natural scalp skin, visible pores, realistic skin texture, natural lighting".
        * For COLOR: "same face unchanged, photorealistic, keep exact same hairstyle, haircut, and hair length, only change hair color, realistic tones, preserve hair structure, strands and volume, seamless integration, natural lighting".
    
    - "negative_prompt": (MANDATORY: depends on the style)
        
        * For "bald":
        "hair, long hair, short hair, stubble hair, hair shadow, hair strands,
        wig, fake hair, plastic hair, unrealistic hairline,
        deformed face, distorted face, different person, identity change,
        altered face, asymmetric face, blurry, bad anatomy, cartoon, cgi,
        render, artificial, oversaturated, modified eyebrows, modified eyes, modified skin, face retouch"

        * For "long":
        "bald, shaved head, receding hairline, missing hair patches,
        deformed face, distorted face, different person, identity change,
        altered face, asymmetric face, blurry, bad anatomy, cartoon, cgi,
        render, artificial, oversaturated, wig, fake hair, plastic hair,
        unrealistic hairline, modified eyebrows, modified eyes, modified skin, face retouch"

        * For "short":
        "long hair, very long hair, excessive volume hair,
        deformed face, distorted face, different person, identity change,
        altered face, asymmetric face, blurry, bad anatomy, cartoon, cgi,
        render, artificial, oversaturated, wig, fake hair, plastic hair,
        unrealistic hairline, modified eyebrows, modified eyes, modified skin, face retouch"

        * For "color":
        "grayscale hair, desaturated hair, unnatural color patches,
        long hair, short hair, bald, shaved head, receding hairline, 
        curly hair, wavy hair, straight hair (if not requested),
        deformed face, distorted face, different person, identity change,
        altered face, asymmetric face, blurry, bad anatomy, cartoon, cgi,
        render, artificial, oversaturated, wig, fake hair, plastic hair,
        unrealistic hairline, modified eyebrows, modified eyes, modified skin, face retouch"
    
    - "dilation_iterations": 
        * 3–4 for "long"
        * 2 for "bald"
        * 1 for "short" or "color"
        
    Rules:
    - Face must stay EXACTLY the same.
    - Skin color must be the same
    - Hair must look real, not like a filter or CGI.
    - Always output a valid JSON object with the keys above.
    - For "color": The hairstyle, haircut, hair length, and hair texture MUST remain exactly the same. 
        Only the hair color can change.

    """
    
    formatted_prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {prompt_system}
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    The user wants: {user_prompt}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """

    native_request = {
        "prompt":formatted_prompt,
        "max_gen_len": 512,
        "temperature":0.1,
        "top_p": 0.9
    }
    request = json.dumps(native_request)
    
    try:
        response = client.invoke_model(
            modelId = ll_model_id,
            body = request
        )
        
    except Exception as e:
        raise RuntimeError(f"No se pudo invocar el modelo LLM '{ll_model_id}': {e}") from e
    
    model_response = json.loads(response.get("body").read())
    response_text = model_response['generation'].strip()
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Limpieza de emergencia si el modelo añade texto extra
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start < 0 or end <= start:
            raise ValueError("El LLM no devolvió un JSON válido.") from None
        return json.loads(response_text[start:end])

def _extend_mask_downward_for_long_hair(mask: np.ndarray) -> np.ndarray:
    """
    Extiende la máscara de cabello hacia abajo para dar espacio a pelo largo,
    sin invadir el área de la cara. Simula la zona donde caería el cabello
    sobre cuello y hombros.
    """
    h, w = mask.shape[:2]
    extended = mask.copy()
    
    # Extensión hacia abajo: para cada columna, extender desde el último pixel de pelo
    for col in range(w):
        col_mask = mask[:, col]
        rows_with_hair = np.where(col_mask > 0)[0]
        if len(rows_with_hair) > 0:
            bottom_hair = rows_with_hair[-1]
            # Extender hacia abajo: ~40% de la altura (donde caería pelo largo)
            extension_pixels = min(int(h * 0.42), h - bottom_hair - 1)
            if extension_pixels > 0:
                start_row = bottom_hair + 1
                end_row = min(bottom_hair + extension_pixels, h)
                for i, row in enumerate(range(start_row, end_row)):
                    progress = i / max(extension_pixels - 1, 1)
                    alpha = 1.0 - (progress * 0.65)  # Gradiente suave
                    extended[row, col] = max(extended[row, col], int(255 * alpha))
    
    # Extensión lateral suave (el pelo largo puede caer a los lados)
    kernel_wide = np.ones((5, 25), np.uint8)
    extended = cv2.dilate(extended, kernel_wide, iterations=1)
    
    # Suavizado en bordes para transición natural, evitar corte artificial
    extended = cv2.GaussianBlur(extended, (15, 15), 0)
    extended = np.clip(extended, 0, 255).astype(np.uint8)
    extended = np.where(extended > 80, 255, 0).astype(np.uint8)
    #extended = extended.astype(np.uint8)
    
    return extended


def processs_dynamic_mask(mask, instructions, hair_info):
    
    style = str(instructions.get('style', '')).lower()
    try:
        iters = int(instructions.get('dilation_iterations', 1))
    except (TypeError, ValueError):
        iters = 1
    
    kernel = np.ones((11, 11), np.uint8)
    
    if style == "long" and hair_info['length'] != 'long':
        # Extensión agresiva: dilatación inicial + extensión hacia abajo para pelo largo
        #mask = cv2.dilate(mask, kernel, iterations=max(2, int(iters)))
        mask = _extend_mask_downward_for_long_hair(mask)
    elif style == "short":
        # Cabello corto: ligera dilatación para limpiar bordes
        mask = cv2.dilate(mask, kernel, iterations=iters)
    elif style == "bald":
        # Dilatación moderada para limpiar bordes de pelo viejo
        mask = cv2.dilate(mask, kernel, iterations=int(iters)+1)

    elif style == "color":
        pass
    
    return mask

def generate_new_style(image: bytes, prompt: str):
    instructions = get_technical_instructions(prompt)  

    # Procesar imagen y realizar segmentacion
    image_mediapipe = process_image(image)
    hair_mask = segmenter_hair(image_mediapipe)
    
    # Convertir a binaria
    hair_mask = np.where(hair_mask.squeeze() == 1, 255, 0).astype(np.uint8)
    face_mask = get_face_oval_mask(image)
    
    hair_info = analyze_hair(hair_mask,face_mask)
    
    if hair_info["is_bald"]:
        # Si no tiene mucho cabello genera un mascara automatica
        hair_mask = get_scalp_mask_for_bald(image)

    mask_dynamic = processs_dynamic_mask(hair_mask, instructions, hair_info)
    #mask_dynamic = processs_dynamic_mask(mask, instructions)
    
    # Protección del rostro: la máscara NUNCA debe cubrir la cara
    mask_dynamic = _apply_face_protection(mask_dynamic, image_mediapipe,hair_info)    

    # Personas calvas o poco cabello: si hay muy poca región de cabello, usar máscara de scalp
    """ hair_pixels = np.sum(mask_dynamic > 0)
    total_pixels = mask_dynamic.shape[0] * mask_dynamic.shape[1]
    if hair_pixels < total_pixels * 0.02 or hair_pixels < 1500:
        scalp_mask = get_scalp_mask_for_bald(image)
        if scalp_mask is not None and np.sum(scalp_mask > 0) > 1500:
            style_val = str(instructions.get("style", "")).lower()
             # AJUSTE SEGÚN ESTILO
            if style_val in ["long", "add_hair"]:
                scalp_mask = _extend_mask_downward_for_long_hair(scalp_mask)

            elif style_val == "short":
                kernel = np.ones((7, 7), np.uint8)
                scalp_mask = cv2.dilate(scalp_mask, kernel, iterations=1)

            elif style_val == "bald":
                kernel = np.ones((5, 5), np.uint8)
                scalp_mask = cv2.dilate(scalp_mask, kernel, iterations=1)

            elif style_val == "color":
                # no expandir
                pass
            scalp_mask = cv2.GaussianBlur(scalp_mask, (21, 21), 0)
            scalp_mask = np.where(scalp_mask > 50, 255, 0).astype(np.uint8)
             
             # REAPLICAR PROTECCIÓN FACIAL (CRÍTICO)
            scalp_mask = _apply_face_protection(
                scalp_mask,
                image_mediapipe,
                style_val
            )

            mask_dynamic = scalp_mask
        elif hair_pixels == 0:
            raise ValueError(
                "No se detectó región de cabello para editar. Verifica que la imagen muestre claramente "
                "la cabeza y el rostro de la persona (frontal, buena iluminación)."
            ) """
    
    # Convertimos el mural en el "paquete" de un archivo PNG
    exito, buffer = cv2.imencode('.png', mask_dynamic)
    mask_bytes = buffer.tobytes()
    input_image = base64.b64encode(image).decode('utf-8')
    mask_image = base64.b64encode(mask_bytes).decode('utf-8')
    
    seed = random.randint(0,2147483646)
    
    # grow_mask para transición suave (máx 20). Mayor valor para cabello largo o agregar a calvo.
    style_val = str(instructions.get('style', '')).lower()
    if style_val == 'long':
        grow_mask_val = 14
    elif style_val == 'bald':
        grow_mask_val = 2
    elif style_val == 'color':
        grow_mask_val = 0
    else:
        grow_mask_val = 2
    refined = instructions.get('refined_prompt') or 'same face unchanged, photorealistic natural hair'
    negative = instructions.get('negative_prompt') or 'deformed face, distorted, blurry, cartoon, filter, artificial, wig, fake hair'
    native_request = {
            "prompt": refined,
            "negative_prompt": negative,
            "image": input_image,   # imagen original en base64
            "mask": mask_image,    # máscara en base64
            "seed": seed,
            "grow_mask": grow_mask_val,
            "style_preset": "photographic"
    } 
    

    request = json.dumps(native_request)

    try:
        response = client.invoke_model(body=request, modelId=model_id)
    except Exception as e:
        raise RuntimeError(f"Error al invocar el modelo de inpainting: {e}") from e

    response_model = json.loads(response["body"].read())
    images = response_model.get('images', [])
    if not images:
        raise RuntimeError("El modelo de inpainting no devolvió ninguna imagen.")

    base64_image_data = images[0]
    image_data = base64.b64decode(base64_image_data)

    # Convertir a array numpy
    nparr = np.frombuffer(image_data, np.uint8)

    result_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if result_image is None:
        raise RuntimeError("No se pudo decodificar la imagen generada.")

    return result_image