from services.segmentation_service import segmenter, process_image, segmenter_hair
import boto3
import base64, cv2, json, random
import numpy as np
from dotenv import load_dotenv


load_dotenv()

#Bedrock client
client = boto3.client('bedrock-runtime', region_name='us-east-1')

#model_id = "amazon.titan-image-generator-v2:0"
model_id = "arn:aws:bedrock:us-east-1:911701613368:inference-profile/us.stability.stable-image-inpaint-v1:0"

#Model for LLM
ll_model_id= "us.meta.llama3-1-70b-instruct-v1:0"

#User prompt
#user_prompt="""I want a short haircut that reaches my eyebrows and straight hair styled formally."""


def get_technical_instructions(user_prompt):
    prompt_system ="""
    You are a technical image editing assistant specialized in hair transformations.
    Analyze the user's request carefully and output ONLY a valid JSON object with these keys:
    - "style": (choose one: "long", "short", "bald", "color")
    - "refined_prompt": (a precise, natural English description for an image generator.
    Always describe the hairstyle clearly, e.g. "a person with completely bald head",
    "a person with long flowing hair", "short cropped hairstyle", "hair dyed bright red").
    Avoid vague terms or unrealistic combinations.
    - "negative_prompt": (standard English negative prompt to avoid artifacts:
    "blurry, distorted, extra limbs, bad anatomy, low quality, unrealistic hair").
    - "dilation_iterations": (integer 1-2. Use 2 for bald or long hair to ensure mask expansion,
    1 for color change or short hair).
    Rules:
    - Always ensure the hairstyle description matches the chosen style.
    - For "bald", explicitly state "completely bald head" to avoid partial hair.
    - For "long", emphasize "long flowing hair" or "extended hairstyle".
    - For "short", emphasize "short cropped hair".
    - For "color", describe the hair color change clearly.
    - Do not invent extra objects, backgrounds, or unrelated features.
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
        
    except (Exception) as e:
        print(f"ERROR: Can't invoke '{ll_model_id}'. Reason: {e}")
        exit(1)
    
    model_response = json.loads(response.get("body").read())
    response_text = model_response['generation'].strip()
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Limpieza de emergencia si el modelo añade texto extra
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        return json.loads(response_text[start:end])

def processs_dynamic_mask(mask_np, instructions):
    mask = np.where(mask_np.squeeze() == 1, 255, 0).astype(np.uint8)
    
    style = instructions['style']
    iters = instructions['dilation_iterations']
    
    kernel = np.ones((11, 11), np.uint8)
    
    if style == "long":
        # Expandir significativamente hacia abajo para dar espacio al pelo largo
        # Creamos una dilatación selectiva o más agresiva
        mask = cv2.dilate(mask, kernel, iterations=int(iters))
    elif style == "bald":
        # Dilatación moderada para limpiar bordes de pelo viejo
        mask = cv2.dilate(mask, kernel, iterations=int(iters)+1)
    
    # Suavizado para evitar cortes bruscos ("manchas")
    #mask = cv2.GaussianBlur(mask, (51, 51), 0)
    return mask

def generate_new_style(image: bytes, prompt: str):
    #image_path = "../Inputs/FotoFrontal1.jpeg"
    instructions = get_technical_instructions(prompt)  
    #print(instructions) 
    #with open(image_path, "rb") as image_file:
    #    imagen_en_bytes = image_file.read()


    image_mediapipe = process_image(image)
    mask = segmenter_hair(image_mediapipe, segmenter)
    mask_dynamic = processs_dynamic_mask(mask, instructions)
    
    # Convertimos el mural en el "paquete" de un archivo PNG
    exito, buffer = cv2.imencode('.png', mask_dynamic)
    mask_bytes = buffer.tobytes()
    input_image = base64.b64encode(image).decode('utf-8')
    mask_image = base64.b64encode(mask_bytes).decode('utf-8')
    
    seed = random.randint(0,2147483646)

    native_request = {
            "prompt": instructions['refined_prompt'],
            "negative_prompt": instructions['negative_prompt'],
            "image": input_image,   # imagen original en base64
            "mask": mask_image,    # máscara en base64
            "seed": seed,
            "grow_mask":0,
    } 
    

    request = json.dumps(native_request)

    response = client.invoke_model(body=request, modelId=model_id)

    response_model = json.loads(response["body"].read())

    base64_image_data = response_model['images'][0]

    image_data = base64.b64decode(base64_image_data)

    # Convertir a array numpy
    nparr = np.frombuffer(image_data, np.uint8)

    result_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  
    return result_image
    
"""     
def generareTes():
    mask,mask_dynamic = generate_new_style()
    cv2.imshow("Resultado", mask_dynamic)
    cv2.imshow("Resultado mascara dinamica", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 """