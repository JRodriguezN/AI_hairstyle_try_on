from fastapi import APIRouter, UploadFile, File
from services.segmentation_service import segmenter, process_image, segmenter_hair
import cv2, base64

segmentation_route = APIRouter()

@segmentation_route.post("/segmentation")
async def segmentation_image(image: UploadFile = File( ... )):
        image_bytes = await image.read()
        
        image_rgb = process_image(image_bytes)
        
        mask_bin = segmenter_hair(image_rgb, segmenter)
        
        # Codificar como PNG y luego a base64
        _, buffer = cv2.imencode(".png", mask_bin)
        mask_base64 = base64.b64encode(buffer).decode("utf-8")

        return {
                "category_mask": mask_base64
        }
