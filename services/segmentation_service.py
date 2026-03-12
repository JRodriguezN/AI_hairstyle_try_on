import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.image import Image
from mediapipe.tasks.python import BaseOptions
import cv2
import numpy as np

#Initialize segmenter
model_path="./Models/hair_segmenter.tflite"

options = vision.ImageSegmenterOptions(
    base_options = BaseOptions(model_path),
    running_mode = vision.RunningMode.IMAGE,
    output_category_mask = True
)
segmenter = vision.ImageSegmenter.create_from_options(options)


def process_image(image_byte: bytes):
    nparr = np.frombuffer(image_byte, np.uint8)
    img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    
    #Convert color to RGB and to mediaPipe object
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_rgb = mp.Image(mp.ImageFormat.SRGB, image_rgb)

    return image_rgb

def segmenter_hair(image: Image, segmenter):
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
    
 
    
    