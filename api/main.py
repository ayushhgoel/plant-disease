from fastapi import FastAPI, HTTPException
import uvicorn
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
import io
import cv2
from rembg import remove
import os
from datetime import datetime

app = FastAPI()


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "v2.keras")


try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print(f"Attempted to load from: {MODEL_PATH}")
    raise

CLASS_NAMES = ['Aloe Vera__Healthy', 'Aloe Vera__Leaf Spot', 'Aloe Vera__Rust', 'Aloe Vera__Sunburn', 
 'Aloe Vera__Anthracnose', 'Money Plant__Bacterial Wilt', 'Money Plant__Manganese Toxicity', 
 'Money Plant__Healthy', 'Potato__Early Blight', 'Potato__Healthy', 'Potato__Late Blight', 
 'Tomato__Bacterial Spot', 'Tomato__Early Blight', 'Tomato__Healthy', 'Tomato__Late Blight', 
 'Tomato__Leaf Mold', 'Tomato__Septoria Leaf Spot', 
 'Tomato__Target Spot', 'Tomato__Tomato Mosaic Virus']


def remove_background_black(image_array):
    pil_image = Image.fromarray(image_array)


    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format="PNG")  
    img_bytes = img_bytes.getvalue()


    output_bytes = remove(img_bytes)
    output_image = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

   
    black_bg = Image.new("RGBA", output_image.size, (0, 0, 0, 255))  
    final_image = Image.alpha_composite(black_bg, output_image).convert("RGB")

    return np.array(final_image)  


def zoom_and_crop(image, zoom_factor=1.2):
    height, width, _ = image.shape
    new_h, new_w = int(height / zoom_factor), int(width / zoom_factor)

    start_x, start_y = (width - new_w) // 2, (height - new_h) // 2
    cropped = image[start_y:start_y + new_h, start_x:start_x + new_w]
    
    return cropped


def adjust_brightness_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)  


def resize_image(image, size=(128, 128)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


# def preprocess_image(image_base64, save_directory="api/temp"):
def preprocess_image(image_base64):
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    image_np = np.array(image)
    bg_removed = remove_background_black(image_np)
    # zoomed_cropped = zoom_and_crop(bg_removed)
    # brightness_fixed = adjust_brightness_contrast(zoomed_cropped)
    brightness_fixed = adjust_brightness_contrast(bg_removed)
    final_resized = resize_image(brightness_fixed, size=(128, 128) )
    # final_resized = resize_image(image_np, size=(160, 160)) 
    # filename = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    # filepath = os.path.join(save_directsory, filename)


    # processed_image_pil = Image.fromarray(final_resized)
    # processed_image_pil.save(filepath)

    # print(f"Processed image saved at: {filepath}")  

    # return final_resized / 255.0 
    return final_resized 

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

@app.post("/predict")
async def predict(data: dict):
    if "image_base64" not in data:
        raise HTTPException(status_code=400, detail="Missing 'image_base64' key in request")

    try:
        processed_image = preprocess_image(data["image_base64"])
        img_batch = np.expand_dims(processed_image, axis=0)
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        print(predicted_class)
        print(float(confidence))
        return {
            'plantname_disease_name': predicted_class,
            'confidence': float(confidence)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
