from pathlib import Path
from io import BytesIO
import json

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "flower_vision_final.keras"
CLASS_PATH = BASE_DIR / "models" / "class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_data = json.load(f)

if isinstance(class_data, list):
    class_names = class_data
elif isinstance(class_data, dict):
    class_names = class_data.get("class_names", class_data.get("classes"))
else:
    raise ValueError("Invalid class names file")

app = FastAPI(
    title="FlowerVision API",
    description="Flower image classification API",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "message": "FlowerVision API is running",
        "model": "FlowerVision",
        "classes": class_names
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image")

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize((180, 180))

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))

        return {
            "predicted_class": class_names[predicted_index],
            "confidence": float(predictions[predicted_index]),
            "confidence_percentage": round(float(predictions[predicted_index]) * 100, 2),
            "probabilities": {
                class_names[i]: float(predictions[i])
                for i in range(len(class_names))
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")
