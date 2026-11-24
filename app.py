
from pathlib import Path
import json

import numpy as np
import onnxruntime as ort
from PIL import Image
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "flower_vision_final.onnx"
LABELS_PATH = BASE_DIR / "models" / "class_names.json"

with open(LABELS_PATH, "r") as file:
    CLASS_NAMES = json.load(file)

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name

app = FastAPI(
    title="FlowerVision API",
    description="Flower classification API powered by ONNX Runtime",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "FlowerVision API is running!",
        "model": "ONNX Runtime",
        "classes": CLASS_NAMES
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPEG or PNG image."
        )

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image = image.resize((180, 180))

        image_array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        image_array = np.expand_dims(image_array, axis=0)

        predictions = session.run(
            [OUTPUT_NAME],
            {INPUT_NAME: image_array}
        )[0][0]

        predicted_index = int(np.argmax(predictions))

        return {
            "predicted_class": CLASS_NAMES[predicted_index],
            "confidence": float(predictions[predicted_index]),
            "probabilities": {
                CLASS_NAMES[i]: float(predictions[i])
                for i in range(len(CLASS_NAMES))
            }
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
