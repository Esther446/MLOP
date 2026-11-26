
import os
import json
import uuid
from pathlib import Path
from io import BytesIO

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from supabase import create_client


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "flower_vision_final.onnx"
LABELS_PATH = BASE_DIR / "models" / "class_names.json"

with open(LABELS_PATH, "r") as file:
    CLASS_NAMES = json.load(file)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
BUCKET_NAME = "flower-images"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")


def get_supabase():
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        raise HTTPException(
            status_code=503,
            detail="Supabase storage is not configured."
        )

    return create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)


# --------------------------------------------------
# LOAD PREDICTION MODEL
# --------------------------------------------------

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="FlowerVision API",
    description="Flower classification and training data upload API",
    version="2.0.0"
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


# --------------------------------------------------
# FLOWER PREDICTION
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPEG or PNG image."
        )

    try:
        contents = await file.read()

        if len(contents) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image must be smaller than 10 MB."
            )

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


# --------------------------------------------------
# UPLOAD LABELED TRAINING IMAGES
# --------------------------------------------------

@app.post("/upload-training-image")
async def upload_training_image(
    file: UploadFile = File(...),
    flower_label: str = Form(...)
):

    if flower_label not in CLASS_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid flower label. Choose from: {CLASS_NAMES}"
        )

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported."
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Empty image.")

    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 10 MB."
        )

    try:
        image = Image.open(BytesIO(contents))
        image.verify()

        image = Image.open(BytesIO(contents)).convert("RGB")

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    # Normalize the image format before storing it.
    output = BytesIO()
    image.save(output, format="JPEG", quality=95)
    image_bytes = output.getvalue()

    filename = f"{uuid.uuid4().hex}.jpg"
    storage_path = f"{flower_label}/{filename}"

    try:
        supabase = get_supabase()

        # Save the image to private Supabase Storage.
        supabase.storage.from_(BUCKET_NAME).upload(
            path=storage_path,
            file=image_bytes,
            file_options={
                "content-type": "image/jpeg",
                "upsert": "false"
            }
        )

        # Save its metadata in the database.
        result = supabase.table("uploaded_images").insert({
            "filename": filename,
            "storage_path": storage_path,
            "flower_label": flower_label,
            "status": "pending"
        }).execute()

        return {
            "message": "Training image uploaded successfully!",
            "filename": filename,
            "flower_label": flower_label,
            "status": "pending"
        }

    except HTTPException:
        raise

    except Exception as error:
        # Remove the stored image if its database record could not be saved.
        try:
            supabase.storage.from_(BUCKET_NAME).remove([storage_path])
        except Exception:
            pass

        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(error)}"
        )
