from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

model = tf.keras.models.load_model("seed_classifier.h5")
CLASS_NAMES = ['Chickpea', 'Kidney Bean', 'Maize', 'Mung Bean', 'Pigeon Pea', 'Rice', 'Urad Bean', 'Wheat']

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((64, 64))  # adjust if your model expects another size
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = preprocess_image(image_bytes)
    prediction = model.predict(img)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return JSONResponse({"class": predicted_class, "confidence": confidence})
