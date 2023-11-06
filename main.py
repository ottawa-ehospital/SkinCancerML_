from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Allow any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("./skin_cancer_final_model4.h5")

# Class names for the two classes
CLASS_NAMES = ["benign", "malignant"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    # Dynamic thresholding based on the predicted class
    if CLASS_NAMES[np.argmax(predictions[0])] == "malignant":
        threshold = 0.5  # Invert the threshold for "malignant" class
    else:
        threshold = 0.5  # Use the same threshold for "benign" class

    if predictions[0] >= threshold:
        predicted_class = "malignant"
    else:
        predicted_class = "benign"

    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)

