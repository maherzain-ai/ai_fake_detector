
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from model_loader import load_model, predict_image
app = FastAPI()

# Allow requests from your frontend
origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "https://zain-deepfake-ai.netlify.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
model = load_model()
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Open uploaded image
    image = Image.open(file.file).convert("RGB")

    # Get prediction
    
    result = predict_image(model, image)
    return result
import requests
from fastapi import Body

API_URL = "https://api-inference.huggingface.co/models/roberta-base-openai-detector"

headers = {
    "Authorization": "Bearer import requests
from fastapi import Body

API_URL = "https://api-inference.huggingface.co/models/roberta-base-openai-detector"

headers = {
    "Authorization": "Bearer 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"
}

@app.post("/detect-text")
async def detect_text(data: dict = Body(...)):

    text = data["text"]

    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": text}
    )

    result = response.json()[0]

    return {
        "prediction": result["label"],
        "confidence": result["score"] * 10"
}

@app.post("/detect-text")
async def detect_text(data: dict = Body(...)):

    text = data["text"]

    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": text}
    )

    result = response.json()[0]

    return {
        "prediction": result["label"],
        "confidence": result["score"] * 100
    }
import uvicorn
import os
if __name__ == "__main__":
    port=int(os.environ.get("PORT",10000))
    uvicorn.run("main:app",
                host="0.0.0.0",port=port)

    
    
