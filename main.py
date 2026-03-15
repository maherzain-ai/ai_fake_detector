
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
    allow_origins=origins,
    allow_credentials=True,
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

import uvicorn
import os
if __name__ == "__main__":
    port=int(os.environ.get("PORT",10000))
    uvicorn.run("main:app",
                host="0.0.0",port=port)

    
    

    return result
