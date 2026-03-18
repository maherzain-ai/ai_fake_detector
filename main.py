import cv2
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

@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    # Save uploaded video
    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(video_path)

    frames_checked = 0
    human_count = 0
    fake_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frames_checked >= 5:  # only 5 frames (SAFE)
            break

        # Convert frame to image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        # Use your existing model
        result = predict_image(model, pil_img)

        if result["prediction"] == "Human":
            human_count += 1
        else:
            fake_count += 1

        frames_checked += 1

    cap.release()

    # Final decision
    if human_count >= fake_count:
        final_result = "Human"
        confidence = (human_count / frames_checked) * 100
    else:
        final_result = "Fake"
        confidence = (fake_count / frames_checked) * 100

    return {
        "prediction": final_result,
        "confidence": confidence
    }

import uvicorn
import os
if __name__ == "__main__":
    port=int(os.environ.get("PORT",10000))
    uvicorn.run("main:app",
                host="0.0.0.0",port=port)

    
    
