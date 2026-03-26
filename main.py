from openai import OpenAI
client=OpenAI(api_key="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")
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
 @app.post("/detect-text")
async def detect_text(request: Request):

    data = await request.json()
    text = data["text"]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Detect whether the text is AI generated or human written. Reply only: Likely AI or Likely Human."
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )

    result = response.choices[0].message.content

    return {"result": result}
import uvicorn
import os
if __name__ == "__main__":
    port=int(os.environ.get("PORT",10000))
    uvicorn.run("main:app",
                host="0.0.0.0",port=port)

    
    
