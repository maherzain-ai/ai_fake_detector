import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from utils import preprocess_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['fake', 'real']

# Load model architecture and weights
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Predict image
def predict_image(model, image):
    image = preprocess_image(image)

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return {
        "prediction": classes[predicted.item()],
        "confidence": round(confidence.item() * 100, 2)
    }