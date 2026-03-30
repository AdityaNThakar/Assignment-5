from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import models
from typing import List

app = FastAPI()

# Recreate architecture
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("resnet10_model.pth", map_location=torch.device('cpu')))
model.eval()

class ImageInput(BaseModel):
    data: List[float] # Expects 3072 values (3*32*32)

@app.get("/")
def home():
    return {"message": "CIFAR-10 Classifier Active"}

@app.post("/predict")
def predict(input_data: ImageInput):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    tensor_data = torch.tensor(input_data.data).view(1, 3, 32, 32)
    with torch.no_grad():
        output = model(tensor_data)
        prediction = torch.argmax(output, dim=1).item()
    return {"prediction": classes[prediction]}