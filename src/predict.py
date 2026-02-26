import torch
from PIL import Image
from torchvision import transforms
from model import FakeImageDetector
import config
import argparse

def predict(image_path):
    model = FakeImageDetector().to(config.DEVICE)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
    model.eval()

    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        prob = model(img_tensor).item()
    
    result = "FAKE" if prob > 0.5 else "REAL"
    print(f"Prediction: {result} (Confidence: {prob if prob > 0.5 else 1-prob:.2%})")

if __name__ == "__main__":
    # Example: python src/predict.py --image test.jpg
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    predict(args.image)