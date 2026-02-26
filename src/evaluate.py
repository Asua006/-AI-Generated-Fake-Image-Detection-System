import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import FakeImageDetector
import config

def evaluate():
    model = FakeImageDetector().to(config.DEVICE)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_loader = DataLoader(datasets.ImageFolder('data/processed/test', transform=transform), batch_size=config.BATCH_SIZE)
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(config.DEVICE)).squeeze()
            all_preds.extend((outputs > 0.5).cpu().numpy())
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
    
    # Save Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig('outputs/plots/confusion_matrix.png')
    print("✓ Plots saved in outputs/plots/")

if __name__ == "__main__":
    evaluate()