import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import config
from model import FakeImageDetector

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({'loss': running_loss/total, 'acc': 100*correct/total})
    return running_loss/len(loader), 100*correct/total

def train():
    device = config.DEVICE
    print(f"Starting Training on {device}...")
    
    model = FakeImageDetector().to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Update paths to point to the processed data
    train_path = os.path.join('data', 'processed', 'train')
    train_loader = DataLoader(datasets.ImageFolder(train_path, transform=transform), 
                              batch_size=config.BATCH_SIZE, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.PHASE1_LR)

    # Phase 1: Train Classifier Only
    print("\n" + "="*50)
    print("PHASE 1: TRAINING CLASSIFIER ONLY")
    print("="*50)
    
    loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Save the model
    os.makedirs(os.path.dirname(config.BEST_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.BEST_MODEL_PATH)
    print(f"\n✓ Model saved to {config.BEST_MODEL_PATH}")

if __name__ == "__main__":
    train()