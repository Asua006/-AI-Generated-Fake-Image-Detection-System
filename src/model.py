import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class FakeImageDetector(nn.Module):
    def __init__(self, num_classes=1):
        super(FakeImageDetector, self).__init__()
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.base_model(x)

    def unfreeze_base(self, unfreeze_layers=30):
        all_params = list(self.base_model.parameters())
        for param in all_params[-unfreeze_layers:]:
            param.requires_grad = True