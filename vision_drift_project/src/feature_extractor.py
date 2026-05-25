import torch
import torch.nn as nn
import torchvision.models as models
import os

class IntegratedVisionModel(nn.Module):
    def __init__(self, device=None):
        super(IntegratedVisionModel, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'resnet18_weights.pth')
        if os.path.exists(weights_path):
            resnet.load_state_dict(torch.load(weights_path))
            print("Loaded ResNet-18 weights from local 'models/' directory.")
        else:
            print(f"Loading official pre-trained ResNet-18 weights (local weights not found at {weights_path}).")
            
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc_layer = resnet.fc
        
        self.to(self.device)
        self.eval()
        
    def forward(self, x):
        x = x.to(self.device)
        embedding = self.feature_extractor(x).flatten(1)
        logits = self.fc_layer(embedding)
        return logits, embedding
