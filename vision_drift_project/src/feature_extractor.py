import torch
import torch.nn as nn
import torchvision.models as models
class VisionEncoder:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import os
        resnet = models.resnet18(weights=None)
        weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'resnet18_weights.pth')
        if os.path.exists(weights_path):
            resnet.load_state_dict(torch.load(weights_path))
            print("Loaded ResNet-18 weights from local 'models/' directory.")
        else:
            print(f"Warning: Offline weights not found at {weights_path}. Loading random weights.")
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.encoder.to(self.device)
        self.encoder.eval()
    def extract_dataset_embeddings(self, dataloader, max_batches=15):
        all_embeddings = []
        with torch.no_grad():
            for idx, (images, _) in enumerate(dataloader):
                if idx >= max_batches:
                    break
                images = images.to(self.device)
                feats = self.encoder(images).flatten(1)
                all_embeddings.append(feats.cpu())
        return torch.cat(all_embeddings, dim=0)
    def predict_single_batch(self, images):
        self.encoder.eval()
        with torch.no_grad():
            images = images.to(self.device)
            return self.encoder(images).flatten(1).cpu()
