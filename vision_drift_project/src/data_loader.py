import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
def get_loaders(data_root='./data', batch_size=64):
    ref_transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    drift_transform = T.Compose([
        T.Resize(224),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    ref_dataset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=ref_transform)
    drift_dataset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=drift_transform)
    ref_loader = DataLoader(ref_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    prod_loader = DataLoader(drift_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return ref_loader, prod_loader
