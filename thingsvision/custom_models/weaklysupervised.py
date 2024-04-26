
import torch
from torchvision import transforms
from .custom import Custom

__all__ = ['ResNeXt101_32x8d_wsl', 'ResNeXt101_32x16d_wsl', 'ResNeXt101_32x32d_wsl', 'ResNeXt101_32x48d_wsl' ]


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ResNeXt101_32x8d_wsl(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"

    def create_model(self) -> Any:
        model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        model.eval()
        return model, preprocess

class ResNeXt101_32x16d_wsl(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"

    def create_model(self) -> Any:
        model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
        model.eval()
        return model, preprocess
    
class ResNeXt101_32x32d_wsl(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"

    def create_model(self) -> Any:
        model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x32d_wsl')
        model.eval()
        return model, preprocess
    
class ResNeXt101_32x48d_wsl(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"

    def create_model(self) -> Any:
        model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')
        model.eval()
        return model, preprocess