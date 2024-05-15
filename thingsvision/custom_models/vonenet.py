from typing import Any

import torch
import torchvision.models as models
from torchvision import transforms

import vonenet #the fork KogniJannis/vonenet
from .custom import Custom

__all__ = ['VOneNet_AlexNet', 'VOneNet_ResNet50', 'VOneNet_ResNet50_at', 'VOneNet_CORnetS']

def load_vonenet_preprocessing():
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    return transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=norm_mean, std=norm_std),
                    ])


class VOneNet_AlexNet(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"
    
    def create_model(self) -> Any:
        model = vonenet.get_model(model_arch='alexnet', pretrained=True, map_location='cpu')
        return model, load_vonenet_preprocessing

class VOneNet_ResNet50(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"

    def create_model(self) -> Any:    
        model = vonenet.get_model(model_arch='resnet50', pretrained=True, map_location='cpu')
        return model, load_vonenet_preprocessing

class VOneNet_ResNet50_at(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"
    
    def create_model(self) -> Any:
        model = vonenet.get_model(model_arch='resnet50_at', pretrained=True, map_location='cpu')
        return model, load_vonenet_preprocessing

class VOneNet_CORnetS(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"
    
    def create_model(self) -> Any:
        model = vonenet.get_model(model_arch='cornets', pretrained=True, map_location='cpu')
        return model, load_vonenet_preprocessing

