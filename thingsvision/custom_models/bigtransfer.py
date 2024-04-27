from typing import Any
import tensorflow_hub as hub
from .custom import Custom

__all__ = ['BitS_ResNet50x1']

class BitS_ResNet50x1(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"

    def create_model(self) -> Any:
        #don't expect this to work, due to layers not being accessible:
        model = hub.KerasLayer("https://www.kaggle.com/models/google/bit/TensorFlow2/s-r50x1-ilsvrc2012-classification/1")
        return model, None
