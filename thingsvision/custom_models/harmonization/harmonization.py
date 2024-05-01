from typing import Any

from harmonization.models import (load_EfficientNetB0, load_LeViT_small,
                                  load_ResNet50, load_tiny_ConvNeXT,
                                  load_tiny_MaxViT, load_VGG16, load_ViT_B16, preprocess_input)

from thingsvision.custom_models.custom import Custom
from tensorflow.keras import layers

class Harmonization(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"
        self.variant = parameters.get("variant", "ViT_B16")

    def check_available_variants(self):
        variants = [
            "ViT_B16",
            "ResNet50",
            "VGG16",
            "EfficientNetB0",
            "tiny_ConvNeXT",
            "tiny_MaxViT",
            "LeViT_small",
        ]

        if self.variant not in variants:
            raise ValueError(f"\nVariant must be one of {variants}")

    def harmonization_preprocessing(self, img):
        img = layers.experimental.preprocessing.Resizing(224, 224)(img)
        return preprocess_input(img)

    def create_model(self) -> Any:
        self.check_available_variants()
        variant_function_dict = {
            "ViT_B16": load_ViT_B16,
            "ResNet50": load_ResNet50,
            "VGG16": load_VGG16,
            "EfficientNetB0": load_EfficientNetB0,
            "tiny_ConvNeXT": load_tiny_ConvNeXT,
            "tiny_MaxViT": load_tiny_MaxViT,
            "LeViT_small": load_LeViT_small,
        }
        model = variant_function_dict[self.variant]()
        return model, self.harmonization_preprocessing
