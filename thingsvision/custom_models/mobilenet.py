from typing import Any
from .custom import Custom
import tensorflow.keras.applications as keras_models
'''
this implements tensorflow mobilenets specifically those available on Brain-Score (naming matches Brain-Score)
the standard keras.applications implementation is not functional because it cannot access the checkpoints of different depth multipliers
'''

__all__ = [
 'MobileNetV2_0_35_96',
 'MobileNetV2_0_35_160',
 'MobileNetV2_0_35_192',
 'MobileNetV2_0_35_224',
 'MobileNetV2_0_5_96',
 'MobileNetV2_0_5_160',
 'MobileNetV2_0_5_192',
 'MobileNetV2_0_5_224',
 'MobileNetV2_0_75_96',
 'MobileNetV2_0_75_160',
 'MobileNetV2_0_75_192',
 'MobileNetV2_0_75_224',
 'MobileNetV2_1_96',
 'MobileNetV2_1_160',
 'MobileNetV2_1_192',
 'MobileNetV2_1_224',
 'MobileNetV2_1_3_224',
 'MobileNetV2_1_4_224']

def resolve_mobilenet(input_size, model_width):
    return keras_models.MobileNetV2((input_size, input_size, 3), alpha=model_width, weights="imagenet",include_top=True, classifier_activation=None)

class MobileNetV2_0_35_96(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"
    def create_model(self) -> Any:
        model = resolve_mobilenet(input_size=96, model_width=0.35)
        preprocess = keras_models.mobilenet_v2.preprocess_input
        return model, preprocess

class MobileNetV2_0_35_160(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"
    def create_model(self) -> Any:
        model = resolve_mobilenet(input_size=160, model_width=0.35)
        preprocess = keras_models.mobilenet_v2.preprocess_input
        return model, preprocess

class MobileNetV2_0_35_192(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"
    def create_model(self) -> Any:
        model = resolve_mobilenet(input_size=192, model_width=0.35)
        preprocess = keras_models.mobilenet_v2.preprocess_input
        return model, preprocess
    
class MobileNetV2_0_35_224(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"
    def create_model(self) -> Any:
        model = resolve_mobilenet(input_size=224, model_width=0.35)
        preprocess = keras_models.mobilenet_v2.preprocess_input
        return model, preprocess
    
class MobileNetV2_0_5_96(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"
    def create_model(self) -> Any:
        model = resolve_mobilenet(input_size=96, model_width=0.5)
        preprocess = keras_models.mobilenet_v2.preprocess_input
        return model, preprocess
    
class MobileNetV2_0_5_160(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"
    def create_model(self) -> Any:
        model = resolve_mobilenet(input_size=160, model_width=0.5)
        preprocess = keras_models.mobilenet_v2.preprocess_input
        return model, preprocess
    
class MobileNetV2_0_5_192(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"
    def create_model(self) -> Any:
        model = resolve_mobilenet(input_size=192, model_width=0.5)
        preprocess = keras_models.mobilenet_v2.preprocess_input
        return model, preprocess
    
class MobileNetV2_0_5_224(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"
    def create_model(self) -> Any:
        model = resolve_mobilenet(input_size=224, model_width=0.5)
        preprocess = keras_models.mobilenet_v2.preprocess_input
        return model, preprocess
    
class MobileNetV2_0_75_96(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"
    def create_model(self) -> Any:
        model = resolve_mobilenet(input_size=96, model_width=0.75)
        preprocess = keras_models.mobilenet_v2.preprocess_input
        return model, preprocess
    
class MobileNetV2_0_75_160(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"
    def create_model(self) -> Any:
        model = resolve_mobilenet(input_size=160, model_width=0.75)
        preprocess = keras_models.mobilenet_v2.preprocess_input
        return model, preprocess
    
class MobileNetV2_0_75_192(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"
    def create_model(self) -> Any:
        model = resolve_mobilenet(input_size=192, model_width=0.75)
        preprocess = keras_models.mobilenet_v2.preprocess_input
        return model, preprocess
    
class MobileNetV2_0_75_224(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"
    def create_model(self) -> Any:
        model = resolve_mobilenet(input_size=224, model_width=0.75)
        preprocess = keras_models.mobilenet_v2.preprocess_input
        return model, preprocess
    
class MobileNetV2_1_96(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"
    def create_model(self) -> Any:
        model = resolve_mobilenet(input_size=96, model_width=1)
        preprocess = keras_models.mobilenet_v2.preprocess_input
        return model, preprocess
    
class MobileNetV2_1_160(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"
    def create_model(self) -> Any:
        model = resolve_mobilenet(input_size=160, model_width=1)
        preprocess = keras_models.mobilenet_v2.preprocess_input
        return model, preprocess
    
class MobileNetV2_1_192(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"
    def create_model(self) -> Any:
        model = resolve_mobilenet(input_size=192, model_width=1)
        preprocess = keras_models.mobilenet_v2.preprocess_input
        return model, preprocess
    
class MobileNetV2_1_224(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"
    def create_model(self) -> Any:
        model = resolve_mobilenet(input_size=224, model_width=1)
        preprocess = keras_models.mobilenet_v2.preprocess_input
        return model, preprocess
    
class MobileNetV2_1_3_224(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"
    def create_model(self) -> Any:
        model = resolve_mobilenet(input_size=224, model_width=1.3)
        preprocess = keras_models.mobilenet_v2.preprocess_input
        return model, preprocess
    
class MobileNetV2_1_4_224(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"
    def create_model(self) -> Any:
        model = resolve_mobilenet(input_size=224, model_width=1.4)
        preprocess = keras_models.mobilenet_v2.preprocess_input
        return model, preprocess
    