'''
Based On:
the robust models from MadryLab, their implementation in model-vs-human which was derived from Cadena's implementation
'''

from .custom import Custom

import torch
from torch.hub import load_state_dict_from_url
from torchvision.models import resnet50

model_urls = {"resnet50_l2_eps0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.ckpt",
              "resnet50_l2_eps0_01": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.01.ckpt",
              "resnet50_l2_eps0_03": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.03.ckpt",
              "resnet50_l2_eps0_05": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.05.ckpt",
              "resnet50_l2_eps0_1": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.1.ckpt",
              "resnet50_l2_eps0_25": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.25.ckpt",
              "resnet50_l2_eps0_5": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.5.ckpt",
              "resnet50_l2_eps1": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps1.ckpt",
              "resnet50_l2_eps3": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps3.ckpt",
              "resnet50_l2_eps5": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps5.ckpt"}

def _model(arch, model_fn, pretrained, progress, use_data_parallel, **kwargs):
    
    model = model_fn(pretrained=False)   

    if pretrained:
        checkpoint = load_state_dict_from_url(model_urls[arch], progress=progress)
        sd = {k[len('module.model.'):]:v for k,v in checkpoint['model'].items()\
              if k[:len('module.model.')] == 'module.model.'}  # Consider only the model and not normalizers or attacker
        model.load_state_dict(sd)
        
    model = torch.nn.DataParallel(model) if use_data_parallel else model 
        
    return model

class ResNet50_l2_eps0(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"

    def create_model(self) -> Any:
        model = _model('resnet50_l2_eps0', resnet50, pretrained, progress, use_data_parallel, **kwargs)
        return model, None
