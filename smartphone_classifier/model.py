import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    model = vit_b_16(weights=weights)
    model.heads.head = nn.Linear(768, num_classes)
    return model
