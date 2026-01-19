import torch.nn as nn
import torchvision.models as models

class ResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights="IMAGENET1K_V1")
        self.layers = nn.Sequential(*list(backbone.children())[:-2])

    def forward(self, x):
        return self.layers(x)