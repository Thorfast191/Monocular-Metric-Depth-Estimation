import torch.nn as nn
from .encoders.resnet50 import ResNet50Encoder
from .decoders.uresnet_decoder import UResNetDecoder

class MonoDepthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet50Encoder()
        self.decoder = UResNetDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.squeeze(1)