import torch.nn as nn

class UResNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)