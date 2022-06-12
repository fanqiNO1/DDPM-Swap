# Author: fanqiNO1
# Date: 2022-06-10
# Description:
# Based on the https://github.com/neuralchen/SimSwap/blob/main/models/fs_networks_fix.py

import torch
import torch.nn as nn

from ae_part import Down, BottleNeck, Up

class AE(nn.Module):
    def __init__(self, in_chanels=3, out_channels=3, latent_size=512, n_blocks=7):
        super(AE, self).__init__()
        self.down = Down(in_chanels, latent_size)
        self.bottleneck = BottleNeck(latent_size, latent_size, n_blocks)
        self.up = Up(latent_size, out_channels)

    def forward(self, x, dlatents=None):
        x = self.down(x)
        if dlatents is not None:
            x = self.bottleneck(x, dlatents)
        x = self.up(x)
        return x

    def cosin_metric(self, x1, x2):
        #return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))


if __name__ == "__main__":
    from torchinfo import summary
    model = AE()
    x = torch.zeros(1, 3, 224, 224)
    dlatents = torch.zeros(1, 512)
    with open("AE.txt", "w") as f:
        results = str(summary(model, input_data=[x, dlatents], device='cpu', depth=10))
        f.write(results)
