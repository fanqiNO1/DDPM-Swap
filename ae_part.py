# Author: fanqiNO1
# Date: 2022-06-10
# Description:
# Based on the https://github.com/neuralchen/SimSwap/blob/main/models/fs_networks_fix.py


import torch
import torch.nn as nn


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x)  # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class ApplyStyle(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """

    def __init__(self, latent_size, channels):
        super(ApplyStyle, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        # x = x * (style[:, 0] + 1.) + style[:, 1]
        x = x * (style[:, 0] * 1 + 1.) + style[:, 1] * 1
        return x


class ResnetBlock_Adain(nn.Module):
    def __init__(self, dim, latent_size, padding_type, activation=nn.ReLU(True)):
        super(ResnetBlock_Adain, self).__init__()

        p = 0
        conv1 = []
        if padding_type == 'reflect':
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        conv1 += [nn.Conv2d(dim, dim, kernel_size=3,
                            padding=p), InstanceNorm()]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle(latent_size, dim)
        self.act1 = activation

        p = 0
        conv2 = []
        if padding_type == 'reflect':
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        conv2 += [nn.Conv2d(dim, dim, kernel_size=3,
                            padding=p), InstanceNorm()]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle(latent_size, dim)

    def forward(self, x, dlatents_in_slice):
        y = self.conv1(x)
        y = self.style1(y, dlatents_in_slice)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.style2(y, dlatents_in_slice)
        out = x + y
        return out


class Down(nn.Module):
    def __init__(self, in_channels, dim):
        super(Down, self).__init__()
        activation = nn.ReLU(True)

        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(in_channels, 64, kernel_size=7, padding=0),
                                         nn.BatchNorm2d(64), activation)
        # downsample
        self.down1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(128), activation)
        self.down2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(256), activation)
        self.down3 = nn.Sequential(nn.Conv2d(256, dim, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(dim), activation)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x


class BottleNeck(nn.Module):
    def __init__(self, dim, latent_size, n_blocks, padding_type="reflect", activation=nn.ReLU(True)):
        super(BottleNeck, self).__init__()
        BN = []
        for i in range(n_blocks):
            BN += [
                ResnetBlock_Adain(dim, latent_size=latent_size, padding_type=padding_type, activation=activation)]
        self.BottleNeck = nn.Sequential(*BN)

    def forward(self, x, dlatents):
        for i in range(len(self.BottleNeck)):
            x = self.BottleNeck[i](x, dlatents)
        return x


class Up(nn.Module):
    def __init__(self, dim, out_channels):
        super(Up, self).__init__()
        activation = nn.ReLU(True)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dim, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), activation
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), activation
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), activation
        )
        self.last_layer = nn.Sequential(nn.ReflectionPad2d(
            3), nn.Conv2d(64, out_channels, kernel_size=7, padding=0))

    def forward(self, x):
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.last_layer(x)
        return x
