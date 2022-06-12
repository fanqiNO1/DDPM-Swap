# Author: fanqiNO1
# Date: 2022-06-10
# Description:
# Based on the https://nn.labml.ai/diffusion/ddpm/unet.html


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x) * x


class TimeEmbedding(nn.Module):
    def __init__(self, n_channels):
        super(TimeEmbedding, self).__init__()
        self.n_channels = n_channels
        self.linear_1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.activation = SiLU()
        self.linear_2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t):
        half_dim = self.n_channels // 8
        emb = -math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.tensor(torch.arange(
            half_dim) * emb, dtype=torch.float32)).to(t.device)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)

        emb = self.linear_1(emb)
        emb = self.activation(emb)
        emb = self.linear_2(emb)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, n_groups):
        super(ResidualBlock, self).__init__()
        self.norm_1 = nn.GroupNorm(n_groups, in_channels)
        self.activation_1 = SiLU()
        self.conv_1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1)
        self.norm_2 = nn.GroupNorm(n_groups, out_channels)
        self.activation_2 = SiLU()
        self.conv_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        self.time_emb = nn.Linear(time_channels, out_channels)

    def forward(self, x, t):
        h = self.norm_1(x)
        h = self.activation_1(h)
        h = self.conv_1(h)
        h += self.time_emb(t)[:, :, None, None]
        h = self.norm_2(h)
        h = self.activation_2(h)
        h = self.conv_2(h)
        h += self.shortcut(x)
        return h


class AttentionBlock(nn.Module):
    def __init__(self, n_channels, n_groups, n_heads=1, d_k=None):
        super(AttentionBlock, self).__init__()
        if d_k is None:
            d_k = n_channels

        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x, t=None):

        b, c, h, w = x.shape
        x = x.reshape(b, c, -1).permute(0, 2, 1)
        qkv = self.projection(x).reshape(b, -1, self.n_heads, self.d_k * 3)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=-1)

        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.reshape(b, -1, self.n_heads * self.d_k)

        res = self.output(res)
        res += x
        res = res.permute(0, 2, 1).reshape(b, c, h, w)
        return res


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, n_groups, has_attn):
        super(DownBlock, self).__init__()
        self.res = ResidualBlock(
            in_channels, out_channels, time_channels, n_groups)
        if has_attn:
            self.attn = AttentionBlock(out_channels, n_groups)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, n_groups, has_attn):
        super(UpBlock, self).__init__()
        self.res = ResidualBlock(
            in_channels + out_channels, out_channels, time_channels, n_groups)
        if has_attn:
            self.attn = AttentionBlock(out_channels, n_groups)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channels, time_channels, n_groups):
        super(MiddleBlock, self).__init__()
        self.res_1 = ResidualBlock(
            n_channels, n_channels, time_channels, n_groups)
        self.attn = AttentionBlock(n_channels, n_groups)
        self.res_2 = ResidualBlock(
            n_channels, n_channels, time_channels, n_groups)

    def forward(self, x, t):
        x = self.res_1(x, t)
        x = self.attn(x)
        x = self.res_2(x, t)
        return x


class Upsample(nn.Module):
    def __init__(self, n_channels):
        super(Upsample, self).__init__()
        self.conv = nn.ConvTranspose2d(
            n_channels, n_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, n_channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels,
                              kernel_size=4, stride=2, padding=1)

    def forward(self, x, t):
        return self.conv(x)
