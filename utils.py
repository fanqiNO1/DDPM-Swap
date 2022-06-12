# Author: fanqiNO1
# Date: 2022-06-10
# Description:
# Based on the https://github.com/neuralchen/SimSwap/blob/main/models/base_model.py

import torch
import torch.nn.functional as F
import numpy as np
import imageio

def save(model, path):
    torch.save(model.state_dict(), path)

def load(model, path):
    pass

def sample(ddpm, ae, arcface, source_image, target_image, args):
    ddpm.eps_model.eval()
    ae.eval()
    with torch.no_grad():
        x = torch.randn([args.n_samples, source_image.shape[1], source_image.shape[2], source_image.shape[3]]).to(args.device)
        for i in range(args.n_steps):
            t = args.n_steps - i - 1
            x = ddpm.p_sample(x, x.new_full((args.n_samples,), t, dtype=torch.long))
            if i > args.range_t:
                source_image = ddpm.q_sample(source_image, t, eps=torch.randn(source_image.shape).to(args.device))
                target_image = F.interpolate(target_image, size=(112, 112), mode="bicubic")
                target_latent = arcface(target_latent)
                x = x - ae(x, None) + ae(source_image, target_latent)
    x = x.cpu().numpy()
    return x[0]


def plot(ddpm, ae, arcface, source_images, target_images, epoch, args):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    n, w, h = source_images.shape[1], source_images.shape[2], source_images.shape[3]
    result = np.zeros((w * (args.batch_size + 1), h * (args.batch_size + 1), n))
    for i in range(args.batch_size + 1):
        for j in range(args.batch_size + 1):
            if i == 0 and j != 0:
                target_image = target_images[j - 1, :, :, :].cpu().numpy()
                target_image = target_image * std + mean
                target_image = target_image.transpose(1, 2, 0)
                result[j * w:(j + 1) * w, :, :] = target_image
            elif i != 0 and j == 0:
                source_image = source_images[i - 1, :, :, :].cpu().numpy()
                source_image = source_image * std + mean
                source_image = source_image.transpose(1, 2, 0)
                result[:, i * h:(i + 1) * h, :] = source_image
            elif i != 0 and j != 0:
                source_image = source_images[i-1, :, :, :]
                target_image = target_images[j-1, :, :, :]
                sample_image = sample(ddpm, ae, arcface, source_image, target_image, args)
                sample_image = sample_image * std + mean
                sample_image = sample_image.transpose(1, 2, 0)
                result[j * w:(j + 1) * w, i * h:(i + 1) * h, :] = sample_image
    imageio.imwrite(f"{args.sample_path}/{epoch}.png", result)