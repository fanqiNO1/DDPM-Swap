import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from Unet import Unet
from ddpm import DenoiseDiffusion
from ae import AE
from dataloader import SwapDataset, get_DataLoader
from utils import save, load, plot


def main(args):
    if args.load:
        # Load model
        pass
    unet = Unet(n_channels=args.n_channels,
                n_groups=args.n_groups).to(args.device)
    unet_optim = torch.optim.Adam(unet.parameters(), lr=args.unet_lr)
    ddpm = DenoiseDiffusion(
        eps_model=unet, n_steps=args.n_steps, device=args.device)
    ae = AE(n_blocks=args.ae_blocks).to(args.device)
    ae_optim = torch.optim.Adam(ae.parameters(), lr=args.ae_lr)
    arcface_ckpt = torch.load(
        args.arcface_path, map_location=torch.device("cpu"))
    arcface = arcface_ckpt['model'].module
    arcface = arcface.to(args.device)
    arcface.eval()
    arcface.requires_grad_(False)
    # Load data
    dataset = SwapDataset(args.data_path)
    dataloader = get_DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    open(f"{args.loss_path}/loss.txt", "w").close()
    # Train
    for epoch in range(args.epochs):
        ddpm.eps_model.train()
        ae.train()
        source_images, target_images = next(iter(dataloader))
        source_images = source_images.to(args.device)
        target_images = target_images.to(args.device)
        target_images_112 = F.interpolate(
            target_images, size=(112, 112), mode="bicubic")
        target_latent = arcface(target_images_112)
        target_latent = F.normalize(target_latent, p=2, dim=1)
        # Train ddpm
        unet_loss = ddpm.loss(source_images)
        unet_optim.zero_grad()
        unet_loss.backward()
        unet_optim.step()
        # Train ae
        fake_images = ae(source_images, target_latent)
        fake_images_112 = F.interpolate(
            fake_images, size=(112, 112), mode="bicubic")
        fake_latent = arcface(fake_images_112)
        fake_latent = F.normalize(fake_latent, p=2, dim=1)
        loss_id = (1 - ae.cosin_metric(fake_latent, target_latent)).mean()
        loss_rec = nn.L1Loss()(fake_images, source_images)
        ae_loss = args.lambda_id * loss_id + args.lambda_rec * loss_rec
        ae_optim.zero_grad()
        ae_loss.backward()
        ae_optim.step()
        # Save model
        if (epoch + 1) % args.save_interval == 0:
            save(ae, f"{args.save_path}/ae_{epoch}.pth")
            save(ddpm.eps_model, f"{args.save_path}/unet_{epoch}.pth")
        # Print loss
        if (epoch + 1) % args.loss_interval == 0:
            with open(f"{args.loss_path}/loss.txt", "a") as f:
                f.write(f"{epoch + 1}: unet: {unet_loss.item():.4f}, ae: {ae_loss.item():.4f}\n")
        # Plot
        if (epoch + 1) % args.sample_interval == 0:
            plot(ddpm, ae, arcface, source_images, target_images, epoch, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # For dataset
    parser.add_argument("--data_path", type=str,
                        default=".", help="dataset path")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of workers")
    # For Unet
    parser.add_argument("--n_channels", type=int,
                        default=16, help="number of channels")
    parser.add_argument("--n_groups", type=int, default=8,
                        help="number of group norm")
    parser.add_argument("--unet_lr", type=float,
                        default=2e-5, help="unet learning rate")
    # For DDPM
    parser.add_argument("--n_steps", type=int,
                        default=1000, help="number of steps")
    parser.add_argument("--n_samples", type=int,
                        default=16, help="number of samples")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--range_t", type=int, default=0, help="range of t")
    # For AE
    parser.add_argument("--lambda_id", type=float,
                        default=30.0, help="weight for id loss")
    parser.add_argument("--lambda_rec", type=float,
                        default=10.0, help="weight for rec loss")
    parser.add_argument("--ae_lr", type=float,
                        default=2e-5, help="ae learning rate")
    parser.add_argument("--ae_blocks", type=int,
                        default=7, help="number of blocks")
    # For Arcface
    parser.add_argument("--arcface_path", type=str,
                        default=".", help="arcface path")
    # For train
    parser.add_argument("--load_path", type=str, default=".", help="load path")
    parser.add_argument("--load", type=bool, default=False, help="load")
    parser.add_argument("--epochs", type=int,
                        default=200000, help="number of epochs")
    parser.add_argument("--save_interval", type=int,
                        default=50000, help="save interval")
    parser.add_argument("--save_path", type=str, default=".", help="save path")
    parser.add_argument("--loss_interval", type=int,
                        default=100, help="loss interval")
    parser.add_argument("--loss_path", type=str, default=".", help="loss path")
    parser.add_argument("--sample_interval", type=int,
                        default=2000, help="sample interval")
    parser.add_argument("--sample_path", type=str,
                        default=".", help="sample path")

    args = parser.parse_args()
    main(args)
