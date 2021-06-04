import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from source.dataset import CelebA
from source.autoencoders.vqvae import VQVAE



def train(epoch, loader, model, optimizer, scheduler, device):
    # if dist.is_primary():
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25

    for i, data in enumerate(loader):
        img = data['input']

        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        #comm = dist.all_gather(comm)

        # for part in comm:
        mse_sum = comm["mse_sum"]
        mse_n = comm["mse_n"]

        #if dist.is_primary():
        lr = optimizer.param_groups[0]["lr"]

        loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

def eval(epoch, loader, model, device):

    loader = tqdm(loader)

    criterion = nn.MSELoss()

    model.eval()

    mse_sum = 0
    mse_n = 0

    for i, data in enumerate(loader):
        img = data['input']
        img = img.to(device)

        out, latent_loss = model(img)
        out = out.detach()
        latent_loss.detach()
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}

        mse_sum += comm["mse_sum"]
        mse_n += comm["mse_n"]

        loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                )
            )

    model.train()

def main(args):
    device = "cuda"


    dataset = CelebA(args.path, split='train')
    loader = DataLoader(dataset, batch_size=128 // args.n_gpu, shuffle=True)

    validation_dataset = CelebA(args.path, split='valid')
    validation_loader = DataLoader(validation_dataset, batch_size=128 // args.n_gpu)

    model = VQVAE().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None


    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device)

        os.makedirs("/scratch/xgitiaux/checkpoint/vqvae", exist_ok=True)
        torch.save(model.state_dict(), f"/scratch/xgitiaux/checkpoint/vqvae/vqvae_{str(i + 1).zfill(3)}.pt")

        eval(i, validation_loader, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    #parser.add_argument("--sched", type=str)
    parser.add_argument("--path", type=str)

    args = parser.parse_args()

    print(args)

    main(args)

    #dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))