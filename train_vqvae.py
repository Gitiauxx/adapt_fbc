import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from torch.nn.parallel.data_parallel import DataParallel

from source.dataset import CelebA
from source.autoencoders.vqvae import VQVAE
from source.auditors.pixel_cnn import PixelCNN
from source.auditors.pixelsnail import PixelSNAIL
from source.losses.ce_loss import CECondLoss
from source.losses.discmixlogistic_loss import DiscMixLogisticLoss
from source.utils import get_logger
import torchvision.transforms as tf

import webdataset as wds

logger = get_logger(__name__)

class _CustomDataParallel(DataParallel):
    """
    DataParallel distribute batches across multiple GPUs

    https://github.com/pytorch/pytorch/issues/16885
    """

    def __init__(self, model):
        super(_CustomDataParallel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(_CustomDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def train(epoch, loader, model, optimizer, scheduler, device, entropy_coder, poptimizer):
    # if dist.is_primary():
    loader = tqdm(loader)

    criterion = DiscMixLogisticLoss()
    ent_loss = CECondLoss()
    #nn.MSELoss()

    mse_sum = 0
    mse_n = 0
    acc_sum = 0

    latent_loss_weight = 0.25 * 100000
    beta = 1.0

    for img, s in loader:
        # img = data['input']
        # s = data['sensitive']

        model.zero_grad()
        ent_loss.zero_grad()

        # beta += 10**(-2) * i / len(loader)
        # beta = min(1.0, beta)

        img = img.to(device)
        s = s.to(device)

        out, latent_loss, id_t = model(img, s)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()

        logits, _ = entropy_coder(id_t)
        prior_loss = ent_loss(logits, id_t).reshape(img.shape[0], -1).sum(1).mean()

        loss = recon_loss + latent_loss_weight * latent_loss + beta * prior_loss
        loss.backward()

        logits, _ = entropy_coder(id_t.detach())
        prior_loss = ent_loss(logits, id_t).reshape(img.shape[0], -1).sum(1).mean()

        prior_loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()
        poptimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        #comm = dist.all_gather(comm)

        # for part in comm:
        mse_sum += comm["mse_sum"]
        mse_n +=comm["mse_n"]

        pred = logits.argmax(1)
        acc_sum += (pred == id_t).float().reshape(img.shape[0], -1).mean(1).sum()

        #if dist.is_primary():
        lr = optimizer.param_groups[0]["lr"]

        loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                    f" prior loss: {prior_loss.item(): .3f}"
                    f" accuracy: {acc_sum / mse_n: .3f}"
                    f"beta: {beta}"
                )
            )


def identity(x):

    s = torch.zeros(2)
    if x == 1:
        s[0] = 1
    else:
        s[1] = 1

    return s


def main(args):
    device = "cuda"


    preproc = tf.Compose([tf.Resize(256), tf.CenterCrop(256), tf.ToTensor()])

    url = '../data_celeba_tar/train_{0..162}.tar'
    dataset = (wds.Dataset(url, length=162000 // 64)
               .shuffle(200)
               .decode("pil")
               .to_tuple("input.jpg", "sensitive.cls")
               .map_tuple(preproc, identity)
               .batched(64)
               )

    loader = DataLoader(dataset, batch_size=None, num_workers=8)
    #loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = VQVAE(cout=30).to(device)

    if torch.cuda.device_count() > 1:
        logger.info(f'Number of gpu is {torch.cuda.device_count()}')
        model = _CustomDataParallel(model)

    entropy_coder = PixelSNAIL(
            [32, 32],
            512,
            64,
            5,
            4,
            4,
            64,
            n_out_res_block=0,
        ).to(device)

    if torch.cuda.device_count() > 1:
        logger.info(f'Number of gpu is {torch.cuda.device_count()}')
        entropy_coder = _CustomDataParallel(entropy_coder)
        #PixelCNN(ncode=512, channels_in=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    poptimizer = optim.Adam(entropy_coder.parameters(), lr=args.lr)
    scheduler = None


    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device, entropy_coder, poptimizer)

        os.makedirs("/scratch/xgitiaux/checkpoint/vqvae", exist_ok=True)
        torch.save(model.state_dict(), f"/scratch/xgitiaux/checkpoint/vqvae/vqvae_{str(i + 1).zfill(3)}.pt")

        #eval(i, validation_loader, model, device)


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