import argparse
import sys
import os

import torch
import torchvision.utils as utils
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from torch.nn.parallel.data_parallel import DataParallel

from source.dataset import CelebA
from source.autoencoders.vqvae import VQVAE
from source.distributions import DiscMixLogistic
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


def train(epoch, loader, model, optimizer, scheduler, device, entropy_coder, entropy_coder_bottom, poptimizer, sample_size=8):
    # if dist.is_primary():
    loader = tqdm(loader)

    criterion = DiscMixLogisticLoss()
    ent_loss = CECondLoss()
    #nn.MSELoss()

    mse_sum = 0
    mse_n = 0
    acc_sum = 0

    latent_loss_weight = 0.25 * 100000
    beta = 10

    for i, data in enumerate(loader):
        img = data[0]
        s = data[1]
        # img = data['input']
        # s = data['sensitive']

        model.zero_grad()
        entropy_coder.zero_grad()
        entropy_coder_bottom.zero_grad()

        # beta += 10**(-2) * i / len(loader)
        # beta = min(1.0, beta)

        img = img.to(device)
        s = s.to(device)

        out, latent_loss, id_t, id_b = model(img, s)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()

        if i % 2 == 0:

            logits, _ = entropy_coder(id_t)
            prior_loss = ent_loss(logits, id_t).reshape(img.shape[0], -1).sum(1).mean()

            logits_b, _ = entropy_coder_bottom(id_b, condition=id_t)
            prior_loss += ent_loss(logits_b, id_b).reshape(img.shape[0], -1).sum(1).mean()

            loss = recon_loss + latent_loss_weight * latent_loss + beta * prior_loss
            loss.backward()
            optimizer.step()

        else:
            logits, _ = entropy_coder(id_t.detach())
            prior_loss = ent_loss(logits, id_t.detach()).reshape(img.shape[0], -1).sum(1).mean()

            logits_b, _ = entropy_coder_bottom(id_b.detach(), condition=id_t.detach())
            prior_loss += ent_loss(logits_b, id_b.detach()).reshape(img.shape[0], -1).sum(1).mean()

            prior_loss.backward()
            poptimizer.step()

        if scheduler is not None:
            scheduler.step()



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
                (   f"Iteration: {i}"
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                    f" prior loss: {prior_loss.item(): .3f}"
                    f" accuracy: {acc_sum / mse_n: .3f}"
                    f"beta: {beta}"
                )
            )

        os.makedirs("/scratch/xgitiaux/samples/vqvae", exist_ok=True)
        if i % 200 == 0:
            model.eval()

            sample = img[:sample_size]
            s = s[:sample_size]

            with torch.no_grad():
                out, _, _, _ = model(sample, s)
                num_mix = int(out.shape[1] / 10)
                disc = DiscMixLogistic(out, num_mix=num_mix)
                out = disc.sample()

            utils.save_image(
                torch.cat([sample, out], 0),
                f"/scratch/xgitiaux/samples/vqvae/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                nrow=sample_size,
            )

            model.train()


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
    dataset = (wds.Dataset(url, length=162000 // 16)
               .shuffle(500)
               .decode("pil")
               .to_tuple("input.jpg", "sensitive.cls")
               .map_tuple(preproc, identity)
               .batched(16)
               )

    loader = DataLoader(dataset, batch_size=None, num_workers=16)
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
            2,
            2,
            64,
            n_out_res_block=0,
        ).to(device)

    entropy_coder_bottom = PixelSNAIL(
        [64, 64],
        512,
        64,
        5,
        2,
        2,
        64,
        n_out_res_block=0,
        n_cond_res_block=2,
        cond_res_channel=64,
        attention=False
    ).to(device)

    if torch.cuda.device_count() > 1:
        logger.info(f'Number of gpu is {torch.cuda.device_count()}')
        entropy_coder = _CustomDataParallel(entropy_coder)
        entropy_coder_bottom = _CustomDataParallel(entropy_coder_bottom)
        #PixelCNN(ncode=512, channels_in=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    poptimizer = optim.Adam(list(entropy_coder.parameters()) + list(entropy_coder_bottom.parameters()),
                            lr=args.lr)
    scheduler = None


    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device, entropy_coder, entropy_coder_bottom, poptimizer)

        os.makedirs("/scratch/xgitiaux/checkpoint/vqvae", exist_ok=True)
        torch.save(model.state_dict(), f"/scratch/xgitiaux/checkpoint/vqvae/two_q_vqvae_{str(i + 1).zfill(3)}.pt")

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