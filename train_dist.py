import os
import builtins
import argparse
import torch
import numpy as np
import tqdm
import random
import torch.distributed as dist

from source.autoencoders.vqvae import VQVAE
from source.dataset import CelebA
from source.losses.discmixlogistic_loss import DiscMixLogisticLoss
from source.utils import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='start epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='local rank for distributed training')
    args = parser.parse_args()
    return args


def main(args):
    # DDP setting
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1

    os.makedirs("/scratch/xgitiaux/checkpoint/vqvae_dist", exist_ok=True)
    print("Create folder vqvae_dist")

    if args.distributed:
        if args.local_rank != -1:  # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ:  # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    #
    # # suppress printing if not on master gpu
    # if args.rank != 0:
    #     def print_pass(*args):
    #         pass
    #
    #     builtins.print = print_pass
    #
    # ### model ###
    model = VQVAE(cout=30)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)

    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    #if args.rank == 0:


    ### optimizer ###
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # ### resume training if necessary ###
    # if args.resume:
    #     pass
    #
    # ### data ###
    dataset = CelebA(args.path, split='train', range_data=10000)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True)

    print(len(train_loader))
    #
    # # val_dataset = MyDataset(mode='val')
    # # val_sampler = None
    # # val_loader = torch.utils.data.DataLoader(
    # #     val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
    # #     num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)
    #
    # torch.backends.cudnn.benchmark = True
    #
    # ### main loop ###
    # for epoch in range(args.start_epoch, args.epochs):
    #     np.random.seed(epoch)
    #     random.seed(epoch)
    #     # fix sampling seed such that each gpu gets different part of dataset
    #     if args.distributed:
    #         train_loader.sampler.set_epoch(epoch)
    #
    #     # adjust lr if needed #
    #
    #     train(epoch, train_loader, model, optimizer)
    #     # if args.rank == 0:  # only val and save on master node
    #     #     validate(val_loader, model, criterion, epoch, args)
    #         # save checkpoint if needed #
    #
    #     if args.rank == 0:
    #         os.makedirs("/scratch/xgitiaux/checkpoint/vqvae_dist", exist_ok=True)
    #         torch.save(model.state_dict(), f"/scratch/xgitiaux/checkpoint/vqvae/vqvae_{str(epoch + 1).zfill(3)}.pt")


def train(epoch, loader, model, optimizer):
    # if dist.is_primary():
    loader = tqdm(loader)

    criterion = DiscMixLogisticLoss()
    #nn.MSELoss()

    mse_sum = 0
    mse_n = 0
    acc_sum = 0

    latent_loss_weight = 0.25 * 100000
    beta = 1.0

    for i, data in enumerate(loader):
        img = data['input']
        s = data['sensitive']

        model.zero_grad()
        #ent_loss.zero_grad()

        # beta += 10**(-2) * i / len(loader)
        # beta = min(1.0, beta)

        img = img.cuda()
        s = s.cuda()

        out, latent_loss, id_t = model(img, s)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()

        logger.info(recon_loss.item())

        # logits, _ = entropy_coder(id_t)
        # prior_loss = ent_loss(logits, id_t).reshape(img.shape[0], -1).sum(1).mean()

        loss = recon_loss + latent_loss_weight * latent_loss \
               #+ beta * prior_loss
        loss.backward()

        # logits, _ = entropy_coder(id_t.detach())
        # prior_loss = ent_loss(logits, id_t).reshape(img.shape[0], -1).sum(1).mean()
        #
        # prior_loss.backward()
        #
        # if scheduler is not None:
        #     scheduler.step()
        optimizer.step()

        #poptimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        #comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        #comm = dist.all_gather(comm)

        # for part in comm:
        # mse_sum += comm["mse_sum"]
        # mse_n +=comm["mse_n"]
        #
        # pred = logits.argmax(1)
        # acc_sum += (pred == id_t).float().reshape(img.shape[0], -1).mean(1).sum()

        #if dist.is_primary():
        #lr = optimizer.param_groups[0]["lr"]

        loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    # f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    # f"lr: {lr:.5f}"
                    # f" prior loss: {prior_loss.item(): .3f}"
                    # f" accuracy: {acc_sum / mse_n: .3f}"
                    # f"beta: {beta}"
                )
            )


def validate(val_loader, model, criterion, epoch, args):
    pass


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)