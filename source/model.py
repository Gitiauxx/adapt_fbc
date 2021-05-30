import torch
import numpy as np

from torch.nn.parallel.data_parallel import DataParallel

from source.utils import get_logger, count_parameters_in_M
from source.losses import *
from source.autoencoders import *
from source.auditors import *

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


class Model(object):
    """
    pytorch model with loss and neural network autoencoder/fairness auditor
    """

    def __init__(self, net, loss, ploss, pmodel, learning_rate={'autoencoder': 0.001, 'lr_min': 0.0001, 'nepochs': 100},
                 device='cpu', beta=0, gamma=0, method='compression', annealing_epochs=0, warmup_epochs=0):

        device = torch.device(device)
        self.net = net.to(device)

        self.loss = loss

        self.pmodel = pmodel.to(device)
        self.ploss = ploss

        logger.info('param size for autoencoder = %fM ', count_parameters_in_M(net))
        logger.info('param size for entropy encoder = %fM ', count_parameters_in_M(pmodel))

        self.learning_rate = learning_rate['autoencoder']
        self.learning_rate_p = learning_rate['pmodel']

        self.device = device
        self.beta = beta
        self.gamma = gamma
        self.method = method
        self.annealing_epochs = annealing_epochs
        self.warmup_epochs = warmup_epochs

        self.optimizer = torch.optim.Adam(list(self.net.parameters()), lr=self.learning_rate, betas=(0.5, 0.999),
                                              weight_decay=1e-5)

        self.optimizer_pmodel = torch.optim.Adam(list(self.pmodel.parameters()), lr=self.learning_rate_p,
                                                betas=(0.5, 0.999), weight_decay=1e-5)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, float(learning_rate['nepochs'] - warmup_epochs - 1), eta_min=learning_rate['lr_min'])

    @classmethod
    def from_dict(cls, config_dict):
        """
        Create a model input configuration from a config dictionary

        Parameters
        ----------
        config_dict : configuration dictionary

        """
        name_net = config_dict['net'].pop('name')
        beta = config_dict['beta']
        gamma = config_dict['gamma']

        method = config_dict['method']

        annealing_epochs = 0
        if 'annealing_epochs' in config_dict:
            annealing_epochs = config_dict['annealing_epochs']

        warmup_epochs = 0
        if 'warmup_epochs' in config_dict:
            warmup_epochs = config_dict['warmup_epochs']

        learning_rate = config_dict['net'].pop('learning_rate')
        lr = {'autoencoder': learning_rate, 'lr_min': config_dict['lr_min'],
              'nepochs': config_dict['n_epochs']}

        learning_rate_p = config_dict['pmodel'].pop('learning_rate')
        lr['pmodel'] = learning_rate_p

        net = globals()[name_net].from_dict(config_dict['net'])

        if torch.cuda.device_count() > 1:
            logger.info(f'Number of gpu is {torch.cuda.device_count()}')
            net = _CustomDataParallel(net)

        name_loss = config_dict['loss'].pop('name')
        device = config_dict['device']

        loss = globals()[name_loss].from_dict(config_dict['loss'])

        name_ploss = config_dict['ploss'].pop('name')
        name_pmodel = config_dict['pmodel'].pop('name')
        ploss = globals()[name_ploss].from_dict(config_dict['ploss'])
        pmodel = globals()[name_pmodel].from_dict(config_dict['pmodel'])

        if torch.cuda.device_count() > 1:
            pmodel = _CustomDataParallel(pmodel)

        model = cls(net, loss, ploss, pmodel, learning_rate=lr,
                    device=device, beta=beta, gamma=gamma, method=method,
                    annealing_epochs=annealing_epochs, warmup_epochs=warmup_epochs)

        return model

    def optimize_parameters(self, x, y, s, autoencoder=True):

        """
        Optimization of both autoencoder
        :param x: input
        :param target:
        :param sensitive:
        :return:
        """
        self.optimizer.zero_grad()
        self.optimizer_pmodel.zero_grad()

        beta = self.beta * torch.rand_like(s[:, 0])
        output, q, mask, centers, commit_loss, embed_loss = self.net.forward(x, s, beta)

        logits = self.pmodel.forward(q)

        loss = self.loss.forward(y, output)

        ploss = self.ploss.forward(centers, logits)
        ploss = ploss.reshape(x.shape[0], -1) * mask

        loss = loss + self.gamma * (beta * ploss.sum(dim=[1])).mean(0) + 0.0 * commit_loss.mean() + 0.0 * embed_loss.mean()

        if autoencoder:
            loss.backward()
            self.optimizer.step()

        q = q.detach()
        centers = centers.detach()
        logits = self.pmodel.forward(q)
        ploss = self.ploss.forward(centers, logits)
        ploss_mean = ploss.sum(dim=[1, 2]).mean(0)

        ploss_mean.backward()
        self.optimizer_pmodel.step()

        return loss

    def train(self, train_loader, validation_loader, n_epochs, writer,
              chkpt_dir=None, save=True):

        if save:
            assert chkpt_dir is not None

        for epoch in range(n_epochs):

            train_loss = 0

            for batch_idx, batch in enumerate(train_loader):

                if epoch > self.warmup_epochs:
                    self.scheduler.step()

                if epoch < self.annealing_epochs:
                    annealing_factor = (float(batch_idx + epoch * len(train_loader)) /
                                        float(self.annealing_epochs * len(train_loader)))

                else:
                    annealing_factor = 1.0

                autoencoder = True

                self.gamma = annealing_factor

                input = batch['input'].to(self.device)
                target = batch['target'].to(self.device)
                sensitive = batch['sensitive'].to(self.device)

                loss = self.optimize_parameters(input, target, sensitive, autoencoder=autoencoder)
                train_loss += loss.detach() * len(input) / len(train_loader.dataset)

            writer['training']['rec_loss'][epoch] = train_loss.cpu().item()

            logger.info(f'Epoch: {epoch} Train loss: {train_loss}')

            if ((epoch % 5 == 0) | (epoch == n_epochs - 1)):
                for beta in [0.0, self.beta * self.gamma / 2 , self.beta * self.gamma]:
                    val_loss, accuracy, s_loss, entr_loss, active_bits = self.eval(validation_loader, beta)
                    logger.info(f'Epoch: {epoch} Beta {beta}: Validation loss: {val_loss}')
                    logger.info(f'Epoch: {epoch} Beta {beta}: Accuracy of Entropy Model: {accuracy}')
                    logger.info(f'Epoch: {epoch} Beta {beta}: Entropy Model: {entr_loss}')
                    logger.info(f'Epoch: {epoch} Beta {beta}: Differences of Q by S: {s_loss}')
                    logger.info(f'Epoch: {epoch} Beta {beta}: Max values code: {self.net.quantize.embed.max()}, '
                                f'{self.net.quantize.embed.min()}')
                    logger.info(f'Epoch: {epoch} Beta {beta}: Number of points per code: {self.net.quantize.cluster_size}}')


            if (save) & ((epoch % 10 == 0) | (epoch == n_epochs - 1)):
                model_dict = {'epoch': epoch,
                              'loss': self.loss,
                              'model_state_dict': self.net.state_dict(),
                              'optimizer_state_dict': self.optimizer.state_dict()}

                torch.save(model_dict, f'{chkpt_dir}/epoch_{epoch}')

                pmodel_dict = {'epoch': epoch,
                               'loss': self.loss,
                               'model_state_dict': self.pmodel.state_dict(),
                               'optimizer_state_dict': self.optimizer_pmodel.state_dict()}

                torch.save(pmodel_dict, f'{chkpt_dir}/pmodel_epoch_{epoch}')

                code = self.net.quantize.embed.detach().cpu().numpy()
                np.save(f'{chkpt_dir}/code_epoch_{epoch}.npy', code)

    def eval(self, data_loader, beta):
        """
        Measure reconstruction loss for self.net and loss for self.auditor
        :param data_loader: torch.DataLoader
        :return: reconstruction loss, auditor accuracy
        """
        rec_loss = 0
        accuracy = 0
        s_loss = 0
        entr_loss = 0
        active_bits = 0

        #self.net.eval()

        beta = torch.tensor([beta])

        for _, batch in enumerate(data_loader):
            x = batch['input'].to(self.device)
            s = batch['sensitive'].to(self.device)
            y = batch['target'].to(self.device)

            b = beta.expand_as(s[:, 0]).to(self.device)
            out, q, mask, centers, z, _ = self.net.forward(x, s, b, training=False)

            q = q.detach()
            out = out.detach()
            mask = mask.detach()
            z = z.detach()

            loss = self.loss.forward(y, out)
            rec_loss += loss.detach() * len(x) / len(data_loader.dataset)

            logits = self.pmodel.forward(q)
            pred = logits.argmax(1)
            acc = (pred == centers).float().mean()
            accuracy += acc.detach() * len(x) / len(data_loader.dataset)

            # bs = b[:, None, ...] * mask[:, None, ...]
            # se = s[:, :, None]
            # b_loss = torch.abs((bs * se).sum(0) / se.sum(0) - bs.mean(0)).sum(dim=[1, 0])
            # s_loss += b_loss.cpu().detach() * len(x) / len(data_loader.dataset)

            ploss = self.ploss.forward(centers, logits)
            entr_loss += ploss.sum(dim=[1, 2]).mean().detach() * len(x) / len(data_loader.dataset)

            act = mask.sum(1).mean(0)
            active_bits += act.detach() * len(x) / len(data_loader.dataset)

        return rec_loss.cpu(), accuracy.cpu(), s_loss, entr_loss.cpu(), active_bits.cpu()
