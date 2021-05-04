import torch
from torch.distributions.bernoulli import Bernoulli

from source.losses.templates import TemplateLoss


class BernouilliLoss(TemplateLoss):
    """
    Compute L2 reconstruction loss
    """

    def __init__(self, beta=0):
        super().__init__()
        self.name = 'BernouilliLoss'
        self.beta = beta

    def forward(self, target, output):
        """

        :param output: reconstructed input (B, C, W, H)
        :param target: initial input (B, C, W, H)
        :return: mean squared loss
        """
        dist = Bernoulli(logits=output)
        rec_loss = - dist.log_prob(target)
        rec_loss = torch.mean(rec_loss.sum(dim=[1, 2, 3]))

        return rec_loss
