import torch

from source.distributions import DiscMixLogistic
from source.losses.templates import TemplateLoss

class DiscMixLogisticLoss(TemplateLoss):
    """
    Discrete logistic loss.
    """

    def __init__(self, beta=0):
        super().__init__()
        self.name = 'DiscMixLogisticLoss'
        self.beta = beta

    def forward(self, target, output):
        """

        :param output: reconstructed input (B, M * 6 + M, W, H)
        :param target: initial input (B, C, W, H)
        :return: mean squared loss
        """
        num_mix = int(output.shape[1] / 10)
        dist = DiscMixLogistic(output, num_mix=num_mix)
        rec_loss = - dist.log_prob(target)
        rec_loss = torch.mean(rec_loss.sum(dim=[1, 2]))

        return rec_loss