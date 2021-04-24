import torch
import torch.nn as nn

from source.losses.templates import TemplateLoss


class CECondLoss(TemplateLoss):
    """
    Implement a cross entropy loss with logits as torch BCEWithLogitsLoss
    """

    def __init__(self):
        super().__init__()

    def forward(self, target, prelogits):
        """

        :param target: (B, *)
        :param prelogits: (B, C, *)
        :return: (zdim, k) - target * log(sigmoid(prelogits)) - (1 - target) * log(1 - sigmoid(prelogits))
        """

        loss = nn.CrossEntropyLoss(reduction='none').forward(prelogits, target.long())
        return loss