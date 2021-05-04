from source.losses.l2_loss import L2Loss
from source.losses.ce_loss import CECondLoss
from source.losses.cross_entropy_loss import MCELoss
from source.losses.bernouilli_loss import BernouilliLoss
from source.losses.disclogistic_loss import DiscLogisticLoss
from source.losses.discmixlogistic_loss import DiscMixLogisticLoss

__all__ = ['L2Loss', 'CECondLoss', 'MCELoss', 'BernouilliLoss', 'DiscLogisticLoss', 'DiscMixLogisticLoss']