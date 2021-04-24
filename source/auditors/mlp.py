import torch
import torch.nn as nn

from source.template_model import TemplateModel
from source.model_utils import activations


class MLP(TemplateModel):
    """
    Implement a basic fully connected mlp with RELU/PRELU/Sigmoid/tanh activation
    to predict sensitive attribute. Returns logit or activation_out(logit)
    """

    def __init__(self, zdim, depth=2, width=8, nclass=1, activation="RELU", activation_out=None):
        super().__init__()

        fc_list = []
        in_dim = zdim
        out_dim = width
        for _ in range(depth - 1):
            fc_list.append(nn.Sequential(nn.Linear(in_dim, out_dim),
                                         nn.BatchNorm1d(out_dim),
                                         nn.ELU()))
            in_dim = out_dim

        self.hidden_layers = nn.Sequential(*fc_list)
        self.final_layer = nn.Linear(width, nclass)

        if activation_out is not None:
            self.final_layer = nn.Sequential(nn.Linear(in_dim, nclass),
                                         nn.BatchNorm1d(out_dim),
                                         activations[activation_out])

        self.param_init()

    def forward(self, x):
        """
        :param x: (B, zdim)
        :return: (B, 1)
        """
        output = self.hidden_layers(x)
        output = self.final_layer(output)

        return output

    def param_init(self):
        """
        Xavier's initialization
        """
        for layer in self.modules():
            if hasattr(layer, 'weight'):

                if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.PReLU, nn.Tanh)):
                    nn.init.normal_(layer.weight, mean=1., std=0.02)
                else:
                    nn.init.xavier_normal_(layer.weight)
            if hasattr(layer, 'bias'):
                nn.init.constant_(layer.bias, 0.)
