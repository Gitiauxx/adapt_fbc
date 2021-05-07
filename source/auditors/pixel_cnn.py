import torch
import torch.nn as nn
from source.template_model import TemplateModel

class MaskedCNN(nn.Conv2d):
    """
    Implementation of Masked CNN Class as explained in A Oord et. al.
    Taken from https://github.com/jzbontar/pixelcnn-pytorch
    """

    def __init__(self, mask_type, *args, residual=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_type = mask_type
        assert mask_type in ['A', 'B'], "Unknown Mask Type"

        self.register_buffer('mask', self.weight.data.clone())

        _, depth, height, width = self.weight.size()
        self.mask.fill_(1)
        if mask_type == 'A':
            self.mask[:, :, height // 2, width // 2:] = 0
            self.mask[:, :, height // 2 + 1:, :] = 0
        else:
            self.mask[:, :, height // 2, width // 2 + 1:] = 0
            self.mask[:, :, height // 2 + 1:, :] = 0

        self.residual = residual

    def forward(self, x):
        self.weight.data *= self.mask
        out = super(MaskedCNN, self).forward(x)
        if self.residual is not None:
            out = self.residual * x + out

        return out


class PixelCNN(TemplateModel):

    def __init__(self, kernel=7, channels=24, ncode=10, residual=None, depth=2):
        super().__init__()

        self.conv1 = nn.Sequential(MaskedCNN('A', 1, channels, kernel, 1, kernel // 2, bias=True),
                                   nn.ELU())

        model_list = []
        for _ in range(depth):
            model = nn.Sequential(MaskedCNN('B', channels, channels, kernel, 1, kernel // 2, bias=True, residual=residual),
                                  nn.ELU())
            model_list.append(model)

        self.model = nn.Sequential(*model_list)
        self.final = MaskedCNN('B', channels, ncode, kernel, 1, kernel // 2, bias=True)

        self.ncode = ncode
        self.param_init()

    def param_init(self):
        """
        Xavier's initialization
        """
        for layer in self.modules():
            if hasattr(layer, 'weight'):

                if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.PReLU, nn.Tanh, MaskedCNN)):
                    nn.init.normal_(layer.weight, mean=1., std=0.02)
                else:
                    nn.init.xavier_normal_(layer.weight)
            if hasattr(layer, 'bias'):
                nn.init.constant_(layer.bias, 0.)

    def forward(self, x):

        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        out = self.conv1(x)
        residual = out

        out = self.model(out)
        out = out + 0.5 * residual

        out = self.final(out)

        return out