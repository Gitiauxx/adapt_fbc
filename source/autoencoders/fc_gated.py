import torch
import torch.nn as nn

from source.template_model import TemplateModel
from source.model_utils import activations

class FCGated(TemplateModel):
    """
    Implement a fully connected encoder and decoder with a quantization laayer and attention
    gates to identify bits that are predictive of the sensitive attribute
    """

    def __init__(self, input_dim, depth=2, width=64, zk=8, k=8, sdim=2, activation_out=None, sigma=1, ncode=2):

        super().__init__()

        encoder_list = []
        in_dim = input_dim
        out_dim = width
        for _ in range(depth - 1):
            encoder = nn.Sequential(nn.Linear(in_dim, out_dim),
                                    nn.BatchNorm1d(out_dim),
                                    nn.ELU())
            encoder_list.append(encoder)

            in_dim = out_dim

        encoder_list.append(nn.Sequential(nn.Linear(out_dim, zk * k),
                                          nn.BatchNorm1d( zk * k),
                                          nn.Tanh()))
        self.encoder = nn.Sequential(*encoder_list)

        decoder_list = []
        in_dim = zk * k + sdim
        out_dim = width
        for _ in range(depth - 1):
            decoder = nn.Sequential(nn.Linear(in_dim, out_dim),
                                    nn.BatchNorm1d(out_dim),
                                    nn.ELU())
            decoder_list.append(decoder)
            in_dim = out_dim

        self.decoder = nn.Sequential(*decoder_list)

        if activation_out is not None:
            self.decoder_final = nn.Sequential(nn.Linear(in_dim, input_dim),
                                         nn.BatchNorm1d(input_dim),
                                         activations[activation_out])
        else:
            self.decoder_final = nn.Linear(in_dim, input_dim)

        self.k = k
        self.zk = zk
        self.sigma = sigma
        self.code = 2 * torch.rand(ncode) - 1
        self.param_init()

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

    def encode(self, x):
        """
        Encoder into a zk * k vectors
        :param x:
        :return:
        """
        return self.encoder(x)

    def decode(self, b):
        """
        Decode quantized representation
        :param b:
        :return:
        """
        out = self.decoder(b)
        return self.decoder_final(out)

    def quantize(self, z):
        """
        Quantization layer (right now a simple binarization using
        nearest neighbor to 0 or 1
        :return:
        """
        z = (z + 1) / 2
        code = torch.arange(self.code.shape[0])
        code = code[None, None, :]
        z_code = (z.unsqueeze(-1) - code) ** 2

        z_soft = torch.sum(nn.Softmax(dim=-1)(- self.sigma * z_code) * code, dim=-1)
        z_hard = torch.round(z)

        return (z_hard - z_soft).detach() + z_soft

    def forward(self, x, s):

        z = self.encode(x)
        q = self.quantize(z)

        q_with_s = torch.cat([q, s], -1)
        out = self.decode(q_with_s)

        return out, q.reshape(q.shape[0], self.zk, self.k)





