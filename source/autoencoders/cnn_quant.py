import torch
import numpy as np
import torch.nn as nn

from source.template_model import TemplateModel
from source.model_utils import activations, _to_one_hot, ResNetBasicBlock, ResNetDecCondBlock, CondFC, Quantize

class CNNQuant(TemplateModel):
    """
    Implement a fully connected encoder and decoder with a quantization laayer and attention
    gates to identify bits that are predictive of the sensitive attribute
    """

    def __init__(self, ichan=[1, 64, 128, 256, 256], kernel=3, embed_dim=4,
                 zk=8, k=8, sdim=2, cout=None, ncode=2, dim=8):

        super().__init__()

        padding = kernel // 2
        self.embed_dim = embed_dim

        preconv = [nn.Conv2d(ichan[0], ichan[1], kernel_size=kernel, stride=1, padding=padding)]

        for i in range(1, len(ichan) - 1):
            preconv.append(ResNetBasicBlock(ichan[i], ichan[i + 1]))

        self.preconv = nn.Sequential(*preconv)

        self.gate = nn.Sequential(CondFC(zk * k, 1, 1, activation='sigmoid'))

        postconv = []
        for i in reversed(range(2, len(ichan))):
            postconv.append(ResNetDecCondBlock(ichan[i], ichan[i - 1], sdim + 1))

        self.postconv = nn.Sequential(*postconv)

        if cout is not None:
            self.image = nn.Sequential(nn.Conv2d(ichan[1], cout, kernel_size=3, stride=1, padding=1))
        else:
            self.image = nn.Conv2d(ichan[1], ichan[0], kernel_size=3, stride=1, padding=1)

        self.quantize_conv = nn.Conv2d(ichan[-1], embed_dim, 1)
        self.quantize = Quantize(embed_dim, ncode)

        self.k = dim * int(np.sqrt(embed_dim))
        self.zk = dim * int(np.sqrt(embed_dim))

        self.quantize_deconv = nn.Conv2d(embed_dim, ichan[-1], 1)

        self.embed_dim = embed_dim
        self.code = nn.Parameter(torch.arange(ncode, dtype=float, requires_grad=True).float() / (ncode - 1))
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
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.)

    def encode(self, x):
        """
        Encoder into a zk * k vectors
        :param x:
        :return:
        """
        h = self.preconv(x)
        return h
        # h = h.reshape(x.shape[0], -1)
        # return self.encoder(h)

    def decode(self, b, beta):
        """
        Decode quantized representation
        :param b:
        :return:
        """
        # out, beta = self.decoder((b, beta))
        # out = out.reshape(out.shape[0], -1, self.embed_dim, self.embed_dim)
        out = b
        out, beta = self.postconv((out, beta))

        return self.image(out)

    def compute_gate(self, z, beta):
        """
        Generate a gate that returns the maximum number of
        bits to include in the bitstream and forget the
        remaining bits.
        :param z:
        :return: mask_soft
        """
        gate, beta = self.gate((z, beta))
        gate = self.k * self.zk * (1 - gate * beta)

        mask_range = torch.arange(0, self.zk * self.k, device=z.device)
        mask_range = mask_range.unsqueeze(0).expand_as(z)

        mask = gate - mask_range
        mask = torch.sigmoid(mask)

        mask_hard = torch.round(mask)
        mask_soft = (mask_hard - mask).detach() + mask

        return mask_soft

    def forward(self, x, s, beta):

        z = self.encode(x)
        b = beta.unsqueeze(1)

        mask = torch.zeros((z.shape[0], self.k * self.zk)).to(x.device)

        quant = self.quantize_conv(z).permute(0, 2, 3, 1)
        quant, centers, commit_diff = self.quantize(quant)
        quant = quant.permute(0, 3, 1, 2)

        b_with_s = torch.cat([b, s], -1)
        quant_expanded = self.quantize_deconv(quant)
        out = self.decode(quant_expanded, b_with_s)

        q = quant.reshape(quant.shape[0], self.zk, self.k)
        centers = centers.reshape(centers.shape[0], self.zk, self.k)

        return out, q, mask, centers, commit_diff




