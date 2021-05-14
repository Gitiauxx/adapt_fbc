import torch
import numpy as np
import torch.nn as nn

from source.template_model import TemplateModel
from source.model_utils import activations, _to_one_hot, ResNetBasicBlock, ResNetDecCondBlock, CondFC, Conv2d

class CNNGated(TemplateModel):
    """
    Implement a fully connected encoder and decoder with a quantization laayer and attention
    gates to identify bits that are predictive of the sensitive attribute
    """

    def __init__(self, ichan=[1, 64, 128, 256, 256], kernel=3, embed_dim=4,
                 zk=8, k=8, sdim=2, cout=None, sigma=1, ncode=2):

        super().__init__()

        padding = kernel // 2
        self.embed_dim = embed_dim

        preconv = [nn.Conv2d(ichan[0], ichan[1], kernel_size=kernel, stride=1, padding=padding)]

        for i in range(1, len(ichan) - 1):
            preconv.append(ResNetBasicBlock(ichan[i], ichan[i + 1]))

        self.preconv = nn.Sequential(*preconv)

        # self.preconv = nn.Sequential(nn.Conv2d(ichan[0], ichan[1], kernel_size=kernel, stride=1, padding=padding),
        #                              ResNetBasicBlock(ichan[1], ichan[2], downsampling=2),
        #                              ResNetBasicBlock(ichan[2], ichan[3], downsampling=2),
        #                              ResNetBasicBlock(ichan[3], ichan[3], downsampling=2))

        self.encoder = nn.Sequential(nn.Linear(ichan[-1] * embed_dim ** 2, zk * k),
                                    nn.BatchNorm1d(zk * k),
                                    nn.Tanh())

        self.gate = nn.Sequential(CondFC(zk * k, 1, 1, activation='sigmoid'))

        self.decoder = CondFC(zk * k, ichan[-1] * embed_dim ** 2, sdim + 1)

        postconv = []
        for i in reversed(range(2, len(ichan))):
            postconv.append(ResNetDecCondBlock(ichan[i], ichan[i - 1], sdim + 1))

        self.postconv = nn.Sequential(*postconv)

        # self.postconv = nn.Sequential(ResNetDecCondBlock(ichan[3], ichan[3], sdim + 1, upsampling=2),
        #                               ResNetDecCondBlock(ichan[3], ichan[2], sdim + 1, upsampling=2),
        #                               ResNetDecCondBlock(ichan[2], ichan[1], sdim + 1, upsampling=2)
        #                               )

        if cout is not None:
            self.image = nn.Sequential(nn.Conv2d(ichan[1], cout, kernel_size=3, stride=1, padding=1))
        else:
            self.image = nn.Conv2d(ichan[1], ichan[0], kernel_size=3, stride=1, padding=1)



        self.k = int(np.sqrt(ichan[-1])) * embed_dim
        self.zk = int(np.sqrt(ichan[-1])) * embed_dim
        self.sigma = sigma
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

    def quantize(self, z):
        """
        Quantization layer (right now a simple binarization using
        nearest neighbor to 0 or 1
        :return:
        """
        z = (z + 1) / 2
        code = self.code.detach()
        code_idx = torch.arange(self.code.shape[0], device=code.device)
        code = code[None, None, None, None, :]
        z_code = (z.unsqueeze(-1) - code) ** 2

        z_soft = torch.sum(nn.Softmax(dim=-1)(- self.sigma * z_code) * code, dim=-1)

        centers = z_code.argmin(-1)
        center_code = _to_one_hot(centers, self.code.shape[0])
        center_code = center_code * code
        z_hard = center_code.sum(-1)

        centers_soft = torch.sum(nn.Softmax(dim=-1)(- self.sigma * z_code) * code_idx, dim=-1)

        q = (z_hard - z_soft).detach() + z_soft
        c = (centers - centers_soft).detach() + centers_soft

        return q, c, center_code.mean(dim=[0, 1, 2, 3])

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
            #self.compute_gate(z, b)

        q, centers, code = self.quantize(z)
        #
        b_with_s = torch.cat([b, s], -1)
        out = self.decode(q, b_with_s)

        q = q.reshape(q.shape[0], self.zk, self.k)
        centers = centers.reshape(q.shape[0], self.zk, self.k)

        #q = mask.reshape(-1, self.zk, self.k)
        centers = q

        return out, q, mask, centers, z




