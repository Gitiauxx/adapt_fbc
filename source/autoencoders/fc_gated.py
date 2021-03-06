import torch
import torch.nn as nn

from source.template_model import TemplateModel
from source.model_utils import activations, _to_one_hot, CondFC

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
        in_dim = zk * k
        out_dim = width
        for _ in range(depth - 1):
            decoder = nn.Sequential(CondFC(in_dim, out_dim, sdim + 1))
            decoder_list.append(decoder)
            in_dim = out_dim

        self.decoder = nn.Sequential(*decoder_list)

        if activation_out is not None:
            self.decoder_final = CondFC(in_dim, input_dim, sdim + 1, activation=activation_out)
        else:
            self.decoder_final = nn.Linear(in_dim, input_dim)

        self.gate = nn.Sequential(CondFC(zk * k, 1, 1, activation='sigmoid'))
        # self.gate_beta = nn.Sequential(nn.Linear(1, zk * k),
        #                                nn.Tanh())

        # self.encode_beta = nn.Sequential(nn.Linear(1, zk * k),
        #                                nn.Tanh())
        #
        # self.decode_beta = nn.Sequential(nn.Linear(1, zk * k * 2),
        #                                  nn.Tanh())

        self.k = k
        self.zk = zk
        self.sigma = sigma
        self.code = nn.Parameter(torch.arange(ncode, dtype=float, requires_grad=True).float() / (ncode - 1))
        self.scale = nn.Parameter(torch.tensor([10.0], requires_grad=True)).float()
        self.scale_decode = nn.Parameter(torch.tensor([100.0], requires_grad=True)).float()
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

    def decode(self, b, beta):
        """
        Decode quantized representation
        :param b:
        :return:
        """
        out, beta = self.decoder((b, beta))
        out, _ = self.decoder_final((out, beta))

        return out

    def quantize(self, z):
        """
        Quantization layer (right now a simple binarization using
        nearest neighbor to 0 or 1
        :return:
        """
        z = (z + 1) / 2
        code = self.code.detach()
        code_idx = torch.arange(self.code.shape[0])
        code = code[None, None, :]
        z_code = (z.unsqueeze(-1) - code) ** 2

        z_soft = torch.sum(nn.Softmax(dim=-1)(- self.sigma * z_code) * code, dim=-1)

        centers = z_code.argmin(-1)
        center_code = _to_one_hot(centers, self.code.shape[0])
        center_code = center_code * code
        z_hard = center_code.sum(-1)

        centers_soft = torch.sum(nn.Softmax(dim=-1)(- self.sigma * z_code) * code_idx, dim=-1)

        q = (z_hard - z_soft).detach() + z_soft
        c = (centers - centers_soft).detach() + centers_soft

        return q, c, center_code.mean(dim=[0, 1])

    def compute_gate(self, z, beta):
        """
        Generate a gate that returns the maximum number of
        bits to include in the bitstream and forget the
        remaining bits.
        :param z:
        :return: mask_soft
        """
        beta = beta.unsqueeze(1)
        # z = z * (1 + self.gate_beta(beta))
        #
        # z_with_beta = torch.cat([z, beta], 1)
        gate, beta = self.gate((z, beta))
        gate = self.k * self.zk * ( 1 - gate * beta)

        mask_range = torch.arange(0, self.zk * self.k, device=z.device)
        mask_range = mask_range.unsqueeze(0).expand_as(z)

        mask = gate - mask_range
        mask = torch.sigmoid(mask)

        #gate = 1 - beta * gate

        mask_hard = torch.round(mask)
        mask_soft = (mask_hard - mask).detach() + mask

        return mask_soft

    def forward(self, x, s, beta):

        z = self.encode(x)
        b = beta.unsqueeze(1)

        mask = self.compute_gate(z, beta)

        q, centers, code = self.quantize(z * mask)

        # adjustment = self.decode_beta(b)
        # scale, bias = torch.chunk(adjustment, 2, dim=1)
        # qb = scale * q + bias
        #qb = torch.cat([q, b], 1)
        #srand = s[torch.randperm(s.shape[0])]

        b_with_s = torch.cat([b, s], -1)
        out = self.decode(q, b_with_s)

        q = q.reshape(q.shape[0], self.zk, self.k)
        centers = centers.reshape(q.shape[0], self.zk, self.k)

        return out, q, mask, centers, code




