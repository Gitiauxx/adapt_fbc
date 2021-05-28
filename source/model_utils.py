import torch.nn as nn
import torch

activations = {'elu': nn.ELU(), 'sigmoid': nn.Sigmoid()}

def _to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1).long()
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype, device=y.device)

    return zeros.scatter(scatter_dim, y_tensor, 1)

class CondFC(nn.Module):

    def __init__(self, in_dim, out_dim, sdim, activation='elu'):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)

        self.scale = nn.Sequential(nn.Linear(sdim, out_dim),
                                   nn.Tanh())
        self.offset = nn.Sequential(nn.Linear(sdim, out_dim),
                                  nn.ELU())

        self.batch = nn.BatchNorm1d(out_dim)
        self.act = activations[activation]

        self.param_init()

    def forward(self, x):
        z = x[0]
        beta = x[1]

        out = self.linear(z)
        scale = self.scale(beta)
        bias = self.offset(beta)

        #out = out * (scale + 1) + bias
        out = self.batch(out)
        out = self.act(out)

        return out, beta

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


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
        pass

    def forward(self, x):
        if isinstance(x, tuple):
            z = x[0]
            return nn.functional.interpolate(z, scale_factor=2, mode='nearest'), x[1]
        else:
            return nn.functional.interpolate(x, scale_factor=2, mode='nearest')


class Conv2d(nn.Conv2d):

    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, data_init=True, weight_norm=True):
        super().__init__(C_in, C_out, kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)

        self.data_init = data_init
        self.init_done = False

    def forward(self, x):
        """

        Parameters
        ----------
        x (torch.Tensor): of size (B, C_in, H, W).

        Returns
        -------

        """
        if self.data_init and not self.init_done:
            self.initialize_parameters(x)
            self.init_done = True

        weight = self.weight

        return nn.functional.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def initialize_parameters(self, x):

        with torch.no_grad():
            weight = self.weight / \
                     (torch.sqrt(torch.sum(self.weight * self.weight, dim=[1, 2, 3])).view(-1, 1, 1, 1) + 1e-5)

            bias = None
            out = nn.functional.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
            mn = torch.mean(out, dim=[0, 2, 3])
            st = 5 * torch.std(out, dim=[0, 2, 3])

            if self.bias is not None:
                self.bias.data = - mn / (st + 1e-5)

            self.weight =  weight



def conv_bn(in_channels, out_channels, kernel=3, stride=1, bias=True):
    padding = kernel // 2
    return nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                   kernel_size=kernel, padding=padding, stride=stride, bias=bias),
                         nn.BatchNorm2d(out_channels)
                        )


class CondConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, sdim, kernel=3, stride=1, bias=True, data_init=True):
        super().__init__()
        padding = kernel // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, bias=bias, padding=padding)

        self.scale = nn.Sequential(nn.Linear(sdim, out_channels), nn.Tanh())
        self.offset = nn.Sequential(nn.Linear(sdim, out_channels), nn.ELU())

        self.batch = nn.BatchNorm2d(out_channels)

        self.data_init = data_init
        self.init_done = True

    def initialize_parameters(self, x):

        with torch.no_grad():
            weight = self.conv.weight / \
                     (torch.sqrt(torch.sum(self.conv.weight * self.conv.weight, dim=[1, 2, 3])).view(-1, 1, 1, 1) + 1e-5)

            bias = None
            out = nn.functional.conv2d(x, weight, bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
            mn = torch.mean(out, dim=[0, 2, 3])
            st = 5 * torch.std(out, dim=[0, 2, 3])

            if self.conv.bias is not None:
                self.conv.bias.data = - mn / (st + 1e-5)

            self.conv.weight = weight

    def forward(self, x):

        z = x[0]
        beta = x[1]

        if self.data_init and not self.init_done:
            self.initialize_parameters(z)
            self.init_done = True

        out = self.conv(z)
        scale = self.scale(beta)
        offset = self.offset(beta)

        out = scale.unsqueeze(2).unsqueeze(3) * out + offset.unsqueeze(2).unsqueeze(3)
        out = self.batch(out)

        return out, beta


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='elu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activations[activation]
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x = residual + x
        #x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, kernel=3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)

        self.expansion, self.kernel = expansion, kernel
        self.downsampling = 2 if self.should_apply_shortcut == True else 1

        if self.should_apply_shortcut:
            self.shortcut = nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1, stride=self.downsampling, bias=True)
        else:
            self.shortcut = None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion


class ResNetDecBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)

        self.kernel = kernel_size

        if self.should_apply_shortcut:
            self.shortcut = nn.Sequential(UpSample(), nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=True))

    @property
    def should_apply_shortcut(self):
        return (self.in_channels != self.out_channels)

class ResNetDecBasicBlock(ResNetDecBlock):
    def __init__(self, in_channels, out_channels, *args, **kwargs):

        super().__init__(in_channels, out_channels, *args, **kwargs)

        if self.upsampling == 2:
            self.blocks = nn.Sequential(UpSample(),
                                        conv_bn(self.in_channels, self.out_channels, kernel=self.kernel, bias=True, stride=1),
                                        activations[self.activation],
                                        conv_bn(self.out_channels, self.out_channels, kernel=self.kernel, bias=True, stride=1),
        )
        else:
            self.blocks = nn.Sequential(conv_bn(self.in_channels, self.out_channels, kernel=self.kernel, bias=True,
                                                stride=1),
                                        activations[self.activation],
                                        conv_bn(self.out_channels, self.out_channels, kernel=self.kernel, bias=True,
                                                stride=1))

class ResNetDecCondBlock(ResNetDecBlock):
    def __init__(self, in_channels, out_channels, sdim, *args, **kwargs):

        super().__init__(in_channels, out_channels, *args, **kwargs)

        self.upsampling = self.should_apply_shortcut

        if self.upsampling:
            self.blocks = nn.Sequential(UpSample(),
                                        CondConv2d(self.in_channels, self.out_channels, sdim, kernel=self.kernel, bias=True, stride=1),
                                        ActivationRec(self.activation),
                                        CondConv2d(self.out_channels, self.out_channels, sdim, kernel=self.kernel, bias=True, stride=1),
        )
        else:
            self.blocks = nn.Sequential(CondConv2d(self.in_channels, self.out_channels, sdim, kernel=self.kernel, bias=True,
                                                stride=1),
                                        ActivationRec(self.activation),
                                        CondConv2d(self.out_channels, self.out_channels, sdim, kernel=self.kernel, bias=True,
                                                stride=1))

    def forward(self, x):
        z = x[0]
        beta = x[1]

        residual = z

        if self.should_apply_shortcut:
            residual = self.shortcut(z)

        z, beta = self.blocks((z, beta))
        z = residual + z
        #z = self.activate(z)

        return z, beta


class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, kernel=self.kernel, bias=True, stride=self.downsampling),
            activations[self.activation],
            conv_bn(self.out_channels, self.expanded_channels, kernel=self.kernel, bias=True, stride=1),
        )

class ActivationRec(nn.Module):

    def __init__(self, activation):
        super().__init__()
        self.act = activations[activation]

    def forward(self, x):

        return self.act(x[0]), x[1]


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.zeros(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.ones(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)

        embed_onehot = torch.nn.functional.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        quant = nn.Softmax(dim=-1)(- dist) @ self.embed.permute(1, 0)
        quant = quant.view(*input.shape)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2)
        diff = diff.view(*input.shape)
        diff = diff.mean()
        quantize = quant + (quantize - quant).detach()

        embed_loss = (quantize - input.detach()).pow(2)
        embed_loss = embed_loss.mean()

        return quantize, embed_ind, diff, embed_loss

    def embed_code(self, embed_id):
        return torch.nn.functional.embedding(embed_id, self.embed.transpose(0, 1))