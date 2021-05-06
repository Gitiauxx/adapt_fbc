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

    def forward(self, x):
        z = x[0]
        beta = x[1]

        out = self.linear(z)
        scale = self.scale(beta)
        bias = self.offset(beta)

        out = out * (scale + 1) + bias
        out = self.batch(out)
        out = self.act(out)

        return out, beta


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


def conv_bn(in_channels, out_channels, kernel=3, stride=1, bias=True):
    padding = kernel // 2
    return nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                   kernel_size=kernel, padding=padding, stride=stride, bias=bias),
                         nn.BatchNorm2d(out_channels)
                        )


class CondConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, sdim, kernel=3, stride=1, bias=True):
        super().__init__()
        padding = kernel // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, bias=bias, padding=padding)

        self.scale = nn.Sequential(nn.Linear(sdim, out_channels), nn.Tanh())
        self.offset = nn.Sequential(nn.Linear(sdim, out_channels), nn.ELU())

        self.batch = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        z = x[0]
        beta = x[1]

        out = self.conv(z)
        scale = self.scale(beta)
        offset = self.offset(beta)

        out = scale.unsqueeze(2).unsqueeze(3) * out + offset.unsqueeze(2).unsqueeze(3)

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
        x = self.activate(x)
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
        z = self.activate(z)

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