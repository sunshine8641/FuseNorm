import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F


# -------------------
# Parametric Bounded ReLU with learnable lower & upper bounds
# -------------------
class ParametricBReLU(nn.Module):
    def __init__(self, init_low=2.0, init_high=2.0, per_channel=False, num_channels=None):
        super().__init__()
        self.per_channel = per_channel
        if per_channel:
            assert num_channels is not None, "num_channels must be specified for per-channel BReLU"
            self.l = nn.Parameter(torch.ones(num_channels) * init_low)
            self.c = nn.Parameter(torch.ones(num_channels) * init_high)
        else:
            self.l = nn.Parameter(torch.tensor(init_low))
            self.c = nn.Parameter(torch.tensor(init_high))

    def forward(self, x):
        if self.per_channel:
            l = self.l.view(1, -1, 1, 1)
            c = self.c.view(1, -1, 1, 1)
            return torch.clamp(x, min=-l*l, max=c*c)
        else:
            return torch.clamp(x, min=-self.l*self.l, max=self.c*self.c)


class BoundedPReLU(nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()
        # unconstrained parameter
        self.theta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        a = 0.2*torch.sigmoid(self.theta)  # 保证在 [0,1]
        a = a.view(1, -1, 1, 1)
        return torch.where(x >= 0, x, a * x)

# -------------------
class ParametricBoundedLeakyReLU(nn.Module):
    def __init__(self, init_high=2.0, negative_slope=0.01, per_channel=False, num_channels=None):
        super().__init__()
        self.negative_slope = negative_slope
        self.per_channel = per_channel

        if per_channel:
            assert num_channels is not None, "num_channels must be specified for per-channel version"
            self.c = nn.Parameter(torch.ones(num_channels) * init_high)  # learnable upper bound
        else:
            self.c = nn.Parameter(torch.tensor(init_high))

    def forward(self, x):
        # normal leaky relu
        x = F.leaky_relu(x, negative_slope=self.negative_slope)

        # apply learnable upper bound (positive side only)
        if self.per_channel:
            c = self.c.view(1, -1, 1, 1)
            return torch.clamp(x, max=c * c)  # 上界可学习
        else:
            return torch.clamp(x, max=self.c * self.c)

# -------------------
# 激活函数字典 / 工厂函数
# -------------------
ACTIVATIONS = {
    "relu": lambda **kwargs: nn.ReLU(inplace=True),
    "leaky_relu": lambda **kwargs: nn.LeakyReLU(0.1, inplace=True),
    "gelu": lambda **kwargs: nn.GELU(),
    "swish": lambda **kwargs: nn.SiLU(),
    "brelue": lambda **kwargs: ParametricBReLU(init_low=0.0, init_high=6.0),
    "brelue_channel": lambda num_channels, **kwargs: ParametricBReLU(init_low=0.0, init_high=6.0, per_channel=True, num_channels=num_channels),
    "pbleaky_relu": lambda num_channels, **kwargs: ParametricBoundedLeakyReLU(init_high=6.0, negative_slope=0.1, per_channel=True, num_channels=num_channels),
    "BoundedPReLU": lambda num_channels, **kwargs: BoundedPReLU(num_channels=num_channels),
}

def get_activation(name, num_channels=None):
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}")
    return ACTIVATIONS[name](num_channels=num_channels)

