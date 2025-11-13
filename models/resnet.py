import torch
import torch.nn as nn
from .activation import get_activation
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

# # -------------------
# # Parametric Bounded ReLU with learnable lower & upper bounds
# # -------------------
# class ParametricBReLU(nn.Module):
#     def __init__(self, init_low=2.0, init_high=2.0, per_channel=False, num_channels=None):
#         super().__init__()
#         self.per_channel = per_channel
#         if per_channel:
#             assert num_channels is not None, "num_channels must be specified for per-channel BReLU"
#             self.l = nn.Parameter(torch.ones(num_channels) * init_low)
#             self.c = nn.Parameter(torch.ones(num_channels) * init_high)
#         else:
#             self.l = nn.Parameter(torch.tensor(init_low))
#             self.c = nn.Parameter(torch.tensor(init_high))
#
#     def forward(self, x):
#         if self.per_channel:
#             l = self.l.view(1, -1, 1, 1)
#             c = self.c.view(1, -1, 1, 1)
#             return torch.clamp(x, min=-l*l, max=c*c)
#         else:
#             return torch.clamp(x, min=-self.l*self.l, max=self.c*self.c)
#
#
# class BoundedPReLU(nn.Module):
#     def __init__(self, num_channels=1):
#         super().__init__()
#         # unconstrained parameter
#         self.theta = nn.Parameter(torch.zeros(num_channels))
#
#     def forward(self, x):
#         a = 0.2*torch.sigmoid(self.theta)  # 保证在 [0,1]
#         a = a.view(1, -1, 1, 1)
#         return torch.where(x >= 0, x, a * x)
#
# # -------------------
# class ParametricBoundedLeakyReLU(nn.Module):
#     def __init__(self, init_high=2.0, negative_slope=0.01, per_channel=False, num_channels=None):
#         super().__init__()
#         self.negative_slope = negative_slope
#         self.per_channel = per_channel
#
#         if per_channel:
#             assert num_channels is not None, "num_channels must be specified for per-channel version"
#             self.c = nn.Parameter(torch.ones(num_channels) * init_high)  # learnable upper bound
#         else:
#             self.c = nn.Parameter(torch.tensor(init_high))
#
#     def forward(self, x):
#         # normal leaky relu
#         x = F.leaky_relu(x, negative_slope=self.negative_slope)
#
#         # apply learnable upper bound (positive side only)
#         if self.per_channel:
#             c = self.c.view(1, -1, 1, 1)
#             return torch.clamp(x, max=c * c)  # 上界可学习
#         else:
#             return torch.clamp(x, max=self.c * self.c)
#
# # -------------------
# # 激活函数字典 / 工厂函数
# # -------------------
# ACTIVATIONS = {
#     "relu": lambda **kwargs: nn.ReLU(inplace=True),
#     "leaky_relu": lambda **kwargs: nn.LeakyReLU(0.1, inplace=True),
#     "gelu": lambda **kwargs: nn.GELU(),
#     "swish": lambda **kwargs: nn.SiLU(),
#     "brelue": lambda **kwargs: ParametricBReLU(init_low=0.0, init_high=6.0),
#     "brelue_channel": lambda num_channels, **kwargs: ParametricBReLU(init_low=0.0, init_high=6.0, per_channel=True, num_channels=num_channels),
#     "pbleaky_relu": lambda num_channels, **kwargs: ParametricBoundedLeakyReLU(init_high=6.0, negative_slope=0.1, per_channel=True, num_channels=num_channels),
#     "BoundedPReLU": lambda num_channels, **kwargs: BoundedPReLU(num_channels=num_channels),
# }
#
# def get_activation(name, num_channels=None):
#     if name not in ACTIVATIONS:
#         raise ValueError(f"Unknown activation: {name}")
#     return ACTIVATIONS[name](num_channels=num_channels)
#

# -------------------
# 卷积工具函数
# -------------------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# -------------------
# BasicBlock
# -------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, activation="relu"):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

        if activation in ["brelue_channel","pbleaky_relu","BoundedPReLU"]:
            self.act = get_activation(activation, num_channels=planes)
        else:
            self.act = get_activation(activation)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act(out)
        return out


# -------------------
# Bottleneck
# -------------------
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, activation="relu"):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample

        if activation in ["brelue_channel","pbleaky_relu","BoundedPReLU"]:
            self.act = get_activation(activation, num_channels=planes * self.expansion)
        else:
            self.act = get_activation(activation)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act(out)
        return out


# -------------------
# ResNet
# -------------------
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, cifar=False, activation="relu",last_block_activation=None):
        super().__init__()
        self.cifar = cifar
        self.activation_name = activation
        self.last_block_activation = last_block_activation
        self.inplanes = 64
        self.relu = nn.ReLU(inplace=True)

        if cifar:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.act = get_activation(activation, num_channels=64 if activation in ["brelue_channel","pbleaky_relu","BoundedPReLU"] else None)
            self.maxpool = nn.Identity()
            self.layer1 = self._make_layer(block, 64, layers[0],activation=self.activation_name)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,activation=self.activation_name)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,activation=self.activation_name)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                    activation=self.activation_name,
                                    last_block_activation=self.last_block_activation)
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.act = get_activation(activation, num_channels=64 if activation in ["brelue_channel","pbleaky_relu","BoundedPReLU"]  else None)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0],activation=self.activation_name)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,activation=self.activation_name)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,activation=self.activation_name)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           activation=self.activation_name,
                                    last_block_activation=self.last_block_activation)
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, activation="relu", last_block_activation=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, activation=activation)]
        self.inplanes = planes * block.expansion

        # 前面的block
        for i in range(1, blocks):
            if last_block_activation is not None and i == blocks - 1:
                # 最后一个block单独用 last_block_activation
                layers.append(block(self.inplanes, planes, activation=last_block_activation))
            else:
                layers.append(block(self.inplanes, planes, activation=activation))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    def features(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    def forward_features(self, x):
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat=torch.flatten(feat, 1)
        return feat


    def forward_head(self, feat):
        out = self.fc(feat)
        return out

    def forward_features_blockwise(self, x, blockwise=True):
        """
        前向提取特征，可选择按 block 或按 stage 输出
        :param x: 输入张量
        :param blockwise: True -> 每个 block 输出; False -> 每个 stage 最后一个 block 输出
        :return: list of feature maps
        """
        features = []

        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)
        # 遍历每一层 (stage)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            if blockwise:
                # 每个 block 输出
                for block in layer:
                    x = block(x)
                    features.append(x)
            else:
                # stage 输出最后一个 block
                for i, block in enumerate(layer):
                    x = block(x)
                    if i == len(layer) - 1:
                        features.append(x)

        return features
    # def forward_features_blockwise(self, x):
    #     """
    #     默认8层里选
    #     :param x:
    #     :return:
    #     """
    #     features = []
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.act(x)
    #     x = self.maxpool(x)
    #
    #     x = self.layer1[0](x)
    #     features.append(x)
    #     x = self.layer1[1](x)
    #     features.append(x)
    #
    #     x = self.layer2[0](x)
    #     features.append(x)
    #     x = self.layer2[1](x)
    #     features.append(x)
    #
    #     x = self.layer3[0](x)
    #     features.append(x)
    #     x = self.layer3[1](x)
    #     features.append(x)
    #
    #     x = self.layer4[0](x)
    #     features.append(x)
    #     x = self.layer4[1](x)
    #     features.append(x)
    #     return features


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

def resnet18_cifar(pretrained=False, progress=True, **kwargs):
    kwargs['cifar'] = True
    # kwargs['activation'] = 'BoundedPReLU'
    # kwargs['last_block_activation'] ='BoundedPReLU'

    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet34_cifar(pretrained=False, progress=True, **kwargs):
    kwargs['cifar'] = True
    # kwargs['activation'] = 'relu'
    # kwargs['last_block_activation'] ='relu'
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet50_cifar(pretrained=False, progress=True, **kwargs):
    kwargs['cifar'] = True
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


if __name__ == "__main__":
    model = resnet18_cifar(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    print("logits:", logits.shape)  # [2, 10]
    ft = model.forward_features_blockwise(x)
    print("ft:", ft[7].shape)  # [2, 10]
    print(model)
