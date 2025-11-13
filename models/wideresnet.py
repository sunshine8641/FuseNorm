import torch
import torch.nn as nn
from .activation import get_activation

# -------------------
# conv helper
# -------------------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# -------------------
# Wide BasicBlock (与 WRN 论文类似，带 dropout)
# -------------------
class WideBasic(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dropout_rate=0.0, activation="relu"):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        act1 = get_activation(activation, num_channels=in_planes if activation in ["brelue_channel","pbleaky_relu","BoundedPReLU"] else None)
        self.act1 = act1
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_planes)
        act2 = get_activation(activation, num_channels=out_planes if activation in ["brelue_channel","pbleaky_relu","BoundedPReLU"] else None)
        self.act2 = act2
        self.conv2 = conv3x3(out_planes, out_planes, stride=1)

        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.act1(self.bn1(x))
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.act2(self.bn2(out))
        out = self.conv2(out)
        shortcut = self.shortcut(x)
        out += shortcut
        return out


# -------------------
# WideResNet 主体
# -------------------
class WideResNet(nn.Module):
    """
    WideResNet implementation (WRN-d-k): depth = d, widen_factor = k
    Typical depth choices: 16, 28, 40 (must satisfy (depth-4) % 6 == 0)
    """
    def __init__(self, depth=28, widen_factor=10, num_classes=1000, dropout_rate=0.0, cifar=True, activation="relu",last_block_activation=None):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        n = (depth - 4) // 6
        self.n = n
        self.widen_factor = widen_factor
        self.dropout_rate = dropout_rate
        self.cifar = cifar
        self.activation_name = activation

        # number of channels for each group (following WRN paper base channels [16,32,64])
        channels = [16, 32, 64]
        channels = [c * widen_factor for c in channels]

        # initial conv
        if cifar:
            # CIFAR: no maxpool, kernel 3
            self.maxpool = nn.Identity()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.act = get_activation(activation, num_channels=16 if activation in ["brelue_channel","pbleaky_relu","BoundedPReLU"] else None)
            self.layer1 = self._make_layer(16, channels[0], n, stride=1)
            self.layer2 = self._make_layer(channels[0], channels[1], n, stride=2)
            self.layer3 = self._make_layer(channels[1], channels[2], n, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(channels[2], num_classes)
        else:
            # ImageNet-style: initial conv7x7 + maxpool, then widen groups built on base 64?
            # We follow a simple scheme: initial conv -> groups
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.act = get_activation(activation, num_channels=64 if activation in ["brelue_channel","pbleaky_relu","BoundedPReLU"] else None)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # for ImageNet, use channels scaled to 64/128/256 * widen_factor // 4 to keep sizes reasonable
            img_channels = [64 * widen_factor // 4, 128 * widen_factor // 4, 256 * widen_factor // 4]
            self.layer1 = self._make_layer(64, img_channels[0], n, stride=1)
            self.layer2 = self._make_layer(img_channels[0], img_channels[1], n, stride=2)
            self.layer3 = self._make_layer(img_channels[1], img_channels[2], n, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(img_channels[2], num_classes)

        # weight init (与 ResNet 风格一致)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, out_planes, num_blocks, stride):
        layers = []
        # first block may change spatial resolution
        layers.append(WideBasic(in_planes, out_planes, stride=stride, dropout_rate=self.dropout_rate, activation=self.activation_name))
        for _ in range(1, num_blocks):
            layers.append(WideBasic(out_planes, out_planes, stride=1, dropout_rate=self.dropout_rate, activation=self.activation_name))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

    # 返回最后卷积特征（未pool）
    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def forward_features(self, x):
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        return feat

    def forward_head(self, feat):
        return self.fc(feat)

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
        for layer in [self.layer1, self.layer2, self.layer3]:
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

# -------------------
# 构造函数
# -------------------
def _wrn(arch,depth, widen_factor,pretrained=False, progress=True, **kwargs):
    return WideResNet(depth=depth, widen_factor=widen_factor, **kwargs)


def wrn16_8(pretrained=False, progress=True, **kwargs):
    return _wrn('wrn16_8',16, 8,  **kwargs)



def wrn16_8_cifar(pretrained=False, progress=True, **kwargs):
    kwargs['cifar'] = True
    return _wrn('wrn16_8',16, 8,  **kwargs)
# -------------------
if __name__ == "__main__":
    model = wrn16_8_cifar(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    print("logits:", logits.shape)  # [2, 10]

    feats = model.forward_features_blockwise(x)
    print(f"total blocks/features returned: {len(feats)}")
    for i, f in enumerate(feats[:6]):
        print(f"{i}: shape={f.shape}")