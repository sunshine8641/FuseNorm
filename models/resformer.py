import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Basic ResNet building blocks
# -------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes*4)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

# -------------------------
# Global Self-Attention Block
# -------------------------
class GlobalAttentionBlock(nn.Module):
    """
    轻量全局自注意力：在空间维度上做 MHSA。
    输入: [B, C, H, W]  ->  输出同形状
    """
    def __init__(self, channels, num_heads=4, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn  = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True, dropout=attn_drop)
        self.drop1 = nn.Dropout(proj_drop)

        self.norm2 = nn.LayerNorm(channels)
        self.mlp   = nn.Sequential(
            nn.Linear(channels, channels*4),
            nn.GELU(),
            nn.Linear(channels*4, channels),
            nn.Dropout(proj_drop),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # [B, C, H, W] -> [B, HW, C]
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        # Attn + Residual
        h = self.norm1(x_flat)
        attn_out, _ = self.attn(h, h, h, need_weights=False)  # [B, HW, C]
        x_flat = x_flat + self.drop1(attn_out)
        # MLP + Residual
        h = self.norm2(x_flat)
        x_flat = x_flat + self.mlp(h)
        # 回到 [B, C, H, W]
        x = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

# -------------------------
# Multi-Scale Fusion before GAP
# -------------------------
class MultiScaleFusion(nn.Module):
    """
    融合 layer2, layer3(attended), layer4 的特征:
    1) 通过 1x1 conv 对齐通道
    2) 上采样到与 layer4 同分辨率
    3) concat -> 1x1 reduce -> BN+ReLU
    """
    def __init__(self, c2, c3, c4, out_channels):
        super().__init__()
        inter = out_channels // 2  # 中间通道
        self.proj2 = nn.Conv2d(c2, inter, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(inter)
        self.proj3 = nn.Conv2d(c3, inter, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(inter)
        self.proj4 = nn.Conv2d(c4, inter, 1, bias=False)
        self.bn4   = nn.BatchNorm2d(inter)

        self.reduce = nn.Conv2d(inter*3, out_channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, f2, f3, f4):
        _, _, H4, W4 = f4.shape
        f2p = self.relu(self.bn2(self.proj2(f2)))
        f3p = self.relu(self.bn3(self.proj3(f3)))
        f4p = self.relu(self.bn4(self.proj4(f4)))

        f2u = F.interpolate(f2p, size=(H4, W4), mode='bilinear', align_corners=False)
        f3u = F.interpolate(f3p, size=(H4, W4), mode='bilinear', align_corners=False)

        fused = torch.cat([f2u, f3u, f4p], dim=1)   # [B, inter*3, H4, W4]
        fused = self.relu(self.bn_out(self.reduce(fused)))  # [B, out_channels, H4, W4]
        return fused

# -------------------------
# ResNet Backbone + Global modules
# -------------------------
class ResNetGlobal(nn.Module):
    """
    CIFAR 版本：conv1 3x3 s=1，无 maxpool；layer2/3 stride=2 下采样
    选项：layer4 的 stride 可设 1 或 2（默认为 1，保持 8x8）
    """
    def __init__(self, block=BasicBlock, layers=(2,2,2,2), num_classes=10, layer4_stride=1, heads=4):
        super().__init__()
        self.inplanes = 64

        # stem
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        # stages
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=layer4_stride)

        # Global attention after layer3 (acts on 256*exp channels)
        c3 = 256 * block.expansion
        self.global_attn = GlobalAttentionBlock(c3, num_heads=heads)

        # Multi-scale fusion before GAP: fuse layer2, layer3_attn, layer4
        c2 = 128 * block.expansion
        c4 = 512 * block.expansion
        self.fusion = MultiScaleFusion(c2=c2, c3=c3, c4=c4, out_channels=c4)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc      = nn.Linear(c4, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        inplanes = self.inplanes
        outplanes = planes * block.expansion
        if stride != 1 or inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )

        layers = [block(inplanes, planes, stride, downsample)]
        self.inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))      # [B, 64, 32, 32]
        x1 = self.layer1(x)                         # [B, 64*exp, 32, 32]
        x2 = self.layer2(x1)                        # [B, 128*exp, 16, 16]
        x3 = self.layer3(x2)                        # [B, 256*exp,  8,  8]
        x3a = self.global_attn(x3)                  # global attention at stage3
        x4 = self.layer4(x3a)                       # [B, 512*exp,  8 or 4,  8 or 4]

        fused = self.fusion(x2, x3a, x4)            # upsample & fuse -> [B, 512*exp, H4, W4]
        out = self.avgpool(fused)                   # [B, 512*exp, 1, 1]
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def forward_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, 32, 32]
        x1 = self.layer1(x)  # [B, 64*exp, 32, 32]
        x2 = self.layer2(x1)  # [B, 128*exp, 16, 16]
        x3 = self.layer3(x2)  # [B, 256*exp,  8,  8]
        x3a = self.global_attn(x3)  # global attention at stage3
        x4 = self.layer4(x3a)  # [B, 512*exp,  8 or 4,  8 or 4]

        fused = self.fusion(x2, x3a, x4)  # upsample & fuse -> [B, 512*exp, H4, W4]
        out = self.avgpool(fused)  # [B, 512*exp, 1, 1]
        out = torch.flatten(out, 1)
        return out
    def forward_head(self, feat):
        out = self.fc(feat)
        return out


# -------------------------
# Factory
# -------------------------
def resnet18_cifar_global(pretrained=False, progress=True, **kwargs):
    return ResNetGlobal(block=BasicBlock, layers=(2,2,2,2),**kwargs)

def resnet34_cifar_global(pretrained=False, progress=True, **kwargs):
    return ResNetGlobal(block=BasicBlock, layers=(3,4,6,3),**kwargs)

# -------------------------
# Quick test
# -------------------------
if __name__ == "__main__":
    model = resnet18_cifar_global(num_classes=10, layer4_stride=1, heads=4)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    print("logits:", logits.shape)  # [2, 10]
