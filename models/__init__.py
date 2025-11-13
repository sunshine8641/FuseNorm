from .activation import  (ParametricBReLU,ParametricBoundedLeakyReLU,BoundedPReLU)
from .resnet import (
    resnet18, resnet50,resnet34,
    resnet18_cifar, resnet50_cifar,resnet34_cifar
)
from .wideresnet import  wrn16_8_cifar,wrn16_8
from .resformer import (resnet18_cifar_global,resnet34_cifar_global)
import  torch

def get_model(model_cfg):
    """
    根据模型配置返回模型实例。
    参数格式示例：
    model_cfg = {
        'type': 'resnet18',           # 模型名称
        'pretrained': False,          # 是否使用预训练权重（仅对 ImageNet 有效）
        'num_classes': 10,            # 分类数
        ...                           # 其他传入模型构造函数的参数，如 in_channels 等
    }
    """
    model_type = model_cfg.get("type", "").lower()
    pretrained = model_cfg.get("pretrained", False)
    # activation= model_cfg.get("activation").lower()
    # last_block_activation= model_cfg.get("last_block_activation").lower()
    # 支持的模型映射
    model_dict = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet18_cifar": resnet18_cifar,
        "resnet50_cifar": resnet50_cifar,
        "resnet34_cifar": resnet34_cifar,
         'resnet18_cifar_global': resnet18_cifar_global,
        'resnet34_cifar_global': resnet34_cifar_global,
        'wrn16_8':wrn16_8,
        'wrn16_8_cifar':wrn16_8_cifar
    }

    if model_type not in model_dict:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 剩余参数传给模型构造函数
    kwargs = {k: v for k, v in model_cfg.items() if k not in ("type", "pretrained")}
    return model_dict[model_type](pretrained=pretrained, **kwargs)


