"""
前向传播方法模块

提供各种 OOD 检测方法的前向传播实现，包括：
- Base: 基础前向传播
- REACT: 基于阈值裁剪的方法
- BATS: 基于统计截断的方法
- LAPS: 基于 Lipschitz 常数的截断方法
- Forward Features: 返回所有层特征

这些方法主要用于在 OOD 检测中修改模型的特征表示，
以提高模型对分布内（ID）和分布外（OOD）样本的区分能力。
"""

import torch
import os
from .get_threshold import compute_threshold_react
from .get_threshold import compute_activation_stats
from accelerate import Accelerator


def forward_base(inputs, model, config):
    """
    基础前向传播函数

    直接使用模型的标准前向传播，返回特征和 logits。

    Args:
        inputs: 输入张量，形状为 [B, C, H, W]
        model: PyTorch 模型
        config: 配置字典（此函数不使用）

    Returns:
        features: 模型的特征张量，形状为 [B, D]
        outputs: 模型的输出 logits，形状为 [B, num_classes]
    """
    # 获取模型的特征表示
    features = model.forward_features(inputs)
    # 获取模型的头部输出（分类 logits）
    outputs = model.forward_head(features)
    return features, outputs


def forward_all_features(inputs, model, config):
    """
    返回所有层的特征（逐块前向传播）

    与基础前向传播不同，此函数返回模型中所有层的特征，
    而不仅仅是最终的池化特征。

    Args:
        inputs: 输入张量，形状为 [B, C, H, W]
        model: PyTorch 模型
        config: 配置字典（此函数不使用）

    Returns:
        features: 包含所有层特征的列表
        None: 此函数不返回 logits
    """
    # 返回逐块计算的特征列表
    features = model.forward_features_blockwise(inputs)
    return features, None


def forward_react(inputs, model, config):
    """
    REACT 方法的前向传播

    REACT (Rectified Activation) 通过对特征进行阈值裁剪来减少 OOD 检测的混淆。
    核心思想：将特征值裁剪到阈值以下，减少高激活值对 OOD 检测的干扰。

    Reference:
        Sun, Y., Guo, C., & Li, Y. (2021).
        "REACT: Out-of-Distribution Detection with Rectified Activations."
        NeurIPS 2021.
    Args:
        inputs: 输入张量，形状为 [B, C, H, W]
        model: PyTorch 模型
        config: 配置字典，必须包含 'react_threshold' 键

    Returns:
        features: 裁剪后的特征张量
        logits: 模型的输出 logits

    Raises:
        ValueError: 当 config['react_threshold'] 为 None 时抛出
    """
    # 获取模型的特征表示
    features = model.forward_features(inputs)
    
    # 获取 REACT 阈值
    react_threshold = config['react_threshold']
    
    # 检查阈值是否有效
    if react_threshold is None:
        raise ValueError("react_threshold cannot be None")
    
    # 将特征值裁剪到阈值以下
    # torch.where(condition, x, y): 如果 condition 为 True，选择 x，否则选择 y
    # 这里将大于 react_threshold 的值替换为 react_threshold
    features = torch.where(features < react_threshold, features, react_threshold)
    
    # 使用裁剪后的特征计算 logits
    logits = model.forward_head(features)
    return features, logits


def forward_bats(inputs, model, config):
    """
    BATS 方法的前向传播

    BATS (Bounded Activation clipping with Two-Sided thresholding) 通过双向阈值裁剪
    来限制特征值的范围，减少异常激活值的影响。
    
    Reference:
        Kong, H., & Li, H. (2023).
        "BFAct: Out-of-Distribution Detection with Butterworth Filter Rectified Activations."
        ICCSIP 2022. Springer, CCIS 1787.

    裁剪公式：
        lower_bound = -feature_std * lam + feature_mean
        upper_bound = feature_std * lam + feature_mean
        features = clamp(features, lower_bound, upper_bound)

    Args:
        inputs: 输入张量，形状为 [B, C, H, W]
        model: PyTorch 模型
        config: 配置字典，必须包含：
            - 'bats_lam': 缩放因子（论文建议：CIFAR 用 3.25，ImageNet 用 1.05）
            - 'feature_mean': 特征均值
            - 'feature_std': 特征标准差

    Returns:
        features: 裁剪后的特征张量
        logits: 模型的输出 logits
    """
    # 获取 BATS 的缩放因子
    lam = config['bats_lam']
    
    # 获取特征的统计信息
    feature_mean = config['feature_mean']
    feature_std = config['feature_std']

    # 获取模型的特征表示
    features = model.forward_features(inputs)
    
    # 上界裁剪：将大于 upper_bound 的值替换为 upper_bound
    # upper_bound = feature_std * lam + feature_mean
    features = torch.where(
        features < (feature_std * lam + feature_mean), 
        features, 
        feature_std * lam + feature_mean
    )
    
    # 下界裁剪：将小于 lower_bound 的值替换为 lower_bound
    # lower_bound = -feature_std * lam + feature_mean
    features = torch.where(
        features > (-feature_std * lam + feature_mean), 
        features, 
        -feature_std * lam + feature_mean
    )
    
    # 使用裁剪后的特征计算 logits
    logits = model.forward_head(features)
    return features, logits


def forward_laps(inputs, model, config):
    """
    LAPS 方法的前向传播

    LAPS (Lipschitz-based Adaptive feature clipping) 是一种基于 Lipschitz 常数的
    自适应特征裁剪方法。与 BATS 不同，LAPS 使用非对称的双边裁剪。

    Reference:
        He, R., Yuan, Y., Han, Z., Wang, F., Su, W., Yin, Y., Liu, T., & Gong, Y. (2024).
        "Exploring Channel-Aware Typical Features for Out-of-Distribution Detection."
        AAAI 2024.

    裁剪公式：
        upper_bound = feature_std * lam1 + feature_mean
        lower_bound = -feature_std * lam2 + feature_mean
        features = clamp(features, lower_bound, upper_bound)

    其中 lam1 和 lam2 是基于特征统计信息自适应计算的。

    Args:
        inputs: 输入张量，形状为 [B, C, H, W]
        model: PyTorch 模型
        config: 配置字典，必须包含：
            - 'laps_lam': 基础 Lipschitz 常数（论文建议值：1.5）
            - 'laps_lam1': 上界缩放因子（自适应计算）
            - 'laps_lam2': 下界缩放因子（自适应计算）
            - 'feature_mean': 特征均值
            - 'feature_std': 特征标准差

    Returns:
        features: 裁剪后的特征张量
        logits: 模型的输出 logits
    """
    # 获取特征的统计信息
    feature_mean = config['feature_mean']
    feature_std = config['feature_std']

    # 获取 LAPS 的参数
    lam1 = config['laps_lam1']
    lam2 = config['laps_lam2']

    # 获取模型的特征表示
    features = model.forward_features(inputs)

    # 上界裁剪：限制特征值的上限
    # 使用 lam1 进行自适应缩放
    features = torch.where(
        features < (feature_std * lam1 + feature_mean),
        features,
        feature_std * lam1 + feature_mean
    )

    # 下界裁剪：限制特征值的下限
    # 使用 lam2 进行自适应缩放
    features = torch.where(
        features > (-feature_std * lam2 + feature_mean),
        features,
        -feature_std * lam2 + feature_mean
    )

    # 使用裁剪后的特征计算 logits
    logits = model.forward_head(features)
    return features, logits


def forward_cadref(inputs, model, config):
    """
    CADRef 方法的前向传播

    CADRef (Class-Aware Decoupled Relative Feature) 通过以下步骤计算 OOD 分数：
    1. 计算样本特征与类平均特征的相对误差
    2. 按与最大 logit 权重的符号对齐关系，将误差解耦为 Positive 和 Negative 两部分
    3. Positive 误差由 Energy score 动态调制，Negative 误差由训练集均值调制

    Reference:
        Ling, Z., Chang, Y., Zhao, H., Zhao, X., Chow, K., & Deng, S. (2025).
        "CADRef: Robust Out-of-Distribution Detection via Class-Aware Decoupled Relative Feature Leveraging."
        CVPR 2025.

    Args:
        inputs: 输入张量，形状为 [B, C, H, W]
        model: PyTorch 模型
        config: 配置字典，必须包含：
            - 'class_centroids': 每个类的平均特征，形状 [num_classes, feature_dim]
            - 'mean_energy': ID 训练集 Energy score 均值（标量）

    Returns:
        decoupled_info: 包含以下键的字典：
            - 'Ep': Positive 误差，形状 [B]
            - 'En': Negative 误差，形状 [B]
            - 'S_logit': Energy score，形状 [B]
        logits: 模型的输出 logits，形状 [B, num_classes]
    """
    # 加载类平均特征和均值 energy
    class_centroids = config['class_centroids']
    mean_energy = config['mean_energy']

    # 提取特征和 logits
    features = model.forward_features(inputs)           # [B, feature_dim]
    logits = model.forward_head(features)              # [B, num_classes]

    # 预测类别 T = argmax(L)
    pred_class = logits.argmax(dim=1)                # [B]

    # 取得每个样本对应类别的平均特征：class_centroids[T] → [B, feature_dim]
    class_feats = class_centroids[pred_class]         # [B, feature_dim]

    # 相对特征：F(x) - F̄^T_T
    rel_feats = features - class_feats                 # [B, feature_dim]

    # W_max：取预测类别那一行的分类器权重（[num_classes, feature_dim] 的第 T 行）
    # 兼容不同模型的分类器命名：fc（DenseNet/ResNet）、classifier（ViT）、head（Swin/ViT）
    if hasattr(model, 'fc'):
        classifier_weights = model.fc.weight           # [num_classes, feature_dim]
    elif hasattr(model, 'classifier'):
        classifier_weights = model.classifier.weight
    elif hasattr(model, 'head'):
        classifier_weights = model.head.weight
    elif hasattr(model, 'get_classifier_weights'):
        classifier_weights = model.get_classifier_weights()
    else:
        raise AttributeError(
            f"Model {type(model).__name__} has no recognized classifier layer "
            "(tried: fc, classifier, head, get_classifier_weights)"
        )
    W_max = classifier_weights[pred_class]            # [B, feature_dim]

    # Element-wise product: W_max · (F - F̄)
    weighted_rel = W_max * rel_feats                 # [B, feature_dim]

    # L1 范数（用于归一化）
    feat_l1 = features.norm(p=1, dim=1) + 1e-8       # [B]

    # Positive / Negative 划分
    # POS: weighted_rel >= 0  →  对 logit 有正贡献
    # NEG: weighted_rel < 0  →  对 logit 有负贡献
    pos_mask = (weighted_rel >= 0).float()           # [B, feature_dim]
    neg_mask = 1.0 - pos_mask                        # [B, feature_dim]

    # 绝对相对特征
    abs_rel = rel_feats.abs()                         # [B, feature_dim]

    # Positive / Negative 误差（公式 7、8）
    Ep = (abs_rel * pos_mask).sum(dim=1) / feat_l1   # [B]
    En = (abs_rel * neg_mask).sum(dim=1) / feat_l1  # [B]

    # Energy score（公式 9 的 S_logit）
    S_logit = torch.logsumexp(logits, dim=1)          # [B]

    decoupled_info = {
        'Ep': Ep.detach().cpu(),
        'En': En.detach().cpu(),
        'S_logit': S_logit.detach().cpu(),
    }

    return decoupled_info, logits


def get_forward(name="base", config=None):
    """
    根据名称获取对应的前向传播函数

    这是一个工厂函数，根据配置和需求返回适当的前向传播方法。
    如果需要加载预计算的阈值或统计信息，会自动计算并加载。

    Args:
        name: 前向传播方法的名称，可选值：
            - "base": 基础前向传播
            - "react": REACT 方法
            - "bats": BATS 方法
            - "laps": LAPS 方法
            - "cadref": CADRef 方法
            - "forward_features": 返回所有层特征
        config: 配置字典，用于存储和获取方法特定的参数

    Returns:
        对应名称的前向传播函数

    Raises:
        ValueError: 当传入未知的前向方法名称时抛出

    Note:
        对于 "react"、"bats"、"laps" 和 "cadref" 方法，如果配置文件中没有预计算的
        阈值或统计信息，函数会自动计算并保存。
    """
    # 构建模型保存目录路径
    # 格式：save_dir/exp_name/test_model/
    model_save_dir = os.path.join(
        config.get("save_dir", "checkpoints"),
        config["exp_name"]
    )

    if name == "base":
        # 返回基础前向传播函数
        return forward_base

    elif name == "react":
        # REACT 方法：需要加载预计算的阈值
        load_path = os.path.join(model_save_dir, config["test_model"], "react_threshold.pt")

        # 如果阈值文件不存在，自动计算
        if not os.path.exists(load_path):
            compute_threshold_react(config)

        # 加载阈值并存储到配置中
        config['react_threshold'] = torch.load(load_path)
        return forward_react

    elif name == "bats":
        # BATS 方法：需要加载特征的统计信息
        load_path = os.path.join(model_save_dir, config["test_model"], "feature_stats.pt")

        # 如果统计文件不存在，自动计算
        if not os.path.exists(load_path):
            compute_activation_stats(config)

        # 加载统计信息
        feature_stats = torch.load(load_path)

        # 存储到配置中，并确保在正确的设备上
        config['feature_std'] = feature_stats['feature_std']
        config['feature_mean'] = feature_stats['feature_mean']

        # 将张量移动到正确的设备（支持 GPU）
        config['feature_std'] = config['feature_std'].to(config["accelerator"].device)
        config['feature_mean'] = config['feature_mean'].to(config["accelerator"].device)

        return forward_bats

    elif name == "laps":
        # LAPS 方法：需要加载特征的统计信息
        # 如果配置中没有统计信息，则尝试加载
        if config.get('feature_std', None) is None:
            load_path = os.path.join(model_save_dir, config["test_model"], "feature_stats.pt")

            # 如果统计文件不存在，自动计算
            if not os.path.exists(load_path):
                compute_activation_stats(config)

            # 加载统计信息
            feature_stats = torch.load(load_path)
            config['feature_std'] = feature_stats['feature_std']
            config['feature_mean'] = feature_stats['feature_mean']

            # 移动到正确设备
            config['feature_std'] = config['feature_std'].to(config["accelerator"].device)
            config['feature_mean'] = config['feature_mean'].to(config["accelerator"].device)

        # 获取基础 Lipschitz 常数
        lam = config['laps_lam']
        feature_mean = config['feature_mean']
        feature_std = config['feature_std']

        # 自适应计算上界和下界的缩放因子
        # 这些参数根据特征的统计特性动态调整
        config['laps_lam1'] = lam + (torch.mean(feature_mean) - feature_mean) * config['laps_m'] \
               + (torch.mean(feature_std) - feature_std) * config['laps_n']
        config['laps_lam2'] = lam - (torch.mean(feature_mean) - feature_mean) * config['laps_m'] \
               + (torch.mean(feature_std) - feature_std) * config['laps_n']

        return forward_laps

    elif name == "cadref":
        # CADRef 方法：需要加载类平均特征和均值 energy
        load_path = os.path.join(model_save_dir, config["test_model"], "cadref_stats.pt")

        # 如果统计文件不存在，自动计算
        if not os.path.exists(load_path):
            from .get_threshold import compute_cadref_stats
            compute_cadref_stats(config)

        # 加载统计信息
        cadref_stats = torch.load(load_path)
        config['class_centroids'] = cadref_stats['class_centroids']
        config['mean_energy'] = cadref_stats['mean_energy']

        # 移动到正确设备（支持 GPU）
        config['class_centroids'] = config['class_centroids'].to(config["accelerator"].device)
        if isinstance(config['mean_energy'], torch.Tensor):
            config['mean_energy'] = config['mean_energy'].to(config["accelerator"].device)

        return forward_cadref

    elif name == "forward_features":
        # 返回包含所有层特征的前向传播函数
        return forward_all_features

    else:
        # 未知的前向方法名称
        raise ValueError(f"Unknown forward method {name}")


# 模块测试入口
if __name__ == "__main__":
    # 此模块可以作为脚本运行，用于测试各方法的基本功能
    pass
