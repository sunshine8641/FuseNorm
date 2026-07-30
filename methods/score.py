"""
OOD 评分方法模块

包含各种 OOD 检测评分方法的实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Optional, Dict, Any

from .forward import forward_all_features, get_forward
from .utils import get_test_model_dir, extract_avgpool_features, collect_avgpool_features_with_labels


def get_msp_score(inputs: torch.Tensor, model: nn.Module, forward_func: callable, config: Optional[dict] = None) -> np.ndarray:
    """
    计算 Maximum Softmax Probability (MSP) 得分

    Reference:
        Hendrycks, D. & Gummell, K. (2017).
        "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks."
        Proceedings of ICLR. (Originally arXiv:1610.02136, 2016)

    Args:
        inputs: 输入图像，形状 [B, C, H, W]
        model: PyTorch 模型
        forward_func: 前向函数，返回 (features, logits)
        config: 配置字典

    Returns:
        scores: MSP 分数，形状 [B]，值越大表示越可能是 ID
    """
    with torch.no_grad():
        features, logits = forward_func(inputs, model, config)

    # 取 softmax 概率的最大值作为得分
    scores = np.max(F.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
    return scores


def get_energy_score(inputs: torch.Tensor, model: nn.Module, forward_func: callable, config: dict) -> np.ndarray:
    """
    计算 Energy-based score（logsumexp）

    Reference:
        Liu, W., Wang, X., Owens, J., & Li, Y. (2020).
        "Energy-based Out-of-Distribution Detection."
        Advances in Neural Information Processing Systems, 33, 21464–21475.

    Args:
        inputs: 输入图像，形状 [B, C, H, W]
        model: PyTorch 模型
        forward_func: 前向函数，返回 (features, logits)
        config: 配置字典

    Returns:
        scores: Energy 分数，形状 [B]，值越大表示越可能是 ID
    """
    with torch.no_grad():
        features, logits = forward_func(inputs, model, config)

    # 计算 logsumexp 作为能量得分
    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    return scores


def get_topk_energy_score(inputs: torch.Tensor, model: nn.Module, forward_func: callable, config: dict) -> np.ndarray:
    """
    计算 Top-K Energy 得分

    Args:
        inputs: 输入图像，形状 [B, C, H, W]
        model: PyTorch 模型
        forward_func: 前向函数，返回 (features, logits)
        config: 配置字典，需包含 energy_k

    Returns:
        scores: Top-K Energy 分数，形状 [B]
    """
    with torch.no_grad():
        features, logits = forward_func(inputs, model, config)

    # 取 logits 的 top-k
    topk_logits, _ = torch.topk(logits, k=config["energy_k"], dim=1)

    # 在 top-k 上计算 logsumexp
    scores = torch.logsumexp(topk_logits.cpu(), dim=1).numpy()

    return scores


def get_energy_entropy_score(inputs: torch.Tensor, model: nn.Module, forward_func: callable, config: dict) -> np.ndarray:
    """
    融合 Energy 和 Entropy 的 OOD score

    Args:
        inputs: 输入图像，形状 [B, C, H, W]
        model: PyTorch 模型
        forward_func: 前向函数，返回 (features, logits)
        config: 配置字典，需包含 alpha, temperature

    Returns:
        scores: 融合分数，形状 [B]，值越大表示越可能是 ID
    """
    alpha = config.get("alpha", 0.5)      # 权重系数
    T = config.get("temperature", 2.0)    # 温度参数

    with torch.no_grad():
        features, logits = forward_func(inputs, model, config)

    # 计算能量分数 Energy
    energy = torch.logsumexp(logits / T, dim=1)  # [B]

    # 计算预测概率分布
    probs = torch.softmax(logits / T, dim=1)     # [B, num_classes]

    # 计算熵 -Entropy
    entropy = (probs * torch.log(probs + 1e-12)).sum(dim=1)  # [B]

    # 融合分数
    scores = alpha * energy + (1 - alpha) * entropy

    return scores.cpu().numpy()


def get_odin_score(inputs: torch.Tensor, model: nn.Module, forward_func: callable, config: dict, accelerator) -> np.ndarray:
    """
    计算 ODIN 分数

    通过输入添加小扰动 + 温度缩放来增强 OOD 识别。

    Reference:
        Liang, S., Li, Y., & Srikant, R. (2018).
        "Enhancing the Reliability of Out-of-Distribution Image Detection in Neural Networks."
        ICLR. (Originally arXiv:1710.01528)

    Args:
        inputs: 输入图像，形状 [B, C, H, W]
        model: PyTorch 模型
        forward_func: 前向函数，返回 (features, logits)
        config: 配置字典，需包含 odin_temperature, odin_magnitude
        accelerator: Accelerator 对象

    Returns:
        scores: ODIN 分数，形状 [B]
    """
    temper = config['odin_temperature']
    noiseMagnitude1 = config['odin_magnitude']

    criterion = nn.CrossEntropyLoss()

    # 将 inputs 放到 accelerator 设备并设置 requires_grad
    inputs = inputs.to(accelerator.device)
    inputs = torch.autograd.Variable(inputs, requires_grad=True)

    features, outputs = forward_func(inputs, model, config)

    # 找出最大类别索引作为伪标签
    maxIndexTemp = torch.argmax(outputs.detach(), dim=1)

    # 温度缩放
    outputs = outputs / temper

    # 构造伪标签
    labels = maxIndexTemp.to(accelerator.device)

    loss = criterion(outputs, labels)

    # 用 accelerator 管理反向传播
    accelerator.backward(loss)

    # 获取归一化梯度
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # 添加扰动
    tempInputs = inputs.data - noiseMagnitude1 * gradient
    tempInputs = tempInputs.to(accelerator.device)

    with torch.no_grad():
        features, outputs = forward_func(tempInputs, model, config)
        outputs = outputs / temper

    # Softmax 计算最大概率得分
    nnOutputs = outputs.data.cpu().numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    scores = np.max(nnOutputs, axis=1)

    return scores


def get_mahalanobis_score(inputs: torch.Tensor, model: nn.Module, config: dict) -> np.ndarray:
    """
    计算 Mahalanobis OOD score

    Reference:
        Lee, K., Lee, K., Lee, H., & Shin, J. (2018).
        "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks."
        NeurIPS 2018.
    Args:
        inputs: 输入图像，形状 [B, C, H, W]
        model: PyTorch 模型
        config: 配置字典
    Returns:
        scores: Mahalanobis 分数，形状 [B]，值越小越可能是 OOD
    """
    stats_path = os.path.join(get_test_model_dir(config), "mahalanobis_stats.pt")

    if os.path.exists(stats_path):
        stats = torch.load(stats_path)
        sample_mean = stats["sample_mean"]
        precision = stats["precision"]
    else:
        print("⚠️ mahalanobis_stats.pt not found, computing statistics...")
        sample_mean, precision = compute_mahalanobis_stats(config)

    device = next(model.parameters()).device
    sample_mean = sample_mean.to(device)
    precision = precision.to(device)

    model.eval()
    with torch.no_grad():
        feats = extract_avgpool_features(model, inputs)

    # 计算每个样本到每类均值的 Mahalanobis 距离
    B = feats.shape[0]
    num_classes = sample_mean.shape[0]
    feature_dim = feats.shape[1]

    # Mahalanobis distance: d(x) = (x - mean)^T @ precision @ (x - mean)
    feats_exp = feats.unsqueeze(1).expand(B, num_classes, feature_dim)  # [B, num_classes, C]
    mean_exp = sample_mean.unsqueeze(0).expand(B, num_classes, feature_dim)  # [B, num_classes, C]
    diff = feats_exp - mean_exp  # [B, num_classes, C]

    # batch-wise 矩阵乘法
    diff_precision = torch.matmul(diff, precision)  # [B, num_classes, C]
    mahalanobis_dist = torch.sum(diff_precision * diff, dim=2)  # [B, num_classes]

    # 对每个样本取最小距离作为 OOD score（加负号使越大越可能是 ID）
    scores = -torch.min(mahalanobis_dist, dim=1)[0]

    return scores.cpu().numpy()


def compute_mahalanobis_stats(config: dict) -> tuple:
    """
    计算 Mahalanobis 需要的统计量

    Args:
        config: 配置字典

    Returns:
        sample_mean: 每类特征均值，形状 [num_classes, feature_dim]
        precision: 协方差精度矩阵，形状 [feature_dim, feature_dim]
    """
    from .utils import prepare_eval_model_with_id_loader

    model, id_train_loader = prepare_eval_model_with_id_loader(config, split="train")
    model.eval()

    num_classes = config["num_classes"]
    all_features, all_labels = collect_avgpool_features_with_labels(model, id_train_loader)

    sample_mean = []
    class_features = []
    for cls in range(num_classes):
        cls_feats = all_features[all_labels == cls]
        class_features.append(cls_feats)
        sample_mean.append(np.mean(cls_feats, axis=0))

    sample_mean = np.stack(sample_mean, axis=0)
    all_features = np.concatenate(class_features, axis=0)

    cov = np.cov(all_features, rowvar=False)
    precision = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))

    sample_mean = torch.tensor(sample_mean, dtype=torch.float32)
    precision = torch.tensor(precision, dtype=torch.float32)

    stats = {
        "sample_mean": sample_mean,
        "precision": precision,
    }

    output_dir = get_test_model_dir(config)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(stats, os.path.join(output_dir, "mahalanobis_stats.pt"))

    print(f"✅ Saved Mahalanobis stats to {os.path.join(output_dir, 'mahalanobis_stats.pt')}")
    return sample_mean, precision


def get_cadref_score(inputs: torch.Tensor, model: nn.Module, forward_func: callable, config: dict) -> np.ndarray:
    """
    计算 CADRef（Class-Aware Decoupled Relative Feature）分数

    CADRef 将样本的相对特征误差按符号对齐关系解耦为 Positive（E_p）和 Negative（E_n）两部分，
    并用 Energy score 对 Positive 误差做动态调制：
        score = E_n / mean_energy + E_p / S_logit
    值越大表示越可能是 ID。

    Reference:
        Ling, Z., Chang, Y., Zhao, H., Zhao, X., Chow, K., & Deng, S. (2025).
        "CADRef: Robust Out-of-Distribution Detection via Class-Aware Decoupled Relative Feature Leveraging."
        CVPR 2025.

    Args:
        inputs: 输入图像，形状 [B, C, H, W]
        model: PyTorch 模型
        forward_func: 前向函数，返回 (decoupled_info, logits)
            decoupled_info 包含：'Ep'（Positive 误差）、'En'（Negative 误差）、'S_logit'（Energy score）
        config: 配置字典，需包含：
            - 'mean_energy': ID 训练集 Energy score 均值（标量）

    Returns:
        scores: CADRef 分数，形状 [B]，值越大越可能是 ID
    """
    decoupled_info, logits = forward_func(inputs, model, config)

    Ep = decoupled_info['Ep']          # [B]
    En = decoupled_info['En']          # [B]
    S_logit = decoupled_info['S_logit']  # [B]

    mean_energy = config['mean_energy']
    if isinstance(mean_energy, torch.Tensor):
        mean_energy = mean_energy.item()

    # 公式 (10)：E(x) = E_n / S̄_logit + E_p / S_logit
    # SCORE = -E(x)，取负使 ID 样本得高分
    E = En / mean_energy + Ep / S_logit
    scores = (-E.numpy())

    return scores


def get_block_norm_score(inputs: torch.Tensor, model: nn.Module, forward_func: Optional[callable] = None, config: Optional[dict] = None) -> np.ndarray:
    """
    计算特征图的 L2 范数作为 OOD score

    Reference:
        Yu, Y., Shin, S., Lee, S., Jun, C., & Lee, K. (2023).
        "Block Selection Method for Using Feature Norm in Out-of-Distribution Detection."
        CVPR 2023.

    Args:
        inputs: 输入图像，形状 [B, C, H, W]
        model: PyTorch 模型
        forward_func: 前向函数（可选）
        config: 配置字典，需包含 sblock

    Returns:
        norm: L2 范数分数，形状 [B]
    """
    if forward_func is None:
        forward_func = forward_all_features

    with torch.no_grad():
        features, logits = forward_func(inputs, model, config)

    # 取指定 block 的特征
    block_idx = config.get("sblock", 0)
    block_features = features[block_idx]

    # 计算 L2 范数
    norm = torch.norm(F.relu(block_features), dim=[2, 3]).mean(1)

    return norm.detach().cpu().numpy()


# 统一注册所有方法对应的函数
SCORE_FUNCS: Dict[str, callable] = {
    "msp": lambda inputs, model, forward_func, config, accelerator=None:
        get_msp_score(inputs, model, forward_func, config),
    "odin": lambda inputs, model, forward_func, config, accelerator=None:
        get_odin_score(inputs, model, forward_func, config, accelerator),
    "energy": lambda inputs, model, forward_func, config, accelerator=None:
        get_energy_score(inputs, model, forward_func, config),
    "top_energy": lambda inputs, model, forward_func, config, accelerator=None:
        get_topk_energy_score(inputs, model, forward_func, config),
    "energy_entropy": lambda inputs, model, forward_func, config, accelerator=None:
        get_energy_entropy_score(inputs, model, forward_func, config),
    "mahalanobis": lambda inputs, model, forward_func, config, accelerator=None:
        get_mahalanobis_score(inputs, model, config),
    "featurenorm": lambda inputs, model, forward_func, config, accelerator=None:
        get_block_norm_score(inputs, model, forward_func, config),
    "cadref": lambda inputs, model, forward_func, config, accelerator=None:
        get_cadref_score(inputs, model, forward_func, config),
}


def get_score(
    inputs: torch.Tensor,
    model: nn.Module,
    forward_func: callable,
    method: str,
    config: dict,
    accelerator=None
) -> np.ndarray:
    """
    根据指定方法获取 OOD 分数

    Args:
        inputs: 输入图像，形状 [B, C, H, W]
        model: PyTorch 模型
        forward_func: 前向函数或前向函数名称（字符串）
        method: 评分方法名称
        config: 配置字典
        accelerator: Accelerator 对象（可选）

    Returns:
        scores: OOD 分数，形状 [B]

    Raises:
        NotImplementedError: 未知的评分方法
    """
    from .forward import get_forward
    
    # 如果 forward_func 是字符串，则通过 get_forward 获取实际函数
    if isinstance(forward_func, str):
        forward_func = get_forward(forward_func, config)
    
    if method not in SCORE_FUNCS:
        raise NotImplementedError(f"Unknown scoring method: {method}")
    return SCORE_FUNCS[method](inputs, model, forward_func, config, accelerator)


def get_all_scores(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    forward_func: callable,
    method: str,
    config: dict,
    n_iter: Optional[int] = None,
    accelerator=None
) -> np.ndarray:
    """
    获取数据加载器中所有样本的 OOD 分数

    Args:
        model: PyTorch 模型
        data_loader: 数据加载器
        forward_func: 前向函数
        method: 评分方法名称
        config: 配置字典
        n_iter: 最大迭代次数（可选）
        accelerator: Accelerator 对象（可选）

    Returns:
        scores_all: 所有样本的 OOD 分数，形状 [N]
    """
    model.eval()
    scores_all = []
    for i, batch in enumerate(data_loader):
        x, y = batch
        score = get_score(x, model, forward_func, method, config, accelerator)
        scores_all.append(score)
        if n_iter is not None and i + 1 >= n_iter:
            break
    return np.concatenate(scores_all, axis=0)