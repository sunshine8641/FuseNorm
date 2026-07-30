"""
特征范数计算模块

包含各种特征范数计算方法，用于 OOD 检测中的特征强度评估。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


def l2norm(x: torch.Tensor) -> torch.Tensor:
    """
    计算特征图的 L2 范数

    Args:
        x: 输入特征图，形状 [B, C, H, W]

    Returns:
        norm: L2 范数分数，形状 [B]
    """
    return torch.norm(x, dim=[2, 3]).mean(1)


def topk_l2norm(x: torch.Tensor, k: float = 0.1) -> torch.Tensor:
    """
    计算 Top-K L2 范数

    在空间维度上选择能量最大的 top-k 区域进行范数计算。

    Args:
        x: 输入特征图，形状 [B, C, H, W]
        k: 选择比例（0.0-1.0）

    Returns:
        norm: Top-K L2 范数分数，形状 [B]
    """
    B, C, H, W = x.shape
    x_flat = x.view(B, C, -1)  # [B, C, H*W]
    k_top = int(k * x_flat.shape[-1])  # 每个通道的 top-k 数量
    if k_top < 1:
        k_top = 1

    # 计算每个 channel 的平方激活并取 top-k
    energy = x_flat.pow(2)
    top_vals, _ = torch.topk(energy, k_top, dim=-1)  # [B, C, k_top]

    # 每个 channel 的 L2 范数（对 top-k 值开平方再求平均）
    per_channel_norm = top_vals.mean(-1).sqrt()  # [B, C]

    # 对所有 channel 取平均
    return per_channel_norm.mean(-1)  # [B]


# def adaptive_topk_l2norm(
#     x: torch.Tensor,
#     k_min: float = 0.05,
#     k_max: float = 0.2,
#     eps: float = 1e-8
# ) -> torch.Tensor:
#     """
#     自适应 Top-K L2 范数

#     根据特征能量分布的熵动态调整选择比例。

#     Args:
#         x: 输入特征图，形状 [B, C, H, W]
#         k_min: 最小选择比例
#         k_max: 最大选择比例
#         eps: 数值稳定性参数

#     Returns:
#         norm: 自适应 Top-K L2 范数分数，形状 [B]
#     """
#     B, C, H, W = x.shape
#     x_flat = x.view(B, C, -1)
#     energy = x_flat.pow(2)

#     # 计算能量分布的熵
#     p = energy / (energy.sum(dim=-1, keepdim=True) + eps)
#     entropy = -(p * (p + eps).log()).sum(dim=-1) / torch.log(torch.tensor(float(H * W)))
#     entropy_mean = entropy.mean(dim=-1)  # [B]

#     # 自适应计算每个样本的 k 比例
#     k_ratio = k_min + (k_max - k_min) * entropy_mean
#     k_vals = (k_ratio * (H * W)).long().clamp(min=1)

#     norms = []
#     for i in range(B):
#         k_top = k_vals[i].item()
#         top_vals, _ = torch.topk(energy[i], k_top, dim=-1)
#         per_channel_norm = top_vals.mean(-1).sqrt()
#         norms.append(per_channel_norm.mean())
#     return torch.stack(norms)


# def channel_consistency_topk(x: torch.Tensor, k: float = 0.1, eps: float = 1e-8) -> torch.Tensor:
#     """
#     通道一致性 Top-K 范数

#     根据通道间的方差动态加权能量值。

#     Args:
#         x: 输入特征图，形状 [B, C, H, W]
#         k: 选择比例
#         eps: 数值稳定性参数

#     Returns:
#         norm: 通道一致性 Top-K 范数分数，形状 [B]
#     """
#     B, C, H, W = x.shape
#     var_map = x.var(dim=1, keepdim=True)  # 通道方差 [B, 1, H, W]
#     weight = 1 / (var_map + eps)
#     weighted_energy = (x.pow(2) * weight)
#     x_flat = weighted_energy.view(B, C, -1)
#     k_top = int(k * x_flat.shape[-1])
#     top_vals, _ = torch.topk(x_flat, k_top, dim=-1)
#     return top_vals.mean(-1).sqrt().mean(-1)


def adaptive_topk_channel_weight(x, k_min=0.1, k_max=0.25, alpha=0.0, eps=1e-8):
    """
    Adaptive Top-k Channel Weighted Norm (ATF)
    ------------------------------------------
    This function computes an uncertainty-aware spatial-channel aggregation
    of feature activations. It adaptively determines:
      (1) the spatial selection ratio (Top-k) per sample, and
      (2) the channel contribution weights based on entropy.

    Args:
        x (torch.Tensor): Feature map of shape [B, C, H, W].
        k_min (float): Minimum spatial selection ratio.
        k_max (float): Maximum spatial selection ratio.
        alpha (float): Smoothing factor controlling uniform prior strength.
        eps (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Aggregated feature norms of shape [B].
    """
    B, C, H, W = x.shape
    x_flat = x.view(B, C, -1)
    energy = x_flat.pow(2)

    # ---- (1) Compute channel-wise entropy ----
    p = energy / (energy.sum(dim=-1, keepdim=True) + eps)
    entropy = -(p * (p + eps).log()).sum(dim=-1)
    entropy = entropy / torch.log(torch.tensor(float(H * W), device=x.device))  # Normalize entropy to [0, 1]

    # ---- (2) Derive channel weights (low entropy → high confidence) ----
    weights = 1.0 - entropy
    weights = weights / (weights.sum(dim=-1, keepdim=True) + eps)

    # Apply entropy smoothing (encouraging balanced weighting)
    uniform = torch.full_like(weights, 1.0 / C)
    weights = (1 - alpha) * weights + alpha * uniform
    weights = weights / (weights.sum(dim=-1, keepdim=True) + eps)

    # ---- (3) Determine adaptive Top-k per sample ----
    entropy_mean = entropy.mean(dim=-1)
    k_ratio = k_min + (k_max - k_min) * entropy_mean  # higher uncertainty → larger k
    k_vals = (k_ratio * (H * W)).long().clamp(min=1)

    # ---- (4) Compute weighted Top-k norm ----
    norms = []
    for i in range(B):
        k_top = int(k_vals[i])
        top_vals, _ = torch.topk(energy[i], k_top, dim=-1)
        per_ch_norm = top_vals.mean(-1).sqrt()  # Spatial aggregation
        norm_val = (per_ch_norm * weights[i]).sum()  # Channel fusion
        norms.append(norm_val)

    return torch.stack(norms)


def get_norm(
    inputs: torch.Tensor,
    model: torch.nn.Module,
    norm_func: callable = l2norm
) -> List[np.ndarray]:
    """
    获取模型各层的特征范数

    Args:
        inputs: 输入张量，形状 [B, C, H, W]
        model: PyTorch 模型（需支持 forward_features_blockwise 方法）
        norm_func: 范数计算函数

    Returns:
        norms: 各层范数列表，每个元素形状为 [B]
    """
    with torch.no_grad():
        features = model.forward_features_blockwise(inputs)  # list of [B, C, W, H]
        features = [F.relu(feature) for feature in features]

    norms = []
    for i in range(len(features)):
        norm = norm_func(features[i])  # [B]
        norms.append(norm.detach().cpu().numpy())

    return norms  # list of [B] arrays


def get_all_activation_strength(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    norm_func: callable,
    n_iter: Optional[int] = None,
    *args,
    **kwargs
) -> List[np.ndarray]:
    """
    获取所有样本的激活强度（特征范数）

    Args:
        model: PyTorch 模型
        data_loader: 数据加载器
        norm_func: 范数计算函数
        n_iter: 最大迭代次数（可选）
        *args: 额外位置参数（保持向后兼容）
        **kwargs: 额外关键字参数（保持向后兼容）

    Returns:
        result: 各层激活强度，形状为 (L, N)
    """
    model.eval()
    all_norms = []
    for i, batch in enumerate(data_loader):
        x, y = batch
        norms = get_norm(x, model, norm_func)
        all_norms.append(norms)
        if n_iter is not None and i + 1 >= n_iter:
            break

    # 转换为 (L, N) 形状
    L = len(all_norms[0])
    result = [np.concatenate([batch[i] for batch in all_norms]) for i in range(L)]
    return result