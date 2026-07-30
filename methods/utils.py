"""
工具函数模块

包含通用的辅助函数，用于特征处理、统计计算和数据操作。
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Union


def compute_id_statistics(
    scores_in: np.ndarray,
    qs: float = 0.1,
    qe: float = 99.9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 ID 数据的统计量（均值、标准差、百分位数）

    Args:
        scores_in: 形状为 (L, N) 的数组，L 为层数，N 为样本数
        qs: 下百分位数（默认 0.1）
        qe: 上百分位数（默认 99.9）

    Returns:
        means: 每层的均值，形状 (L,)
        stds: 每层的标准差，形状 (L,)
        ps: 每层的下百分位数，形状 (L,)
        pe: 每层的上百分位数，形状 (L,)
    """
    scores_in = np.asarray(scores_in)
    means = np.mean(scores_in, axis=1)
    stds = np.std(scores_in, axis=1) + 1e-8  # 避免除零
    ps = np.percentile(scores_in, qs, axis=1)
    pe = np.percentile(scores_in, qe, axis=1)
    return means, stds, ps, pe


def compute_standard_score(
    scores_test: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray
) -> np.ndarray:
    """
    使用预计算的 ID 均值和标准差对测试分数进行标准化（Z-score）

    Args:
        scores_test: 形状为 (L, N) 的测试分数数组
        means: 每层的均值，形状 (L,)
        stds: 每层的标准差，形状 (L,)

    Returns:
        z_scores: 标准化后的分数，形状 (L, N)
    """
    scores_test = np.asarray(scores_test)
    means = np.asarray(means)
    stds = np.asarray(stds)

    z_scores = (scores_test - means[:, None]) / stds[:, None]
    return z_scores


# def get_vaod_score(scores_test: np.ndarray) -> np.ndarray:
#     """
#     基于跨层变异性计算 VaOD 分数

#     Args:
#         scores_test: 原始激活分数，形状 (L, N)

#     Returns:
#         vaod_scores: VaOD 分数，形状 (N,)，值越大表示越可能是 ID
#     """
#     vaod_scores = -np.std(scores_test, axis=0)
#     return vaod_scores





def prepare_eval_model_with_id_loader(
    config: dict,
    split: str = "train"
) -> Tuple[torch.nn.Module, torch.utils.data.DataLoader]:
    """
    准备评估模型和 ID 数据加载器（从 get_threshold.py 提取）

    Args:
        config: 配置字典
        split: 数据分割（train/val/test）

    Returns:
        model: 加载好权重的模型
        data_loader: 对应分割的数据加载器
    """
    from accelerate import Accelerator
    from data_utils.build_dataset import build_id_dataloaders
    from models import get_model
    from train.train_utils import load_clean_model_state

    accelerator = Accelerator(mixed_precision=config.get("mixed_precision", "no"))

    model_save_dir = os.path.join(config.get("save_dir", "checkpoints"), config["exp_name"])
    load_path = os.path.join(model_save_dir, config["test_model"], "model.pt")

    state_dict = load_clean_model_state(load_path)
    model = get_model(config["model"])
    model.load_state_dict(state_dict)

    if split == "train":
        id_train_loader, id_test_loader, id_val_loader = build_id_dataloaders(config, accelerator)
        return model, id_train_loader
    elif split == "val":
        id_train_loader, id_test_loader, id_val_loader = build_id_dataloaders(config, accelerator)
        return model, id_val_loader
    elif split == "test":
        id_train_loader, id_test_loader, id_val_loader = build_id_dataloaders(config, accelerator)
        return model, id_test_loader
    else:
        raise ValueError(f"Unknown split: {split}")


def get_test_model_dir(config: dict) -> str:
    """
    获取测试模型的保存目录

    Args:
        config: 配置字典

    Returns:
        model_dir: 模型目录路径
    """
    return os.path.join(config.get("save_dir", "checkpoints"), config["exp_name"], config["test_model"])


def extract_avgpool_features(
    model: torch.nn.Module,
    inputs: torch.Tensor
) -> torch.Tensor:
    """
    提取模型的 avgpool 层特征

    Args:
        model: PyTorch 模型
        inputs: 输入张量，形状 [B, C, H, W]

    Returns:
        feats: avgpool 特征，形状 [B, feature_dim]
    """
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    handle = model.avgpool.register_forward_hook(get_activation('avgpool'))
    with torch.no_grad():
        model(inputs)
    handle.remove()

    feats = activation['avgpool'].squeeze(-1).squeeze(-1)
    return feats


def collect_avgpool_features_with_labels(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader
) -> Tuple[np.ndarray, np.ndarray]:
    """
    收集所有样本的 avgpool 特征和标签

    Args:
        model: PyTorch 模型
        data_loader: 数据加载器

    Returns:
        all_features: 所有特征，形状 [N, feature_dim]
        all_labels: 所有标签，形状 [N]
    """
    all_features = []
    all_labels = []

    for batch in data_loader:
        x, y = batch
        with torch.no_grad():
            feats = extract_avgpool_features(model, x)
        all_features.append(feats.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)