import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.mahalanobis_lib import get_Mahalanobis_score
from .forward import  forward_all_features, get_forward
import os
from accelerate import Accelerator
from data_utils.build_dataset import build_id_dataloaders,build_ood_dataloaders,build_jigsaw_dataloaders
from models import get_model
import random
import numpy as np
from PIL import Image
import torchvision.transforms as T

## FuseNorm##




def compute_id_statistics(scores_in,qs=0.1,qe=99.9):
    """
    Compute per-layer statistics (mean, std, 0.5% and 99.5% percentiles)
    for in-distribution scores.

    Args:
        scores_in (np.ndarray): Array of shape (L, N),
            where L is the number of layers and N is the number of samples.

    Returns:
        means (np.ndarray): Mean score for each layer, shape (L,).
        stds (np.ndarray): Standard deviation for each layer, shape (L,).
        p005 (np.ndarray): 0.5th percentile for each layer, shape (L,).
        p995 (np.ndarray): 99.5th percentile for each layer, shape (L,).
    """
    scores_in = np.asarray(scores_in)
    means = np.mean(scores_in, axis=1)
    stds = np.std(scores_in, axis=1) + 1e-8  # avoid division by zero
    ps = np.percentile(scores_in, qs, axis=1)
    pe = np.percentile(scores_in, qe, axis=1)
    return means, stds, ps, pe





def l2norm(x):
    # x: [B, C, W, H]
    return torch.norm(x, dim=[2, 3]).mean(1)  # -> [B]




# 一些基础操作
augmentations = [
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.RandomRotation(15),
    T.RandomHorizontalFlip(p=1.0),
    T.RandomVerticalFlip(p=1.0),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
]
# 批量版本：输入 tensor[B,C,H,W] -> 输出 tensor[B,C,H,W]
def augmix_batch_fn(batch_tensor):
    """
    batch_tensor: [B, C, H, W] in [0,1]
    """
    batch_aug = []
    for img_tensor in batch_tensor:
        pil_img = T.ToPILImage()(img_tensor.cpu())
        aug_img = augmix_fn(pil_img)
        batch_aug.append(aug_img)
    return torch.stack(batch_aug).to(batch_tensor.device)

def augmix_fn(image, severity=1, width=3, depth=-1, alpha=1.):
    """
    最简 AugMix 实现
    image: PIL.Image
    severity: 增强强度
    width: 并行分支数
    depth: 每条分支长度 (-1 表示随机)
    alpha: Beta 分布参数（控制混合权重）
    """
    ws = np.random.dirichlet([alpha] * width).astype(np.float32)  # 每个分支权重
    m = np.random.beta(alpha, alpha)  # 与原图融合权重

    mix = torch.zeros_like(T.ToTensor()(image))
    for i in range(width):
        image_aug = image.copy()
        d = depth if depth > 0 else np.random.randint(1, 4)  # 每条分支深度
        for _ in range(d):
            op = random.choice(augmentations)
            image_aug = op(image_aug)
        mix += ws[i] * T.ToTensor()(image_aug)

    mixed = (1 - m) * T.ToTensor()(image) + m * mix
    return mixed







# 计算 ODIN 分数，通过输入添加小扰动 + 温度缩放来增强 OOD 识别
def get_odin_score(inputs, model, forward_func, config, accelerator):
    temper = config['odin_temperature']
    noiseMagnitude1 = config['odin_magnitude']

    criterion = nn.CrossEntropyLoss()

    # 将 inputs 放到 accelerator 设备并设置 requires_grad
    inputs = inputs.to(accelerator.device)
    inputs = torch.autograd.Variable(inputs, requires_grad=True)

    features,outputs = forward_func(inputs, model,config)

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
        features, outputs = forward_func(tempInputs, model,config)
        outputs = outputs / temper

    # Softmax 计算最大概率得分
    nnOutputs = outputs.data.cpu().numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    scores = np.max(nnOutputs, axis=1)

    return scores


from train.train_utils import load_clean_model_state

import numpy as np
import torch

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

def get_all_activation_strength(model, data_loader, norm_func, forward_name,
                                method="odin", accelerator=None, n_iter=None, config=None):
    """
    Collect activation strengths (feature norms) and OOD scores from all layers.

    Args:
        model: neural network model
        data_loader: data loader providing (x, y)
        norm_func: function to compute per-layer norms (returns list of np arrays)
        forward_name: name of forward pass function
        method: OOD scoring method ("odin", "energy", etc.)
        accelerator: optional accelerator (e.g., HuggingFace)
        n_iter: optional number of iterations (for subset evaluation)
        config: configuration dict

    Returns:
        norms_all: list of length L+1, where each element is np.array of shape [N].
                   The last entry corresponds to the final OOD score.
    """
    model.eval()
    norms_all = None
    forward_func = get_forward(name=forward_name, config=config)

    for i, batch in enumerate(data_loader):
        with torch.no_grad():
            x, y = batch
            batch_norms = get_norm(x, model, norm_func)  # list of length L

            # Initialize container once we know L
            if norms_all is None:
                L = len(batch_norms)
                norms_all = [[] for _ in range(L + 1)]

            # Collect per-layer norms
            for l in range(L):
                norms_all[l].append(batch_norms[l])

        # Collect output-based OOD score (e.g., ODIN / Energy)
        score = get_score(x, model, forward_func, method, config, accelerator)
        norms_all[-1].append(score)

        # Optional iteration limit to balance OOD test size and ID test size
        if n_iter is not None and i >= n_iter - 1:
            break

    # Concatenate per-layer results into single numpy arrays [N]
    norms_all = [np.concatenate(layer_norms, axis=0) for layer_norms in norms_all]
    return norms_all


def df(x):
    """Nonlinear deviation scaling function."""
    return np.minimum(1, np.power(20, x) - 1)


def fuse_norm(scores_main, scores_other):
    """
    Fuse scores from strong and weak experts (FuseNorm).

    Args:
        scores_main: np.array, strong expert scores [N]
        scores_other: np.array, weak experts' normalized scores [L, N]

    Returns:
        fused_score: np.array, adjusted strong expert scores [1, N]
        delta_score: np.array, OOD confidence adjustment [1, N]
    """
    scores_main = np.atleast_2d(np.copy(scores_main))
    scores_other = np.copy(scores_other)

    # Mean strong expert score across multiple strong cues (if any)
    scores_main_mean = np.mean(scores_main, axis=0, keepdims=True)

    # Compute deviation for weak experts
    s_high = scores_other > 1
    s_low = scores_other < 0
    scores_other = np.abs(scores_other * s_low + (scores_other - 1) * s_high)

    # Take max deviation among weak experts
    scores_other = np.max(scores_other, axis=0, keepdims=True)

    # Scale deviation (confidence calibration)
    delta_score = df(scores_other)

    # Fuse: subtract deviation from strong expert
    fused_score = scores_main_mean - delta_score

    return fused_score, delta_score


# 统一注册所有方法对应的函数
SCORE_FUNCS = {
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
}

def get_score(inputs, model, forward_func, method, config, accelerator=None):
    if method not in SCORE_FUNCS:
        raise NotImplementedError(f"Unknown scoring method: {method}")
    return SCORE_FUNCS[method](inputs, model, forward_func, config, accelerator)





def get_all_scores( model,data_loader, forward_func,method, config, n_iter=None,accelerator=None):
    model.eval()
    scores_all=[]
    for i, batch in enumerate(iter(data_loader)):
        x, y = batch
        score = get_score(x, model, forward_func, method, config,accelerator)
        scores_all.append(score)
        if i>=n_iter-2:
            break
    scores_all_tensor = np.concatenate(scores_all, axis=0)
    return scores_all_tensor


