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



def topk_l2norm(x, k=0.1):
    # x: [B, C, H, W]
    B, C, H, W = x.shape
    x_flat = x.view(B, C, -1)                # [B, C, H*W]
    k_top = int(k * x_flat.shape[-1])        # 每个通道的top-k数量
    if k_top < 1:
        k_top = 1
# 计算每个channel的平方激活并取top-k
    energy = x_flat.pow(2)
    top_vals, _ = torch.topk(energy, k_top, dim=-1)  # [B, C, k_top]

    # 每个channel的L2 norm（对top-k值开平方再求平均）
    per_channel_norm = top_vals.mean(-1).sqrt()      # [B, C]

    # 对所有channel取平均
    return per_channel_norm.mean(-1)                 # [B]


def l2norm(x):
    # x: [B, C, W, H]
    return torch.norm(x, dim=[2, 3]).mean(1)  # -> [B]


def adaptive_topk_l2norm(x, k_min=0.05, k_max=0.2, eps=1e-8):  # 空间上的选取激活区域
    # x: [B, C, H, W]
    B, C, H, W = x.shape
    x_flat = x.view(B, C, -1)
    energy = x_flat.pow(2)

    # 计算能量分布的熵
    p = energy / (energy.sum(dim=-1, keepdim=True) + eps)
    entropy = -(p * (p + eps).log()).sum(dim=-1) / torch.log(torch.tensor(float(H * W)))
    entropy_mean = entropy.mean(dim=-1)  # [B]

    # 自适应计算每个样本的k比例
    k_ratio = k_min + (k_max - k_min) * entropy_mean
    k_vals = (k_ratio * (H * W)).long().clamp(min=1)

    norms = []
    for i in range(B):
        k_top = k_vals[i].item()
        top_vals, _ = torch.topk(energy[i], k_top, dim=-1)
        per_channel_norm = top_vals.mean(-1).sqrt()
        norms.append(per_channel_norm.mean())
    return torch.stack(norms)

# def soft_topk_l2norm(feature_map, tau=0.1, p=2, eps=1e-8):
#     """
#     Dynamic Soft Top-k FeatureNorm.
#
#     Args:
#         feature_map (Tensor): shape [B, C, H, W], activation features.
#         tau (float): temperature for softmax weighting (smaller -> more top-k-like).
#         p (float): Lp norm type (default=2).
#         eps (float): numerical stability constant.
#
#     Returns:
#         Tensor: [B] FeatureNorm score per sample.
#     """
#     B, C, H, W = feature_map.shape
#
#     # compute spatial Lp norm per channel
#     energy = feature_map.abs().pow(p).mean(1)  # [B, H, W]
#     energy_flat = energy.view(B, -1)  # [B, HW]
#
#     # compute softmax weights (higher energy → higher weight)
#     weights = F.softmax(energy_flat / tau, dim=1)  # [B, HW]
#
#     # compute weighted feature norm
#     soft_score = (weights * energy_flat).sum(dim=1)  # [B]
#
#     # normalize by expected uniform weight (optional)
#     soft_score = soft_score / (weights.sum(dim=1) + eps)
#
#     return soft_score




def channel_consistency_topk(x, k=0.1, eps=1e-8):
    B, C, H, W = x.shape
    var_map = x.var(dim=1, keepdim=True)  # 通道方差 [B,1,H,W]
    weight = 1 / (var_map + eps)
    weighted_energy = (x.pow(2) * weight)
    x_flat = weighted_energy.view(B, C, -1)
    k_top = int(k * x_flat.shape[-1])
    top_vals, _ = torch.topk(x_flat, k_top, dim=-1)
    return top_vals.mean(-1).sqrt().mean(-1)


def get_norm(inputs, model, norm_func=l2norm):
    with torch.no_grad():
        features = model.forward_features_blockwise(inputs)  # list of [B,C,W,H]
        features=[F.relu(feature) for feature in features]
    norms = []
    for i in range(len(features)):
        norm = norm_func(features[i])  # [B]
        norms.append(norm.detach().cpu().numpy())
    del inputs, features
    return norms  # list of [B] arrays




def compute_standard_score(scores_test, means, stds):
    """
    Standardize test scores using pre-computed ID means and standard deviations.

    Args:
        scores_test (np.ndarray): Array of shape (L, N),
            where L is the number of layers and N is the number of test samples.
        means (np.ndarray): Per-layer mean from ID data, shape (L,).
        stds (np.ndarray): Per-layer std from ID data, shape (L,).

    Returns:
        z_scores (np.ndarray): Standardized scores (z-scores) of shape (L, N).
    """
    scores_test = np.asarray(scores_test)
    means = np.asarray(means)
    stds = np.asarray(stds)  # stability term

    z_scores = (scores_test - means[:, None]) / stds[:, None]
    return z_scores





def get_vaod_score(scores_test):
    """
    Compute VaOD scores for test samples based on cross-layer variability.

    Args:
        scores_test (np.ndarray): Raw activation scores of shape (L, N),
            where L is the number of layers and N is the number of samples.
        means (np.ndarray): Per-layer mean scores from ID data, shape (L,).
        stds (np.ndarray): Per-layer std scores from ID data, shape (L,).

    Returns:
        vaod_scores (np.ndarray): VaOD scores of shape (N,).
            Higher values indicate stronger in-distribution likelihood.
    """
    vaod_scores = -np.std(scores_test, axis=0)

    return vaod_scores




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








# 计算 Maximum Softmax Probability（MSP）得分
def get_msp_score(inputs, model, forward_func, config=None):
    # 如果没有提供 logits，则用 forward_func 计算模型输出
    with torch.no_grad():
        features, logits = forward_func(inputs, model,config)
    # 取 softmax 概率的最大值作为得分（confidence）
    scores = np.max(F.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
    return scores





def get_block_norm_score(inputs, model, forward_func=None, config=None):
    # 如果没有提供 features，则用 forward_func 获取
    forward_func=forward_all_features
    with torch.no_grad():
        features,logits = forward_func(inputs, model, config)
    # 取指定 block 的特征
    block_idx = config.get("sblock", 0)
    block_features = features[block_idx]
    # norm 计算（原版 L2 范数，也可改成 amax）
    norm = torch.norm(F.relu(block_features), dim=[2, 3]).mean(1)
    # 返回 CPU numpy 数组
    return norm.detach().cpu().numpy()

# 计算 Energy-based score（logsumexp）作为得分
def get_energy_score(inputs, model, forward_func, config):
    """
    energy 越小 越表示in distribution， 因此返回的是-energy score
    torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    """
    #默认设置问题T=1，因此不二外设置T。 energy越大
    with torch.no_grad():
        features,logits = forward_func(inputs, model,config)
    # 计算 logsumexp（log-sum-exp trick）作为能量得分
    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    return scores



def get_topk_energy_score(inputs, model, forward_func, config):
    with torch.no_grad():
        features,logits = forward_func(inputs, model, config)

    # 取 logits 的 top-k（按值）
    topk_logits, _ = torch.topk(logits, k=config["energy_k"], dim=1)

    # 在 top-k 上计算 logsumexp
    scores = torch.logsumexp(topk_logits.cpu(), dim=1).numpy()

    return scores

def get_energy_entropy_score(inputs, model, forward_func, config):
    """
    融合 Energy 和 Entropy 的 OOD score。
    越大越可能是 ID，越小越可能是 OOD。

    Args:
        inputs: 输入图像 [B, C, H, W]
        model: PyTorch 模型
        forward_func: 前向函数，返回 (features, logits)
        config: 配置字典，需包含 alpha, temperature
    """
    alpha = config.get("alpha", 0.5)          # 权重系数，默认0.5
    T = config.get("temperature", 2.0)        # 温度，默认1.0

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
    # return [energy.cpu().numpy(),entropy.cpu().numpy()]

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

def compute_mahalanobis_stats(config):
    """
    计算 Mahalanobis 需要的统计量：
        - 每类特征均值 sample_mean: [num_classes, feature_dim]
        - 协方差精度矩阵 precision: [feature_dim, feature_dim]

    Args:
        config: dict 配置，需包含：
            - exp_name
            - test_model
            - model
            - save_dir
            - num_classes
            - batch_size
    Returns:
        sample_mean: torch.tensor [num_classes, feature_dim]
        precision: torch.tensor [feature_dim, feature_dim]
    """

    accelerator = Accelerator(mixed_precision=config.get("training", {}).get("mixed_precision", "no"))

    # 1. 加载模型
    model_save_dir = os.path.join(config.get("save_dir", "checkpoints"), config["exp_name"])
    load_path = os.path.join(model_save_dir, config["test_model"], "model.pt")
    state_dict = load_clean_model_state(load_path)

    model = get_model(config["model"])
    model.load_state_dict(state_dict)
    model.eval()

    # 2. 数据加载
    id_train_loader, _, _ = build_id_dataloaders(config, accelerator)
    model, id_train_loader = accelerator.prepare(model, id_train_loader)

    # 3. 注册 hook
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    layer_remark = 'avgpool'  # 根据实际需要修改
    handle = model.avgpool.register_forward_hook(get_activation(layer_remark))

    # 4. 收集每类特征
    num_classes = config['num_classes']
    features_per_class = [[] for _ in range(num_classes)]

    for x, y in id_train_loader:
        with torch.no_grad():
            _ = model(x)
        feats = activation[layer_remark].squeeze(-1).squeeze(-1).cpu().numpy()  # [B, C]
        y = y.cpu().numpy()
        for cls in range(num_classes):
            cls_mask = (y == cls)
            if cls_mask.sum() > 0:
                features_per_class[cls].append(feats[cls_mask])

        activation.clear()

    handle.remove()

    # 5. 计算每类均值
    sample_mean = []
    all_features = []
    for cls_feats in features_per_class:
        cls_feats = np.concatenate(cls_feats, axis=0)  # [N_cls, C]
        cls_mean = np.mean(cls_feats, axis=0)          # [C]
        sample_mean.append(cls_mean)
        all_features.append(cls_feats)

    sample_mean = np.stack(sample_mean, axis=0)       # [num_classes, C]
    all_features = np.concatenate(all_features, axis=0)  # [N_total, C]

    # 6. 计算协方差并取逆矩阵
    cov = np.cov(all_features, rowvar=False)          # [C, C]
    precision = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))  # 防止奇异

    # 转为 torch tensor 并保存
    sample_mean = torch.tensor(sample_mean, dtype=torch.float32)
    precision = torch.tensor(precision, dtype=torch.float32)

    stats = {
        "sample_mean": sample_mean,
        "precision": precision
    }

    output_dir = os.path.join(model_save_dir, config["test_model"])
    os.makedirs(output_dir, exist_ok=True)
    torch.save(stats, os.path.join(output_dir, "mahalanobis_stats.pt"))

    print(f"✅ Saved Mahalanobis stats to {os.path.join(output_dir, 'mahalanobis_stats.pt')}")
    return sample_mean, precision
# 计算 Mahalanobis 距离分数，基于各类别特征的协方差建模

def get_mahalanobis_score(inputs, model, config):
    """
    计算 Mahalanobis OOD score。越小越可能是 OOD。
    如果存在 mahalanobis_stats.pt，会自动加载 sample_mean 和 precision。
    否则，会调用 compute_mahalanobis_stats(config) 生成。

    Args:
        inputs: torch.Tensor [B, C, H, W]
        model: torch.nn.Module
        config: 配置字典
    Returns:
        scores: np.ndarray [B]，越小越 OOD
    """
    # 1. 尝试加载统计量
    model_save_dir = os.path.join(config.get("save_dir", "checkpoints"), config["exp_name"])
    stats_path = os.path.join(model_save_dir, config["test_model"], "mahalanobis_stats.pt")

    if os.path.exists(stats_path):
        stats = torch.load(stats_path)
        sample_mean = stats["sample_mean"]  # [num_classes, feature_dim]
        precision = stats["precision"]  # [feature_dim, feature_dim]
    else:
        print("⚠️ mahalanobis_stats.pt not found, computing statistics...")
        sample_mean, precision = compute_mahalanobis_stats(config)

    device = next(model.parameters()).device
    sample_mean = sample_mean.to(device)
    precision = precision.to(device)

    model.eval()
    with torch.no_grad():
        # 取 avgpool 层特征
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        handle = model.avgpool.register_forward_hook(get_activation("avgpool"))
        _ = model(inputs)
        feats = activation["avgpool"].squeeze(-1).squeeze(-1)  # [B, feature_dim]
        handle.remove()

    # 计算每个样本到每类均值的 Mahalanobis 距离
    B = feats.shape[0]
    num_classes = sample_mean.shape[0]
    feature_dim = feats.shape[1]

    # Mahalanobis distance: d(x) = (x - mean)^T @ precision @ (x - mean)
    feats_exp = feats.unsqueeze(1).expand(B, num_classes, feature_dim)  # [B, num_classes, C]
    mean_exp = sample_mean.unsqueeze(0).expand(B, num_classes, feature_dim)  # [B, num_classes, C]
    diff = feats_exp - mean_exp  # [B, num_classes, C]

    # batch-wise矩阵乘法： diff @ precision @ diff^T
    # diff [B, num_classes, C], precision [C, C]
    # 先乘 precision: [B, num_classes, C]
    diff_precision = torch.matmul(diff, precision)  # [B, num_classes, C]
    # 逐类 Mahalanobis 距离
    mahalanobis_dist = torch.sum(diff_precision * diff, dim=2)  # [B, num_classes]

    # 对每个样本取最小距离作为 OOD score
    scores = -torch.min(mahalanobis_dist, dim=1)[0]  # 负号为了越大越可能是 ID，越小越 OOD

    return scores.cpu().numpy()


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




##下面为曾经试验过代码
# def get_weight_based_score(inputs, model, forward_func, config=None, mode="cosine"):
#     """
#     OOD 分数计算方式：
#     - mode="logit":    S(x) = max_c W_c^T f(x)
#     - mode="norm":     S(x) = max_c W_c^T f(x) / ||W_c||
#     - mode="cosine":   S(x) = max_c cos(f(x), W_c)
#     """
#     with torch.no_grad():
#         # 提取特征 f(x)
#         feats,logits = forward_func(inputs, model,config) # 假设支持返回特征
#
#         # classifier 权重 W [num_classes, feat_dim]
#         W = model.fc.weight if hasattr(model, "fc") else model.classifier.weight
#         # 一般是 [num_classes, feat_dim]
#         if mode == "logit":
#             """
#                         效果接近，但不如energy
#             """
#             # S(x) = max_c W_c^T f(x)
#             scores = torch.matmul(feats, W.t())  # [batch, num_classes]
#             scores, _ = scores.max(dim=1)
#
#         elif mode == "norm":
#             """
#                 还不如上面，normalization后丢失信息过多。
#             """
#             # S(x) = max_c W_c^T f(x) / ||W_c||
#             W_norm = F.normalize(W, p=2, dim=1)  # 每个类别权重归一化
#             scores = torch.matmul(feats, W_norm.t())
#             scores, _ = scores.max(dim=1)
#
#         elif mode == "cosine":
#             """
#             还不如上面，normalization后丢失信息过多。
#             """
#             # S(x) = max_c cos(f(x), W_c)
#             feats_norm = F.normalize(feats, p=2, dim=1)
#             W_norm = F.normalize(W, p=2, dim=1)
#             scores = torch.matmul(feats_norm, W_norm.t())
#             scores, _ = scores.max(dim=1)
#
#         else:
#             raise ValueError(f"Unsupported mode: {mode}")
#
#     return scores.detach().cpu().numpy()
#
#
#
# def get_cutmix_divergence_score(inputs, model, forward_func, config, accelerator):
#     """
#     inputs: [B,C,H,W] tensor in [0,1] 或已经归一化
#     model: PyTorch 模型
#     forward_func: 前向函数
#     config: 配置字典，支持 'odin_temperature'
#     accelerator: accelerate 管理器
#     num_aug: 每个输入生成多少个 AugMix 版本
#     """
#     temper =3# config['odin_temperature']
#     inputs = inputs.to(accelerator.device)
#
#     with (torch.no_grad()):
#         # 原始输入预测
#         feats,outputs_clean = forward_func(inputs, model, config)
#         outputs_clean=outputs_clean/ temper
#         probs_clean = F.softmax(outputs_clean, dim=1)
#
#         # 多个增强版本预测
#         probs_aug_all = []
#         for _ in range(1):
#             aug_inputs = augmix_batch_fn(inputs)  # 批量增强
#             feats,out_aug = forward_func(aug_inputs, model, config)
#             out_aug=out_aug/temper
#             probs_aug = F.softmax(out_aug, dim=1)
#             probs_aug_all.append(probs_aug)
#
#         # -------- 一致性分数计算 --------
#         # 对称 KL divergence: KL(clean || aug) + KL(aug || clean)
#         scores = []
#         for probs_aug in probs_aug_all:
#             kl1 = F.kl_div(probs_aug.log(), probs_clean,reduction='none')
#             kl1 = kl1.sum(dim=1)  # (B,)  每个样本一个分数
#             # print(kl1)
#
#             kl2 = F.kl_div(probs_clean.log(), probs_aug,reduction='none')
#             kl2 = kl2.sum(dim=1)  # (B,)  每个样本一个分数
#             # print(kl2)
#             scores.append(0.5 * (kl1 + kl2))
#
#         # print(scores)
#         # 取平均 KL divergence 作为 OOD score
#         scores = torch.stack(scores)  # 把 [tensor(...), tensor(...)] 堆成一个张量
#
#         scores = scores.mean(dim=0)
#         score = scores.detach().cpu().numpy()
#
#     return score
