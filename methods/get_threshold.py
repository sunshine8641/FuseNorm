import os
import torch
from accelerate import Accelerator
from data_utils.build_dataset import build_id_dataloaders,build_ood_dataloaders,build_jigsaw_dataloaders
from models import get_model
from safetensors.torch import load_file
from collections import OrderedDict
from metrics import  *
import os
from train.train_utils import load_clean_model_state
import sys
from torch.nn import functional as F


from train.train_utils import load_clean_model_state


def compute_cadref_stats(config):
    """
    计算 CADRef 所需的统计量：类平均特征 + ID 训练集平均 Energy score

    Reference:
        Ling, Z., Chang, Y., Zhao, H., Zhao, X., Chow, K., & Deng, S. (2025).
        "CADRef: Robust Out-of-Distribution Detection via Class-Aware Decoupled Relative Feature Leveraging."
        CVPR 2025.

    Args:
        config: 配置字典，需包含 num_classes
    """
    accelerator = Accelerator(mixed_precision=config.get("training", {}).get("mixed_precision", "no"))

    model_save_dir = os.path.join(config.get("save_dir", "checkpoints"), config["exp_name"])
    load_path = os.path.join(model_save_dir, config["test_model"], "model.pt")
    print(f"[CADRef] Loading model from: {load_path}")
    state_dict = load_clean_model_state(load_path)

    model = get_model(config["model"])
    model.load_state_dict(state_dict)

    id_train_loader, id_test_loader, id_val_loader = build_id_dataloaders(config, accelerator)
    model, id_train_loader = accelerator.prepare(model, id_train_loader)

    num_classes = config["num_classes"]

    # 收集训练集的特征、预测类别、logits
    all_features = []
    all_pred_classes = []
    all_energy_scores = []

    model.eval()
    with torch.no_grad():
        for batch in id_train_loader:
            x, y = batch
            feats = model.forward_features(x)
            logits = model.forward_head(feats)
            energy = torch.logsumexp(logits, dim=1)  # Energy score per sample

            all_features.append(feats.cpu())
            all_pred_classes.append(logits.argmax(dim=1).cpu())
            all_energy_scores.append(energy.cpu())

    all_features = torch.cat(all_features, dim=0)
    all_pred_classes = torch.cat(all_pred_classes, dim=0)
    all_energy_scores = torch.cat(all_energy_scores, dim=0)

    # 按预测类别聚合，计算每个类的平均特征（公式 5）
    class_centroids = torch.zeros(num_classes, all_features.shape[1])
    for k in range(num_classes):
        mask = (all_pred_classes == k)
        if mask.sum() > 0:
            class_centroids[k] = all_features[mask].mean(dim=0)

    # 计算 ID 训练集 Energy score 的均值（公式 9 的 S̄_logit）
    mean_energy = all_energy_scores.mean()

    output_dir = os.path.join(model_save_dir, config["test_model"])
    os.makedirs(output_dir, exist_ok=True)

    stats = {
        "class_centroids": class_centroids,
        "mean_energy": mean_energy,
    }
    torch.save(stats, os.path.join(output_dir, "cadref_stats.pt"))
    print(f"[CADRef] Saved stats to: {os.path.join(output_dir, 'cadref_stats.pt')}")
    print(f"  - class_centroids shape: {class_centroids.shape}")
    print(f"  - mean_energy: {mean_energy:.4f}")


def compute_threshold_react(config):
    accelerator = Accelerator(mixed_precision=config.get("training", {}).get("mixed_precision", "no"))

    # 目录和变量准备
    model_save_dir = os.path.join(config.get("save_dir", "checkpoints"), config["exp_name"])
    load_path = os.path.join(model_save_dir, config["test_model"] ,"model.pt")
    print(load_path)
    state_dict = load_clean_model_state(load_path)

    model = get_model(config["model"])
    model.load_state_dict(state_dict)
    id_train_loader, id_test_loader, id_val_loader = build_id_dataloaders(config, accelerator)

    model, id_val_loader = accelerator.prepare(
        model, id_val_loader)


    # 2. 定义 activation 容器和 hook 函数
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # 3. 注册 hook（以 ResNet 的 avgpool 层为例）
    hooker_handles = []
    layer_remark = 'avgpool'
    hooker_handles.append(
        model.avgpool.register_forward_hook(get_activation(layer_remark))
    )
    activation_log = []
    for index, batch in enumerate(id_val_loader):
        x, y = batch
        curr_batch_size = x.shape[0]
        with torch.no_grad():
            out_puts = model(x)
        avgpool_feats = activation[layer_remark]  # shape: [batch_size, 512, 1, 1] for resnet18
        dim = avgpool_feats.shape[1]
        activation_log.append(avgpool_feats.data.cpu().numpy().reshape(curr_batch_size, dim, -1).mean(2))

        # print(f"Batch : avgpool features shape = {avgpool_feats.shape}")
        activation.clear()
        # if index == 0:
        #     break

    for handle in hooker_handles:
        handle.remove()
    activation_log = np.concatenate(activation_log, axis=0)
    react_threshold=np.percentile(activation_log.flatten(), 90)
    react_threshold=torch.tensor(react_threshold)
    print(f"\nTHRESHOLD at percentile {90} is:")
    print(react_threshold)
    output_dir = os.path.join(model_save_dir, config["test_model"])
    torch.save(react_threshold, os.path.join(output_dir, "react_threshold.pt"))




def compute_activation_stats(config):
    accelerator = Accelerator(mixed_precision=config.get("training", {}).get("mixed_precision", "no"))

    # 目录和变量准备
    model_save_dir = os.path.join(config.get("save_dir", "checkpoints"), config["exp_name"])
    load_path = os.path.join(model_save_dir, config["test_model"] ,"model.pt")
    print(load_path)
    state_dict = load_clean_model_state(load_path)

    model = get_model(config["model"])
    model.load_state_dict(state_dict)

    id_train_loader, id_test_loader, id_val_loader = build_id_dataloaders(config, accelerator)

    model, id_val_loader = accelerator.prepare(model, id_val_loader)

    # 2. 定义 activation 容器和 hook 函数
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # 3. 注册 hook（以 ResNet 的 avgpool 层为例）
    hooker_handles = []
    layer_remark = 'avgpool'
    hooker_handles.append(
        model.avgpool.register_forward_hook(get_activation(layer_remark))
    )

    # 用于记录所有样本的 avgpool 特征（[B, C]）
    activation_log = []
    for index, batch in enumerate(id_val_loader):
        x, y = batch
        with torch.no_grad():
            outputs = model(x)

        avgpool_feats = activation[layer_remark]  # shape: [B, C, 1, 1]
        feats = avgpool_feats.squeeze(-1).squeeze(-1).cpu().numpy()  # [B, C]
        activation_log.append(feats)

        activation.clear()

    # 清理 hook
    for handle in hooker_handles:
        handle.remove()

    # 拼接所有样本的特征：[N, C]
    activation_log = np.concatenate(activation_log, axis=0)  # shape: [N, C]
    print(activation_log.shape)
    # 按通道计算 mean 和 std（shape: [C]）
    feature_mean = torch.tensor(np.mean(activation_log, axis=0))
    feature_std = torch.tensor(np.std(activation_log, axis=0))

    # 保存 mean 和 std（改为保存 tensor）
    output_dir = os.path.join(model_save_dir, config["test_model"])

    stats = {
        "feature_mean": feature_mean,  # tensor [C]
        "feature_std": feature_std     # tensor [C]
    }
    torch.save(stats, os.path.join(output_dir, "feature_stats.pt"))

    print(f"\n✅ Saved per-channel stats to: {os.path.join(output_dir, 'feature_stats.pt')}")
    print("\nPer-channel mean:")
    print(feature_mean)
    print("\nPer-channel std:")
    print(feature_std)

    # 若你需要总体均值和方差（scalar）
    print("\nOverall mean of means:", feature_mean.mean())
    print("Overall mean of stds:", feature_std.mean())


def calculate_layer_norm(config):
    # model, train_loader, blur_loader, accelerator,
    num_blocks = config["num_blocks"]
    accelerator = Accelerator(mixed_precision=config.get("training", {}).get("mixed_precision", "no"))

    # 目录和变量准备
    model_save_dir = os.path.join(config.get("save_dir", "checkpoints"), config["exp_name"])
    load_path = os.path.join(model_save_dir, config["test_model"], "model.pt")
    print(load_path)
    state_dict = load_clean_model_state(load_path)

    model = get_model(config["model"])
    model.load_state_dict(state_dict)


    model.eval()  # 评估模式（不更新BN/Dropout）

    # 存储每个 block 的特征范数
    norm_pred_ori = {i: [] for i in range(num_blocks)}
    norm_pred_jigsaw = {i: [] for i in range(num_blocks)}
    id_train_loader, id_test_loader, id_val_loader = build_id_dataloaders(config, accelerator)
    jigsaw_train_loader, jigsaw_test_loader, jigsaw_val_loader = build_jigsaw_dataloaders(config, accelerator)
    model, id_train_loader,jigsaw_train_loader = accelerator.prepare(model, id_train_loader,jigsaw_train_loader)

    if accelerator.is_main_process:
        print(type(model).__name__, len(id_train_loader), len(jigsaw_train_loader))

    with torch.no_grad():
        # 同时迭代原图和模糊图 loader
        for batch_idx, (data1, data2) in enumerate(zip(id_train_loader, jigsaw_train_loader)):
            # data1[0], data2[0] 已经在 accelerator.prepare 后分发到对应设备
            x = torch.cat([data1[0], data2[0]], dim=0)

            # 提取每个 block 的输出特征
            features = model.forward_features_blockwise(x)

            # 遍历每个 block
            for i in range(num_blocks):
                norm = torch.norm(F.relu(features[i]), dim=[2, 3]).mean(1)
                norm_ori = norm[:len(data1[0])]
                norm_jigsaw = norm[len(data1[0]):]

                norm_pred_ori[i].append(norm_ori)
                norm_pred_jigsaw[i].append(norm_jigsaw)
            if batch_idx >200:
                break

    # 聚合所有 batch 的结果
    for i in range(num_blocks):
        norm_pred_ori[i] = torch.cat(norm_pred_ori[i], dim=0)
        norm_pred_jigsaw[i] = torch.cat(norm_pred_jigsaw[i], dim=0)

        # 在多进程下同步结果
        norm_pred_ori[i] = accelerator.gather(norm_pred_ori[i])
        norm_pred_jigsaw[i] = accelerator.gather(norm_pred_jigsaw[i])

        if accelerator.is_main_process:
            ratio = (norm_pred_ori[i] / norm_pred_jigsaw[i]).mean()
            print(f'NormRatio-Block{i}: {ratio}')


