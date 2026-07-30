import os
import torch
import torch.nn as nn
from accelerate import Accelerator
from data_utils.build_dataset import build_id_dataloaders,build_ood_dataloaders,build_jigsaw_dataloaders
from methods import forward_base,calculate_layer_norm
from models import get_model,ParametricBReLU,ParametricBoundedLeakyReLU,BoundedPReLU
from utils import get_args, merge_args_with_config, load_config, accuracy, MetersGroup,\
format_meters_log
from methods import get_forward,get_score,compute_activation_stats,get_all_scores,\
                    compute_threshold_react,calculate_layer_norm,get_all_scores
from metrics import  *
from train.train_utils import load_clean_model_state

def initialize_experiment(config):
    """
    初始化实验环境，包括：
    - HuggingFace Accelerate 加速器
    - 模型加载
    - ID / OOD 数据加载器
    - 损失函数与指标统计器

    Args:
        config (dict): 配置字典，需包含如下字段：
            - training.mixed_precision: (str) 混合精度模式 ("no" / "fp16" / "bf16")
            - save_dir: (str) 模型保存路径
            - exp_name: (str) 实验名称
            - test_model: (str) 测试模型文件夹名称
            - model: (dict) 模型配置
            - ood_dataset.name: (str) OOD 数据集名称
            - ood_dataset.root: (str) OOD 数据集路径
            - top_5: (bool) 是否计算 top-5 精度

    Returns:
        dict: {
            "accelerator": Accelerator,
            "model": nn.Module,
            "criterion": nn.Module,
            "train_meters": MetersGroup,
            "test_meters": MetersGroup,
            "top_k": tuple,
            "dataloaders": {
                "id_train": DataLoader,
                "id_test": DataLoader,
                "id_val": DataLoader,
                "ood": DataLoader,
            }
        }
    """
    # -----------------------
    # 初始化加速器
    # -----------------------
    accelerator = Accelerator(
        mixed_precision=config.get("training", {}).get("mixed_precision", "no")
    )
    config["accelerator"] = accelerator

    # -----------------------
    # 模型加载
    # -----------------------
    model_save_dir = os.path.join(config.get("save_dir", "checkpoints"), config["exp_name"])
    load_path = os.path.join(model_save_dir, config["test_model"], "model.pt")

    state_dict = load_clean_model_state(load_path)
    model = get_model(config["model"])
    model.load_state_dict(state_dict)

    # -----------------------
    # 数据加载器
    # -----------------------
    id_train_loader, id_test_loader, id_val_loader = build_id_dataloaders(config, accelerator)
    ood_name = config["ood_dataset"]["name"]
    print(f"[Test] Loading OOD Dataloader for {ood_name}...")
    ood_loader = build_ood_dataloaders(config, accelerator)

    # 加速器封装
    model, ood_loader, id_train_loader, id_test_loader, id_val_loader = accelerator.prepare(
        model, ood_loader, id_train_loader, id_test_loader, id_val_loader
    )

    # -----------------------
    # 损失函数
    # -----------------------
    criterion = nn.CrossEntropyLoss()

    # -----------------------
    # 指标
    # -----------------------
    if config.get("top_5", False):
        train_meters = MetersGroup(["Acc@1", "Acc@5", "Loss", "Time"])
        test_meters = MetersGroup(["Acc@1", "Acc@5", "Loss", "Time"])
        top_k = (1, 5)
    else:
        train_meters = MetersGroup(["Acc@1", "Loss", "Time"])
        test_meters = MetersGroup(["Acc@1", "Loss", "Time"])
        top_k = (1,)

    # -----------------------
    # 设备信息与特征分析
    # -----------------------
    print(accelerator.device)

    return accelerator, model, criterion, train_meters, test_meters, top_k, id_train_loader, id_test_loader, id_val_loader, ood_loader


def exp_base(config,forward_names=["base", "react", "bats", "laps"],methods=["msp", "energy", "odin"]):
    """
    基础实验流程:
    - 针对不同的 forward 方法 (base, react, bats, laps)
      和不同的 OOD 打分方法 (msp, energy, odin)，
      计算 In-Distribution (ID) 与 Out-of-Distribution (OOD) 的分数。
    - 结果保存为 .pt 文件，同时返回 scores_in_all / scores_out_all 以便后续可视化/分析。

    参数:
        config: dict, 配置文件，需包含:
            - save_dir: 模型保存目录
            - exp_name: 实验名称
            - test_model: 模型子目录名
            - ood_dataset: {"name": str}，OOD 数据集名称
            - 其他: 传递给 get_forward/get_all_scores 所需的配置

    返回:
        scores_in_all  : dict[forward_name][method] -> list (ID 数据分数)
        scores_out_all : dict[forward_name][method] -> list (OOD 数据分数)
    """
    accelerator, model, criterion, train_meters, test_meters, top_k, id_train_loader, id_test_loader, id_val_loader, ood_loader = initialize_experiment(
        config)

    # 模型保存目录: save_dir/exp_name/test_model/ood_name
    model_save_dir = os.path.join(
        config.get("save_dir", "checkpoints"),
        config["exp_name"],
        config["test_model"],
        config["ood_dataset"]["name"]
    )
    os.makedirs(model_save_dir, exist_ok=True)

    # 用于保存 ID/OOD 分数
    scores_in_all, scores_out_all = {}, {}

    # 确定迭代次数（取 ID 测试集与 OOD 集长度的最小值，保持公平）
    n_iter = min(len(id_test_loader), len(ood_loader))
    print('n_iter:', n_iter)

    # 遍历不同的 forward 策略
    for forward_name in forward_names:
    # for forward_name in ["forward_last_features"]
        scores_in_all[forward_name] = {}
        scores_out_all[forward_name] = {}

        # 获取前向计算函数
        forward_func = get_forward(name=forward_name, config=config)

        # 遍历不同的 OOD 方法
        for method in methods:
        # for method in ["energy"]:
            # ---------------- OOD 分数 ----------------
            scores_out_tensor = get_all_scores(
                model, ood_loader, forward_func, method, config, n_iter, accelerator
            )
            scores_out_all[forward_name][method] = scores_out_tensor

            # 保存 OOD 分数到文件
            out_path = os.path.join(model_save_dir, f"{forward_name}_{method}_out.pt")
            torch.save(torch.tensor(scores_out_tensor), out_path)

            # ---------------- ID 分数 ----------------
            scores_in_tensor = get_all_scores(
                model, id_test_loader, forward_func, method, config, n_iter, accelerator
            )
            scores_in_all[forward_name][method] = scores_in_tensor

            # 保存 ID 分数到文件
            in_path = os.path.join(model_save_dir, f"{forward_name}_{method}_in.pt")
            torch.save(torch.tensor(scores_in_tensor), in_path)

    return scores_in_all, scores_out_all
