import os
import torch.optim as optim
from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import torch
from data_utils.build_dataset import build_id_dataloaders
from models import get_model
from utils import  MetersGroup
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
def average_state_dicts(queue):
    avg_state = {}
    for k in queue[0]:
        avg_state[k] = sum(state[k] for state in queue) / len(queue)
    return avg_state


def load_clean_model_state(path,device=torch.device('cpu')):
    raw = torch.load(path,map_location=device)
    clean = OrderedDict((k.replace("_orig_mod.", ""), v) for k, v in raw.items())
    return clean

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        """
        参数:
            smoothing: 平滑因子, 一般取值范围为 0.0 ~ 0.2
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing

    def forward(self, logits, target):
        """
        logits: 模型输出, shape=(batch_size, num_classes)
        target: 真实标签, shape=(batch_size,)
        """
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)

        # 创建 one-hot 标签
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        # 计算平滑后的交叉熵
        loss = -torch.sum(true_dist * log_probs, dim=1).mean()
        return loss

def get_scheduler(optimizer, config):
    scheduler_type = config.get("scheduler", "cosine").lower()

    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.get("epochs", 200),
            eta_min=config.get("min_lr", 1e-6)
        )
    elif scheduler_type == "step":
        scheduler = StepLR(
            optimizer,
            step_size=config.get("step_size", 30),
            gamma=config.get("step_gamma", 0.1)
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return scheduler

def save_model_and_states(path, model, optimizer, scheduler,accelerator,model_name="model.pt",save_state=False):
    os.makedirs(path, exist_ok=True)
    torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(path, model_name))
    if save_state:
        torch.save({
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        }, os.path.join(path, "optimizer_scheduler.pt"))


def prepare_training(config):

    # print(config)

    # 创建模型保存路径
    model_save_dir = os.path.join(config.get("save_dir", "checkpoints"), config["exp_name"])
    os.makedirs(model_save_dir, exist_ok=True)

    # 初始化指标
    if config.get("top_5", False):
        train_meters = MetersGroup(["Acc@1", "Acc@5", "Loss", "Time"])
        test_meters = MetersGroup(["Acc@1", "Acc@5", "Loss", "Time"])
        val_meters = MetersGroup(["Acc@1", "Acc@5", "Loss", "Time"])

        top_k = (1, 5)
    else:
        train_meters = MetersGroup(["Acc@1", "Loss", "Time"])
        test_meters = MetersGroup(["Acc@1", "Loss", "Time"])
        val_meters = MetersGroup(["Acc@1", "Loss", "Time"])
        top_k = (1,)

    # 初始化加速器
    accelerator = Accelerator(mixed_precision=config.get("training", {}).get("mixed_precision", "fp16"),
                              gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1))
    if accelerator.is_main_process:
        print(f"设备: {accelerator.device}, 进程数: {accelerator.num_processes}, 进程ID: {accelerator.process_index}, "
              f"混合精度: {accelerator.mixed_precision}")
        accelerator.print(f"设备状态:\n{accelerator.state}")

    # 构建数据加载器
    id_train_loader, id_test_loader, id_val_loader = build_id_dataloaders(config, accelerator)

    # 模型与优化器
    model = get_model(config["model"])
    # model.gradient_checkpointing_enable()  # 开启梯度检查点

    if config.get("optimizer", "adam").lower() == "adam":
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=config["lr"],
                              momentum=config.get("momentum", 0.9),
                              weight_decay=config["weight_decay"])

    scheduler = get_scheduler(optimizer, config)

    return {
        "config": config,
        "model_save_dir": model_save_dir,
        "accelerator": accelerator,
        "train_meters": train_meters,
        "test_meters": test_meters,
        "val_meters": val_meters,
        "top_k": top_k,
        "id_train_loader": id_train_loader,
        "id_test_loader": id_test_loader,
        "id_val_loader": id_val_loader,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler
    }
