import os
import yaml
import argparse
import torch
from loguru import logger
from datetime import datetime
import json


def load_config(config_path):
    """
    加载 YAML 配置文件并返回为 dict。

    Args:
        config_path (str): 配置文件路径

    Returns:
        dict: 配置项
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] 配置文件未找到: {config_path}")

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(f"[ERROR] YAML 解析失败: {e}")

    return config


def get_args():
    parser = argparse.ArgumentParser()

    # ✅ 命令行参数定义
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--exp_name", type=str, default=None, help="experiment name")
    args = parser.parse_args()
    return args
def merge_args_with_config(args, config_dict):
    """命令行优先，其次 YAML。返回最终参数 dict"""
    args_dict = vars(args)
    result = config_dict.copy()
    result.update({k: v for k, v in args_dict.items() if v is not None})
    return result


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / batch_size)))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



class MetersGroup:
    def __init__(self, names):
        self.meters = {name: AverageMeter(name) for name in names}

    def update(self, updates: dict, n=1):
        for name, val in updates.items():
            self.meters[name].update(val, n)
    def reset(self):
        for name in self.meters:
            self.meters[name].reset()

    def __str__(self):
        return ' | '.join(str(meter) for meter in self.meters.values())




def format_meters_log(epoch, meters, time_duration, scheduler=None, prefix="Epoch"):
    log_items = [f"{prefix} {epoch}"]
    if "Loss" in meters:
        log_items.append(f"Loss: {meters['Loss'].avg:.4f}")
    for name, meter in meters.items():
        if name.startswith("Acc@"):
            log_items.append(f"{name}: {meter.avg:.2f}%")
    if scheduler is not None:
        lr = scheduler.get_last_lr()[0]
        log_items.append(f"LR: {lr:.4f}")
    log_items.append(f"Time: {time_duration:.2f}")
    return " | ".join(log_items)

# def format_meters_log(epoch, meters, time_duration, scheduler=None, prefix="Epoch"):
#     log_items = [f"{prefix} {epoch}"]
#
#     for name, meter in meters.items():
#         if hasattr(meter, 'avg'):
#             value = meter.avg
#         else:
#             value = meter  # 支持 meter 是标量而不是对象的情况
#
#         # 格式化输出
#         if name.lower().startswith("acc@"):
#             log_items.append(f"{name}: {value:.2f}%")
#         elif name.lower() in ["loss", "lr"]:
#             log_items.append(f"{name}: {value:.4f}")
#         else:
#             log_items.append(f"{name}: {value:.4f}")
#
#     # 加上 scheduler 的 learning rate
#     if scheduler is not None:
#         try:
#             lr = scheduler.get_last_lr()[0]
#             log_items.append(f"LR: {lr:.4f}")
#         except Exception:
#             pass  # 某些 scheduler 不支持 get_last_lr
#
#     log_items.append(f"Time: {time_duration:.2f}s")
#     return " | ".join(log_items)



def setup_logging(config):
    log_cfg = config.get("log", {})
    use_loguru = log_cfg.get("use_loguru", True)
    if use_loguru:
        os.makedirs("logs", exist_ok=True)
        train_method=config["train_method"]
        log_path = os.path.join("logs", f"{train_method}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
        logger.add(log_path, rotation="10 MB", retention="60 days", level="INFO")
        logger.info("Training Config:\n{}", json.dumps(config, indent=4))
    return use_loguru
