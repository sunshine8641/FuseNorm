import os
import yaml
import argparse
import torch
from loguru import logger
from datetime import datetime
import json


def load_config(config_path):
    """
    Load a YAML configuration file and return as a dictionary.

    Args:
        config_path (str): Path to the YAML config file

    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] Config file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(f"[ERROR] YAML parsing failed: {e}")

    return config


def get_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser()

    # Command-line arguments
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="Weight decay")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    args = parser.parse_args()
    return args


def merge_args_with_config(args, config_dict):
    """
    Merge command-line arguments with YAML config.
    Command-line arguments have higher priority.

    Args:
        args (argparse.Namespace): parsed CLI arguments
        config_dict (dict): config loaded from YAML

    Returns:
        dict: merged configuration
    """
    args_dict = vars(args)
    result = config_dict.copy()
    result.update({k: v for k, v in args_dict.items() if v is not None})
    return result


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy for the specified values of k"""
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


class AverageMeter:
    """Compute and store the average and current value of a metric"""
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
    """Group multiple AverageMeter objects for convenient metric tracking"""
    def __init__(self, names):
        self.meters = {name: AverageMeter(name) for name in names}

    def update(self, updates: dict, n=1):
        for name, val in updates.items():
            self.meters[name].update(val, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def __getitem__(self, key):
        return self.meters[key]

    def __str__(self):
        return ' | '.join(str(meter) for meter in self.meters.values())


def format_meters_log(epoch, meters, time_duration, scheduler=None, prefix="Epoch"):
    """
    Format the metric log string for printing.

    Args:
        epoch (int): current epoch number
        meters (MetersGroup or dict): tracked metrics
        time_duration (float): duration of the epoch
        scheduler (torch.optim.lr_scheduler, optional): learning rate scheduler
        prefix (str): prefix for the log

    Returns:
        str: formatted log string
    """
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


def setup_logging(config):
    """
    Initialize logging using loguru. Logs are saved to 'logs/' folder with timestamp.

    Args:
        config (dict): configuration dictionary

    Returns:
        bool: whether loguru is used
    """
    log_cfg = config.get("log", {})
    use_loguru = log_cfg.get("use_loguru", True)
    if use_loguru:
        os.makedirs("logs", exist_ok=True)
        train_method = config.get("train_method", "train")
        log_path = os.path.join("logs", f"{train_method}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
        logger.add(log_path, rotation="10 MB", retention="60 days", level="INFO")
        logger.info("Training Config:\n{}", json.dumps(config, indent=4))
    return use_loguru
