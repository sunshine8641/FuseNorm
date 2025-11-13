import os
import torch
import numpy as np
from accelerate import Accelerator
from data_utils.build_dataset import build_id_dataloaders, build_jigsaw_dataloaders
from models import get_model
from safetensors.torch import load_file
from collections import OrderedDict
from metrics import *
from train.train_utils import load_clean_model_state
from torch.nn import functional as F


def compute_threshold_react(config):
    """
    Compute REACT threshold based on percentile of activation values 
    at a specified layer (e.g., ResNet avgpool) on validation data.
    """
    accelerator = Accelerator(mixed_precision=config.get("training", {}).get("mixed_precision", "no"))

    # Load model
    model_save_dir = os.path.join(config.get("save_dir", "checkpoints"), config["exp_name"])
    load_path = os.path.join(model_save_dir, config["test_model"], "model.pt")
    print(f"Loading model from {load_path}")
    state_dict = load_clean_model_state(load_path)

    model = get_model(config["model"])
    model.load_state_dict(state_dict)

    # Build validation dataloader
    id_train_loader, id_test_loader, id_val_loader = build_id_dataloaders(config, accelerator)
    model, id_val_loader = accelerator.prepare(model, id_val_loader)

    # Hook container
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # Register hook for the avgpool layer
    hooker_handles = []
    layer_remark = 'avgpool'
    hooker_handles.append(model.avgpool.register_forward_hook(get_activation(layer_remark)))

    activation_log = []
    for batch_idx, batch in enumerate(id_val_loader):
        x, y = batch
        curr_batch_size = x.shape[0]
        with torch.no_grad():
            outputs = model(x)
        avgpool_feats = activation[layer_remark]  # [B, C, 1, 1]
        dim = avgpool_feats.shape[1]
        activation_log.append(avgpool_feats.data.cpu().numpy().reshape(curr_batch_size, dim, -1).mean(2))
        activation.clear()

    for handle in hooker_handles:
        handle.remove()

    activation_log = np.concatenate(activation_log, axis=0)
    react_threshold = np.percentile(activation_log.flatten(), 90)
    react_threshold = torch.tensor(react_threshold)

    print(f"\nREACT Threshold at 90th percentile: {react_threshold}")
    output_dir = os.path.join(model_save_dir, config["test_model"])
    torch.save(react_threshold, os.path.join(output_dir, "react_threshold.pt"))


def compute_activation_stats(config):
    """
    Compute per-channel mean and std for a given layer (e.g., avgpool)
    on validation data. Saves results as tensors for later use.
    """
    accelerator = Accelerator(mixed_precision=config.get("training", {}).get("mixed_precision", "no"))

    # Load model
    model_save_dir = os.path.join(config.get("save_dir", "checkpoints"), config["exp_name"])
    load_path = os.path.join(model_save_dir, config["test_model"], "model.pt")
    print(f"Loading model from {load_path}")
    state_dict = load_clean_model_state(load_path)

    model = get_model(config["model"])
    model.load_state_dict(state_dict)

    id_train_loader, id_test_loader, id_val_loader = build_id_dataloaders(config, accelerator)
    model, id_val_loader = accelerator.prepare(model, id_val_loader)

    # Hook container
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    hooker_handles = []
    layer_remark = 'avgpool'
    hooker_handles.append(model.avgpool.register_forward_hook(get_activation(layer_remark)))

    activation_log = []
    for batch_idx, batch in enumerate(id_val_loader):
        x, y = batch
        with torch.no_grad():
            _ = model(x)
        avgpool_feats = activation[layer_remark]  # [B, C, 1, 1]
        feats = avgpool_feats.squeeze(-1).squeeze(-1).cpu().numpy()  # [B, C]
        activation_log.append(feats)
        activation.clear()

    for handle in hooker_handles:
        handle.remove()

    activation_log = np.concatenate(activation_log, axis=0)  # [N, C]
    print(f"Activation log shape: {activation_log.shape}")

    feature_mean = torch.tensor(np.mean(activation_log, axis=0))
    feature_std = torch.tensor(np.std(activation_log, axis=0))

    output_dir = os.path.join(model_save_dir, config["test_model"])
    stats = {
        "feature_mean": feature_mean,
        "feature_std": feature_std
    }
    torch.save(stats, os.path.join(output_dir, "feature_stats.pt"))

    print(f"\nSaved per-channel stats to: {os.path.join(output_dir, 'feature_stats.pt')}")
    print("Per-channel mean:", feature_mean)
    print("Per-channel std:", feature_std)
    print("Overall mean of means:", feature_mean.mean())
    print("Overall mean of stds:", feature_std.mean())


def calculate_layer_norm(config):
    """
    Compute per-block feature norms for original and jigsawed inputs,
    and calculate the norm ratio for each block.
    """
    num_blocks = config["num_blocks"]
    accelerator = Accelerator(mixed_precision=config.get("training", {}).get("mixed_precision", "no"))

    # Load model
    model_save_dir = os.path.join(config.get("save_dir", "checkpoints"), config["exp_name"])
    load_path = os.path.join(model_save_dir, config["test_model"], "model.pt")
    print(f"Loading model from {load_path}")
    state_dict = load_clean_model_state(load_path)

    model = get_model(config["model"])
    model.load_state_dict(state_dict)
    model.eval()

    # Storage for per-block norms
    norm_pred_ori = {i: [] for i in range(num_blocks)}
    norm_pred_jigsaw = {i: [] for i in range(num_blocks)}

    id_train_loader, _, _ = build_id_dataloaders(config, accelerator)
    jigsaw_train_loader, _, _ = build_jigsaw_dataloaders(config, accelerator)
    model, id_train_loader, jigsaw_train_loader = accelerator.prepare(model, id_train_loader, jigsaw_train_loader)

    if accelerator.is_main_process:
        print(type(model).__name__, len(id_train_loader), len(jigsaw_train_loader))

    with torch.no_grad():
        for batch_idx, (data1, data2) in enumerate(zip(id_train_loader, jigsaw_train_loader)):
            x = torch.cat([data1[0], data2[0]], dim=0)
            features = model.forward_features_blockwise(x)

            for i in range(num_blocks):
                norm = torch.norm(F.relu(features[i]), dim=[2, 3]).mean(1)
                norm_ori = norm[:len(data1[0])]
                norm_jigsaw = norm[len(data1[0]):]
                norm_pred_ori[i].append(norm_ori)
                norm_pred_jigsaw[i].append(norm_jigsaw)

            if batch_idx > 200:  # limit to first 200 batches
                break

    # Aggregate all batches
    for i in range(num_blocks):
        norm_pred_ori[i] = torch.cat(norm_pred_ori[i], dim=0)
        norm_pred_jigsaw[i] = torch.cat(norm_pred_jigsaw[i], dim=0)
        norm_pred_ori[i] = accelerator.gather(norm_pred_ori[i])
        norm_pred_jigsaw[i] = accelerator.gather(norm_pred_jigsaw[i])

        if accelerator.is_main_process:
            ratio = (norm_pred_ori[i] / norm_pred_jigsaw[i]).mean()
            print(f'NormRatio-Block{i}: {ratio:.4f}')
