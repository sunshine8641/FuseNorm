import os
import torch
import torch.nn as nn
from accelerate import Accelerator
from data_utils.build_dataset import build_id_dataloaders, build_ood_dataloaders, build_jigsaw_dataloaders
from methods import forward_base, calculate_layer_norm
from models import get_model, ParametricBReLU, ParametricBoundedLeakyReLU, BoundedPReLU
from utils import get_args, merge_args_with_config, load_config, accuracy, MetersGroup, format_meters_log
from methods import get_forward, get_score, compute_activation_stats, get_all_scores, \
                    compute_threshold_react, calculate_layer_norm, get_all_scores
from metrics import *
from train.train_utils import load_clean_model_state


def initialize_experiment(config):
    """
    Initialize the experiment environment, including:
    - HuggingFace Accelerate accelerator
    - Model loading
    - ID / OOD dataloaders
    - Loss function and metrics meters

    Args:
        config (dict): Configuration dictionary, must contain:
            - training.mixed_precision: (str) mixed precision mode ("no" / "fp16" / "bf16")
            - save_dir: (str) directory to save models
            - exp_name: (str) experiment name
            - test_model: (str) test model subfolder name
            - model: (dict) model configuration
            - ood_dataset.name: (str) OOD dataset name
            - ood_dataset.root: (str) OOD dataset path
            - top_5: (bool) whether to compute top-5 accuracy

    Returns:
        tuple: (
            accelerator: Accelerator,
            model: nn.Module,
            criterion: nn.Module,
            train_meters: MetersGroup,
            test_meters: MetersGroup,
            top_k: tuple,
            id_train_loader: DataLoader,
            id_test_loader: DataLoader,
            id_val_loader: DataLoader,
            ood_loader: DataLoader
        )
    """
    # -----------------------
    # Initialize accelerator
    # -----------------------
    accelerator = Accelerator(
        mixed_precision=config.get("mixed_precision", "fp16")
    )
    config["accelerator"] = accelerator

    # -----------------------
    # Load model
    # -----------------------
    model_save_dir = os.path.join(config.get("save_dir", "checkpoints"), config["exp_name"])
    load_path = os.path.join(model_save_dir, config["test_model"], "model.pt")

    state_dict = load_clean_model_state(load_path)
    model = get_model(config["model"])
    model.load_state_dict(state_dict)

    # -----------------------
    # Build dataloaders
    # -----------------------
    id_train_loader, id_test_loader, id_val_loader = build_id_dataloaders(config, accelerator)
    ood_name = config["ood_dataset"]["name"]
    print(f"[Test] Loading OOD Dataloader for {ood_name}...")
    ood_loader = build_ood_dataloaders(config, accelerator)

    # Wrap with accelerator
    model, ood_loader, id_train_loader, id_test_loader, id_val_loader = accelerator.prepare(
        model, ood_loader, id_train_loader, id_test_loader, id_val_loader
    )

    # -----------------------
    # Loss function
    # -----------------------
    criterion = nn.CrossEntropyLoss()

    # -----------------------
    # Metrics
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
    # Device info & feature analysis
    # -----------------------
    print(accelerator.device)

    return accelerator, model, criterion, train_meters, test_meters, top_k, id_train_loader, id_test_loader, id_val_loader, ood_loader


def exp_base(config, forward_names=["base", "react", "bats", "laps"], methods=["msp", "energy", "odin"]):
    """
    Base experiment pipeline:
    - For different forward methods (base, react, bats, laps)
      and different OOD scoring methods (msp, energy, odin),
      compute scores for In-Distribution (ID) and Out-of-Distribution (OOD) data.
    - Results are saved as .pt files, and scores_in_all / scores_out_all
      are returned for further visualization/analysis.

    Args:
        config: dict, configuration must include:
            - save_dir: directory to save models
            - exp_name: experiment name
            - test_model: model subfolder name
            - ood_dataset: {"name": str}, OOD dataset name
            - others: passed to get_forward/get_all_scores

    Returns:
        scores_in_all  : dict[forward_name][method] -> list (ID scores)
        scores_out_all : dict[forward_name][method] -> list (OOD scores)
    """
    accelerator, model, criterion, train_meters, test_meters, top_k, id_train_loader, id_test_loader, id_val_loader, ood_loader = initialize_experiment(config)

    # Model save directory: save_dir/exp_name/test_model/ood_name
    model_save_dir = os.path.join(
        config.get("save_dir", "checkpoints"),
        config["exp_name"],
        config["test_model"],
        config["ood_dataset"]["name"]
    )
    os.makedirs(model_save_dir, exist_ok=True)

    # Dictionaries to store ID / OOD scores
    scores_in_all, scores_out_all = {}, {}

    # Determine number of iterations (min of ID test loader and OOD loader lengths for fairness)
    n_iter = min(len(id_test_loader), len(ood_loader))
    print('n_iter:', n_iter)

    # Loop over different forward strategies
    for forward_name in forward_names:
        scores_in_all[forward_name] = {}
        scores_out_all[forward_name] = {}

        # Get forward function
        forward_func = get_forward(name=forward_name, config=config)

        # Loop over different OOD methods
        for method in methods:
            # ---------------- OOD scores ----------------
            scores_out_tensor = get_all_scores(
                model, ood_loader, forward_func, method, config, n_iter, accelerator
            )
            scores_out_all[forward_name][method] = scores_out_tensor

            # Save OOD scores to file
            out_path = os.path.join(model_save_dir, f"{forward_name}_{method}_out.pt")
            torch.save(torch.tensor(scores_out_tensor), out_path)

            # ---------------- ID scores ----------------
            scores_in_tensor = get_all_scores(
                model, id_test_loader, forward_func, method, config, n_iter, accelerator
            )
            scores_in_all[forward_name][method] = scores_in_tensor

            # Save ID scores to file
            in_path = os.path.join(model_save_dir, f"{forward_name}_{method}_in.pt")
            torch.save(torch.tensor(scores_in_tensor), in_path)

    return scores_in_all, scores_out_all
