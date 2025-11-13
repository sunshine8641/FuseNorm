from .train_utils import  save_model_and_states,LabelSmoothingCrossEntropy,\
    prepare_training,load_clean_model_state,get_model,average_state_dicts
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from utils import format_meters_log, setup_logging, accuracy
from torch.optim.swa_utils import SWALR, update_bn
from collections import deque
from copy import deepcopy
import time
import contextlib
from models import  ParametricBReLU



def train_baseline(config):
    context = prepare_training(config)
    model = context["model"]
    optimizer = context["optimizer"]
    scheduler = context["scheduler"]
    train_meters = context["train_meters"]
    test_meters = context["test_meters"]
    val_meters = context["val_meters"]
    accelerator = context["accelerator"]
    model_save_dir = context["model_save_dir"]
    id_train_loader = context["id_train_loader"]
    id_test_loader = context["id_test_loader"]
    id_val_loader = context["id_val_loader"]
    top_k = context["top_k"]

    criterion = nn.CrossEntropyLoss()
    if config["label_smooth"]:
        criterion = LabelSmoothingCrossEntropy(config["smoothing"])
    if config["start_epoch"] > 0:
        resume_path = os.path.join(model_save_dir, config["train_method"] + "_best")
        model.load_state_dict(load_clean_model_state(os.path.join(resume_path, "start_model.pt")))
        optimizer_scheduler = torch.load(os.path.join(resume_path, "optimizer_scheduler.pt"))
        optimizer.load_state_dict(optimizer_scheduler["optimizer_state_dict"])
        scheduler.load_state_dict(optimizer_scheduler["scheduler_state_dict"])

    model, optimizer, id_train_loader, id_test_loader, id_val_loader = accelerator.prepare(
        model, optimizer, id_train_loader, id_test_loader, id_val_loader)


    use_loguru = setup_logging(config)
    best_acc1 = 0.0
    if config.get("use_swa", False):
        swa_model = deepcopy(accelerator.unwrap_model(model))
        swa_start = config.get("swa_start", int(config["epochs"] * 0.75))
        swa_window = config.get("swa_window", 10)
        swa_queue = deque(maxlen=swa_window)

    swa_scheduler = SWALR(optimizer, swa_lr=config.get("swa_lr", 1e-3), anneal_strategy='linear', anneal_epochs=1) \
        if config.get("use_swa", False) else scheduler

    for epoch in range(config.get("start_epoch", 0), config["epochs"]):
        train_time = run_one_epoch_base(model,optimizer, id_train_loader, criterion, accelerator, top_k, train_meters, mode="Train")
        if accelerator.is_main_process:
            logger.info(format_meters_log(epoch, train_meters.meters, train_time, scheduler, prefix="Train at Epoch"))

        if config.get("use_swa", False) and epoch >= swa_start:
            swa_queue.append(deepcopy(accelerator.unwrap_model(model).state_dict()))
            swa_model.load_state_dict(average_state_dicts(swa_queue))
            update_bn(id_train_loader, swa_model)
            swa_scheduler.step()
        else:
            scheduler.step()

        if config.get("use_swa", False) and epoch == swa_start - 1 and accelerator.is_main_process:
            swa_save_path = os.path.join(model_save_dir, config["train_method"] + "_best")
            save_model_and_states(swa_save_path, model, optimizer, scheduler, accelerator, "start_model.pt", save_state=True)

        # Evaluation
        active_model = swa_model if config.get("use_swa", False) and epoch >= swa_start else model
        test_time = run_one_epoch_base(active_model, optimizer, id_test_loader, criterion, accelerator, top_k, test_meters, "Test")
        val_time = run_one_epoch_base(active_model, optimizer, id_val_loader, criterion, accelerator, top_k, val_meters, "Val")

        if accelerator.is_main_process:
            acc1 = val_meters.meters["Acc@1"].avg
            logger.info(format_meters_log(epoch, test_meters.meters, test_time, scheduler, prefix="Test at Epoch"))
            logger.info(format_meters_log(epoch, val_meters.meters, val_time, scheduler, prefix="Val at Epoch"))

            if acc1 > best_acc1:
                best_acc1 = acc1
                acc_test = test_meters.meters["Acc@1"].avg
                best_path = os.path.join(model_save_dir, config["train_method"] + "_best")
                save_model_and_states(best_path, active_model, optimizer, scheduler, accelerator, "model.pt", save_state=False)
                logger.info(f"\n✔️ Best model updated at epoch {epoch}, val Acc@1 = {acc1:.2f}%, test Acc@1 = {acc_test:.2f}%")

    if accelerator.is_main_process:
        last_path = os.path.join(model_save_dir, config["train_method"] + "_last")
        save_model_and_states(last_path, swa_model if config.get("use_swa", False) else model, optimizer, scheduler, accelerator, "model.pt", save_state=False)

def train_base(config):
    config["use_swa"]=False
    train_baseline(config)
