from .train_utils import  save_model_and_states,LabelSmoothingCrossEntropy,\
    prepare_training,load_clean_model_state,get_model,average_state_dicts
from .train_base import run_one_epoch_base
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


class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=4.0, label_smoothing=0):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.ce_loss = LabelSmoothingCrossEntropy(label_smoothing)
        print("KD LOSS SETTING:", alpha, temperature, label_smoothing)
    def forward(self, student_logits, targets, teacher_logits):
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits / self.T, dim=1),
            reduction='batchmean'
        ) * (self.T ** 2)

        ce_loss = self.ce_loss(student_logits, targets)
        # print(ce_loss)
        # print(kd_loss)
        # print(self.alpha)
        return self.alpha * kd_loss + (1 - self.alpha) * ce_loss



def run_one_epoch_distill(student_model, teacher_model, optimizer, dataloader,\
                          criterion, accelerator, top_k, meters, mode="Train"):
    student_model.train() if mode == "Train" else student_model.eval()
    teacher_model.eval()
    meters.reset()
    start = time.time()

    with torch.no_grad() if mode != "Train" else contextlib.nullcontext():
        for inputs, targets in dataloader:
            student_outputs = student_model(inputs)
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            loss = criterion(student_outputs, targets, teacher_outputs)

            if mode == "Train":
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            acc = accuracy(student_outputs, targets, topk=top_k)
            acc_dict = {f"Acc@{k}": a.item() for k, a in zip(top_k, acc)}
            acc_dict["Loss"] = loss.item()
            meters.update(acc_dict, n=targets.size(0))

    return time.time() - start


def train_kd(config):
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

    criterion = DistillationLoss(
        alpha=config["distill_alpha"],
        temperature=config["distill_temperature"],
        label_smoothing=config["smoothing"]
    )
    ce_loss = nn.CrossEntropyLoss()

    if config["start_epoch"] > 0:
        resume_path = os.path.join(model_save_dir, config["train_method"] + "_best")
        model.load_state_dict(load_clean_model_state(os.path.join(resume_path, "start_model.pt")))
        optimizer_scheduler = torch.load(os.path.join(resume_path, "optimizer_scheduler.pt"))
        optimizer.load_state_dict(optimizer_scheduler["optimizer_state_dict"])
        scheduler.load_state_dict(optimizer_scheduler["scheduler_state_dict"])

    model, optimizer, id_train_loader, id_test_loader, id_val_loader = accelerator.prepare(
        model, optimizer, id_train_loader, id_test_loader, id_val_loader)

    teacher = get_model(config["teacher"])
    teacher.load_state_dict(load_clean_model_state(os.path.join(model_save_dir, config["teacher_path"])))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher = accelerator.prepare(teacher)


    use_loguru = setup_logging(config)
    best_acc1 = 0.0

    swa_model = deepcopy(accelerator.unwrap_model(model))
    swa_start = config.get("swa_start", int(config["epochs"] * 0.75))
    swa_window = config.get("swa_window", 10)
    swa_queue = deque(maxlen=swa_window)

    swa_scheduler = SWALR(optimizer, swa_lr=config.get("swa_lr", 1e-3), anneal_strategy='linear', anneal_epochs=1) \
        if config.get("use_swa", False) else scheduler
    test_time = run_one_epoch_base(teacher, optimizer, id_test_loader, ce_loss, accelerator, top_k, test_meters,
                                   "Test")
    if accelerator.is_main_process:
        logger.info(format_meters_log(-1, test_meters.meters, test_time, scheduler, prefix="Teacher Accuracy "))

    for epoch in range(config.get("start_epoch", 0), config["epochs"]):
        train_time = run_one_epoch_distill(model, teacher, optimizer, id_train_loader, criterion, accelerator, top_k, train_meters, mode="Train")
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
        test_time = run_one_epoch_base(active_model, optimizer, id_test_loader, ce_loss, accelerator, top_k, test_meters, "Test")
        val_time = run_one_epoch_base(active_model, optimizer, id_val_loader, ce_loss, accelerator, top_k, val_meters, "Val")

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
