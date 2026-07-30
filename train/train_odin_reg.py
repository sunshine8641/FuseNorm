# === 新增 / 修改的 import ===
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import time

# ======================
# 1) ODIN 回归损失（MSE）
# ======================
class ODINRegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, student_scores, target_scores):
        # student_scores: tensor (B,) or (B,1)
        # target_scores: tensor (B,)
        if student_scores.dim() == 2 and student_scores.size(1) == 1:
            student_scores = student_scores.squeeze(1)
        return self.mse(student_scores, target_scores)


# ====================================================
# 2) 计算一个 batch 的 ODIN score（基于 teacher）函数
#    —— 注意：这是逐 batch 在线计算，比较慢
# ====================================================
def compute_odin_score_batch(inputs, teacher_model, forward_func, config, device):
    """
    inputs: tensor [B,C,H,W], on CPU or device
    teacher_model: teacher (eval, params frozen)
    forward_func: returns (features, logits) or logits depending on your impl
    config: must contain 'odin_temperature' and 'odin_magnitude'
    returns: torch.tensor shape (B,) on device (same device as inputs)
    """
    T = config.get('odin_temperature', 1.0)
    eps = config.get('odin_magnitude', 0.0)

    # Ensure on device and require grad
    inputs = inputs.to(device)
    inputs = inputs.clone().detach().requires_grad_(True)

    # forward once
    with torch.enable_grad():
        features, logits = forward_func(inputs, teacher_model, config)
        # logits shape [B, C]
        # find pseudo-label = argmax
        _, pred = torch.max(logits.detach(), dim=1)  # [B]

        # temperature scaling
        logits_T = logits / T

        # compute CE loss with pseudo-labels to backprop to inputs
        loss = F.cross_entropy(logits_T, pred.to(device))
        # compute gradients wrt inputs
        grads = torch.autograd.grad(loss, inputs, create_graph=False)[0]  # [B,C,H,W]

        # gradient sign as in ODIN (original uses sign of grad)
        # you used earlier: gradient = torch.ge(inputs.grad.data, 0); gradient = (gradient.float() - 0.5) * 2
        # Here we'll use sign:
        grad_sign = torch.sign(grads)  # values in {-1,0,1}

        # create perturbed inputs (note: follow ODIN original: subtract epsilon * sign)
        perturbed = inputs - eps * grad_sign
        perturbed = perturbed.detach()

    # forward perturbed (no grad)
    with torch.no_grad():
        _, logits_pert = forward_func(perturbed, teacher_model, config)
        logits_pert_T = logits_pert / T
        probs = F.softmax(logits_pert_T, dim=1)
        # ODIN score commonly uses max softmax probability
        max_probs, _ = torch.max(probs, dim=1)  # [B]
    return max_probs.detach()  # on device


# ====================================================
# 3) 训练/评估一个 epoch（回归 ODIN score）
# ====================================================
def run_one_epoch_odin_reg(student_model, teacher_model, optimizer, dataloader,
                           criterion, accelerator, meters, mode="Train",
                           forward_func=None, config=None):
    """
    student_model: returns a scalar per sample (B,) or (B,1)
    teacher_model: used to compute ODIN score per batch
    forward_func: function(inputs, model, config) -> (features, logits) ; must match teacher API
    criterion: ODINRegressionLoss (MSE)
    """
    is_train = (mode == "Train")
    if is_train:
        student_model.train()
    else:
        student_model.eval()
    teacher_model.eval()
    meters.reset()
    start = time.time()

    device = accelerator.device

    # we may want to avoid no_grad when computing teacher odin (we need grads)
    for inputs, _ in dataloader:
        inputs = inputs.to(device)

        # 1) compute teacher ODIN score for this batch (target)
        # WARNING: this computes gradients through teacher wrt inputs; expensive.
        target_scores = compute_odin_score_batch(inputs, teacher_model, forward_func, config, device)
        # target_scores shape [B], on device

        # 2) student forward -> predicted scores
        # assume student_model returns (B,) or (B,1)
        student_out = student_model(inputs)
        # ensure shape
        if student_out.dim() == 2 and student_out.size(1) == 1:
            student_scores = student_out.squeeze(1)
        else:
            student_scores = student_out.view(inputs.size(0))

        loss = criterion(student_scores, target_scores)

        if is_train:
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        # logging: we can compute a simple RMSE or correlation as metric
        with torch.no_grad():
            mse = F.mse_loss(student_scores, target_scores).item()
            # optionally compute Pearson corr
            ts = target_scores.detach().cpu().numpy()
            ps = student_scores.detach().cpu().numpy()
            try:
                corr = np.corrcoef(ts, ps)[0,1]
            except Exception:
                corr = 0.0

            meters.update({"Loss": loss.item(), "MSE": mse, "Corr": float(corr)}, n=inputs.size(0))

    return time.time() - start


# ====================================================
# 4) 修改 train_kd 主流程为训练 ODIN 回归（示例）
#    你可以把此函数替换原 train_kd 或并行放置
# ====================================================
def train_odin_reg(config):
    context = prepare_training(config)
    student = context["model"]
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

    # Loss = MSE to ODIN score
    criterion = ODINRegressionLoss()

    # prepare models & data with accelerator
    student, optimizer, id_train_loader, id_test_loader, id_val_loader = accelerator.prepare(
        student, optimizer, id_train_loader, id_test_loader, id_val_loader)

    teacher = get_model(config["teacher"])
    teacher.load_state_dict(load_clean_model_state(os.path.join(model_save_dir, config["teacher_path"])))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher = accelerator.prepare(teacher)

    use_loguru = setup_logging(config)
    best_metric = float("inf")  # use mse lower better

    for epoch in range(config.get("start_epoch", 0), config["epochs"]):
        train_time = run_one_epoch_odin_reg(student, teacher, optimizer, id_train_loader,
                                           criterion, accelerator, train_meters,
                                           mode="Train", forward_func=forward_func, config=config)
        if accelerator.is_main_process:
            logger.info(format_meters_log(epoch, train_meters.meters, train_time, scheduler, prefix="Train at Epoch"))

        scheduler.step()

        # eval on val
        val_time = run_one_epoch_odin_reg(student, teacher, optimizer, id_val_loader,
                                         criterion, accelerator, val_meters,
                                         mode="Eval", forward_func=forward_func, config=config)
        if accelerator.is_main_process:
            logger.info(format_meters_log(epoch, val_meters.meters, val_time, scheduler, prefix="Val at Epoch"))
            cur_mse = val_meters.meters["MSE"].avg
            if cur_mse < best_metric:
                best_metric = cur_mse
                save_path = os.path.join(model_save_dir, config["train_method"] + "_best")
                save_model_and_states(save_path, student, optimizer, scheduler, accelerator, "model.pt", save_state=False)
                logger.info(f"Saved best model at epoch {epoch}, val MSE {cur_mse:.6f}")

    # final save
    if accelerator.is_main_process:
        last_path = os.path.join(model_save_dir, config["train_method"] + "_last")
        save_model_and_states(last_path, student, optimizer, scheduler, accelerator, "model.pt", save_state=False)
