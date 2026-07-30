from __future__ import print_function
import argparse
import os
import sys

from scipy import misc
import numpy as np
from collections import defaultdict


def get_curve(known, novel, threshold=None):
    """
    known 是 positive 类（In-distribution，ID），
    novel 是 negative 类（Out-of-distribution，OOD）。
    """
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    # 对 known 和 novel 分数排序（从小到大）
    known.sort()
    novel.sort()

    # 计算得分的最大最小值（用于后续分析）
    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known), np.min(novel)])

    # 拼接所有数据进行整体排序，用于遍历阈值
    all = np.concatenate((known, novel))
    all.sort()

    # 分别统计 known（ID） 和 novel（OOD） 的样本数
    num_k = known.shape[0]
    num_n = novel.shape[0]

    # 选择用于计算 TPR=95% 时 FPR 的阈值（约为 known 分布前5%的分数）
    if  threshold  is None:
        threshold = known[round(0.05 * num_k)]

    # 初始化 tp（True Positive）和 fp（False Positive）曲线数组
    tp = -np.ones([num_k + num_n + 1], dtype=int)
    fp = -np.ones([num_k + num_n + 1], dtype=int)
    tp[0], fp[0] = num_k, num_n  # 初始时，全部预测为正类

    k, n = 0, 0  # k 代表已遍历的 known 数，n 代表已遍历的 novel 数
    for l in range(num_k + num_n):
        if k == num_k:
            # 如果已遍历完所有 known，只剩 novel
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l] - 1, -1, -1)
            break
        elif n == num_n:
            # 如果已遍历完所有 novel，只剩 known
            tp[l+1:] = np.arange(tp[l] - 1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            # 将 novel 和 known 的分数升序比较
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]      # novel 为负类，tp 不变
                fp[l+1] = fp[l] - 1  # novel 被错误当成正类，fp 减少
            else:
                k += 1
                tp[l+1] = tp[l] - 1  # known 被正确识别为正类，tp 减少
                fp[l+1] = fp[l]      # fp 不变

    # 去除 score 相等时重复值带来的跳跃（后处理）
    j = num_k + num_n - 1
    for l in range(num_k + num_n - 1):
        if all[j] == all[j - 1]:
            tp[j] = tp[j + 1]
            fp[j] = fp[j + 1]
        j -= 1

    # 计算在 TPR 为 95% 时的 FPR，等价于找出 threshold 上面被判为正的 novel 比例
    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95



def cal_metric(known, novel, threshold=None):
    # 输入为 in-distribution 数据的分数 known 和 OOD 数据的分数 novel
    # method 可选，用于计算时使用不同排序方法（暂未使用）

    # 获取 TPR 曲线、FPR 曲线、以及 95% TPR 时的 FPR 值
    tp, fp, fpr_at_tpr95 = get_curve(known, novel, threshold)

    results = dict()
    mtypes = ['FPR', 'AUROC', 'DTERR', 'AUIN', 'AUOUT']  # 5种常用指标

    # ----------- FPR@95TPR (False Positive Rate at 95% True Positive Rate) -----------
    mtype = 'FPR'
    results[mtype] = fpr_at_tpr95  # 这是 get_curve 函数直接返回的

    # ----------- AUROC (Area Under ROC Curve) -----------
    mtype = 'AUROC'
    # 构造 TPR 和 FPR 曲线，前后加上起始点和终止点（保证积分完整性）
    tpr = np.concatenate([[1.], tp / tp[0], [0.]])
    fpr = np.concatenate([[1.], fp / fp[0], [0.]])
    # 使用 trapezoidal rule 进行积分，注意 1 - FPR 是因为方向要正
    results[mtype] = -np.trapz(1. - fpr, tpr)

    # ----------- DTERR (Detection Error: 最小检测错误率) -----------
    mtype = 'DTERR'
    # (FN + FP) / 总样本数，其中 tp[0] 是正类总数，fp[0] 是负类总数
    # 该式子是在不同阈值下找一个最小的错误率
    results[mtype] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # ----------- AUIN (Area Under Inverse Precision-Recall Curve for ID) -----------
    mtype = 'AUIN'
    denom = tp + fp  # precision 的分母
    denom[denom == 0.] = -1.  # 避免除以零
    # 只保留分母 > 0 的位置用于积分（否则无定义）
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    # precision 曲线（插入 0 和 0.5 起始终止点）
    pin = np.concatenate([[.5], tp / denom, [0.]])
    # 对 precision vs recall 曲线进行积分（面积越大越好）
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # ----------- AUOUT (Area Under Inverse Precision-Recall Curve for OOD) -----------
    mtype = 'AUOUT'
    # 反向 precision 分母：FP + TN
    denom = tp[0] - tp + fp[0] - fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    # OOD precision 曲线（查全率维度）
    pout = np.concatenate([[0.], (fp[0] - fp) / denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1. - fpr[pout_ind])

    return results

def compute_all_metrics(scores_in_all, scores_out_all):
    all_results = defaultdict(dict)

    for forward_name in scores_in_all:
        for score in scores_in_all[forward_name]:
            # 确保 out 集合也有对应的 score
            if score in scores_out_all.get(forward_name, {}):
                scores_in = scores_in_all[forward_name][score]
                scores_out = scores_out_all[forward_name][score]

                # 计算指标
                result = cal_metric(scores_in, scores_out)

                # 保存结果
                all_results[forward_name][score] = result
            else:
                print(f"[Warning] Missing out scores for: {forward_name}, {score}")
    return all_results