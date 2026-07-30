from __future__ import print_function  # 兼容 Python2 和 Python3 的 print 函数
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from scipy.spatial.distance import pdist, cdist, squareform
def sample_estimator(model, num_classes, feature_list, train_loader):
    """
    估计每个类别的特征均值 (mean vector) 和精度矩阵 (协方差的逆)
    用于 Mahalanobis 距离计算。
    返回:
        - sample_class_mean: 每类每层的特征均值
        - precision: 每层的精度矩阵
    """
    import sklearn.covariance  # 用于计算经验协方差

    model.eval()  # 设置模型为推理模式
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)

    correct, total = 0, 0
    num_output = len(feature_list)  # 要提取的特征层数
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)

    # 初始化结构：list_features[layer][class] = 特征列表
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    for data, target in train_loader:
        total += data.size(0)
        print(total)
        if total > 50000:  # 限制最大样本数
            break

        data = Variable(data).cuda()
        output, out_features = model.feature_list(data)  # 提取各层特征

        # 对每层的特征进行平均池化处理
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)

        # 计算准确率（可选步骤）
        pred = output.data.max(1)[1]
        correct += pred.eq(target.cuda()).cpu().sum()

        # 保存每类特征向量
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = torch.cat(
                        (list_features[out_count][label], out[i].view(1, -1)), 0
                    )
                    out_count += 1
            num_sample_per_class[label] += 1

    # 计算每个类别的均值向量
    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    # 计算精度矩阵（协方差的逆）
    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            centered = list_features[k][i] - sample_class_mean[k][i]
            X = centered if i == 0 else torch.cat((X, centered), 0)

        group_lasso.fit(X.cpu().numpy())  # 拟合协方差
        temp_precision = torch.from_numpy(group_lasso.precision_).double().cuda()
        precision.append(temp_precision)

    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision

def get_Mahalanobis_score(inputs, model, num_classes, sample_mean, precision, num_output, magnitude):
    """
    输入测试图像，计算其在每一层与各类别均值的马氏距离，输出 OOD 分数。
    使用梯度方向扰动进行 input preprocessing。
    """

    for layer_index in range(num_output):
        data = Variable(inputs, requires_grad=True).cuda()

        # 提取当前层特征
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        # 计算与所有类均值的马氏距离高斯得分
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            term_gau = term_gau.view(-1, 1)
            gaussian_score = term_gau if i == 0 else torch.cat((gaussian_score, term_gau), 1)

        # 找最大得分类（最小距离）
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        # 计算输入的扰动方向（符号梯度）
        gradient = torch.ge(data.grad.data, 0).float() * 2 - 1

        # 添加扰动（input preprocessing）
        tempInputs = torch.add(data.data, -magnitude, gradient)

        # 再次提取特征并计算马氏得分（扰动后）
        noise_out_features = model.intermediate_forward(Variable(tempInputs), layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)

        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            noise_gaussian_score = term_gau.view(-1, 1) if i == 0 else torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

        # 选择最大得分作为该层 OOD 得分
        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        noise_gaussian_score = np.asarray(noise_gaussian_score.cpu().numpy(), dtype=np.float32)

        # 将该层得分拼接到最终结果
        if layer_index == 0:
            Mahalanobis_scores = noise_gaussian_score.reshape((noise_gaussian_score.shape[0], -1))
        else:
            Mahalanobis_scores = np.concatenate(
                (Mahalanobis_scores, noise_gaussian_score.reshape((noise_gaussian_score.shape[0], -1))),
                axis=1
            )

    return Mahalanobis_scores
