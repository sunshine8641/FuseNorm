from __future__ import print_function  # Compatibility for Python2 and Python3 print function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from scipy.spatial.distance import pdist, cdist, squareform


def sample_estimator(model, num_classes, feature_list, train_loader):
    """
    Estimate the mean vector and precision matrix (inverse covariance)
    for each class, used for Mahalanobis distance computation.

    Returns:
        - sample_class_mean: mean vectors for each class and layer
        - precision: precision matrices for each layer
    """
    import sklearn.covariance  # For empirical covariance computation

    model.eval()  # Set model to evaluation mode
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)

    correct, total = 0, 0
    num_output = len(feature_list)  # Number of feature layers to extract
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)

    # Initialize structure: list_features[layer][class] = list of features
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    for data, target in train_loader:
        total += data.size(0)
        print(total)
        if total > 50000:  # Limit maximum number of samples
            break

        data = Variable(data).cuda()
        output, out_features = model.feature_list(data)  # Extract features from all layers

        # Apply average pooling to each layer's features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)

        # Compute accuracy (optional)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.cuda()).cpu().sum()

        # Save features per class
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

    # Compute mean vectors for each class
    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    # Compute precision matrices (inverse covariance)
    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            centered = list_features[k][i] - sample_class_mean[k][i]
            X = centered if i == 0 else torch.cat((X, centered), 0)

        group_lasso.fit(X.cpu().numpy())  # Fit covariance
        temp_precision = torch.from_numpy(group_lasso.precision_).double().cuda()
        precision.append(temp_precision)

    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision


def get_Mahalanobis_score(inputs, model, num_classes, sample_mean, precision, num_output, magnitude):
    """
    Given test images, compute Mahalanobis distance to class means
    for each layer, outputting OOD scores.
    Input preprocessing is applied using gradient-based perturbation.
    """

    for layer_index in range(num_output):
        data = Variable(inputs, requires_grad=True).cuda()

        # Extract features from the current layer
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        # Compute Gaussian scores using Mahalanobis distance to all class means
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            term_gau = term_gau.view(-1, 1)
            gaussian_score = term_gau if i == 0 else torch.cat((gaussian_score, term_gau), 1)

        # Select class with max score (min distance)
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        # Compute input perturbation direction (sign of gradient)
        gradient = torch.ge(data.grad.data, 0).float() * 2 - 1

        # Apply input preprocessing perturbation
        tempInputs = torch.add(data.data, -magnitude, gradient)

        # Re-extract features and compute Mahalanobis scores after perturbation
        noise_out_features = model.intermediate_forward(Variable(tempInputs), layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)

        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            noise_gaussian_score = term_gau.view(-1, 1) if i == 0 else torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

        # Take max score as OOD score for this layer
        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        noise_gaussian_score = np.asarray(noise_gaussian_score.cpu().numpy(), dtype=np.float32)

        # Concatenate layer scores to final result
        if layer_index == 0:
            Mahalanobis_scores = noise_gaussian_score.reshape((noise_gaussian_score.shape[0], -1))
        else:
            Mahalanobis_scores = np.concatenate(
                (Mahalanobis_scores, noise_gaussian_score.reshape((noise_gaussian_score.shape[0], -1))),
                axis=1
            )

    return Mahalanobis_scores
