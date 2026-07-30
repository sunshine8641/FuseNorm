import matplotlib.pyplot as plt
from metrics import  *
from scipy.stats import norm
import torch
from torchvision import transforms
import torchvision.utils as T


def plot_histograms(scores_in_all, scores_out_all, bins=50, title="Histograms of OOD Scores", save_path=None):
    """
    绘制 k 列子图的直方图，每列对应一组分布，并支持保存图片
    :param scores_in_all: shape (k, N)，ID数据的得分
    :param scores_out_all: shape (k, N)，OOD数据的得分
    :param bins: 直方图分箱数
    :param title: 总标题
    :param save_path: 图片保存路径 (例如 'results/ood_hist.png')，若为 None 则仅显示
    """
    k = len(scores_in_all)
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))  # 1行k列

    if k == 1:
        axes = [axes]  # 保证可迭代

    for i in range(k):
        results = cal_metric(scores_in_all[i], scores_out_all[i])
        # 构造指标字符串（只显示 AUROC 和 FPR）
        metrics_str = '\n'.join([f'{key}: {val:.3f}' for key, val in results.items() if key in ['AUROC', 'FPR']])

        # 在右上角显示指标
        axes[i].text(
            0.05, 0.95, metrics_str,
            transform=axes[i].transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

        # 直方图
        axes[i].hist(scores_in_all[i], bins=bins, alpha=0.6, label="ID", color="blue", density=True)
        axes[i].hist(scores_out_all[i], bins=bins, alpha=0.6, label="OOD", color="red", density=True)
        # axes[i].set_title(f"Group {i+1}")
        axes[i].set_xlabel("Score")
        if i == 0:
            axes[i].set_ylabel("Density")
        axes[i].legend(loc='upper right')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    # 保存或显示
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()


def plot_histograms_one(scores_in_all,   means, stds, bins=50, title="Histograms of OOD Scores"):
    """
    绘制 k 列子图的直方图 + 高斯分布曲线
    :param scores_in_all: shape (k, N)，ID数据的得分
    :param bins: 直方图分箱数
    :param title: 总标题
    """

    k = len(scores_in_all)
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))  # 1行k列

    if k == 1:
        axes = [axes]  # 保证可迭代

    for i in range(k):
        # 直方图
        axes[i].hist(scores_in_all[i], bins=bins, alpha=0.6,
                     label="ID Histogram", color="blue", density=True)

        # 生成x范围
        xmin, xmax =np.min(scores_in_all[i]),np.max(scores_in_all[i])
        x = np.linspace(xmin, xmax, 200)

        # 高斯分布曲线
        pdf = norm.pdf(x, means[i], stds[i])
        axes[i].plot(x, pdf, "r-", lw=2, label=f"N({means[i]:.2f}, {stds[i]:.2f}²)")

        # 设置标题和坐标轴
        axes[i].set_title(f"Group {i+1}")
        axes[i].set_xlabel("Score")
        axes[i].set_ylabel("Density")   # ✅ 每个子图都设置 y 轴

        axes[i].legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def unnormalize(t, mean, std):
    # CIFAR10 mean/std
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return torch.clamp(t * std + mean, 0, 1)

def show_images(x, y, mean, std, n=16,save_path="Figs/x.png"):
    x = x[:n]
    y = y[:n]
    x = torch.stack([unnormalize(img.cpu(), mean, std) for img in x])
    grid_img = transforms.ToPILImage()(T.make_grid(x, nrow=int(n ** 0.5)))
    plt.figure(figsize=(4, 4))
    plt.imshow(grid_img)
    plt.axis("off")
    plt.title("Sample Images")
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图像已保存到: {save_path}")
    plt.show()


def plot_score_dicts_subplots(scores_in_all, scores_out_all, title):
    """
    绘制多个 forward_name 与 score 指标的子图对比（ID vs OOD 分布 + AUROC/FPR 指标）

    参数:
        scores_in_all  : dict[forward_name][score] -> list/ndarray (ID 数据的分数)
        scores_out_all : dict[forward_name][score] -> list/ndarray (OOD 数据的分数)
        title          : str, 总图标题
    """
    forward_names = list(scores_in_all.keys())  # 模型前向传递的不同名称/层
    scores = list(scores_in_all[forward_names[0]].keys())  # 每个 forward_name 对应的分数类型
    num_rows, num_cols = len(forward_names), len(scores)

    # 创建子图
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), squeeze=False)

    # 将 axes 强制转换成二维数组，保证后续索引一致
    axes = np.atleast_2d(axes)  # 自动转换为二维，不用手动 if-else

    for i, forward_name in enumerate(forward_names):
        for j, score in enumerate(scores):
            ax = axes[i, j]

            scores_in = scores_in_all[forward_name][score]
            scores_out = scores_out_all[forward_name][score]

            # 绘制直方图
            ax.hist(scores_in, bins=50, alpha=0.5, label='ID', color='blue', density=False)
            ax.hist(scores_out, bins=50, alpha=0.5, label='OOD', color='red', density=False)

            # 计算评估指标（如 AUROC, FPR95）
            results = cal_metric(scores_in, scores_out)
            # 保留 3 位小数
            rounded_results = {k: round(v, 3) for k, v in results.items()}

            # 设置标题
            ax.set_title(f'{forward_name}-{score}')

            # 在右上角显示指标
            metrics_str = '\n'.join(
                [f'{k}: {v:.3f}' for k, v in rounded_results.items() if k in ['AUROC', 'FPR']]
            )
            ax.text(
                0.95, 0.95, metrics_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )

            ax.legend(loc='upper left')

    # 设置总标题
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 给总标题留空间
    plt.show()
