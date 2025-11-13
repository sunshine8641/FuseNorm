import matplotlib.pyplot as plt
from metrics import  *
from scipy.stats import norm
import torch
from torchvision import transforms
import torchvision.utils as T


def plot_histograms(scores_in_all, scores_out_all, bins=50, title="Histograms of OOD Scores", save_path=None):
    """
    Plot histograms for k groups of scores with subplots, optionally save the figure.

    :param scores_in_all: shape (k, N), ID scores
    :param scores_out_all: shape (k, N), OOD scores
    :param bins: number of bins for the histogram
    :param title: overall title for the figure
    :param save_path: path to save the figure (e.g., 'results/ood_hist.png'); if None, display only
    """
    k = len(scores_in_all)
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))  # 1 row, k columns

    if k == 1:
        axes = [axes]  # Ensure iterable

    for i in range(k):
        results = cal_metric(scores_in_all[i], scores_out_all[i])
        # Construct metrics string (show only AUROC and FPR)
        metrics_str = '\n'.join([f'{key}: {val:.3f}' for key, val in results.items() if key in ['AUROC', 'FPR']])

        # Display metrics at top-left corner
        axes[i].text(
            0.05, 0.95, metrics_str,
            transform=axes[i].transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

        # Histogram
        axes[i].hist(scores_in_all[i], bins=bins, alpha=0.6, label="ID", color="blue", density=True)
        axes[i].hist(scores_out_all[i], bins=bins, alpha=0.6, label="OOD", color="red", density=True)
        axes[i].set_title(f"Group {i+1}")
        axes[i].set_xlabel("Score")
        if i == 0:
            axes[i].set_ylabel("Density")
        axes[i].legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Save or display
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()


def plot_histograms_one(scores_in_all, means, stds, bins=50, title="Histograms of OOD Scores"):
    """
    Plot histograms for k groups with Gaussian distribution overlay.

    :param scores_in_all: shape (k, N), ID scores
    :param bins: number of histogram bins
    :param title: overall figure title
    """
    k = len(scores_in_all)
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))  # 1 row, k columns

    if k == 1:
        axes = [axes]  # Ensure iterable

    for i in range(k):
        # Histogram
        axes[i].hist(scores_in_all[i], bins=bins, alpha=0.6,
                     label="ID Histogram", color="blue", density=True)

        # Generate x range
        xmin, xmax = np.min(scores_in_all[i]), np.max(scores_in_all[i])
        x = np.linspace(xmin, xmax, 200)

        # Gaussian distribution curve
        pdf = norm.pdf(x, means[i], stds[i])
        axes[i].plot(x, pdf, "r-", lw=2, label=f"N({means[i]:.2f}, {stds[i]:.2f}²)")

        # Set titles and axis labels
        axes[i].set_title(f"Group {i+1}")
        axes[i].set_xlabel("Score")
        axes[i].set_ylabel("Density")  # ✅ y-axis for each subplot
        axes[i].legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def unnormalize(t, mean, std):
    # Unnormalize tensor (e.g., CIFAR10 images)
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return torch.clamp(t * std + mean, 0, 1)


def show_images(x, y, mean, std, n=16, save_path="Figs/x.png"):
    """
    Show a grid of sample images.

    :param x: input images tensor
    :param y: labels tensor
    :param mean: normalization mean
    :param std: normalization std
    :param n: number of images to display
    :param save_path: path to save the figure
    """
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
        print(f"Figure saved to: {save_path}")
    plt.show()


def plot_score_dicts_subplots(scores_in_all, scores_out_all, title):
    """
    Plot subplots comparing multiple forward_name and score metrics (ID vs OOD distributions + AUROC/FPR metrics).

    :param scores_in_all: dict[forward_name][score] -> list/ndarray of ID scores
    :param scores_out_all: dict[forward_name][score] -> list/ndarray of OOD scores
    :param title: overall figure title
    """
    forward_names = list(scores_in_all.keys())  # Different forward passes/layers
    scores = list(scores_in_all[forward_names[0]].keys())  # Score types for each forward_name
    num_rows, num_cols = len(forward_names), len(scores)

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), squeeze=False)

    # Ensure axes is a 2D array for consistent indexing
    axes = np.atleast_2d(axes)

    for i, forward_name in enumerate(forward_names):
        for j, score in enumerate(scores):
            ax = axes[i, j]

            scores_in = scores_in_all[forward_name][score]
            scores_out = scores_out_all[forward_name][score]

            # Plot histograms
            ax.hist(scores_in, bins=50, alpha=0.5, label='ID', color='blue', density=False)
            ax.hist(scores_out, bins=50, alpha=0.5, label='OOD', color='red', density=False)

            # Compute metrics (e.g., AUROC, FPR95)
            results = cal_metric(scores_in, scores_out)
            rounded_results = {k: round(v, 3) for k, v in results.items()}

            # Set subplot title
            ax.set_title(f'{forward_name}-{score}')

            # Display metrics at top-right corner
            metrics_str = '\n'.join(
                [f'{k}: {v:.3f}' for k, v in rounded_results.items() if k in ['AUROC', 'FPR']]
            )
            ax.text(
                0.95, 0.95, metrics_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )

            ax.legend()

    # Set overall figure title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
    plt.show()
