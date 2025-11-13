from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from torchvision.datasets.folder import ImageFolder
from torch.utils.data import random_split
import os
import torch
from pathlib import Path
from PIL import Image
import shutil
from torchvision import datasets

from datasets import load_dataset

# def get_id_dataset(name, transform_train, transform_test, root="./data"):
#     data_path = root
#     if name == "cifar10":
#         train_data = CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
#         test_data = CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
#
#     elif name == "cifar100":
#         train_data = CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
#         test_data = CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
#
#     elif name == "imagenet1k":
#         train_path = os.path.join(root, "imagenet1k/train")
#         val_path = os.path.join(root, "imagenet1k/val")
#         if not os.path.exists(train_path) or not os.path.exists(val_path):
#             raise FileNotFoundError("ImageNet1K 路径不存在，请检查 ./data/imagenet1k/train 和 /val")
#         train_data = ImageFolder(root=train_path, transform=transform_train)
#         test_data = ImageFolder(root=val_path, transform=transform_test)
#
#     else:
#         raise ValueError(f"Unsupported ID dataset: {name}")
#
#     return train_data, test_data
# CIFAR-100 fine label 到 coarse label 的映射表
fine_to_coarse = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13
]


class CIFAR100SuperClass(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root=root, train=train, transform=transform,
                         target_transform=target_transform, download=download)

        # 手动生成 coarse_labels
        self.coarse_labels = [fine_to_coarse[f] for f in self.targets]

        # 把 targets 改为 coarse_labels
        self.targets = self.coarse_labels

        # 超级类名字
        self.superclass_names = [
            'aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
            'household electrical devices', 'household furniture', 'insects', 'large carnivores',
            'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
            'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles',
            'small mammals', 'trees', 'vehicles 1', 'vehicles 2'
        ]

def get_id_dataset(name, transform_train, transform_test, root="./data", val_split_ratio=0.3, seed=42):
    data_path = root
    if name == "cifar10":
        train_data = CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        test_data_full = CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

    elif name == "cifar100":
        train_data = CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
        test_data_full = CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
    elif name == "cifar100super":
        train_data = CIFAR100SuperClass(root=data_path, train=True, download=True, transform=transform_train)
        test_data_full = CIFAR100SuperClass(root=data_path, train=False, download=True, transform=transform_test)
    elif name == "imagenet":
        train_path = os.path.join(root, "imagenet100/train")
        test_path = os.path.join(root, "imagenet100/test")
        train_data = ImageFolder(root=train_path, transform=transform_train)
        test_data_full = ImageFolder(root=test_path, transform=transform_test)
    else:
        raise ValueError(f"Unsupported ID dataset: {name}")

    # 拆分 test_data_full 为 test / val
    total_len = len(test_data_full)
    val_len = int(total_len * val_split_ratio)
    test_len = total_len - val_len

    generator = torch.Generator().manual_seed(seed)  # 保证可复现
    test_data, val_data = random_split(test_data_full, [test_len, val_len], generator=generator)

    return train_data, test_data, val_data



if __name__ == "__main__":
    train_data = CIFAR100SuperClass(root="/", train=True, download=True, transform=None)

