from torchvision.datasets import SVHN, ImageFolder
import os
from torch.utils.data import Dataset
from torchvision.datasets import SVHN, ImageFolder
import os
from PIL import Image
from pathlib import Path


def load_all_val_paths(labels_dir, images_root):
    """
    合并 val1.txt 到 val10.txt 中的图像路径，返回完整路径列表。

    :param labels_dir: labels 文件夹路径，如 'data/dtd/labels'
    :param images_root: 图像根目录路径，如 'data/dtd/images'
    :return: test 图像的完整路径列表
    """
    val_image_paths = []

    for i in range(1, 11):
        val_file = os.path.join(labels_dir, f'val{i}.txt')
        with open(val_file, 'r') as f:
            for line in f:
                rel_path = line.strip()  # e.g. 'banded/banded_0010.jpg'
                full_path = os.path.join(images_root, rel_path)
                val_image_paths.append(full_path)

    return val_image_paths


class DTDTextureDataset(Dataset):
    def __init__(self, img_root, img_list, transform=None):
        self.img_root = img_root
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root, self.img_list[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.img_list[idx].split("/")[0]  # 类别就是路径的第一级
        if self.transform:
            image = self.transform(image)
        return image, label



def get_ood_dataset(name, transform, root="./data"):
    # name = name.lower()
    if not os.path.exists(root):
        raise FileNotFoundError(f"OOD dataset path not found: {root}")
    if name == "svhn":
        return SVHN(root=root, split="test", download=True, transform=transform)
    elif name == "textures":
        path = os.path.join(root, "textures")
        # 示例：读取 train1.txt 中的图像路径

        labels_dir=os.path.join(path, "labels")
        images_dir=os.path.join(path, "images")
        val_image_paths = load_all_val_paths(labels_dir,images_dir)

        return DTDTextureDataset(img_root=images_dir, img_list=val_image_paths, transform=transform)
    elif name in ["LSUN_R", "LSUN_C","iSUN","iNaturalist","SUN","Places"]:
        path = os.path.join(root, name)
        print(path)
        return  ImageFolder(root=path, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset name: {name}")



