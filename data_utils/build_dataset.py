import torch
from .id_datasets import get_id_dataset
from .ood_datasets import get_ood_dataset
from .transforms import get_id_transform
from .transforms import get_ood_transform
import random
import torch.utils.data as data

class jigsaw_dataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]

        s = int(float(x.size(1)) / 3)

        x_ = torch.zeros_like(x)
        tiles_order = random.sample(range(9), 9)
        for o, tile in enumerate(tiles_order):
            i = int(o / 3)
            j = int(o % 3)

            ti = int(tile / 3)
            tj = int(tile % 3)
            # print(i, j, ti, tj)
            x_[:, i * s:(i + 1) * s, j * s:(j + 1) * s] = x[:, ti * s:(ti + 1) * s, tj * s:(tj + 1) * s]
        return x_, y

def build_id_dataloaders(cfg, accelerator):
    batch_size = cfg["batch_size"]
    num_workers =cfg["num_workers"]
    dataset_name =cfg["id_dataset"]["name"]
    dataset_root = cfg["id_dataset"].get("root", "./data")

    transform_train, transform_test = get_id_transform(dataset_name)
    id_train, id_test, id_val = get_id_dataset(dataset_name, transform_train, transform_test, dataset_root)

    def make_loader(ds, shuffle):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )

    id_train_loader = make_loader(id_train, shuffle=True)
    id_test_loader = make_loader(id_test, shuffle=True)
    id_val_loader = make_loader(id_val, shuffle=True)

    # id_train_loader,id_test_loader, id_val_loader = accelerator.prepare(id_train_loader, id_test_loader,id_val_loader)

    return id_train_loader,id_test_loader,id_val_loader

def build_ood_dataloaders(cfg, accelerator):
    """
    dataset_cfg: dict, eg. {"batch_size": 128, "num_workers": 8}
    ood_cfg: dict, eg. {"name": "SVHN", "root": "./data/SVHN"}
    accelerator: accelerate.Accelerator 实例
    """
    id_name=cfg["id_dataset"]["name"]
    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]
    ood_name = cfg["ood_dataset"]["name"]
    ood_root = cfg["ood_dataset"]["root"]
    transform_test = get_ood_transform(id_name)

    ood_dataset = get_ood_dataset(ood_name, transform_test, ood_root)

    def make_loader(ds, shuffle):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )

    ood_dataloder=make_loader(ood_dataset, shuffle=True)
    # ood_dataloder = accelerator.prepare(ood_dataloder)
    return ood_dataloder


def build_jigsaw_dataloaders(cfg, accelerator):
    batch_size = cfg["batch_size"]
    num_workers =cfg["num_workers"]
    dataset_name =cfg["id_dataset"]["name"]
    dataset_root = cfg["id_dataset"].get("root", "./data")

    transform_train, transform_test = get_id_transform(dataset_name)
    id_train, id_test, id_val = get_id_dataset(dataset_name, transform_test, transform_test, dataset_root)

    def make_loader(ds, shuffle):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )
    jigsaw_train = jigsaw_dataset(id_train)
    jigsaw_test = jigsaw_dataset(id_test)
    jigsaw_val = jigsaw_dataset(id_val)

    jigsaw_train_loader = make_loader(jigsaw_train, shuffle=True)
    jigsaw_test_loader = make_loader( jigsaw_test, shuffle=True)
    jigsaw_val_loader = make_loader(jigsaw_val, shuffle=True)

    # igsaw_train_loader, jigsaw_test_loader, jigsaw_val_loader = accelerator.prepare(jigsaw_train_loader, jigsaw_test_loader,jigsaw_val_loader)

    return jigsaw_train_loader, jigsaw_test_loader,jigsaw_val_loader