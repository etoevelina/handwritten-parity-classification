from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


MEAN = (0.973,)
STD = (0.137,)


def transforms_for(augment: bool):
    items = [transforms.Grayscale(num_output_channels=1), transforms.Resize((64, 64))]
    if augment:
        items += [
            # Domain-safe subset of RandAugment-style transforms. Operations such
            # as Solarize/Invert were excluded because they change this dataset's
            # white-paper acquisition domain.
            transforms.RandomAffine(degrees=10, translate=(0.06, 0.06), scale=(0.94, 1.06), fill=255),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.7))], p=0.18),
            transforms.ColorJitter(brightness=0.08, contrast=0.12),
        ]
    items += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if augment:
        items += [transforms.RandomErasing(p=0.08, scale=(0.005, 0.025), value=0)]
    return transforms.Compose(items)


def make_loaders(data_dir: str | Path, batch_size: int = 32, augment: bool = True):
    root = Path(data_dir)
    train = datasets.ImageFolder(root / "train", transform=transforms_for(augment))
    val = datasets.ImageFolder(root / "val", transform=transforms_for(False))
    test = datasets.ImageFolder(root / "test", transform=transforms_for(False))
    kwargs = dict(batch_size=batch_size, num_workers=0)
    return (
        DataLoader(train, shuffle=True, **kwargs),
        DataLoader(val, shuffle=False, **kwargs),
        DataLoader(test, shuffle=False, **kwargs),
        train.class_to_idx,
    )
