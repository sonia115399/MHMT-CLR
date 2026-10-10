import hashlib
from pathlib import Path

from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, random_split

from mhmt.transforms import downstream_transform

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file()
                  and p.suffix.lower() in IMAGE_EXTENSIONS
                  and not any(part.startswith(".") for part in p.relative_to(root).parts))


def content_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_datasets(root, fraction=1.0):

    root = Path(root)
    train_source = ImageFolder(root / "train", transform=downstream_transform(True))
    val_source = ImageFolder(root / "train", transform=downstream_transform(False))
    test = ImageFolder(root / "test", transform=downstream_transform(False))
    if train_source.class_to_idx != test.class_to_idx:
        raise ValueError("train 与 test 的类别目录必须一致。")
    train_size = int(0.8 * len(train_source))
    train, validation_indices = random_split(train_source, [train_size, len(train_source) - train_size])

    val = Subset(val_source, validation_indices.indices)
    if fraction < 1.0:
        selected = []
        for label in range(len(train_source.classes)):
            group = [i for i in train.indices if train_source.targets[i] == label]
            selected.extend(group[:int(len(group) * fraction)])
        train = Subset(train_source, selected)
    if not len(train) or not len(val) or not len(test):
        raise ValueError("训练、验证和测试集均需要图像。")
    return {"train": train, "val": val, "test": test}, train_source.classes
