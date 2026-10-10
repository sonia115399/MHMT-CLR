import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from mhmt.data import image_files
from mhmt.engine import choose_device, info_nce_loss, save_json, seed_everything, seed_worker
from mhmt.models import SourceModel
from mhmt.transforms import ContrastiveViews, downstream_transform

ROOT = Path(__file__).resolve().parents[1]


class SourceDataset(Dataset):
    def __init__(self, root, transform, strict=False):
        root = Path(root)

        partitions = [root / "train", root / "test"] if (root / "train").is_dir() else [root]
        self.classes = sorted({p.name for partition in partitions for p in partition.iterdir()
                               if p.is_dir() and not p.name.startswith(".")})
        self.samples = []
        for index, name in enumerate(self.classes):
            files = sorted(p for partition in partitions for p in image_files(partition / name))
            if strict and len(files) != 1100:
                raise ValueError(f"{name} 有 {len(files)} 张，论文源域要求每类 1100 张。")
            self.samples.extend((path, index) for path in files)
        if strict and len(self.classes) != 10:
            raise ValueError("源数据需要 10 个类别；根目录应直接包含类别目录。")
        if not self.samples or len(self.classes) > 10:
            raise ValueError("源数据为空或超过 10 类。")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            images = self.transform(image.convert("RGB"))
        return images, target


def pretrain_main(method):
    parser = argparse.ArgumentParser(description=f"{'SimCLR' if method == 'clr' else '监督'}番茄源域预训练")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--strict-paper-source", action="store_true", help="额外检查源域 10 类、每类 1100 张")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))["pretrain"][method]
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if config["epochs"] < 1:
        raise ValueError("训练轮数必须为正数。")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("输出目录已有内容；请使用新的 --output-dir。")
    seed_everything(args.seed)
    device = choose_device(args.device)
    dataset = SourceDataset(args.data_root,
                            ContrastiveViews() if method == "clr" else downstream_transform(True),
                            strict=args.strict_paper_source)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                        num_workers=args.workers, drop_last=method == "clr",
                        worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(args.seed),
                        pin_memory=device.type == "cuda")
    if not len(loader):
        raise ValueError("源数据数量少于预训练批次大小。")
    model = SourceModel(method).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    scheduler = None
    if method == "clr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = torch.nn.CrossEntropyLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    metadata = {"method": method, "stage": "pretrain", "seed": args.seed,
                "config": config, "classes": dataset.classes,
                "sample_count": len(dataset), "strict_paper_source": args.strict_paper_source}
    save_json(args.output_dir / "config.json", metadata)
    for epoch in range(config["epochs"]):
        model.train()
        running_loss, samples = 0.0, 0
        for images, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            if method == "clr":
                images = torch.cat(images, dim=0).to(device)
                loss = info_nce_loss(model(images), config["temperature"])
                batch = targets.shape[0]
            else:
                images, targets = images.to(device), targets.to(device)
                loss = criterion(model(images), targets)
                batch = targets.shape[0]
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch
            samples += batch
        history.append({"epoch": epoch + 1, "loss": running_loss / samples,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        print(json.dumps(history[-1]), flush=True)
        if scheduler is not None:
            scheduler.step()
        torch.save({**metadata, "state_dict": model.state_dict(), "epoch": epoch + 1},
                   args.output_dir / "source.pth")
        save_json(args.output_dir / "history.json", history)
