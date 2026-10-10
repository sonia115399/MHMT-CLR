import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from mhmt.engine import choose_device, evaluate, save_json
from mhmt.models import MHMT
from mhmt.transforms import downstream_transform


def main():
    parser = argparse.ArgumentParser(description="评估 MHMT 下游 checkpoint")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path, help="直接包含各类别目录的测试目录")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/metrics.json"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("stage") != "downstream":
        raise ValueError("评估需要本版本生成的下游 checkpoint。")
    dataset = ImageFolder(args.data_root, transform=downstream_transform(False))
    if dataset.classes != checkpoint["classes"]:
        raise ValueError("测试集类别顺序与训练 checkpoint 不一致。")
    model = MHMT(checkpoint["species"], checkpoint["use_head"], checkpoint["use_tail"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    metrics = evaluate(model, DataLoader(dataset, batch_size=args.batch_size), device, dataset.classes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, metrics)
    print({key: value for key, value in metrics.items() if key.startswith(("top1", "macro"))})


if __name__ == "__main__":
    main()
