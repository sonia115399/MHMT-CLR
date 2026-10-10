import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image

from mhmt.engine import choose_device
from mhmt.models import MHMT
from mhmt.transforms import downstream_transform


def main():
    parser = argparse.ArgumentParser(description="MHMT 单张叶片预测")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("stage") != "downstream":
        raise ValueError("预测需要本版本生成的下游 checkpoint。")
    device = choose_device(args.device)
    model = MHMT(checkpoint["species"], checkpoint["use_head"], checkpoint["use_tail"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    with Image.open(args.image) as image:
        tensor = downstream_transform(False)(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = model(tensor).softmax(dim=1)[0].cpu().tolist()
    print(dict(zip(checkpoint["classes"], probabilities)))


if __name__ == "__main__":
    main()
