import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")


def parse_original_args(filename, species, stage, seed, epochs):
    folder = Path(filename).resolve().parent
    method = "net" if folder.name == "code" else "clr"
    parser = argparse.ArgumentParser(description=f"原始 {method}/{species}/{stage} 下游脚本")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="与原脚本网络和键名匹配的本地历史 checkpoint")
    parser.add_argument("--data-root", type=Path, default=folder / "multi_data" / species,
                        help="包含 train 和 test 类别目录的数据根目录")
    parser.add_argument("--output-dir", type=Path,
                        default=folder.parent / "outputs" / "original" / method / species / stage)
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("训练轮数须为正，batch size 至少为 2。")
    args.checkpoint = args.checkpoint.resolve()
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.checkpoint.is_file():
        parser.error(f"模型文件不存在：{args.checkpoint}")
    if not (args.data_root / "train").is_dir() or not (args.data_root / "test").is_dir():
        parser.error("数据根目录必须包含 train/ 和 test/。")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args
