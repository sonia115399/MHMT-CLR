import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mhmt.data import content_hash, make_datasets
from mhmt.engine import choose_device, save_json, seed_everything, seed_worker, summarize, train_branch
from mhmt.models import MHMT, SPECIES

ROOT = Path(__file__).resolve().parents[1]


def downstream_main(method):
    parser = argparse.ArgumentParser(description=f"MHMT-{'CLR' if method == 'clr' else 'Net'} 下游训练")
    parser.add_argument("--species", required=True, choices=SPECIES)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--epochs", type=int, help="覆盖轮数；覆盖后属于自定义实验")
    parser.add_argument("--fraction", type=float, default=1.0, choices=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--no-head", action="store_true", help="表5消融：移除 CNN 头")
    parser.add_argument("--no-tail", action="store_true", help="表5消融：CNN 尾替换为 MLP")
    args = parser.parse_args()
    paper = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = args.seeds or paper["seeds"]
    if len(seeds) != len(set(seeds)):
        raise ValueError("运行种子不可重复。")
    config = {**paper["downstream"], **paper["species"][args.species]}
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if config["epochs"] < 1:
        raise ValueError("训练轮数必须为正数。")
    device = choose_device(args.device)
    variant = f"head{int(not args.no_head)}_tail{int(not args.no_tail)}_fraction{args.fraction:g}"
    output = args.output_dir / method / args.species / variant
    if any((output / f"seed_{seed}").exists() for seed in seeds):
        raise FileExistsError("输出中已有同种子运行；请使用新的 --output-dir。")
    source_hash = content_hash(args.checkpoint)
    results = []
    for seed in seeds:
        seed_everything(seed)
        datasets, classes = make_datasets(args.data_root, args.fraction)
        if len(classes) != SPECIES[args.species]:
            raise ValueError("数据类别数与物种分支不一致。")
        loaders = {
            split: DataLoader(dataset, batch_size=config["batch_size"], shuffle=split == "train",
                              num_workers=args.workers, worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(seed),
                              pin_memory=device.type == "cuda")
            for split, dataset in datasets.items()
        }
        model = MHMT(args.species, not args.no_head, not args.no_tail)
        model.load_source(args.checkpoint, method)
        metadata = {
            "method": method, "stage": "downstream", "species": args.species,
            "seed": seed, "classes": classes,
            "source_sha256": source_hash,
            "split_protocol": "existing_train_test_plus_seeded_random_80_20",
            "use_head": not args.no_head, "use_tail": not args.no_tail,
            "fraction": args.fraction,
        }
        run_output = output / f"seed_{seed}"
        run_output.mkdir(parents=True)

        split_record = {
            split: [str(Path(dataset.dataset.samples[i][0]).relative_to(args.data_root))
                    for i in dataset.indices]
            for split, dataset in datasets.items() if split != "test"
        }
        split_record["test"] = [str(Path(path).relative_to(args.data_root))
                                for path, _ in datasets["test"].samples]
        save_json(run_output / "split.json", split_record)
        results.append(train_branch(model, loaders, device, config, run_output, metadata))
    save_json(output / "runs.json", results)
    if len(results) > 1:
        save_json(output / "summary.json", summarize(results))
    print(f"结果保存至 {output}，所有指标使用 0–1 比例。", flush=True)
