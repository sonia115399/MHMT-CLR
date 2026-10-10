import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def info_nce_loss(features, temperature=0.05):

    count = features.shape[0]
    if count < 4 or count % 2 or temperature <= 0:
        raise ValueError("InfoNCE 需要至少两个样本的双视图及正温度。")
    features = torch.nn.functional.normalize(features, dim=1)
    logits = features @ features.T / temperature
    diagonal = torch.eye(count, dtype=torch.bool, device=features.device)
    logits = logits.masked_fill(diagonal, float("-inf"))
    targets = (torch.arange(count, device=features.device) + count // 2) % count
    return torch.nn.functional.cross_entropy(logits, targets)


def evaluate(model, loader, device, classes):
    model.eval()
    truth, prediction = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            truth.extend(labels.tolist())
            prediction.extend(outputs.argmax(1).cpu().tolist())
    report = classification_report(
        truth, prediction, labels=list(range(len(classes))), target_names=classes,
        output_dict=True, zero_division=0,
    )
    return {
        "top1_accuracy": accuracy_score(truth, prediction),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "classification_report": report,
        "confusion_matrix": confusion_matrix(truth, prediction,
                                              labels=list(range(len(classes)))).tolist(),
    }


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def train_branch(model, loaders, device, config, output, metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["step_size"], gamma=config["gamma"]
    )
    criterion = torch.nn.CrossEntropyLoss()
    best, history = -1.0, []
    classes = metadata["classes"]
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        for images, labels in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
        val = evaluate(model, loaders["val"], device, classes)
        row = {"epoch": epoch + 1, "loss": total_loss / len(loaders["train"].dataset),
               "val_accuracy": val["top1_accuracy"], "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["top1_accuracy"] > best:
            best = val["top1_accuracy"]
            torch.save({**metadata, "state_dict": model.state_dict(), "epoch": epoch + 1,
                        "best_val_accuracy": best, "config": config}, output / "best.pth")
        scheduler.step()
        save_json(output / "history.json", history)
    checkpoint = torch.load(output / "best.pth", map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    metrics = evaluate(model, loaders["test"], device, classes)
    metrics.update({"seed": metadata["seed"], "best_epoch": checkpoint["epoch"]})
    save_json(output / "metrics.json", metrics)
    return metrics


def summarize(results):
    if len(results) < 2:
        raise ValueError("样本标准差需要至少两次运行。")
    return {
        metric: {"mean": float(np.mean([r[metric] for r in results])),
                 "sample_std": float(np.std([r[metric] for r in results], ddof=1))}
        for metric in ("top1_accuracy", "macro_precision", "macro_recall", "macro_f1")
    }
