import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from mhmt.data import make_datasets
from mhmt.engine import info_nce_loss, seed_everything, summarize, train_branch
from mhmt.models import MHMT, SourceModel
from mhmt.transforms import ContrastiveViews, downstream_transform

torch.set_num_threads(1)


@pytest.mark.parametrize("species,classes", [("corn", 4), ("datepalm", 3), ("tea", 8)])
def test_paper_shapes_and_frozen_encoder(species, classes):
    model = MHMT(species)
    model.train()
    assert not model.encoder.training
    before = {name: value.clone() for name, value in model.encoder.state_dict().items()}
    images = torch.randn(2, 3, 256, 256)
    head = model.head(images)
    assert head.shape == (2, 3, 128, 128)
    feature = model.encoder(head)
    assert feature.shape == (2, 512, 4, 4)
    logits = model.tail(feature)
    assert logits.shape == (2, classes)
    logits.square().mean().backward()
    assert model.head.layers[0][0].weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.encoder.parameters())
    for name, value in model.encoder.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize("head,tail", [(False, False), (False, True), (True, False), (True, True)])
def test_ablation_forward(head, tail):
    model = MHMT("tea", head, tail).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 256, 256)).shape == (2, 8)


def test_info_nce_positive_pair_and_gradient():
    features = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]], requires_grad=True)
    loss = info_nce_loss(features, temperature=1.0)
    assert loss.item() == pytest.approx(math.log(math.e + 2) - 1)
    loss.backward()
    assert torch.isfinite(features.grad).all()
    with pytest.raises(ValueError):
        info_nce_loss(features[:3])


def test_checkpoint_load_requires_correct_method_and_complete_encoder(tmp_path):
    source = SourceModel("clr")
    path = tmp_path / "source.pth"
    torch.save({"method": "clr", "stage": "pretrain", "state_dict": source.state_dict()}, path)
    branch = MHMT("datepalm")
    branch.load_source(path, "clr")
    assert torch.equal(branch.encoder[0].weight, source.backbone.conv1.weight)
    with pytest.raises(ValueError):
        branch.load_source(path, "net")
    checkpoint = torch.load(path)
    del checkpoint["state_dict"]["backbone.conv1.weight"]
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError):
        branch.load_source(path, "clr")


def test_original_random_split_is_preserved(tmp_path):
    for split in ("train", "test"):
        for name in ("a", "b"):
            folder = tmp_path / split / name
            folder.mkdir(parents=True)
            for i in range(5):
                Image.new("RGB", (300, 260), color=(i * 20, 100, 50)).save(folder / f"{i}.png")
    seed_everything(42)
    expected, _ = random_split(range(10), [8, 2])
    seed_everything(42)
    datasets, classes = make_datasets(tmp_path)
    assert classes == ["a", "b"]
    assert datasets["train"].indices == expected.indices
    assert len(datasets["test"]) == 10
    assert set(datasets["train"].indices).isdisjoint(datasets["val"].indices)
    first = datasets["val"][0][0]
    assert torch.equal(first, datasets["val"][0][0])
    seed_everything(123)
    alternate, _ = make_datasets(tmp_path)
    assert alternate["train"].indices != datasets["train"].indices


def test_transforms_and_sample_standard_deviation():
    image = Image.new("RGB", (400, 300))
    assert downstream_transform(False)(image).shape == (3, 256, 256)
    assert torch.equal(downstream_transform(False)(image), downstream_transform(False)(image))
    assert all(view.shape == (3, 256, 256) for view in ContrastiveViews()(image))
    records = [{key: value for key in ("top1_accuracy", "macro_precision", "macro_recall", "macro_f1")}
               for value in (0.5, 1.0)]
    assert summarize(records)["macro_f1"]["sample_std"] == pytest.approx(np.std([0.5, 1.0], ddof=1))


def test_best_validation_checkpoint_is_restored(tmp_path, monkeypatch):
    import mhmt.engine as engine
    model = nn.Linear(2, 2)
    loader = DataLoader(TensorDataset(torch.ones(4, 2), torch.tensor([0, 1, 0, 1])), batch_size=2)
    captured = []

    def fake_evaluate(model, loader, device, classes):
        captured.append({key: value.detach().clone() for key, value in model.state_dict().items()})
        return {"top1_accuracy": [0.9, 0.8, 0.7][len(captured) - 1]}

    monkeypatch.setattr(engine, "evaluate", fake_evaluate)
    config = {"epochs": 2, "learning_rate": 0.01, "weight_decay": 0.0001, "step_size": 15, "gamma": 0.85}
    train_branch(model, {s: loader for s in ("train", "val", "test")}, torch.device("cpu"),
                 config, tmp_path, {"classes": ["a", "b"], "seed": 42})
    checkpoint = torch.load(tmp_path / "best.pth")
    assert checkpoint["epoch"] == 1
    for key, value in captured[0].items():
        assert torch.equal(captured[-1][key], value)


def test_paper_config_schedules_and_seeds():
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/paper.json").read_text())
    assert config["seeds"] == [21, 42, 123, 1568, 3407]
    assert [config["species"][s]["epochs"] for s in ("corn", "datepalm", "tea")] == [50, 40, 45]
    assert [config["species"][s]["learning_rate"] for s in ("corn", "datepalm", "tea")] == [0.001, 0.0008, 0.0005]
