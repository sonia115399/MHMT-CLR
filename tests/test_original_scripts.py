import ast
import importlib.util
import runpy
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mhmt.original_runtime import parse_original_args

ROOT = Path(__file__).resolve().parents[1]
LIGHT = ["code/light_corn.py", "code/light_datepalmhead.py", "code/light_tea.py",
         "SimCLR/light_cornhead.py", "SimCLR/light_datepalmhead.py", "SimCLR/light_tea.py"]
SCRIPTS = LIGHT + [f"{folder}/{species}_cnn.py" for folder in ("code", "SimCLR")
                   for species in ("corn", "datepalm", "tea")]


@pytest.mark.parametrize("script", SCRIPTS)
def test_original_scripts_help(script, monkeypatch):
    monkeypatch.setattr(sys, "argv", [script, "--help"])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(ROOT / script), run_name="__main__")
    assert error.value.code == 0


@pytest.mark.parametrize("script", LIGHT)
def test_restored_original_heads_and_tails_forward_and_gradients(script):
    source = ast.parse((ROOT / script).read_text())
    classes = [node for node in ast.walk(source) if isinstance(node, ast.ClassDef)]
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=classes, type_ignores=[]), script, "exec"), namespace)
    count = 4 if "corn" in script else 3 if "datepalm" in script else 8
    head = namespace["HeadMLP"]()
    tail = namespace["LightConvTail"](512, count)
    output = head(torch.randn(2, 3, 256, 256))
    assert output.shape == (2, 3, 128, 128)
    output.square().mean().backward()
    assert head.conv1.weight.grad.abs().sum() > 0
    logits = tail(torch.randn(2, 512, 4, 4))
    assert logits.shape == (2, count)
    logits.square().mean().backward()
    assert tail.conv1.weight.grad.abs().sum() > 0


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("folder,tools", [("code", "utils"), ("SimCLR", "util")])
def test_original_trainer_and_evaluator_output_paths(folder, tools, tmp_path):
    train_module = load_file(f"original_{folder}_train", ROOT / folder / tools / "train.py")
    test_module = load_file(f"original_{folder}_test", ROOT / folder / tools / "test.py")
    model = nn.Linear(2, 2)
    dataset = TensorDataset(torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]]),
                            torch.tensor([0, 1, 0, 1]))
    loader = DataLoader(dataset, batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    trainer = train_module.ModelTrainer(model, loader, loader, optimizer,
                                       nn.CrossEntropyLoss(), torch.device("cpu"))
    trainer.train_model(1, str(tmp_path), "run")
    evaluator = test_module.ModelEvaluator(model, loader, torch.device("cpu"), str(tmp_path), "run")
    evaluator.evaluate_model()
    for name in ("training_logs.log", "model.pth", "training_progress.png", "results.csv", "confusion_matrix.csv"):
        assert (tmp_path / "run" / name).is_file()
    assert not (tmp_path / "corn").exists()


def test_original_data_loader_ignores_metadata_and_keeps_order(tmp_path):
    module = load_file("original_dataset", ROOT / "code/dataprocess/mydataset.py")
    for name in ("a", "b"):
        folder = tmp_path / name
        folder.mkdir()
        Image.new("RGB", (10, 10)).save(folder / "leaf.png")
        (folder / ".DS_Store").write_text("metadata")
    (tmp_path / ".DS_Store").write_text("metadata")
    dataset = module.MyDataset(str(tmp_path))
    assert dataset.classes == ["a", "b"]
    assert len(dataset) == 2
    assert dataset[1][1] == 1


def test_original_runtime_checks_inputs_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["light_corn.py", "--checkpoint", str(tmp_path / "missing.pth")])
    with pytest.raises(SystemExit) as error:
        parse_original_args(ROOT / LIGHT[0], "corn", "head_tail", 21, 50)
    assert error.value.code == 2
