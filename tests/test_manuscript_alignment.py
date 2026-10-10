import ast
import importlib.util
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

from mhmt.models import make_resnet18

ROOT = Path(__file__).resolve().parents[1]
LIGHT = ["code/light_corn.py", "code/light_datepalmhead.py", "code/light_tea.py",
         "SimCLR/light_cornhead.py", "SimCLR/light_datepalmhead.py", "SimCLR/light_tea.py"]


def script_classes(script):

    tree = ast.parse((ROOT / script).read_text())
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=classes, type_ignores=[]), script, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("script", LIGHT)
def test_encoder_stem_freeze_gradients_and_legacy_keys(script):
    torch.manual_seed(3407)
    classes = script_classes(script)
    count = 4 if "corn" in script else 3 if "datepalm" in script else 8
    if script.startswith("code/"):
        backbone = runpy.run_path(str(ROOT / "code/model/resnet.py"))["ResNet18"](10)
        base = backbone.model
        source = backbone if "tea" in script else base
    else:
        base = make_resnet18()
        source = SimpleNamespace(backbone=base)


    legacy_encoder = nn.Sequential(*list(base.children())[:-2])
    legacy_state = legacy_encoder.state_dict()
    custom = classes["CustomModel"](source, classes["LightConvTail"](512, count))
    assert set(custom.resnet.state_dict()) == set(legacy_state)
    custom.resnet.load_state_dict(legacy_state, strict=True)
    model = classes["MultiheadMultitail"](classes["HeadMLP"](), custom)

    assert all(not p.requires_grad for p in custom.resnet.parameters())
    assert all(not module.training for module in custom.resnet.modules())
    for mode in (True, False, True):
        assert model.train(mode) is model
        assert model.headmlp.training is mode
        assert custom.mlp.training is mode
        assert all(not module.training for module in custom.resnet.modules())

    shapes = {}
    handles = [module.register_forward_hook(
        lambda _module, _inputs, output, key=key: shapes.update({key: tuple(output.shape)}))
        for key, module in custom.resnet.named_children()]
    before = {key: value.clone() for key, value in custom.resnet.state_dict().items()}
    head_before = model.headmlp.conv1.weight.detach().clone()
    tail_before = custom.mlp.conv1.weight.detach().clone()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001)
    output = model(torch.randn(2, 3, 256, 256))
    assert output.shape == (2, count)
    if script.startswith("code/"):
        assert shapes["stem_relu"] == (2, 64, 64, 64)
        assert shapes["stem_pool"] == (2, 64, 32, 32)
        assert [shapes[str(i)] for i in range(2, 6)] == [
            (2, 64, 32, 32), (2, 128, 16, 16), (2, 256, 8, 8), (2, 512, 4, 4)]
        assert isinstance(custom.resnet.stem_relu, nn.ReLU)
        assert isinstance(custom.resnet.stem_pool, nn.MaxPool2d)
    else:
        assert shapes["7"] == (2, 512, 4, 4)
    nn.CrossEntropyLoss()(output, torch.tensor([0, count - 1])).backward()
    assert model.headmlp.conv1.weight.grad.abs().sum() > 0
    assert custom.mlp.conv1.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in custom.resnet.parameters())
    optimizer.step()
    assert not torch.equal(head_before, model.headmlp.conv1.weight)
    assert not torch.equal(tail_before, custom.mlp.conv1.weight)
    assert all(torch.equal(before[key], value) for key, value in custom.resnet.state_dict().items())
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("script", ["code/light_datepalmhead.py", "SimCLR/light_datepalmhead.py"])
def test_datepalm_se_receives_second_block_output(script):
    head = script_classes(script)["HeadMLP"]().eval()
    events = []
    tensors = {}

    def second_block(_module, _inputs, output):
        events.append("second_block")
        tensors["second_block"] = output

    def se_input(_module, inputs):
        events.append("SE")
        tensors["SE"] = inputs[0]

    head.relu2.register_forward_hook(second_block)
    se = next(module for module in head.modules() if module.__class__.__name__ == "SqueezeExcitation")
    se.register_forward_pre_hook(se_input)
    with torch.no_grad():
        head(torch.randn(2, 3, 256, 256))
    assert events == ["second_block", "SE"]
    assert tensors["SE"] is tensors["second_block"]
    assert (se.fc1.in_features, se.fc1.out_features, se.fc2.out_features) == (32, 8, 32)


@pytest.mark.parametrize("script", LIGHT)
def test_entrypoint_passes_scheduler_and_decay_occurs_every_15_epochs(script):

    folder, utility = ("code", "utils") if script.startswith("code/") else ("SimCLR", "util")
    spec = importlib.util.spec_from_file_location("alignment_trainer", ROOT / folder / utility / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main = next(node for node in ast.parse((ROOT / script).read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    statements = [node for node in main.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id in
                          {"lr", "optimizer", "scheduler", "trainer"} for target in node.targets)]
    namespace = {"torch": torch, "StepLR": StepLR, "ModelTrainer": module.ModelTrainer,
                 "multi_model": nn.Linear(2, 2), "train_loader": None, "val_loader": None,
                 "criterion": nn.CrossEntropyLoss(), "device": torch.device("cpu")}
    exec(compile(ast.Module(body=statements, type_ignores=[]), script, "exec"), namespace)
    optimizer = namespace["optimizer"]
    scheduler = namespace["trainer"].scheduler
    assert scheduler is namespace["scheduler"]
    initial_lr = optimizer.param_groups[0]["lr"]
    for epoch in range(1, 41):
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(initial_lr * 0.85 ** (epoch // 15))
