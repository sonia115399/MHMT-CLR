import json
import runpy
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from mhmt.cli import downstream_main
from mhmt.data import image_files
from mhmt.pretrain import SourceDataset, pretrain_main
from mhmt.transforms import downstream_transform

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def create_images(folder, classes, count):
    for index in range(classes):
        target = folder / f"class_{index}"
        target.mkdir(parents=True)
        for number in range(count):
            Image.new("RGB", (280, 260), (index * 20, number * 20, 100)).save(target / f"image_{number}.png")


@pytest.mark.parametrize("method", ["net", "clr"])
def test_full_pretraining_downstream_and_inference(tmp_path, monkeypatch, method):
    source = tmp_path / "source"
    create_images(source, 2, 2)
    config = json.loads((ROOT / "configs/paper.json").read_text())
    config["pretrain"][method]["batch_size"] = 2
    config["downstream"]["batch_size"] = 4
    custom = tmp_path / "smoke_config.json"
    custom.write_text(json.dumps(config))
    pretrain_output = tmp_path / "pretraining"
    arguments = ["run.py", "--data-root", str(source), "--output-dir", str(pretrain_output),
                 "--config", str(custom), "--epochs", "1", "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", arguments)
    pretrain_main(method)
    checkpoint = pretrain_output / "source.pth"
    assert torch.load(checkpoint)["epoch"] == 1
    assert torch.load(checkpoint)["method"] == method
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(FileExistsError):
        pretrain_main(method)
    for species, classes in [("corn", 4), ("datepalm", 3), ("tea", 8)]:
        data = tmp_path / species
        create_images(data / "train", classes, 2)
        create_images(data / "test", classes, 1)
        output = tmp_path / "downstream"
        arguments = ["train.py", "--data-root", str(data), "--species", species,
                     "--checkpoint", str(checkpoint), "--config", str(custom),
                     "--output-dir", str(output), "--epochs", "1", "--device", "cpu",
                     "--seeds", "42", "123"]

        monkeypatch.setattr(sys, "argv", arguments)
        downstream_main(method)
        branch = output / method / species / "head1_tail1_fraction1"
        assert (branch / "summary.json").is_file()
        best = branch / "seed_42/best.pth"
        metrics = json.loads((branch / "seed_42/metrics.json").read_text())
        assert sum(sum(row) for row in metrics["confusion_matrix"]) == classes
        split = json.loads((branch / "seed_42/split.json").read_text())
        assert len(split["train"]) == int(0.8 * classes * 2)
        monkeypatch.setattr(sys, "argv", arguments)
        with pytest.raises(FileExistsError):
            downstream_main(method)
        monkeypatch.setattr(sys, "argv", ["evaluate.py", "--checkpoint", str(best),
                                         "--data-root", str(data / "test"),
                                         "--output", str(branch / "retest.json"), "--device", "cpu"])
        runpy.run_path(str(ROOT / "scripts/evaluate.py"), run_name="__main__")
        repeated = json.loads((branch / "retest.json").read_text())
        assert repeated["confusion_matrix"] == metrics["confusion_matrix"]
        monkeypatch.setattr(sys, "argv", ["predict.py", "--checkpoint", str(best),
                                         "--image", str(data / "test/class_0/image_0.png"), "--device", "cpu"])
        runpy.run_path(str(ROOT / "scripts/predict.py"), run_name="__main__")


def test_source_layouts_and_invalid_input(tmp_path):
    create_images(tmp_path / "train", 2, 2)
    create_images(tmp_path / "test", 2, 1)
    dataset = SourceDataset(tmp_path, downstream_transform(False))
    assert len(dataset) == 6
    assert dataset[0][0].shape == (3, 256, 256)
    with pytest.raises(ValueError, match="1100"):
        SourceDataset(tmp_path, downstream_transform(False), strict=True)
    hidden = tmp_path / "train/class_0/.ipynb_checkpoints"
    hidden.mkdir()
    Image.new("RGB", (10, 10)).save(hidden / "hidden.png")
    assert len(image_files(tmp_path / "train/class_0")) == 2


@pytest.mark.parametrize("script", ["code/run.py", "code/train.py", "SimCLR/run.py", "SimCLR/train.py"])
def test_entrypoint_help(script, monkeypatch):
    monkeypatch.setattr(sys, "argv", [script, "--help"])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(ROOT / script), run_name="__main__")
    assert error.value.code == 0
