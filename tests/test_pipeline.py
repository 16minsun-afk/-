import json

import pytest
import yaml
from PIL import Image

from pipeline.artifacts import check_metrics, verify_bundle
from pipeline.data import check_label, inspect_dataset, sha256


@pytest.fixture
def dataset(tmp_path):
    for split, color in (("train", "red"), ("val", "blue")):
        for kind in ("images", "labels"):
            (tmp_path / kind / split).mkdir(parents=True)
        Image.new("RGB", (32, 32), color).save(tmp_path / "images" / split / "a.png")
        (tmp_path / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    config = tmp_path / "data.yaml"
    config.write_text(yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val", "names": ["item"]}))
    return config


@pytest.mark.parametrize("label", ["", "0 0.5 0.5 1 1", "0 0.5 0.5 0.1 0.2\n"])
def test_valid_labels(label):
    check_label(label, 1)


@pytest.mark.parametrize("label", ["0 0.5", "1 0.5 0.5 1 1", "0.5 0.5 0.5 1 1", "0 nan 0.5 1 1", "0 0.5 0.5 0 1", "0 0.9 0.5 1 1", "-1 0.5 0.5 1 1", "0 0.5 0.5 -1 1"])
def test_invalid_labels(label):
    with pytest.raises(ValueError):
        check_label(label, 1)


def test_dataset_fingerprint(dataset):
    first = inspect_dataset(dataset)
    assert first["counts"] == {"train": 1, "val": 1}
    assert first["sha256"] == inspect_dataset(dataset)["sha256"]
    (dataset.parent / "labels/train/a.txt").write_text("0 0.5 0.5 0.2 0.2")
    assert first["sha256"] != inspect_dataset(dataset)["sha256"]


def test_reject_split_leakage(dataset):
    (dataset.parent / "images/val/a.png").write_bytes((dataset.parent / "images/train/a.png").read_bytes())
    with pytest.raises(ValueError, match="leakage"):
        inspect_dataset(dataset)


def test_reject_missing_label(dataset):
    (dataset.parent / "labels/val/a.txt").unlink()
    with pytest.raises(ValueError, match="label"):
        inspect_dataset(dataset)


def test_reject_download_code(dataset):
    data = yaml.safe_load(dataset.read_text())
    data["download"] = "arbitrary code must not run"
    dataset.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="download"):
        inspect_dataset(dataset)


def test_reject_all_background_validation(dataset):
    (dataset.parent / "labels/val/a.txt").write_text("")
    with pytest.raises(ValueError, match="labeled object"):
        inspect_dataset(dataset)


def test_gate_boundary():
    metrics = dict.fromkeys(["map50_95", "map50", "precision", "recall"], 0.5)
    assert check_metrics(metrics, 0.5)
    assert not check_metrics(metrics, 0.51)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True, "0.8", None])
def test_invalid_metrics(value):
    metrics = dict.fromkeys(["map50_95", "map50", "precision", "recall"], 0.8)
    metrics["map50_95"] = value
    with pytest.raises(ValueError):
        check_metrics(metrics, 0.5)


def test_bundle_checksum_and_server_gate(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"dummy trusted test weights; never loaded")
    manifest = {"schema_version": 1, "split": "val", "model_sha256": sha256(weights),
                "metrics": dict.fromkeys(["map50_95", "map50", "precision", "recall"], 0.7)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert verify_bundle(tmp_path)["split"] == "val"
    with pytest.raises(ValueError, match="requirement"):
        verify_bundle(tmp_path, 0.8)
    weights.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(tmp_path)


def test_reject_synthetic_bundle_in_production(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"ci weights")
    report = {"schema_version": 1, "split": "val", "purpose": "synthetic-ci-only",
              "model_sha256": sha256(tmp_path / "best.pt"),
              "metrics": dict.fromkeys(["map50_95", "map50", "precision", "recall"], 1.0)}
    (tmp_path / "manifest.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Synthetic"):
        verify_bundle(tmp_path)
