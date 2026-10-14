"""Tiny synthetic end-to-end CPU test. Its metrics are NOT real-world accuracy."""
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from pipeline.data import inspect_dataset, normalized_config, sha256


def main():
    from ultralytics import YOLO

    root = Path(".ci/dataset").resolve()
    rng = np.random.default_rng(42)
    for split, count in (("train", 4), ("val", 2)):
        for part in ("images", "labels"):
            (root / part / split).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            pixels = rng.integers(0, 80, (64, 64, 3), dtype=np.uint8)
            pixels[16:48, 16:48] = 230
            Image.fromarray(pixels).save(root / "images" / split / f"{index}.png")
            (root / "labels" / split / f"{index}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    config = root / "data.yaml"
    config.write_text(yaml.safe_dump({"path": str(root), "train": "images/train",
                                     "val": "images/val", "names": ["square"]}))
    info = inspect_dataset(config)
    model = YOLO("yolo26n.yaml")
    model.train(data=str(normalized_config(info)), epochs=1, imgsz=64, batch=2,
                device="cpu", workers=0, amp=False, pretrained=False, plots=False,
                optimizer="SGD", close_mosaic=0, project=".ci/runs", name="smoke", exist_ok=True)
    bundle = Path(".ci/bundle")
    bundle.mkdir(exist_ok=True)
    shutil.copyfile(model.trainer.best, bundle / "best.pt")
    reloaded = YOLO(str(bundle / "best.pt"))
    result = reloaded.val(data=str(normalized_config(info)), device="cpu", imgsz=64,
                          workers=0, plots=False, project=".ci/runs", name="val", exist_ok=True)
    metrics = {"map50_95": float(result.box.map), "map50": float(result.box.map50),
               "precision": float(result.box.mp), "recall": float(result.box.mr)}
    manifest = {"schema_version": 1, "split": "val", "model_sha256": sha256(bundle / "best.pt"),
                "dataset_sha256": info["sha256"], "metrics": metrics, "imgsz": 64,
                "purpose": "synthetic-ci-only"}
    (bundle / "manifest.json").write_text(json.dumps(manifest, allow_nan=False))
    assert len(reloaded.predict(str(root / "images/val/0.png"), device="cpu", verbose=False)) == 1


if __name__ == "__main__":
    main()
