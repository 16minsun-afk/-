"""Strict YOLO detection preflight; directory-based datasets only."""
import hashlib
import json
import math
from pathlib import Path

import yaml
from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_label(text: str, classes: int) -> None:
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue  # Empty label files are allowed for background images.
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Label line {number}: expected class x y width height")
        values = [float(value) for value in parts]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Label line {number}: non-finite value")
        cls, x, y, width, height = values
        if not cls.is_integer() or not 0 <= cls < classes:
            raise ValueError(f"Label line {number}: invalid class ID")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"Label line {number}: coordinates must be normalized")
        if min(x - width / 2, y - height / 2) < -1e-5:
            raise ValueError(f"Label line {number}: box outside image")
        if max(x + width / 2, y + height / 2) > 1 + 1e-5:
            raise ValueError(f"Label line {number}: box outside image")


def inspect_dataset(config: str | Path) -> dict:
    config = Path(config).resolve()
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "download" in raw:
        raise ValueError("Use a local YAML mapping without a download script")
    names = raw.get("names")
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError("Class IDs must be contiguous integers starting at zero")
        names = [names[index] for index in range(len(names))]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError("names must be a non-empty class-name list or integer-keyed mapping")
    if len(set(names)) != len(names):
        raise ValueError("Class names must be unique")
    root = Path(raw.get("path", "."))
    root = (config.parent / root).resolve()
    normalized = {"path": str(root), "names": names}
    records, seen, counts = [], {}, {}
    for split in ("train", "val", "test"):
        if split not in raw:
            if split != "test":
                raise ValueError(f"Missing {split} split")
            continue
        if not isinstance(raw[split], str):
            raise ValueError("Each split must reference one local image directory")
        directory = (root / raw[split]).resolve()
        if not directory.is_relative_to(root) or not directory.is_dir():
            raise ValueError(f"{split}: image directory must exist inside dataset root")
        images = sorted(p for p in directory.rglob("*") if p.suffix.lower() in EXTENSIONS)
        if not images:
            raise ValueError(f"{split}: no supported images")
        boxes = 0
        for image in images:
            if not image.resolve().is_relative_to(root):
                raise ValueError("Image symlinks must stay inside the dataset root")
            with Image.open(image) as opened:
                opened.verify()
            parts = list(image.relative_to(root).parts)
            if "images" not in parts:
                raise ValueError("Image paths must contain an images directory")
            index = len(parts) - 1 - parts[::-1].index("images")
            parts[index] = "labels"
            label = root.joinpath(*parts).with_suffix(".txt")
            if not label.is_file() or not label.resolve().is_relative_to(root):
                raise ValueError(f"Missing or unsafe label: {label}")
            text = label.read_text(encoding="utf-8-sig")
            check_label(text, len(names))
            boxes += sum(bool(line.strip()) for line in text.splitlines())
            digest = sha256(image)
            if digest in seen and seen[digest] != split:
                raise ValueError(f"Data leakage: identical image in {seen[digest]} and {split}")
            seen[digest] = split
            records.append([split, image.relative_to(root).as_posix(), digest, sha256(label)])
        if boxes == 0:
            raise ValueError(f"{split}: at least one labeled object is required")
        normalized[split] = str(directory)
        counts[split] = len(images)
    fingerprint = hashlib.sha256(json.dumps([names, records], sort_keys=True).encode()).hexdigest()
    return {"data": normalized, "sha256": fingerprint, "counts": counts}


def normalized_config(info: dict) -> Path:
    path = Path("runs/configs") / f"{info['sha256']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(info["data"], allow_unicode=True), encoding="utf-8")
    return path.resolve()
