"""Bind a validation report to exact model bytes; not a digital signature."""
import json
import math
from pathlib import Path

from pipeline.data import sha256


def check_metrics(metrics: dict, minimum: float) -> bool:
    if not math.isfinite(minimum) or not 0 <= minimum <= 1:
        raise ValueError("Minimum mAP50-95 must be between zero and one")
    for key in ("map50_95", "map50", "precision", "recall"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError(f"Missing or invalid metric: {key}")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Non-finite or out-of-range metric: {key}")
    return metrics["map50_95"] >= minimum


def verify_bundle(directory: Path, minimum: float = 0.5, *, allow_smoke: bool = False) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("split") not in {"val", "test"}:
        raise ValueError("Unsupported model manifest")
    if manifest.get("purpose") == "synthetic-ci-only" and not allow_smoke:
        raise ValueError("Synthetic CI models cannot be deployed to production")
    if not check_metrics(manifest["metrics"], minimum):
        raise ValueError("Model does not meet the server's mAP50-95 requirement")
    if sha256(directory / "best.pt") != manifest.get("model_sha256"):
        raise ValueError("Model checksum mismatch")
    return manifest
