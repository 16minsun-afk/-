"""Evaluate an actual checkpoint and export a gated, checksummed model bundle."""
import argparse
import importlib.metadata
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pipeline.artifacts import check_metrics, verify_bundle
from pipeline.data import inspect_dataset, normalized_config, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data.local.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-map", type=float, default=0.5)
    parser.add_argument("--out", required=True, help="New directory, e.g. artifacts/v1")
    parser.add_argument("--report", default="reports/validation.json")
    args = parser.parse_args()
    check_metrics(dict.fromkeys(("map50_95", "map50", "precision", "recall"), 0.0), args.min_map)
    model_path, output = Path(args.model).resolve(), Path(args.out)
    if not model_path.is_file() or output.exists():
        raise ValueError("Model must exist; output must be a NEW, unused version directory")
    info = inspect_dataset(args.data)
    if args.split not in info["data"]:
        raise ValueError(f"Dataset has no {args.split} split")
    from ultralytics import YOLO

    model_digest = sha256(model_path)
    model = YOLO(str(model_path))  # Only load checkpoints you created or otherwise trust.
    result = model.val(
        data=str(normalized_config(info)), split=args.split, device=args.device,
        imgsz=args.imgsz, workers=0, plots=True, project="runs/val",
    )
    metrics = {"map50_95": float(result.box.map), "map50": float(result.box.map50),
               "precision": float(result.box.mp), "recall": float(result.box.mr)}
    passed = check_metrics(metrics, args.min_map)
    if sha256(model_path) != model_digest:
        raise RuntimeError("Checkpoint changed during evaluation; validate a stable copy")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip() or "unknown"
    manifest = {
        "schema_version": 1, "model_sha256": model_digest,
        "dataset_sha256": info["sha256"], "counts": info["counts"], "split": args.split,
        "metrics": metrics, "minimum_map50_95": args.min_map, "passed": passed,
        "imgsz": args.imgsz, "classes": info["data"]["names"], "code_revision": revision,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "versions": {name: importlib.metadata.version(name) for name in ("ultralytics", "torch")},
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    print(text)
    if not passed:
        raise SystemExit("Quality gate failed; no deployable bundle was created")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(model_path, output / "best.pt")
    (output / "manifest.json").write_text(text, encoding="utf-8")
    verify_bundle(output, args.min_map)
    print(f"Validated model bundle: {output}")


if __name__ == "__main__":
    main()
