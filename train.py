"""Windows GPU training entry point with an explicit spawn-safe main guard."""
import argparse
import json
from datetime import datetime, timezone

from pipeline.data import inspect_dataset, normalized_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.local.yaml")
    parser.add_argument("--model", default="yolo26m.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default=datetime.now(timezone.utc).strftime("custom_%Y%m%d_%H%M%S"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    info = inspect_dataset(args.data)
    print(json.dumps({"counts": info["counts"], "dataset_sha256": info["sha256"]}, indent=2))
    if args.check_only:
        return
    import torch
    from ultralytics import YOLO

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install the cu128 wheels or use --device cpu")
    model = YOLO(args.model)
    model.train(
        data=str(normalized_config(info)), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, workers=args.workers, patience=30,
        amp=args.device != "cpu", seed=42, deterministic=True,
        project="runs/train", name=args.name, exist_ok=False,
    )
    directory = model.trainer.save_dir
    (directory / "dataset_manifest.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Best model: {model.trainer.best}")
    print("Next: validate.py with this exact best.pt and the same dataset version.")


if __name__ == "__main__":
    main()
