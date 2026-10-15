"""Train a follicular-unit detector.

Thin, opinionated wrapper over ultralytics YOLO training. The point of this file
is not features; it is that every training run records its provenance.

Usage:
    python train.py --data <dataset.yaml or dir> --model yolo11n.pt --epochs 60 --name fu-v0.1
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def dataset_digest(data: Path) -> dict | None:
    manifest = data / "manifest.json" if data.is_dir() else None
    if manifest and manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8"))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset dir or dataset.yaml")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="fu-v0.1")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", default="", help="extra args passed verbatim to YOLO()")
    args = ap.parse_args()

    data_path = Path(args.data)
    if data_path.is_dir():
        data_yaml = data_path / "dataset.yaml"
        if not data_yaml.exists():
            raise SystemExit(f"no dataset.yaml in {data_path}; run data-pipeline/convert first")
        args.data = str(data_yaml)

    cmd = [
        "yolo", "detect", "train",
        f"data={args.data}",
        f"model={args.model}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"name={args.name}",
        f"project={args.project}",
        f"seed={args.seed}",
        "exist_ok=True",
        "patience=25",
    ] + [a for a in args.extra.split() if a]

    print("running:", " ".join(cmd))
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("ultralytics not installed -> pip install -r requirements.txt")
        return 1

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=args.project,
        seed=args.seed,
        exist_ok=True,
        patience=25,
        device=0,
    )

    run_dir = Path(args.project) / args.name
    provenance = {
        "command": " ".join(cmd),
        "dataset_manifest": dataset_digest(data_path),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "ultralytics": subprocess.run(
            ["python", "-c", "import ultralytics; print(ultralytics.__version__)"],
            capture_output=True, text=True,
        ).stdout.strip(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str), encoding="utf-8"
    )
    print(f"\ndone. weights: {run_dir / 'weights' / 'best.pt'}")
    print(f"provenance: {run_dir / 'provenance.json'}  (attach to every model card)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
