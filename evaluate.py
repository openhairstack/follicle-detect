"""Evaluate a trained model and append a benchmark row.

Usage:
    python evaluate.py --weights runs/fu-v0.1/weights/best.pt --data ../data-pipeline/data/follicle-obb-v0.1 --split val
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="runs/eval-results.json")
    args = ap.parse_args()

    from ultralytics import YOLO

    data_path = Path(args.data)
    data_yaml = data_path if data_path.suffix == ".yaml" else data_path / "dataset.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"dataset yaml not found: {data_yaml}")

    model = YOLO(args.weights)
    metrics = model.val(data=str(data_yaml), split=args.split, imgsz=args.imgsz, verbose=False)

    per_class: dict[str, dict] = {}
    try:
        names = metrics.names
        for i, cls_name in names.items():
            row = {
                "mAP50": round(float(metrics.box.maps[i]), 4),
                "precision": round(float(metrics.box.p[i]), 4),
                "recall": round(float(metrics.box.r[i]), 4),
            }
            per_class[str(cls_name)] = row
    except Exception as exc:  # noqa: BLE001
        print(f"per-class metrics unavailable: {exc}")

    summary = {
        "weights": args.weights,
        "data": str(data_yaml),
        "split": args.split,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "box": {
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
        },
        "per_class": per_class,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        history = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        history = {"runs": []}
    history["runs"].append(summary)
    out_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("\n=== eval summary ===")
    print(json.dumps(summary["box"], indent=2))
    print("\nper-class:")
    for k, v in per_class.items():
        print(f"  {k:5s} mAP50={v['mAP50']:.3f}  P={v['precision']:.3f}  R={v['recall']:.3f}")
    print(f"\nappended to {out_path} — copy this row into README's benchmark table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
