"""Single-image inference -> JSON.

Deliberately identical in interface to the demo's detector so a model that works
here works in the Space (and vice versa).

Usage:
    python infer.py --weights best.pt --image scalp.jpg --conf 0.25 --out result.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CANON = ["fu1", "fu2", "fu3", "fu4", "fu5"]


def resolve_weights(weights: str) -> Path:
    p = Path(weights)
    if p.exists():
        return p
    if "/" in weights or weights.endswith(".pt"):
        from huggingface_hub import snapshot_download
        p = Path(snapshot_download(repo_id=weights, revision="main"))
        cands = sorted(p.rglob("*.pt"))
        if not cands:
            raise SystemExit(f"no .pt found in HF repo {weights}")
        return cands[-1]
    raise SystemExit(f"weights not found: {weights}")


def load_model(weights: str):
    from ultralytics import YOLO

    return YOLO(str(resolve_weights(weights)))


def predict(model, image_path: str, conf: float = 0.25, imgsz: int = 640) -> dict:
    results = model.predict(source=image_path, conf=conf, imgsz=imgsz, verbose=False)
    r = results[0]
    names = r.names
    detections = []
    for box in r.boxes:
        cls_id = int(box.cls.item())
        detections.append({
            "class_id": cls_id,
            "class": names.get(cls_id, str(cls_id)),
            "confidence": round(float(box.conf.item()), 4),
            "xyxy": [round(v, 1) for v in box.xyxy.cpu().numpy().tolist()[0]],
        })
    counts: dict[str, int] = {}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1
    return {
        "image": str(image_path),
        "n_detections": len(detections),
        "class_counts": counts,
        "detections": detections,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="local .pt path or an HF repo id")
    ap.add_argument("--image", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = load_model(args.weights)
    out = predict(model, args.image, args.conf)
    text = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}: {out['n_detections']} detections {out['class_counts']}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
