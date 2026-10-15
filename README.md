# follicle-detect

Training and benchmarking for follicular-unit detection. One task, done reproducibly: **find every follicular unit in a scalp image and classify its hair count.**

## Status

| Release | Data | Backbone | mAP50 | mAP50-95 | Status |
|---|---|---|---|---|---|
| `fu-v0.1` | `follicle-obb-v0.1` (688 imgs, CC-BY) | yolo11n | — | — | training (Colab notebook below) |
| `fu-v0.1-obb` | same | yolo11n-obb | — | — | planned rc2 |

Weights and eval scripts publish together. Every claim in the table has a `runs/<name>/results.csv` behind it.

## Quick start (free GPU, 90 minutes, no local install)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openhairstack/follicle-detect/blob/main/notebooks/train_colab.ipynb)

The notebook does the whole pipeline: install → download (API key) → convert → train → evaluate → publish weights to Hugging Face (optional). Run it top to bottom; nothing else required.

## Repo layout

```
train.py                 CLI trainer (thin, opinionated wrapper over ultralytics)
infer.py                 single image -> JSON (boxes, classes, conf, stats)
evaluate.py              val/test eval + per-class metrics, appends benchmark row
scripts/               helpers (benchmark markdown writer)
notebooks/train_colab.ipynb   the one-click pipeline above
models/MODEL_CARD.md    template every weights release ships with
```

## Local use (once you have a machine with a GPU or patience)

```bash
pip install -r requirements.txt
python train.py --data ../data-pipeline/data/follicle-obb-v0.1 --model yolo11n.pt --epochs 60 --name fu-v0.1
python evaluate.py --weights runs/fu-v0.1/weights/best.pt --data ../data-pipeline/data/follicle-obb-v0.1
python infer.py --weights runs/fu-v0.1/weights/best.pt --image test.jpg --out result.json
```

## Ground rules (from the Charter)

1. **No weights without data provenance.** A `best.pt` is not mergeable without its dataset manifest hash, train config, and eval CSV.
2. **Report failures.** If a class never learned, the model card says so.
3. **Baselines are baselines.** 688 images is small. The v0.1 numbers will be mediocre and that is the point — they define the floor the community improves on.

## Relations to other work (cited, not copied)

- **YOLO-OHFD** — Zhu et al., oriented-bbox follicle detection (MDPI Appl. Sci. 2025, open access). Code: github.com/Melonpoint/YOLO-OHFD. Our OBB variant follows the paper; we reimplement and cite.
- **Hair-YOLO** — Zhu et al., YOLOv8 follicle detection (Meas. Sci. Technol. 2025). Prior art.
- **DCSB-YOLOX** — structured-light 3-stage detection scheme in transplant robots. Prior art.
- **Fudan follicle dataset** — Gu et al., ICANN 2023, 1,652 images / 20,697 vectors. Research-only.
