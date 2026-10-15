# follicle-detect

> **Train a computer to see hair follicles — in one click, on free hardware, with the scores
> published whether they're good or not.**

### In plain English

A robotic hair transplant can only move what it can see: this project trains the "eyes." You
give it labeled scalp photos — bundles marked and counted — and it learns to find those
bundles on its own. Everything needed to do that is one notebook and a free cloud GPU.

**No wet lab, no special hardware, no $200k machine.** The whole point: the "eyes" half of
robotic hair restoration should be buildable by anyone.

New to the vocabulary? Start here: [GLOSSARY.md](../governance/GLOSSARY.md).

---

## Status

| Release | Data | Backbone | mAP50 | mAP50-95 | Status |
|---|---|---|---|---|---|
| `fu-v0.1` | `follicle-obb-v0.1` (688 imgs, CC-BY) | yolo11n | — | — | training (Colab notebook below) |
| `fu-v0.1-obb` | same | yolo11n-obb | — | — | planned |

Weights and eval scripts publish together. Every number in the table has a `results.csv`
behind it. (Blanks until the baseline finishes training — we don't fill in numbers we
haven't measured.)

## Quick start (free GPU, ~90 minutes, no local install)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openhairstack/follicle-detect/blob/main/notebooks/train_colab.ipynb)

The notebook does the whole pipeline: install → download → convert → train → evaluate →
publish weights to Hugging Face (optional). Run top to bottom; nothing else required. If you
don't have accounts yet, it tells you exactly which ones (all free).

## Technical notes

```
train.py                 CLI trainer (records provenance by default)
infer.py                 single image -> JSON (boxes, classes, conf, stats)
evaluate.py              val/test eval + per-class metrics
notebooks/train_colab.ipynb   the one-click pipeline
models/MODEL_CARD.md     template every weights release ships with
```

```bash
# local
pip install -r requirements.txt
python train.py --data ../data-pipeline/data/follicle-obb-v0.1 --model yolo11n.pt --epochs 60 --name fu-v0.1
python evaluate.py --weights runs/fu-v0.1/weights/best.pt --data ../data-pipeline/data/follicle-obb-v0.1
python infer.py --weights runs/fu-v0.1/weights/best.pt --image test.jpg --out result.json
```

## Ground rules (from the Charter)

1. **No weights without data provenance.** A `best.pt` isn't mergeable without its dataset
   manifest hash, train config, and eval CSV.
2. **Report failures.** If a class never learned, the model card says so.
3. **Baselines are baselines.** 688 images is small. The v0.1 numbers will be mediocre and
   that is the point — they define the floor the community improves on.

## Prior work (cited, not copied)

- **YOLO-OHFD** — Zhu et al., oriented-bbox follicle detection (MDPI Appl. Sci. 2025, open
  access; code released without dataset). Our OBB variant follows the paper; we reimplement
  and cite.
- **Hair-YOLO** — YOLOv8 follicle detection (Meas. Sci. Technol. 2025).
- **Fudan follicle dataset** — Gu et al., ICANN 2023, 1,652 clinical images, research-only.
- **Roboflow CC-BY sets** — the small license-clean sets we train on today.
