# follicle-detect

> **Train a computer to see hair follicles — in one click, on free hardware, with the scores
> published whether they're good or not.**

### In plain English

A robotic hair transplant can only move what it can see: this project trains the "eyes." You
give it labeled scalp photos — bundles marked and counted — and it learns to find those
bundles on its own. Everything needed to do that is one notebook and a free cloud GPU.

**No wet lab, no special hardware, no $200k machine.** The whole point: the "eyes" half of
robotic hair restoration should be buildable by anyone.

New to the vocabulary? Start here: [GLOSSARY.md](https://github.com/openhairstack/governance/blob/main/GLOSSARY.md).

---

## Status & Benchmark Floor

| Release | Data | Backbone | mAP50 | mAP50-95 | Status |
|---|---|---|---|---|---|
| `fu-v0.1` | `follicle-v0.1` (688 imgs, CC-BY; axis-aligned) | yolo11n | 0.001 | 0.000 | **Released** (`fu2` recall: 24.9%) |
| `fu-v0.1-obb` | same source, oriented boxes | yolo11n-obb | — | — | Planned (v0.3 roadmap) |

Weights and evaluation scripts publish together. Our v0.1 model establishes the first open, reproducible benchmark floor. Full per-class metrics and confusion matrices are documented in [`models/MODEL_CARD.md`](models/MODEL_CARD.md).

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
python train.py --data ../data-pipeline/data/follicle-v0.1 --model yolo11n.pt --epochs 60 --name fu-v0.1
python evaluate.py --weights runs/fu-v0.1/weights/best.pt --data ../data-pipeline/data/follicle-v0.1
python infer.py --weights runs/fu-v0.1/weights/best.pt --image test.jpg --out result.json
```

## Ground rules (from the Charter)

1. **No weights without data provenance.** A `best.pt` isn't mergeable without its dataset
   manifest hash, train config, and eval CSV.
2. **Report failures.** If a class never learned, the model card says so.
3. **Baselines are baselines.** 688 images is small. The v0.1 numbers will be mediocre and
   that is the point — they define the floor the community improves on.

## Prior work (cited, not copied)

- **YOLO-OHFD** — oriented-bbox follicle detection (MDPI Appl. Sci. 2025, open access; code
  released without a dataset). Prior art; our v0.3 OBB variant is planned to follow it —
  we reimplement and cite.
- **Hair-YOLO** — Zhu et al., hair-count-per-follicle detection (Meas. Sci. Technol. 2025).
- **Fudan follicle dataset** — Gu et al., ICANN 2023, 1,652 clinical images, research-only.
- **Roboflow CC-BY sets** — the small license-clean sets we train on today.

---

### Questions & Community Support

Have questions about training, fine-tuning, or benchmarking?
* **Connect with Salik Shah:** [DM @salik on X](https://x.com/salik)
* **Discussions & Issues:** [Open Hair Stack Community](https://github.com/orgs/openhairstack/discussions)
