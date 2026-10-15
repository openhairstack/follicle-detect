# Model card template — every published `best.pt` ships one of these

Copy to `models/MODEL_CARD-fu-vX.md` alongside the weights. Tag it into the HF repo's README too.

---

language: en
license: apache-2.0
tags: [hair, follicle, object-detection, yolo, hair-restoration, research-only]
datasets: openhairstack/follicle-obb-v0.1

## Model

- **Architecture:** yolo11n (or yolo11n-obb)
- **Weights file:** `best.pt` — sha256: __________
- **Task:** follicular-unit detection + hair-count class (fu1–fu5)
- **Trained:** date · hardware · runtime
- **Run provenance:** `provenance.json` from `train.py` (dataset manifest, seed, versions)

## Metrics (val split, from evaluate.py)

| mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|
| | | | |

Per-class:

| class | mAP50 | P | R | n_val_instances |
|---|---|---|---|---|
| fu1 | | | | |
| fu2 | | | | |

## Intended uses & limitations

- **Intended:** research, benchmarking, education, the Open Hair Stack demo.
- **Not for:** diagnosis, clinical decisions, surgical planning for real patients, or any use where accuracy expectations exceed the metrics above.
- **Known weaknesses:** (list honestly: small dataset, capture conditions, texture/curl coverage, occlusion behavior, `fu5` recall, etc.)

## Training data

`follicle-obb-v0.1` (CC BY 4.0, 688 source images) — see `data-pipeline/DATASETS.md` for per-source attribution. Data licenses propagate: cite upstream too.

## Ethical & safety notes

No patient-identifiable data in the training set. This model is part of an open research project and comes with the disclaimer in `scalp-analyzer/MEDICAL_DISCLAIMER.md`.

## How to cite

Cite the upstream datasets (from DATASETS.md) first, then this model:

```bibtex
@misc{openhairstack_fu_vX,
  title  = {Follicle-Unit Detector vX, Open Hair Stack},
  author = {Open Hair Stack contributors},
  year   = {2026},
  url    = {https://huggingface.co/openhairstack/follicle-fu-vX}
}
```
