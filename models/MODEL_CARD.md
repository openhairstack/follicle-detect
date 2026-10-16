# Model Card — follicle-fu-v0.1

Baseline follicular-unit detection model for trichoscopy images.

---

language: en
license: apache-2.0
tags: [hair, follicle, object-detection, yolo, hair-restoration, research-only]
datasets: openhairstack/follicle-v0.1

## Model Details

- **Architecture:** YOLO11n (v0.1, axis-aligned detection)
- **Weights files:**
  - `best.pt` (PyTorch, sha256: `568c5a77629c125ea3c0e04c2d9bf7f698103e5ea98451813e24256066d1d7fe`)
  - `model.onnx` (ONNX Runtime Web client-side, sha256: `9c0a09869735d99501bfdedad957ffa22f3be13e9f8cf7ffe7df6f08814dd7b2`)
- **Task:** Follicular-unit detection + hair-count classification (`fu1`–`fu5`; exact counts, `fu5` = 5+)
- **Training Platform:** Google Colab (Tesla T4 GPU, 60 epochs, imgsz=640, batch=16, seed=0)
- **Dataset:** `follicle-v0.1` (688 source trichoscopic images, CC BY 4.0; Roboflow Universe `ranaudio/hair-follicle-detection-gjgak`)

## Evaluation Metrics (val split: 122 images, 1,013 instances)

| mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|
| **0.001** | **0.000** | **0.001** | **0.050** |

### Per-Class Performance

| Class | Label | mAP50-95 | Precision | Recall | Val Instances |
|---|---|---|---|---|---|
| `fu1` | 1 hair | 0.000 | 0.000 | 0.000 | 277 |
| `fu2` | 2 hairs | 0.001 | 0.004 | **0.249** | 587 |
| `fu3` | 3 hairs | 0.000 | 0.000 | 0.000 | 133 |
| `fu4` | 4 hairs | 0.000 | 0.000 | 0.000 | 15 |
| `fu5` | 5+ hairs | 0.000 | 0.000 | 0.000 | 1 |

## Intended Uses & Limitations

- **Intended Use:** Open research, academic benchmarking, educational demonstrations, and client-side browser testing.
- **Not for:** Clinical diagnosis, medical triage, or surgical planning.
- **Known Weaknesses & Honest Analysis:**
  - **Class Imbalance & Dominance:** The training distribution is heavily skewed toward 2-hair units (`fu2`), where the model learned 24.9% recall. Minor classes (`fu3`, `fu4`, `fu5`) suffer from extreme sparsity.
  - **Resolution & Scale:** Follicles in 640px crops are microscopic. Future v0.2 iterations will test higher input resolutions (`imgsz=1280`) and oriented bounding boxes (`yolo11n-obb`).
  - **Lighting & Skin/Hair Diversity:** The 688 images do not cover the full range of hair curl patterns, grey hair, or diverse Fitzpatrick skin tones.

## Training Data Provenance

Licensed under CC BY 4.0 (`ranaudio/hair-follicle-detection-gjgak`). See [`data-pipeline/DATASETS.md`](https://github.com/openhairstack/data-pipeline/blob/main/DATASETS.md) for full attribution.

## Ethical & Safety Notes

No patient-identifiable data in the training set. This model is part of an open research project and comes with the disclaimer in `scalp-analyzer/MEDICAL_DISCLAIMER.md`.

## Contact & Discussions

A project by Salik Shah:
- **Contact on X:** [@salik](https://x.com/salik)
- **Discussions:** [GitHub Discussions](https://github.com/orgs/openhairstack/discussions)
