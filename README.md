# Handwritten Number Parity Classification

Assignment 2 continues the original even/odd handwritten digit project. This repository keeps the original dataset and the Assignment 1 Logistic Regression/CNN artifacts, then adds a compact SE-ResNet, advanced augmentation, Grad-CAM explainability, an ablation experiment, latency measurement, and a camera-friendly web demo.

made by Evelina and Birkhanym

## Results

See `artifacts/comparison.csv` for the measured results on the unchanged 30-image test set. Because the dataset is very small (198 images total), results should be reported with the exact split and not generalized to all handwriting.

## Project layout

- `data/`: original train/validation/test split (138/30/30 images)
- `legacy/`: Assignment 1 models, figures, results, and report
- `src/`: model, data pipeline, inference, and Grad-CAM code
- `train.py`: reproducible SE-ResNet training and augmentation ablation
- `evaluate_baseline.py`: evaluation of the Assignment 1 Logistic Regression model
- `app.py`: FastAPI upload/camera demo
- `artifacts/`: weights, metrics, curves, confusion matrices, and Grad-CAM examples
- `docs/`: report and demo-video script

## Setup and run

Python 3.10-3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py --mode both
python evaluate_baseline.py
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`. On a phone connected to the same network, start with `uvicorn app:app --host 0.0.0.0` and open the computer's local IP address.

## Reproducibility

Training uses seed 42, the original fixed split, AdamW, cosine learning-rate decay, label smoothing, and early stopping on validation F1. The ablation changes only the augmentation flag. `train.py` chooses Apple MPS when available and otherwise CPU. Exact package versions can be recorded with `pip freeze > environment-lock.txt`.

## Ethics and limitations

The images are self-collected handwriting. No face, identity, or sensitive biometric data is required. Obtain consent before adding handwriting from other people. The demo is educational, not an identity or ability assessment. It can fail on cropped, multi-digit, ambiguous, or unfamiliar inputs. No synthetic image is represented as human-collected data; augmentations are generated online and are never included in the test set.
