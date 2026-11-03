# Demo video script 3 to 5 minutes

## 0:00 to 0:30 Introduction

“This project classifies a handwritten digit as even or odd. Assignment 1 compared Logistic Regression with a CNN. Assignment 2 adds a compact SE-ResNet with channel attention, stronger training augmentation, Grad-CAM explainability, an ablation, and an interactive web application.”

## 0:30 to 1:10 Dataset and method

Show the `data` folders and one or two images. State that the fixed split contains 138 training, 30 validation, and 30 test images, balanced between even and odd. Open the architecture diagram in the report. Briefly explain residual connections and Squeeze-and-Excitation attention.

## 1:10 to 2:00 Results

Open `artifacts/comparison.csv` and the report results table. Compare accuracy, precision, recall, F1, and latency. Then show the ablation: the same SE-ResNet without augmentation versus RandAugment plus affine transforms and random erasing. Emphasize that the test set was not augmented.

## 2:00 to 3:20 Live application

Run `uvicorn app:app --reload`, open the page, and upload or photograph at least one even digit and one odd digit. Point out the prediction, confidence, measured latency, and Grad-CAM overlay. Explain that warm colors show image regions that most influenced the predicted class.

## 3:20 to 4:10 Limitations and ethics

Try one difficult example such as a rotated or faint digit. Explain that the dataset is small and contains limited handwriting styles. State that additional handwriting should be collected with consent, augmentations are disclosed, and the system must not be used for identity or ability judgments.

## 4:10 to 4:30 Reproducibility

Show the README commands and finish with the repository structure, report, and saved model weights.
