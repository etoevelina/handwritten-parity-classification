from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="legacy/logistic_regression_model.pkl")
    p.add_argument("--data", default="data/test")
    p.add_argument("--output", default="artifacts/baseline_metrics.json")
    args = p.parse_args()
    model = joblib.load(args.model)
    xs, ys = [], []
    for label, cls in enumerate(["even", "odd"]):
        for path in sorted((Path(args.data) / cls).glob("*.png")):
            xs.append(np.asarray(Image.open(path).convert("L").resize((64, 64)), dtype=np.float32).reshape(-1) / 255.0)
            ys.append(label)
    x = np.stack(xs)
    pred = model.predict(x)
    for _ in range(10): model.predict(x[:1])
    start = time.perf_counter()
    for _ in range(1000): model.predict(x[:1])
    latency = (time.perf_counter() - start)
    result = {
        "accuracy": accuracy_score(ys, pred),
        "precision": precision_score(ys, pred, zero_division=0),
        "recall": recall_score(ys, pred, zero_division=0),
        "f1": f1_score(ys, pred, zero_division=0),
        "latency_ms": latency,
        "samples": len(ys),
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
