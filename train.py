from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn

from src.data import make_loaders
from src.model import build_model


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        p = logits.softmax(1)
        ys.extend(y.numpy().tolist())
        preds.extend(p.argmax(1).cpu().numpy().tolist())
        probs.extend(p[:, 1].cpu().numpy().tolist())
    return {
        "accuracy": accuracy_score(ys, preds),
        "precision": precision_score(ys, preds, zero_division=0),
        "recall": recall_score(ys, preds, zero_division=0),
        "f1": f1_score(ys, preds, zero_division=0),
        "labels": ys,
        "predictions": preds,
        "odd_probabilities": probs,
    }


def train_one(config, augment: bool, tag: str):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, val_loader, test_loader, class_to_idx = make_loaders(
        config.data, config.batch_size, augment
    )
    model = build_model().to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    history, best_f1, best_state, patience_left = [], -1.0, None, config.patience

    for epoch in range(1, config.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            seen += len(y)
        scheduler.step()
        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": running / seen, "val_accuracy": val["accuracy"], "val_f1": val["f1"]}
        history.append(row)
        print(tag, row)
        if val["f1"] > best_f1 + 1e-5:
            best_f1 = val["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best_state)
    model.to(device)
    test = evaluate(model, test_loader, device)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "class_to_idx": class_to_idx, "tag": tag}, output / f"se_resnet_{tag}.pt")
    with (output / f"history_{tag}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)

    # Warm-up and per-image latency on the actual test environment.
    model.eval()
    sample = next(iter(test_loader))[0][:1].to(device)
    for _ in range(10): model(sample)
    if device.type == "mps": torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(100): model(sample)
    if device.type == "mps": torch.mps.synchronize()
    latency = (time.perf_counter() - start) * 10
    result = {k: v for k, v in test.items() if k not in {"labels", "predictions", "odd_probabilities"}}
    result.update({"tag": tag, "augmentation": augment, "latency_ms": latency, "device": str(device), "epochs_ran": len(history)})
    (output / f"metrics_{tag}.json").write_text(json.dumps(result, indent=2))
    (output / f"predictions_{tag}.json").write_text(json.dumps(test, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--output", default="artifacts")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mode", choices=["augmented", "plain", "both"], default="both")
    args = p.parse_args()
    seed_everything()
    results = []
    if args.mode in {"plain", "both"}: results.append(train_one(args, False, "plain"))
    if args.mode in {"augmented", "both"}: results.append(train_one(args, True, "augmented"))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
