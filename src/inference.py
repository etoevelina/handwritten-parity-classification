from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as F

from src.data import MEAN, STD
from src.model import build_model


class ParityPredictor:
    def __init__(self, weights: str | Path):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        self.model = build_model()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval().to(self.device)
        self.activations = None
        self.gradients = None
        target = self.model.features[-1].conv2
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output):
        self.activations = output

    def _save_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0]

    @staticmethod
    def prepare(image: Image.Image):
        gray = ImageOps.grayscale(image)
        gray = ImageOps.autocontrast(gray).resize((64, 64))
        x = F.to_tensor(gray)
        x = F.normalize(x, MEAN, STD).unsqueeze(0)
        return gray, x

    def predict(self, image: Image.Image):
        gray, x = self.prepare(image)
        x = x.to(self.device)
        start = time.perf_counter()
        logits = self.model(x)
        predicted = int(logits.argmax(1).item())
        self.model.zero_grad(set_to_none=True)
        logits[0, predicted].backward()
        if self.device.type == "mps": torch.mps.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000
        probs = logits.softmax(1)[0].detach().cpu().numpy()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(1, keepdim=True).relu()
        cam = torch.nn.functional.interpolate(cam, (64, 64), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        heat = np.zeros((64, 64, 3), dtype=np.uint8)
        heat[..., 0] = (255 * cam).astype(np.uint8)
        heat[..., 1] = (180 * np.clip(1 - np.abs(cam - 0.5) * 2, 0, 1)).astype(np.uint8)
        base = np.asarray(gray.convert("RGB"), dtype=np.float32)
        overlay = Image.fromarray(np.clip(0.58 * base + 0.42 * heat, 0, 255).astype(np.uint8)).resize((384, 384))
        return {
            "label": "odd" if predicted else "even",
            "confidence": float(probs[predicted]),
            "probabilities": {"even": float(probs[0]), "odd": float(probs[1])},
            "latency_ms": latency_ms,
            "overlay": overlay,
        }


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO(); image.save(buffer, format="PNG"); return buffer.getvalue()
