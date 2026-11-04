from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from src.inference import ParityPredictor, image_to_png_bytes


ROOT = Path(__file__).resolve().parent
WEIGHTS = Path(os.getenv("PARITY_WEIGHTS", ROOT / "artifacts/se_resnet_augmented.pt"))
app = FastAPI(title="Handwritten Parity Explorer")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")
predictor = ParityPredictor(WEIGHTS)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, image: UploadFile = File(...)):
    raw = await image.read()
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    result = predictor.predict(pil)
    source = base64.b64encode(image_to_png_bytes(pil)).decode()
    heatmap = base64.b64encode(image_to_png_bytes(result["overlay"])).decode()
    context = {"result": result, "source": source, "heatmap": heatmap}
    return templates.TemplateResponse(request, "index.html", context)
