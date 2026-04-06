"""
FastAPI Backend — Deep Learning Inference Server
=================================================
Serves the frontend UI and exposes REST API endpoints
for image, text, and multimodal classification.

Run:
    uvicorn extensions.app_demo.main:app --reload --port 8000

Or directly:
    python -m extensions.app_demo.main
"""

import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .inference import ImageClassifier, TextClassifier, MultimodalClassifier

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("app_demo")

# ── Globals (populated on startup) ──────────────────────────────────
image_clf: ImageClassifier | None = None
text_clf: TextClassifier | None = None
multimodal_clf: MultimodalClassifier | None = None

STATIC_DIR    = Path(__file__).parent / "static"
DOCS_DEMO_DIR = Path(__file__).resolve().parents[2] / "docs" / "demo"


# ── Lifespan — load models at startup ──────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    global image_clf, text_clf, multimodal_clf
    logger.info("🚀 Starting inference server …")

    image_clf = ImageClassifier()
    text_clf = TextClassifier()
    multimodal_clf = MultimodalClassifier()

    # Pre-load all models at startup so first request is instant
    image_clf.load_model("resnet18")
    image_clf.load_model("vit")
    text_clf.load_model("lstm")
    text_clf.load_model("distilbert")
    multimodal_clf.load_model("clip_zero_shot")
    multimodal_clf.load_model("clip_few_shot")

    logger.info("✅ All classifiers ready.")
    yield
    logger.info("🛑 Shutting down.")


# ── App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="DL Inference Demo",
    description="CO3133 Assignment 1 — Classification Demo (Image / Text / Multimodal)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        # GitHub Pages URL — update <username> and <repo> if different
        "https://tangcongthanhcse.github.io",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Serve Frontend ──────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Serve docs/demo (GitHub Pages version) locally at /demo ─────────
if DOCS_DEMO_DIR.exists():
    app.mount("/demo", StaticFiles(directory=str(DOCS_DEMO_DIR), html=True), name="demo")


@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


# ── Health Check ────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "image_models": image_clf.available_models if image_clf else [],
        "text_models": text_clf.available_models if text_clf else [],
        "multimodal_models": multimodal_clf.available_models if multimodal_clf else [],
    }


# ── API: Image Classification ──────────────────────────────────────
@app.post("/api/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    model: str = Form("resnet18"),
):
    if not image_clf:
        raise HTTPException(503, "Image classifier not loaded")
    if model not in image_clf.available_models:
        raise HTTPException(400, f"Unknown model '{model}'. Choose from: {image_clf.available_models}")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Empty file")

    result = image_clf.predict(image_bytes, model)
    return result


# ── API: Text Classification ───────────────────────────────────────
@app.post("/api/predict/text")
async def predict_text(
    text: str = Form(...),
    model: str = Form("distilbert"),
):
    if not text_clf:
        raise HTTPException(503, "Text classifier not loaded")
    if model not in text_clf.available_models:
        raise HTTPException(400, f"Unknown model '{model}'. Choose from: {text_clf.available_models}")

    text = text.strip()
    if not text:
        raise HTTPException(400, "Text cannot be empty")

    result = text_clf.predict(text, model)
    return result


# ── API: Multimodal Classification ─────────────────────────────────
@app.post("/api/predict/multimodal")
async def predict_multimodal(
    file: UploadFile = File(...),
    text: str = Form(...),
    model: str = Form("clip_zero_shot"),
):
    if not multimodal_clf:
        raise HTTPException(503, "Multimodal classifier not loaded")
    if model not in multimodal_clf.available_models:
        raise HTTPException(400, f"Unknown model '{model}'. Choose from: {multimodal_clf.available_models}")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Empty image file")

    text = text.strip()
    if not text:
        raise HTTPException(400, "Text query cannot be empty")

    result = multimodal_clf.predict(image_bytes, text, model)
    return result


# ── Direct run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("extensions.app_demo.main:app", host="0.0.0.0", port=8000, reload=True)
