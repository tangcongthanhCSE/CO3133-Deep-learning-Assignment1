"""
Multimodal Classification Inference Module
===========================================
Supports Zero-shot and Few-shot classification using CLIP-like models.

HOW TO INTEGRATE:
1. Set `CANDIDATE_LABELS` to your dataset classes
2. Load your CLIP / multimodal model in `load_model`
3. Implement real zero-shot / few-shot logic in `predict`
"""

import io
import time
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "clip_zero_shot":  {"checkpoint": None, "method": "zero_shot", "backbone": "openai/clip-vit-base-patch32"},
    "clip_few_shot":   {"checkpoint": None, "method": "few_shot",  "backbone": "openai/clip-vit-base-patch32"},
}

CANDIDATE_LABELS: list[str] = [
    "Class_0", "Class_1", "Class_2", "Class_3", "Class_4",
]


class MultimodalClassifier:
    """Multimodal classifier for zero-shot & few-shot image+text tasks."""

    def __init__(self):
        self._models: dict = {}
        self._available_models = list(MODEL_REGISTRY.keys())
        logger.info("MultimodalClassifier initialised. Available: %s", self._available_models)

    @property
    def available_models(self) -> list[str]:
        return self._available_models

    def load_model(self, model_name: str) -> bool:
        """
        TODO: Replace with real CLIP loading:
        
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            model.eval()
            self._models[model_name] = {
                "model": model, "preprocess": preprocess, "tokenizer": tokenizer
            }
        """
        if model_name not in MODEL_REGISTRY:
            return False

        cfg = MODEL_REGISTRY[model_name]
        logger.info("Loading multimodal model '%s' (method=%s)", model_name, cfg["method"])

        # ★ MOCK ★
        self._models[model_name] = {"loaded": True, "method": cfg["method"]}
        return True

    def predict(self, image_bytes: bytes, text_query: str, model_name: str) -> dict:
        """
        Run multimodal inference.
        
        TODO: Replace with real CLIP inference:
        
            image = preprocess(Image.open(io.BytesIO(image_bytes))).unsqueeze(0)
            text = tokenizer(CANDIDATE_LABELS)
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                similarity = (image_features @ text_features.T).softmax(dim=-1)[0]
        """
        if model_name not in self._models:
            self.load_model(model_name)

        start = time.perf_counter()

        # ★ MOCK ★
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        n = len(CANDIDATE_LABELS)
        probs = np.random.dirichlet(np.ones(n))
        sorted_idx = np.argsort(probs)[::-1]

        elapsed = (time.perf_counter() - start) * 1000

        predictions = [
            {"label": CANDIDATE_LABELS[i], "confidence": round(float(probs[i]), 4)}
            for i in sorted_idx
        ]

        return {
            "model": model_name,
            "method": MODEL_REGISTRY[model_name]["method"],
            "text_query": text_query,
            "predictions": predictions,
            "inference_time_ms": round(elapsed, 2),
        }
