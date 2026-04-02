"""
Image Classification Inference Module
======================================
Supports CNN (ResNet, EfficientNet) and ViT families.

HOW TO INTEGRATE YOUR TRAINED MODEL:
1. Update `MODEL_REGISTRY` with your checkpoint paths
2. Set `LABELS` to your dataset's class names
3. Adjust `transform` if you used different preprocessing
"""

import io
import time
import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION — Edit these when you have trained models
# ──────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "resnet50":       {"checkpoint": None, "arch": "resnet50"},
    "efficientnet":   {"checkpoint": None, "arch": "efficientnet_b0"},
    "vit":            {"checkpoint": None, "arch": "vit_base_patch16_224"},
}

# Replace with your actual class labels
LABELS: list[str] = [
    "Class_0", "Class_1", "Class_2", "Class_3", "Class_4",
    "Class_5", "Class_6", "Class_7", "Class_8", "Class_9",
]

class ImageClassifier:
    """Unified image classifier that wraps CNN and ViT models."""

    def __init__(self):
        self._models: dict = {}
        self._available_models = list(MODEL_REGISTRY.keys())
        logger.info("ImageClassifier initialised. Available models: %s", self._available_models)

    # ── Public API ──────────────────────────────────────────────────
    @property
    def available_models(self) -> list[str]:
        return self._available_models

    def load_model(self, model_name: str) -> bool:
        """
        Load a model into memory. Returns True on success.
        
        TODO: Replace the mock implementation below with real model loading:
        
            import torch, torchvision.models as models
            
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            
            # Load your fine-tuned weights
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            
            self._models[model_name] = {"model": model, "transform": weights.transforms()}
        """
        if model_name not in MODEL_REGISTRY:
            return False

        cfg = MODEL_REGISTRY[model_name]
        logger.info("Loading image model '%s' (arch=%s, ckpt=%s)", model_name, cfg["arch"], cfg["checkpoint"])

        # ★ MOCK — replace with real loading logic above ★
        self._models[model_name] = {"loaded": True, "arch": cfg["arch"]}
        return True

    def predict(self, image_bytes: bytes, model_name: str) -> dict:
        """
        Run inference on raw image bytes.
        
        Returns:
            {
              "model": str,
              "predictions": [{"label": str, "confidence": float}, ...],
              "inference_time_ms": float,
            }

        TODO: Replace mock with real inference:
        
            from torchvision import transforms
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = self._models[model_name]["transform"](img).unsqueeze(0)
            with torch.no_grad():
                logits = self._models[model_name]["model"](tensor)
                probs = torch.softmax(logits, dim=1)[0]
            top_indices = probs.argsort(descending=True)
        """
        if model_name not in self._models:
            self.load_model(model_name)

        start = time.perf_counter()

        # ★ MOCK — replace with real inference ★
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        n_classes = len(LABELS)
        probs = np.random.dirichlet(np.ones(n_classes))
        sorted_idx = np.argsort(probs)[::-1]

        elapsed = (time.perf_counter() - start) * 1000

        predictions = [
            {"label": LABELS[i], "confidence": round(float(probs[i]), 4)}
            for i in sorted_idx
        ]

        return {
            "model": model_name,
            "predictions": predictions,
            "inference_time_ms": round(elapsed, 2),
        }
