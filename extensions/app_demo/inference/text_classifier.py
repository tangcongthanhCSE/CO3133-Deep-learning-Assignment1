"""
Text Classification Inference Module
=====================================
Supports RNN (LSTM/GRU) and Transformer (BERT/RoBERTa) families.

HOW TO INTEGRATE YOUR TRAINED MODEL:
1. Update `MODEL_REGISTRY` with your checkpoint paths & tokenizer names
2. Set `LABELS` to your dataset's class names
3. For RNN models, also provide the vocabulary / tokenizer
"""

import time
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "lstm":     {"checkpoint": None, "type": "rnn",         "tokenizer": None},
    "bilstm":   {"checkpoint": None, "type": "rnn",         "tokenizer": None},
    "bert":     {"checkpoint": None, "type": "transformer", "tokenizer": "bert-base-uncased"},
    "roberta":  {"checkpoint": None, "type": "transformer", "tokenizer": "roberta-base"},
}

LABELS: list[str] = [
    "Class_0", "Class_1", "Class_2", "Class_3", "Class_4",
]


class TextClassifier:
    """Unified text classifier wrapping RNN and Transformer models."""

    def __init__(self):
        self._models: dict = {}
        self._available_models = list(MODEL_REGISTRY.keys())
        logger.info("TextClassifier initialised. Available models: %s", self._available_models)

    @property
    def available_models(self) -> list[str]:
        return self._available_models

    def load_model(self, model_name: str) -> bool:
        """
        TODO: Replace with real loading, e.g.:
        
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
            model = AutoModelForSequenceClassification.from_pretrained(cfg["tokenizer"])
            state_dict = torch.load(cfg["checkpoint"], map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            self._models[model_name] = {"model": model, "tokenizer": tokenizer}
        """
        if model_name not in MODEL_REGISTRY:
            return False

        cfg = MODEL_REGISTRY[model_name]
        logger.info("Loading text model '%s' (type=%s)", model_name, cfg["type"])

        # ★ MOCK ★
        self._models[model_name] = {"loaded": True, "type": cfg["type"]}
        return True

    def predict(self, text: str, model_name: str) -> dict:
        """
        Run inference on a text string.
        
        TODO: Replace with real inference:
        
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1)[0]
        """
        if model_name not in self._models:
            self.load_model(model_name)

        start = time.perf_counter()

        # ★ MOCK ★
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
            "input_text": text[:120] + ("..." if len(text) > 120 else ""),
            "predictions": predictions,
            "inference_time_ms": round(elapsed, 2),
        }
