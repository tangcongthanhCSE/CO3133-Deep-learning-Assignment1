"""
Image Classification Inference Module
======================================
Intel Image Classification — 6 scene classes
Models: ResNet18 (fine-tuned) | ViT-B/16 (fine-tuned)

Checkpoints expected at:
  models/image_dataset/resnet18_intel_best.pth
  models/image_dataset/ViT_intel_best.pth

Each checkpoint is a dict saved by torch.save() with keys:
  'epoch', 'model_state_dict', 'optimizer_state_dict', 'best_acc'
"""

import io
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights, vit_b_16, ViT_B_16_Weights
from PIL import Image

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[4]   # …/BTL1/
MODELS_DIR = _PROJECT_ROOT / "models" / "image_dataset"

MODEL_REGISTRY = {
    "resnet18": {
        "checkpoint": MODELS_DIR / "resnet18_intel_best.pth",
        "arch": "resnet18",
    },
    "vit": {
        "checkpoint": MODELS_DIR / "ViT_intel_best.pth",
        "arch": "vit_b_16",
    },
}

# Intel Image Classification — 6 scene labels (same order as ImageFolder's sorted classes)
LABELS = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
NUM_CLASSES = len(LABELS)

# Inference transform — identical to the 'test' transform used during training
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Model builders ─────────────────────────────────────────────────────
def _build_resnet18() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def _build_vit() -> nn.Module:
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    return model


_BUILDERS = {"resnet18": _build_resnet18, "vit": _build_vit}


class ImageClassifier:
    """Loads ResNet18 / ViT-B/16 checkpoints and runs inference on raw image bytes."""

    def __init__(self):
        self._models: dict = {}
        self._available = list(MODEL_REGISTRY.keys())
        logger.info("ImageClassifier initialised. Available models: %s", self._available)

    @property
    def available_models(self) -> list[str]:
        return self._available

    def load_model(self, name: str) -> bool:
        """Load a checkpoint into memory. Returns True on success."""
        if name in self._models:
            return True
        if name not in MODEL_REGISTRY:
            logger.error("Unknown image model '%s'", name)
            return False

        cfg = MODEL_REGISTRY[name]
        ckpt_path = cfg["checkpoint"]

        if not ckpt_path.exists():
            logger.error(
                "Checkpoint not found: %s\n"
                "Train the model and save it to that path.",
                ckpt_path,
            )
            return False

        model = _BUILDERS[name]()
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        # Checkpoint saved as dict; fall back to raw state_dict if needed
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()

        self._models[name] = model
        logger.info("Loaded image model '%s' (arch=%s) from %s", name, cfg["arch"], ckpt_path)
        return True

    def predict(self, image_bytes: bytes, model_name: str) -> dict:
        """
        Run inference on raw image bytes.

        Returns:
            {
              "model": str,
              "predictions": [{"label": str, "confidence": float}, ...],   # sorted by confidence
              "inference_time_ms": float,
            }
        """
        if model_name not in self._models:
            if not self.load_model(model_name):
                raise RuntimeError(f"Could not load image model '{model_name}'")

        model = self._models[model_name]
        start = time.perf_counter()

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = INFER_TRANSFORM(img).unsqueeze(0)   # [1, 3, 224, 224]

        with torch.no_grad():
            logits = model(tensor)                    # [1, 6]
            probs = torch.softmax(logits, dim=1)[0]   # [6]

        elapsed = (time.perf_counter() - start) * 1000
        sorted_idx = probs.argsort(descending=True).tolist()

        return {
            "model": model_name,
            "predictions": [
                {"label": LABELS[i], "confidence": round(probs[i].item(), 4)}
                for i in sorted_idx
            ],
            "inference_time_ms": round(elapsed, 2),
        }
