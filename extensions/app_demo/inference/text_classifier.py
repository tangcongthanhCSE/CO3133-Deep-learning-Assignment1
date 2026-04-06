"""
Text Classification Inference Module
======================================
Emotion Classification — 6 classes: sadness, joy, love, anger, fear, surprise
Models:
  - lstm      : BiLSTM with GloVe-100 embeddings
  - distilbert: DistilBERT-base-uncased fine-tuned

Checkpoints expected at:
  models/text_dataset/best_lstm_model.pt       (state_dict)
  models/text_dataset/lstm_vocab.pkl           (word→index dict, save from training notebook)
  models/text_dataset/best_distilbert_model.pt (state_dict)

To save the vocab from the training notebook, add this cell after building `vocab`:
    import pickle
    with open("lstm_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    # then copy to models/text_dataset/lstm_vocab.pkl
"""

import re
import time
import logging
import pickle
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Device ────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Text classifier using device: %s", DEVICE)

# ── Paths ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = _PROJECT_ROOT / "models" / "text_dataset"

# Emotion labels (integer index matches training label encoding)
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
NUM_CLASSES = len(LABELS)

MODEL_REGISTRY = {
    "lstm": {
        "type": "rnn",
        "checkpoint": MODELS_DIR / "best_lstm_model.pt",
        "vocab": MODELS_DIR / "lstm_vocab.pkl",
    },
    "distilbert": {
        "type": "transformer",
        "checkpoint": MODELS_DIR / "best_distilbert_model.pt",
        "pretrained": "distilbert-base-uncased",
    },
}

# ── LSTM tokeniser (must mirror the training notebook exactly) ─────────
_CONTRACTIONS = {
    "dont": "do not",    "doesnt": "does not",  "didnt": "did not",
    "cant": "can not",   "couldnt": "could not", "wont": "will not",
    "wouldnt": "would not", "shouldnt": "should not", "mustnt": "must not",
    "im": "i am",        "ive": "i have",        "ill": "i will",
    "youre": "you are",  "youve": "you have",    "youll": "you will",
    "hes": "he is",      "shes": "she is",
    "theyre": "they are", "theyve": "they have", "theyll": "they will",
    "weve": "we have",   "isnt": "is not",       "arent": "are not",
    "wasnt": "was not",  "werent": "were not",   "havent": "have not",
    "hasnt": "has not",  "hadnt": "had not",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase → strip punctuation → expand contractions → split."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    expanded: list[str] = []
    for t in tokens:
        expanded.extend(_CONTRACTIONS.get(t, t).split())
    return expanded


# ── BiLSTM architecture (mirrors training notebook) ───────────────────
class _LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        # Concatenate last hidden states of both directions
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)


class TextClassifier:
    """Loads BiLSTM and DistilBERT emotion classifiers and runs inference."""

    # Hyper-parameters must match the training notebook
    _LSTM_MAX_LEN    = 128
    _LSTM_EMBED_DIM  = 100
    _LSTM_HIDDEN_DIM = 256
    _LSTM_NUM_LAYERS = 2
    _LSTM_DROPOUT    = 0.3
    _DISTILBERT_MAX_LEN = 128

    def __init__(self):
        self._models: dict     = {}
        self._vocabs: dict     = {}   # lstm name → word→idx dict
        self._tokenizers: dict = {}   # distilbert name → HF tokenizer
        self._available = list(MODEL_REGISTRY.keys())
        logger.info("TextClassifier initialised. Available models: %s", self._available)

    @property
    def available_models(self) -> list[str]:
        return self._available

    # ── Loading ────────────────────────────────────────────────────────
    def _load_lstm(self, cfg: dict) -> bool:
        vocab_path: Path = cfg["vocab"]
        if not vocab_path.exists():
            logger.error(
                "LSTM vocabulary file not found: %s\n"
                "Save it from the training notebook with:\n"
                "  import pickle; pickle.dump(vocab, open('%s', 'wb'))",
                vocab_path, vocab_path,
            )
            return False

        with open(vocab_path, "rb") as f:
            vocab: dict = pickle.load(f)

        ckpt_path: Path = cfg["checkpoint"]
        if not ckpt_path.exists():
            logger.error("LSTM checkpoint not found: %s", ckpt_path)
            return False

        model = _LSTMClassifier(
            vocab_size=len(vocab),
            embed_dim=self._LSTM_EMBED_DIM,
            hidden_dim=self._LSTM_HIDDEN_DIM,
            num_layers=self._LSTM_NUM_LAYERS,
            num_classes=NUM_CLASSES,
            dropout=self._LSTM_DROPOUT,
        )
        state_dict = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        self._models["lstm"]  = model
        self._vocabs["lstm"]  = vocab
        logger.info("Loaded LSTM model (vocab_size=%d) on %s from %s", len(vocab), DEVICE, ckpt_path)
        return True

    def _load_distilbert(self, cfg: dict) -> bool:
        ckpt_path: Path = cfg["checkpoint"]
        if not ckpt_path.exists():
            logger.error("DistilBERT checkpoint not found: %s", ckpt_path)
            return False

        try:
            from transformers import (
                DistilBertForSequenceClassification,
                DistilBertTokenizerFast,
            )
        except ImportError:
            logger.error("'transformers' package is not installed.")
            return False

        pretrained = cfg["pretrained"]
        tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained)
        model = DistilBertForSequenceClassification.from_pretrained(
            pretrained, num_labels=NUM_CLASSES
        )
        state_dict = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        self._models["distilbert"]     = model
        self._tokenizers["distilbert"] = tokenizer
        logger.info("Loaded DistilBERT model on %s from %s", DEVICE, ckpt_path)
        return True

    def load_model(self, name: str) -> bool:
        """Load a model into memory. Returns True on success."""
        if name in self._models:
            return True
        if name not in MODEL_REGISTRY:
            logger.error("Unknown text model '%s'", name)
            return False
        cfg = MODEL_REGISTRY[name]
        if cfg["type"] == "rnn":
            return self._load_lstm(cfg)
        return self._load_distilbert(cfg)

    # ── Inference helpers ──────────────────────────────────────────────
    def _infer_lstm(self, text: str) -> list[float]:
        vocab = self._vocabs["lstm"]
        model = self._models["lstm"]
        unk_idx = vocab.get("<unk>", 1)
        tokens = _tokenize(text)
        indices = [vocab.get(t, unk_idx) for t in tokens[:self._LSTM_MAX_LEN]]
        indices += [0] * (self._LSTM_MAX_LEN - len(indices))   # pad
        tensor = torch.tensor([indices], dtype=torch.long).to(DEVICE)  # [1, 128]
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
        return probs.tolist()

    def _infer_distilbert(self, text: str) -> list[float]:
        tokenizer = self._tokenizers["distilbert"]
        model     = self._models["distilbert"]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._DISTILBERT_MAX_LEN,
            padding="max_length",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs  = torch.softmax(logits, dim=1)[0]
        return probs.tolist()

    def predict(self, text: str, model_name: str) -> dict:
        """
        Run inference on a text string.

        Returns:
            {
              "model": str,
              "input_text": str,
              "predictions": [{"label": str, "confidence": float}, ...],
              "inference_time_ms": float,
            }
        """
        if model_name not in self._models:
            if not self.load_model(model_name):
                raise RuntimeError(f"Could not load text model '{model_name}'")

        start = time.perf_counter()

        if MODEL_REGISTRY[model_name]["type"] == "rnn":
            probs = self._infer_lstm(text)
        else:
            probs = self._infer_distilbert(text)

        elapsed = (time.perf_counter() - start) * 1000
        sorted_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)

        return {
            "model": model_name,
            "input_text": text[:120] + ("..." if len(text) > 120 else ""),
            "predictions": [
                {"label": LABELS[i], "confidence": round(probs[i], 4)}
                for i in sorted_idx
            ],
            "inference_time_ms": round(elapsed, 2),
        }
