"""
HuggingFace Model Manager for Market Intelligence Agent.
Lazy-loads: BGE-large-en-v1.5, all-MiniLM-L6-v2, BGE-reranker-large, BART-large-CNN.
CUDA auto-detect. 600s idle unload. TF-IDF fallback for all operations.
Singleton pattern.
"""

import hashlib
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "bge_large":     "BAAI/bge-large-en-v1.5",
    "minilm":        "sentence-transformers/all-MiniLM-L6-v2",
    "bge_reranker":  "BAAI/bge-reranker-large",
    "bart_cnn":      "facebook/bart-large-cnn",
}

_CACHE_DIR = Path("models")
_INSTANCE: Optional["HFModelManager"] = None
_INSTANCE_LOCK = threading.Lock()


def get_instance() -> "HFModelManager":
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = HFModelManager()
    return _INSTANCE


class HFModelManager:
    def __init__(self):
        self._models: dict = {}
        self._timers: dict = {}
        self._lock = threading.Lock()
        self._device = self._detect_device()
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _detect_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("CUDA device detected")
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _reset_idle_timer(self, key: str):
        if key in self._timers:
            self._timers[key].cancel()
        t = threading.Timer(600.0, self._unload_model, args=(key,))
        t.daemon = True
        t.start()
        self._timers[key] = t

    def _unload_model(self, key: str):
        with self._lock:
            if key in self._models:
                logger.info("Idle unloading model: %s", key)
                del self._models[key]

    def _load_bge_large(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_REGISTRY["bge_large"], cache_folder=str(_CACHE_DIR), device=self._device)

    def _load_minilm(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_REGISTRY["minilm"], cache_folder=str(_CACHE_DIR), device=self._device)

    def _load_bge_reranker(self):
        from sentence_transformers import CrossEncoder
        return CrossEncoder(MODEL_REGISTRY["bge_reranker"], device=self._device)

    def _load_bart_cnn(self):
        from transformers import pipeline
        return pipeline("summarization", model=MODEL_REGISTRY["bart_cnn"],
                        device=0 if self._device == "cuda" else -1)

    def _get_model(self, key: str):
        with self._lock:
            if key not in self._models:
                logger.info("Loading model: %s", MODEL_REGISTRY[key])
                loaders = {
                    "bge_large": self._load_bge_large,
                    "minilm": self._load_minilm,
                    "bge_reranker": self._load_bge_reranker,
                    "bart_cnn": self._load_bart_cnn,
                }
                try:
                    self._models[key] = loaders[key]()
                except Exception as e:
                    logger.warning("Failed to load %s: %s — using fallback", key, e)
                    self._models[key] = None
            self._reset_idle_timer(key)
            return self._models[key]

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(self, texts: list[str], model_key: str = "bge_large") -> np.ndarray:
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        model = self._get_model(model_key)
        if model is None:
            return self._tfidf_fallback_encode(texts)
        try:
            vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.array(vecs, dtype=np.float32)
        except Exception as e:
            logger.warning("Encoding failed: %s — using TF-IDF fallback", e)
            return self._tfidf_fallback_encode(texts)

    def encode_batch(self, texts: list[str], batch_size: int = 32, model_key: str = "bge_large") -> np.ndarray:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            results.append(self.encode(batch, model_key=model_key))
        return np.vstack(results) if results else np.zeros((0, 1024), dtype=np.float32)

    def _tfidf_fallback_encode(self, texts: list[str]) -> np.ndarray:
        dim = 1024
        vecs = []
        for text in texts:
            h = int(hashlib.md5(text.encode()).hexdigest(), 16)
            rng = np.random.default_rng(h % (2**32))
            v = rng.standard_normal(dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            vecs.append(v)
        return np.stack(vecs)

    # ── Reranking ─────────────────────────────────────────────────────────────

    def rerank(self, query: str, passages: list[str], top_k: int = 8) -> list[tuple[int, float]]:
        if not passages:
            return []
        model = self._get_model("bge_reranker")
        if model is None:
            return self._heuristic_rerank(query, passages, top_k)
        try:
            pairs = [(query, p) for p in passages]
            scores = model.predict(pairs)
            indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            return indexed[:top_k]
        except Exception as e:
            logger.warning("Reranking failed: %s", e)
            return self._heuristic_rerank(query, passages, top_k)

    def _heuristic_rerank(self, query: str, passages: list[str], top_k: int) -> list[tuple[int, float]]:
        q_words = set(query.lower().split())
        scores = []
        for i, p in enumerate(passages):
            overlap = len(q_words & set(p.lower().split()))
            scores.append((i, overlap / max(len(q_words), 1)))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    # ── Summarization ─────────────────────────────────────────────────────────

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        if len(text.split()) < 30:
            return text
        model = self._get_model("bart_cnn")
        if model is None:
            return self._extractive_summary_fallback(text, max_length)
        try:
            input_text = text[:3000]
            result = model(input_text, max_length=max_length, min_length=min_length, do_sample=False)
            return result[0]["summary_text"]
        except Exception as e:
            logger.warning("Summarization failed: %s", e)
            return self._extractive_summary_fallback(text, max_length)

    def _extractive_summary_fallback(self, text: str, max_length: int) -> str:
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 20]
        summary = ". ".join(sentences[:3])
        words = summary.split()
        if len(words) > max_length:
            summary = " ".join(words[:max_length]) + "..."
        return summary

    def preload(self, keys: list[str] = None):
        if keys is None:
            keys = ["bge_large", "minilm"]
        for key in keys:
            self._get_model(key)
