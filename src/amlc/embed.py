"""Cached text embeddings of names/addresses (optional add-on, not used by the baseline).

Useful as an extra blocking channel (dense nearest neighbours catch transliterations and
synonyms that char n-grams miss) or as pair features (cosine of name embeddings).
Cached to data/features/<name>.npy (float16): compute once on the biggest GPU, share the file.
~12M strings with multilingual-e5-small likely take over an hour on the laptop GPU; a cloud T4 is faster.
"""
from pathlib import Path

import numpy as np

from .config import FEATURES
from .utils import free_memory, get_device

# Challenge rule: final model must be MIT / Apache-2.0 and <= 8B params. Both below are MIT.
TEXT_MODEL = "intfloat/multilingual-e5-small"   # 118M, 384-d, handles Hindi/Tamil/French; use prefix="query: "
TEXT_MODEL_EN = "BAAI/bge-small-en-v1.5"        # 33M, 384-d, English only


def _cached(name: str | None):
    if name is None:
        return None, None
    path = FEATURES / f"{name}.npy"
    return path, (np.load(path) if path.exists() else None)


def embed_texts(
    texts,
    model_name: str = TEXT_MODEL,
    cache_name: str | None = None,
    batch_size: int = 128,
    max_seq_length: int = 256,
    normalize: bool = True,
    prefix: str = "",
) -> np.ndarray:
    """Sentence-transformers embeddings. For e5 models pass prefix='query: '."""
    path, hit = _cached(cache_name)
    if hit is not None:
        print(f"loaded cached {path.name} {hit.shape}")
        return hit
    from sentence_transformers import SentenceTransformer

    device = get_device()
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    if device == "cuda":
        model.half()
    texts = [prefix + ("" if t is None or t != t else str(t)) for t in texts]
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                       normalize_embeddings=normalize, convert_to_numpy=True).astype(np.float16)
    del model
    free_memory()
    if path is not None:
        np.save(path, emb)
    return emb


def load_features(*names: str) -> np.ndarray:
    """Horizontally stack cached feature files: load_features('txt_bge', 'img_clip')."""
    return np.hstack([np.load(Path(FEATURES) / f"{n}.npy").astype(np.float32) for n in names])
