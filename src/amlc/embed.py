"""Cached text / image embeddings -> features for GBDTs or small heads.

Everything is cached to data/features/<name>.npy (float16), so each embedding is paid
for once per team: compute it on whichever machine has the GPU, share the .npy file.

Laptop GPU is a GTX 1650 (4 GB): stick to small/base encoders here and use a bigger
GPU (Kaggle / SageMaker) for anything larger.
"""
from pathlib import Path

import numpy as np

from .config import FEATURES
from .utils import free_memory, get_device

# Good small defaults that fit in 4 GB with fp16
TEXT_MODEL = "BAAI/bge-small-en-v1.5"          # 33M params, 384-d
TEXT_MODEL_MULTI = "intfloat/multilingual-e5-small"
IMAGE_MODEL = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"  # 512-d, fast, safetensors (openai/* repos lack them)
IMAGE_MODEL_ALT = "google/siglip-base-patch16-224"


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


def embed_images(
    paths,
    model_name: str = IMAGE_MODEL,
    cache_name: str | None = None,
    batch_size: int = 64,
    num_workers: int = 6,
    normalize: bool = True,
) -> np.ndarray:
    """CLIP/SigLIP image embeddings. Missing or corrupt images get an all-zero row."""
    path, hit = _cached(cache_name)
    if hit is not None:
        print(f"loaded cached {path.name} {hit.shape}")
        return hit
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm
    from transformers import AutoModel, AutoProcessor

    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_name, dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)

    class _DS(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            p = paths[i]
            try:
                return Image.open(p).convert("RGB"), True
            except Exception:
                return Image.new("RGB", (224, 224)), False

    def collate(batch):
        imgs, ok = zip(*batch)
        return processor(images=list(imgs), return_tensors="pt")["pixel_values"], torch.tensor(ok)

    paths = [str(p) if p is not None else "" for p in paths]
    dl = DataLoader(_DS(), batch_size=batch_size, num_workers=num_workers, collate_fn=collate)
    out = []
    with torch.inference_mode():
        for px, ok in tqdm(dl, desc="image emb"):
            feats = model.get_image_features(pixel_values=px.to(device, dtype))
            if not torch.is_tensor(feats):  # newer transformers may return a ModelOutput
                emb = getattr(feats, "image_embeds", None)
                feats = emb if emb is not None else feats.pooler_output
            if normalize:
                feats = torch.nn.functional.normalize(feats.float(), dim=-1)
            feats[~ok.to(device)] = 0
            out.append(feats.float().cpu().numpy().astype(np.float16))
    emb = np.concatenate(out)
    del model
    free_memory()
    if path is not None:
        np.save(path, emb)
    return emb


def load_features(*names: str) -> np.ndarray:
    """Horizontally stack cached feature files: load_features('txt_bge', 'img_clip')."""
    return np.hstack([np.load(Path(FEATURES) / f"{n}.npy").astype(np.float32) for n in names])
