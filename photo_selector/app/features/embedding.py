from __future__ import annotations

"""
CLIP embedding pipeline for local-first image processing.

Example:
    from app.features.embedding import process_images, update_index_json

    records = [
        {"id": "img_0001", "path": "/photos/a.jpg"},
        {"id": "img_0002", "path": "/photos/b.jpg"},
    ]
    updated = process_images(records, batch_size=32)
    update_index_json(updated)

Performance notes:
- CPU works on all systems but is slower for large batches.
- GPU (CUDA) is used automatically when available and can be much faster.
- Start with batch_size=32; decrease if you hit memory limits.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from app.config import DATA_DIR, EMBEDDINGS_DIR

# Singleton CLIP state
_CLIP_MODEL: Optional[torch.nn.Module] = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE: Optional[str] = None

INDEX_JSON_PATH = DATA_DIR / "images_index.json"


def load_clip_model() -> tuple[torch.nn.Module, object]:
    """Load OpenAI CLIP ViT-B/32 once and return (model, preprocess)."""
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE

    if _CLIP_MODEL is not None and _CLIP_PREPROCESS is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _CLIP_DEVICE = device

    try:
        import clip  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OpenAI CLIP package is required. Install with: pip install git+https://github.com/openai/CLIP.git"
        ) from exc

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    _CLIP_MODEL = model
    _CLIP_PREPROCESS = preprocess

    print(f"[INFO] CLIP model loaded: ViT-B/32 on device={device}")
    return _CLIP_MODEL, _CLIP_PREPROCESS


def _safe_load_preprocessed(image_path: str, preprocess: object) -> Optional[torch.Tensor]:
    """Load and preprocess image, returning None for unreadable files."""
    try:
        image = Image.open(image_path).convert("RGB")
        tensor = preprocess(image)  # [3, H, W]
        return tensor
    except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
        print(f"[WARN] Skipping invalid image {image_path}: {exc}")
        return None
    except Exception as exc:  # keep pipeline resilient
        print(f"[WARN] Failed preprocessing {image_path}: {exc}")
        return None


def _l2_normalize(embedding: torch.Tensor) -> torch.Tensor:
    """L2-normalize embedding vectors row-wise."""
    return embedding / embedding.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def generate_embedding(image_path: str) -> np.ndarray:
    """Generate one normalized CLIP embedding with shape (512,)."""
    model, preprocess = load_clip_model()
    if _CLIP_DEVICE is None:
        raise RuntimeError("CLIP device is not initialized.")

    image_tensor = _safe_load_preprocessed(image_path, preprocess)
    if image_tensor is None:
        raise ValueError(f"Could not load image: {image_path}")

    image_batch = image_tensor.unsqueeze(0).to(_CLIP_DEVICE)

    with torch.no_grad():
        features = model.encode_image(image_batch)
        features = _l2_normalize(features)

    return features.squeeze(0).cpu().numpy().astype(np.float32)


def batch_generate_embeddings(image_paths: list[str], batch_size: int = 32) -> list[Optional[np.ndarray]]:
    """
    Generate normalized CLIP embeddings in batches.

    Returns a list aligned to image_paths, where unreadable images are None.
    """
    if not image_paths:
        return []

    model, preprocess = load_clip_model()
    if _CLIP_DEVICE is None:
        raise RuntimeError("CLIP device is not initialized.")

    output: list[Optional[np.ndarray]] = [None] * len(image_paths)

    for start in tqdm(range(0, len(image_paths), batch_size), desc="Embedding batches"):
        end = min(start + batch_size, len(image_paths))
        batch_paths = image_paths[start:end]

        valid_tensors: list[torch.Tensor] = []
        valid_indices: list[int] = []

        for offset, path in enumerate(batch_paths):
            tensor = _safe_load_preprocessed(path, preprocess)
            if tensor is None:
                continue
            valid_tensors.append(tensor)
            valid_indices.append(start + offset)

        if not valid_tensors:
            continue

        batch_tensor = torch.stack(valid_tensors).to(_CLIP_DEVICE)

        with torch.no_grad():
            batch_embeddings = model.encode_image(batch_tensor)
            batch_embeddings = _l2_normalize(batch_embeddings)

        batch_np = batch_embeddings.cpu().numpy().astype(np.float32)
        for idx, emb in zip(valid_indices, batch_np):
            output[idx] = emb

    return output


def save_embedding(image_id: str, embedding: np.ndarray) -> str:
    """Save embedding to data/embeddings/{image_id}.npy and return relative path."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = EMBEDDINGS_DIR / f"{image_id}.npy"
    np.save(save_path, embedding)
    return str(Path("data") / "embeddings" / f"{image_id}.npy")


def load_embedding(image_id: str) -> np.ndarray:
    """Load embedding from data/embeddings/{image_id}.npy."""
    path = EMBEDDINGS_DIR / f"{image_id}.npy"
    return np.load(path)


def embedding_exists(image_id: str) -> bool:
    """Check whether embedding file exists for image_id."""
    return (EMBEDDINGS_DIR / f"{image_id}.npy").exists()


def process_images(image_records: list[dict], batch_size: int = 32) -> list[dict]:
    """
    Process records, generate missing embeddings, and update embedding_path.

    Input record shape:
        {"id": "...", "path": "..."}
    """
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    to_embed_paths: list[str] = []
    to_embed_ids: list[str] = []

    for record in image_records:
        image_id = str(record.get("id", "")).strip()
        image_path = str(record.get("path", "")).strip()

        if not image_id or not image_path:
            print(f"[WARN] Invalid record, missing id/path: {record}")
            continue

        if embedding_exists(image_id):
            record["embedding_path"] = str(Path("data") / "embeddings" / f"{image_id}.npy")
            continue

        to_embed_ids.append(image_id)
        to_embed_paths.append(image_path)

    if not to_embed_paths:
        print("[INFO] All embeddings already exist. Nothing to process.")
        return image_records

    embeddings = batch_generate_embeddings(to_embed_paths, batch_size=batch_size)

    id_to_path: dict[str, str] = {}
    for image_id, embedding in zip(to_embed_ids, embeddings):
        if embedding is None:
            print(f"[WARN] Embedding not created for id={image_id}")
            continue
        relative_path = save_embedding(image_id, embedding)
        id_to_path[image_id] = relative_path

    for record in image_records:
        image_id = str(record.get("id", "")).strip()
        if image_id in id_to_path:
            record["embedding_path"] = id_to_path[image_id]
        elif image_id and embedding_exists(image_id):
            record["embedding_path"] = str(Path("data") / "embeddings" / f"{image_id}.npy")

    return image_records


def update_index_json(image_records: list[dict]) -> None:
    """
    Merge updated image records into data/images_index.json safely.

    Supported index formats on disk:
    - list[dict]
    - {"images": list[dict]}
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing_by_id: dict[str, dict] = {}
    wrapper_key = ""

    if INDEX_JSON_PATH.exists():
        try:
            existing = json.loads(INDEX_JSON_PATH.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                for item in existing:
                    if isinstance(item, dict) and "id" in item:
                        existing_by_id[str(item["id"])] = item
            elif isinstance(existing, dict) and isinstance(existing.get("images"), list):
                wrapper_key = "images"
                for item in existing["images"]:
                    if isinstance(item, dict) and "id" in item:
                        existing_by_id[str(item["id"])] = item
            else:
                print("[WARN] Existing images_index.json format not recognized. Rebuilding index.")
        except Exception as exc:
            print(f"[WARN] Failed reading {INDEX_JSON_PATH}: {exc}. Rebuilding index.")

    for record in image_records:
        image_id = str(record.get("id", "")).strip()
        if not image_id:
            continue
        merged = dict(existing_by_id.get(image_id, {}))
        merged.update(record)
        existing_by_id[image_id] = merged

    merged_records = list(existing_by_id.values())
    output_obj = {"images": merged_records} if wrapper_key == "images" else merged_records

    tmp_path = INDEX_JSON_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(output_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(INDEX_JSON_PATH)

    print(f"[INFO] Updated index with {len(image_records)} records: {INDEX_JSON_PATH}")
