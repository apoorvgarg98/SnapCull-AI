from __future__ import annotations

from pathlib import Path
from typing import Optional

from urllib.request import urlretrieve

import numpy as np
import torch

from app.config import DATA_DIR
from app.features.embedding import generate_embedding
from app.utils.device import resolve_torch_device

_AESTHETIC_MODEL: torch.nn.Module | None = None
_AESTHETIC_DEVICE: str | None = None

# LAION public linear head for CLIP ViT-B/32 embeddings.
# Source repository: https://github.com/LAION-AI/aesthetic-predictor
LAION_VIT_B32_LINEAR_URL = (
    "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_b_32_linear.pth"
)
MODEL_DIR = DATA_DIR / "models"
MODEL_PATH = MODEL_DIR / "sa_0_4_vit_b_32_linear.pth"


def _ensure_model_file() -> Path:
    """Ensure LAION predictor weights are available locally."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return MODEL_PATH
    print(f"[INFO] Downloading LAION aesthetic model to {MODEL_PATH}")
    urlretrieve(LAION_VIT_B32_LINEAR_URL, MODEL_PATH)  # nosec: B310
    return MODEL_PATH


def _load_aesthetic_model() -> torch.nn.Module:
    """Load LAION ViT-B/32 aesthetic linear head once."""
    global _AESTHETIC_MODEL, _AESTHETIC_DEVICE
    if _AESTHETIC_MODEL is not None:
        return _AESTHETIC_MODEL

    _ensure_model_file()
    device = resolve_torch_device()

    model = torch.nn.Linear(512, 1)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _AESTHETIC_MODEL = model
    _AESTHETIC_DEVICE = device
    print(f"[INFO] LAION aesthetic model loaded on device={device}")
    return _AESTHETIC_MODEL


def estimate_aesthetic_score(image_path: str | Path, embedding: Optional[np.ndarray] = None) -> float:
    """
    Estimate aesthetic quality using the LAION aesthetic predictor.

    Returns:
        Score clipped to [0.0, 10.0].
    """
    model = _load_aesthetic_model()
    if _AESTHETIC_DEVICE is None:
        raise RuntimeError("Aesthetic model device is not initialized.")

    if embedding is None:
        embedding = generate_embedding(str(image_path))

    embedding_tensor = torch.from_numpy(np.asarray(embedding, dtype=np.float32)).unsqueeze(0).to(_AESTHETIC_DEVICE)

    with torch.no_grad():
        score = float(model(embedding_tensor).squeeze().item())

    return round(float(np.clip(score, 0.0, 10.0)), 4)
