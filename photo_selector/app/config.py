from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")
THUMBNAIL_SIZE: tuple[int, int] = (256, 256)
EMBEDDING_DIM: int = 512

BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "data"
EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
THUMBNAILS_DIR: Path = DATA_DIR / "thumbnails"
DB_PATH: Path = DATA_DIR / "database.sqlite"

# Ranking weights
WEIGHT_AESTHETIC: float = 0.4
WEIGHT_BLUR: float = 0.3
WEIGHT_FACE_COUNT: float = 0.2
WEIGHT_CLUSTER_BALANCE: float = 0.1


def ensure_directories() -> None:
    """Create required data directories if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
