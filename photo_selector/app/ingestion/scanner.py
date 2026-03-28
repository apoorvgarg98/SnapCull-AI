from __future__ import annotations

from pathlib import Path

from app.config import SUPPORTED_EXTENSIONS


def scan_images(folder: str | Path) -> list[Path]:
    """Recursively scan a folder and return supported image paths."""
    root = Path(folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid folder: {root}")

    image_paths = [
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    image_paths.sort()
    return image_paths
