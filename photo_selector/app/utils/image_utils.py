from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError


def load_image_safely(image_path: str | Path) -> Image.Image | None:
    """Safely load an image and return None if the file is invalid/corrupted."""
    path = Path(image_path)
    try:
        image = Image.open(path)
        image.load()
        return image
    except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
        print(f"[WARN] Failed to load image {path}: {exc}")
        return None


def generate_thumbnail(image_path: str | Path, output_path: str | Path, size: tuple[int, int]) -> bool:
    """Generate and save a thumbnail, returning success status."""
    image = load_image_safely(image_path)
    if image is None:
        return False

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    thumbnail_image = image.copy()
    thumbnail_image.thumbnail(size)
    thumbnail_image.save(out_path)
    return True
