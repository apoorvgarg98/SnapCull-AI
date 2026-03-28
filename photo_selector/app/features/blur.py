from __future__ import annotations

from pathlib import Path

import cv2


def calculate_blur_score(image_path: str | Path) -> float:
    """Compute blur score using Laplacian variance (higher = sharper)."""
    path = str(Path(image_path))
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"[WARN] Could not read image for blur scoring: {path}")
        return 0.0

    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    return float(variance)
