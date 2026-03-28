from __future__ import annotations

from pathlib import Path

import random


def estimate_aesthetic_score(image_path: str | Path) -> float:
    """Placeholder aesthetic model returning a random score in [0, 1]."""
    _ = image_path
    return round(random.random(), 4)
