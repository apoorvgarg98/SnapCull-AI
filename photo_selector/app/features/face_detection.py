from __future__ import annotations

from pathlib import Path

import random


def detect_face_count(image_path: str | Path) -> int:
    """Placeholder face detector returning random face count in [0, 5]."""
    _ = image_path
    return random.randint(0, 5)
