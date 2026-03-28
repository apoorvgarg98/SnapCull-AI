from __future__ import annotations

from pathlib import Path

import torch
from facenet_pytorch import MTCNN
from PIL import Image, UnidentifiedImageError

_FACE_DETECTOR: MTCNN | None = None
_FACE_DEVICE: str | None = None

# Keep threshold modest for group shots while filtering weak detections.
MIN_FACE_PROBABILITY = 0.90


def _load_face_detector() -> MTCNN:
    """Load MTCNN face detector once for the current process."""
    global _FACE_DETECTOR, _FACE_DEVICE

    if _FACE_DETECTOR is not None:
        return _FACE_DETECTOR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = MTCNN(keep_all=True, device=device)

    _FACE_DETECTOR = detector
    _FACE_DEVICE = device
    print(f"[INFO] MTCNN face detector loaded on device={device}")
    return _FACE_DETECTOR


def detect_face_count(image_path: str | Path) -> int:
    """Detect faces using MTCNN and return confident face count."""
    detector = _load_face_detector()

    try:
        image = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
        print(f"[WARN] Could not read image for face detection: {image_path} ({exc})")
        return 0

    boxes, probs = detector.detect(image)
    if boxes is None or probs is None:
        return 0

    confident_faces = [score for score in probs.tolist() if score is not None and score >= MIN_FACE_PROBABILITY]
    return len(confident_faces)
