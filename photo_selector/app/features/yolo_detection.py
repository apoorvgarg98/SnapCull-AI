from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

_YOLO_MODEL: YOLO | None = None
_YOLO_DEVICE: str | None = None

# COCO class names used by YOLOv8.
CLASS_PERSON = "person"
CLASS_TIE = "tie"
CLASS_HAND_BAG = "handbag"
CLASS_SUITCASE = "suitcase"
CLASS_DINING_TABLE = "dining table"
CLASS_CHAIR = "chair"
CLASS_CUP = "cup"
CLASS_BOWL = "bowl"
CLASS_VASE = "vase"

GROUP_PERSON_THRESHOLD = 4
MIN_CONFIDENCE = 0.25


def _load_yolo_model() -> YOLO:
    """Load YOLOv8 model once and reuse it across calls."""
    global _YOLO_MODEL, _YOLO_DEVICE
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    _YOLO_MODEL = YOLO("yolov8n.pt")
    _YOLO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] YOLOv8 detector loaded on device={_YOLO_DEVICE}")
    return _YOLO_MODEL


def _class_name(model: YOLO, class_id: int) -> str:
    names = model.names
    if isinstance(names, dict):
        return str(names.get(class_id, ""))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return ""


def detect_wedding_context(image_path: str | Path) -> dict[str, Any]:
    """
    Detect wedding-relevant context using YOLOv8 with lightweight heuristics.

    Returns keys:
      - person_count
      - has_group
      - has_bride_groom (heuristic)
      - has_ritual (heuristic)
      - yolo_labels (detected labels above confidence threshold)
    """
    model = _load_yolo_model()
    path = str(Path(image_path))

    try:
        results = model.predict(source=path, conf=MIN_CONFIDENCE, verbose=False, device=_YOLO_DEVICE)
    except Exception as exc:
        print(f"[WARN] YOLO detection failed for {path}: {exc}")
        return {
            "person_count": 0,
            "has_group": False,
            "has_bride_groom": False,
            "has_ritual": False,
            "yolo_labels": [],
        }

    if not results:
        return {
            "person_count": 0,
            "has_group": False,
            "has_bride_groom": False,
            "has_ritual": False,
            "yolo_labels": [],
        }

    result = results[0]
    boxes = result.boxes
    if boxes is None or boxes.cls is None:
        return {
            "person_count": 0,
            "has_group": False,
            "has_bride_groom": False,
            "has_ritual": False,
            "yolo_labels": [],
        }

    class_ids = boxes.cls.int().tolist()
    labels = [_class_name(model, class_id) for class_id in class_ids]

    person_count = sum(1 for label in labels if label == CLASS_PERSON)
    has_group = person_count >= GROUP_PERSON_THRESHOLD

    # Heuristic: at least two people plus wedding-like accessories can indicate bride/groom shot.
    has_bride_groom = person_count >= 2 and any(
        label in {CLASS_TIE, CLASS_HAND_BAG, CLASS_SUITCASE} for label in labels
    )

    # Heuristic: ritual scenes often include table/chair/cup/bowl/vase context.
    has_ritual = any(
        label in {CLASS_DINING_TABLE, CLASS_CHAIR, CLASS_CUP, CLASS_BOWL, CLASS_VASE} for label in labels
    )

    return {
        "person_count": person_count,
        "has_group": has_group,
        "has_bride_groom": has_bride_groom,
        "has_ritual": has_ritual,
        "yolo_labels": sorted(set(label for label in labels if label)),
    }
