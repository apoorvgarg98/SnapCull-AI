from __future__ import annotations

from collections import Counter

from app.config import (
    WEIGHT_AESTHETIC,
    WEIGHT_BLUR,
    WEIGHT_CLUSTER_BALANCE,
    WEIGHT_FACE_COUNT,
)


def _normalize(value: float, min_val: float, max_val: float) -> float:
    if max_val <= min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def compute_final_scores(images: list[dict]) -> list[dict]:
    """Compute weighted final score for each image."""
    if not images:
        return images

    blur_values = [float(item.get("blur_score") or 0.0) for item in images]
    face_values = [float(item.get("face_count") or 0.0) for item in images]
    cluster_values = [int(item.get("cluster_id") or 0) for item in images]

    min_blur, max_blur = min(blur_values), max(blur_values)
    min_face, max_face = min(face_values), max(face_values)

    cluster_counts = Counter(cluster_values)
    max_cluster_count = max(cluster_counts.values()) if cluster_counts else 1

    for item in images:
        aesthetic = float(item.get("aesthetic_score") or 0.0)
        blur_norm = _normalize(float(item.get("blur_score") or 0.0), min_blur, max_blur)
        face_norm = _normalize(float(item.get("face_count") or 0.0), min_face, max_face)

        cluster_id = int(item.get("cluster_id") or 0)
        cluster_size = cluster_counts.get(cluster_id, 1)
        cluster_balance = 1.0 - (cluster_size / max_cluster_count)

        final_score = (
            WEIGHT_AESTHETIC * aesthetic
            + WEIGHT_BLUR * blur_norm
            + WEIGHT_FACE_COUNT * face_norm
            + WEIGHT_CLUSTER_BALANCE * cluster_balance
        )
        item["final_score"] = round(final_score, 4)

    return images
