from __future__ import annotations

import numpy as np


def assign_clusters(embeddings: np.ndarray, num_clusters: int = 10) -> list[int]:
    """Placeholder cluster assignment by deterministic index bucketing."""
    if embeddings.size == 0:
        return []
    cluster_count = max(1, min(num_clusters, embeddings.shape[0]))
    return [index % cluster_count for index in range(embeddings.shape[0])]
