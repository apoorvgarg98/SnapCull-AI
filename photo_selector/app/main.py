from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

from app.clustering.cluster import assign_clusters
from app.config import DB_PATH, EMBEDDINGS_DIR, THUMBNAIL_SIZE, THUMBNAILS_DIR, ensure_directories
from app.features.aesthetic import estimate_aesthetic_score
from app.features.blur import calculate_blur_score
from app.features.embedding import generate_embedding, update_index_json
from app.features.face_detection import detect_face_count
from app.ingestion.scanner import scan_images
from app.ranking.ranker import compute_final_scores
from app.storage.db import Database
from app.utils.image_utils import generate_thumbnail


def _artifact_stem(image_path: Path) -> str:
    """Return a stable, filesystem-safe stem unique to each image path."""
    digest = hashlib.sha256(str(image_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{image_path.stem}_{digest}"


def run_pipeline(input_folder: str | Path) -> None:
    ensure_directories()
    db = Database(DB_PATH)
    db.initialize()

    image_paths = scan_images(input_folder)
    print(f"[INFO] Found {len(image_paths)} images in {input_folder}")

    embeddings: list[np.ndarray] = []
    image_path_strings: list[str] = []
    index_records: list[dict] = []

    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            embedding = generate_embedding(image_path)
        except ValueError as exc:
            print(f"[WARN] Skipping {image_path}: {exc}")
            continue

        aesthetic_score = estimate_aesthetic_score(image_path, embedding=embedding)
        blur_score = calculate_blur_score(image_path)
        face_count = detect_face_count(image_path)

        artifact_stem = _artifact_stem(image_path)
        embedding_file = EMBEDDINGS_DIR / f"{artifact_stem}.npy"
        np.save(embedding_file, embedding)

        thumbnail_file = THUMBNAILS_DIR / f"{artifact_stem}.jpg"
        generate_thumbnail(image_path, thumbnail_file, THUMBNAIL_SIZE)

        db.upsert_image(
            path=str(image_path),
            embedding_path=str(embedding_file),
            aesthetic_score=aesthetic_score,
            blur_score=blur_score,
            face_count=face_count,
        )

        embeddings.append(embedding)
        image_path_strings.append(str(image_path))
        index_records.append(
            {
                "id": artifact_stem,
                "path": str(image_path),
                "embedding_path": str(embedding_file),
                "aesthetic_score": aesthetic_score,
                "blur_score": blur_score,
                "face_count": face_count,
            }
        )

    if embeddings:
        cluster_ids = assign_clusters(np.vstack(embeddings))
        for image_path, cluster_id in zip(image_path_strings, cluster_ids):
            db.update_cluster(image_path, cluster_id)

    records = db.fetch_all_images()
    ranked = compute_final_scores(records)
    for item in ranked:
        db.update_final_score(item["path"], float(item["final_score"]))

    if index_records:
        update_index_json(index_records)

    print("[INFO] Pipeline complete.")
    db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local photo selection pipeline")
    parser.add_argument("--input-folder", required=True, help="Path to folder containing images")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.input_folder)
