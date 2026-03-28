from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from app.clustering.cluster import assign_clusters
from app.config import EMBEDDINGS_DIR, THUMBNAIL_SIZE, THUMBNAILS_DIR, ensure_directories
from app.features.aesthetic import estimate_aesthetic_score
from app.features.blur import calculate_blur_score
from app.features.embedding import batch_generate_embeddings, update_index_json
from app.features.face_detection import detect_face_count
from app.features.yolo_detection import detect_wedding_context
from app.ingestion.scanner import scan_images
from app.ranking.ranker import compute_final_scores
from app.utils.image_utils import generate_thumbnail


def _artifact_stem(image_path: Path) -> str:
    """Return a stable, filesystem-safe stem unique to each image path."""
    digest = hashlib.sha256(str(image_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{image_path.stem}_{digest}"


def _process_single_image(image_path: Path, embedding: np.ndarray) -> dict:
    """Run non-CLIP feature extraction and artifact generation for one image."""
    aesthetic_score = estimate_aesthetic_score(image_path, embedding=embedding)
    blur_score = calculate_blur_score(image_path)
    face_count = detect_face_count(image_path)
    yolo_context = detect_wedding_context(image_path)

    artifact_stem = _artifact_stem(image_path)
    embedding_file = EMBEDDINGS_DIR / f"{artifact_stem}.npy"
    np.save(embedding_file, embedding)

    thumbnail_file = THUMBNAILS_DIR / f"{artifact_stem}.jpg"
    generate_thumbnail(image_path, thumbnail_file, THUMBNAIL_SIZE)

    return {
        "path": str(image_path),
        "embedding_file": str(embedding_file),
        "aesthetic_score": aesthetic_score,
        "blur_score": blur_score,
        "face_count": face_count,
        "person_count": int(yolo_context["person_count"]),
        "has_bride_groom": bool(yolo_context["has_bride_groom"]),
        "has_ritual": bool(yolo_context["has_ritual"]),
        "has_group": bool(yolo_context["has_group"]),
        "yolo_labels": yolo_context["yolo_labels"],
        "artifact_stem": artifact_stem,
        "embedding": embedding,
    }


def run_pipeline(input_folder: str | Path) -> None:
    ensure_directories()

    image_paths = scan_images(input_folder)
    print(f"[INFO] Found {len(image_paths)} images in {input_folder}")

    if not image_paths:
        print("[INFO] No images found. Nothing to process.")
        return

    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    requested_workers = int(os.getenv("FEATURE_WORKERS", "4"))

    print(f"[INFO] Generating CLIP embeddings in batches (batch_size={batch_size})")
    batched = batch_generate_embeddings([str(path) for path in image_paths], batch_size=batch_size)

    valid_items: list[tuple[Path, np.ndarray]] = []
    for image_path, embedding in zip(image_paths, batched):
        if embedding is None:
            print(f"[WARN] Skipping {image_path}: embedding could not be created")
            continue
        valid_items.append((image_path, embedding))

    if not valid_items:
        print("[INFO] No valid images after embedding step.")
        return

    gpu_available = torch.cuda.is_available()
    worker_count = 1 if gpu_available else max(1, requested_workers)
    print(
        f"[INFO] Feature extraction workers={worker_count} "
        f"(gpu_available={gpu_available}, requested={requested_workers})"
    )

    embeddings: list[np.ndarray] = []
    image_path_strings: list[str] = []
    index_records: list[dict] = []

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_process_single_image, image_path, embedding): (image_path, embedding)
            for image_path, embedding in valid_items
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            image_path, embedding = futures[future]
            try:
                item = future.result()
            except Exception as exc:
                print(f"[WARN] Failed processing {image_path}: {exc}")
                continue

            embeddings.append(item["embedding"])
            image_path_strings.append(item["path"])
            index_records.append(
                {
                    "id": item["artifact_stem"],
                    "path": item["path"],
                    "embedding_path": item["embedding_file"],
                    "aesthetic_score": item["aesthetic_score"],
                    "blur_score": item["blur_score"],
                    "face_count": item["face_count"],
                    "person_count": item["person_count"],
                    "has_group": item["has_group"],
                    "has_bride_groom": item["has_bride_groom"],
                    "has_ritual": item["has_ritual"],
                    "yolo_labels": item["yolo_labels"],
                }
            )

    if embeddings:
        cluster_ids = assign_clusters(np.vstack(embeddings))
        cluster_by_path = {image_path: cluster_id for image_path, cluster_id in zip(image_path_strings, cluster_ids)}
        for item in index_records:
            item["cluster_id"] = int(cluster_by_path.get(item["path"], 0))

    ranked = compute_final_scores(index_records)
    if ranked:
        update_index_json(ranked)

    print("[INFO] Pipeline complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local photo selection pipeline")
    parser.add_argument("--input-folder", required=True, help="Path to folder containing images")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.input_folder)
