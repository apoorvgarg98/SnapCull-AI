from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class Database:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row

    def initialize(self) -> None:
        query = """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            embedding_path TEXT,
            aesthetic_score REAL,
            blur_score REAL,
            face_count INTEGER,
            cluster_id INTEGER,
            final_score REAL
        );
        """
        self.connection.execute(query)
        self.connection.commit()

    def upsert_image(
        self,
        path: str,
        embedding_path: str,
        aesthetic_score: float,
        blur_score: float,
        face_count: int,
    ) -> None:
        query = """
        INSERT INTO images (path, embedding_path, aesthetic_score, blur_score, face_count)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            embedding_path = excluded.embedding_path,
            aesthetic_score = excluded.aesthetic_score,
            blur_score = excluded.blur_score,
            face_count = excluded.face_count;
        """
        self.connection.execute(query, (path, embedding_path, aesthetic_score, blur_score, face_count))
        self.connection.commit()

    def update_cluster(self, image_path: str, cluster_id: int) -> None:
        self.connection.execute("UPDATE images SET cluster_id = ? WHERE path = ?", (cluster_id, image_path))
        self.connection.commit()

    def update_final_score(self, image_path: str, final_score: float) -> None:
        self.connection.execute("UPDATE images SET final_score = ? WHERE path = ?", (final_score, image_path))
        self.connection.commit()

    def fetch_all_images(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM images ORDER BY id ASC").fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self.connection.close()
