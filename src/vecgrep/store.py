"""SQLite-backed vector store with numpy cosine similarity search."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    chunk_hash TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_hash ON chunks(chunk_hash);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class VectorStore:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = index_dir / "index.db"
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # File hash helpers
    # ------------------------------------------------------------------

    def get_file_hashes(self) -> dict[str, str]:
        """Return {file_path: file_hash} for all indexed files."""
        rows = self._conn.execute(
            "SELECT DISTINCT file_path, file_hash FROM chunks"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def delete_file_chunks(self, file_path: str) -> None:
        """Remove all chunks for a given file."""
        self._conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
        self._conn.commit()

    def add_chunks(self, rows: list[dict], vectors: np.ndarray) -> None:
        """
        Insert chunk rows with their embeddings.

        rows: list of dicts with keys: file_path, start_line, end_line,
              content, file_hash, chunk_hash
        vectors: float32 array of shape (len(rows), 384) — pre-normalised
        """
        assert len(rows) == len(vectors), "rows/vectors length mismatch"
        params = [
            (
                r["file_path"],
                r["start_line"],
                r["end_line"],
                r["content"],
                r["file_hash"],
                r["chunk_hash"],
                vectors[i].tobytes(),
            )
            for i, r in enumerate(rows)
        ]
        self._conn.executemany(
            "INSERT INTO chunks "
            "(file_path, start_line, end_line, content, file_hash, chunk_hash, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            params,
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vec: np.ndarray, top_k: int = 8) -> list[dict]:
        """
        Cosine similarity search. query_vec must be pre-normalised (shape 384,).
        Returns list of dicts: {file_path, start_line, end_line, content, score}.
        """
        rows = self._conn.execute(
            "SELECT id, file_path, start_line, end_line, content, embedding FROM chunks"
        ).fetchall()

        if not rows:
            return []

        ids = [r[0] for r in rows]
        meta = [(r[1], r[2], r[3], r[4]) for r in rows]
        vectors = np.frombuffer(b"".join(r[5] for r in rows), dtype=np.float32).reshape(
            len(rows), -1
        )

        scores = vectors @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "file_path": meta[i][0],
                "start_line": meta[i][1],
                "end_line": meta[i][2],
                "content": meta[i][3],
                "score": float(scores[i]),
            }
            for i in top_indices
        ]

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        total_chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        total_files = self._conn.execute(
            "SELECT COUNT(DISTINCT file_path) FROM chunks"
        ).fetchone()[0]
        last_indexed = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'last_indexed'"
        ).fetchone()
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "last_indexed": last_indexed[0] if last_indexed else "never",
            "index_size_bytes": db_size,
        }

    def touch_last_indexed(self) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_indexed', ?)",
            (ts,),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
