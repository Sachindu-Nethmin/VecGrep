"""FastMCP server exposing VecGrep tools."""

from __future__ import annotations

import fnmatch
import hashlib
import os
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP

from vecgrep.chunker import chunk_file
from vecgrep.embedder import embed
from vecgrep.store import VectorStore

# ---------------------------------------------------------------------------
# MCP server setup
# ---------------------------------------------------------------------------

mcp = FastMCP("vecgrep")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VECGREP_HOME = Path.home() / ".vecgrep"

ALWAYS_SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".env",
    "dist",
    "build",
    "target",
    ".next",
    ".nuxt",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "eggs",
    ".eggs",
    "htmlcov",
}

ALWAYS_SKIP_PATTERNS = [
    "*.min.js",
    "*.bundle.js",
    "*.lock",
    "*.pyc",
    "*.class",
    "*.o",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.exe",
    "*.DS_Store",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.svg",
    "*.ico",
    "*.pdf",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.whl",
    "*.egg",
]

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".rs", ".go",
    ".java", ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",
    ".rb", ".swift", ".kt", ".cs", ".md", ".txt", ".yaml",
    ".yml", ".toml", ".json", ".sh", ".bash", ".zsh",
    ".fish", ".html", ".css", ".scss", ".less", ".sql",
    ".graphql", ".proto", ".tf", ".hcl", ".dockerfile",
    ".vue", ".svelte",
}

MAX_FILE_BYTES = 512 * 1024  # 512 KB — skip very large files
EMBED_BATCH = 64

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_hash(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()[:16]


def _get_store(path: str) -> VectorStore:
    index_dir = VECGREP_HOME / _project_hash(path)
    return VectorStore(index_dir)


def _sha256_file(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _load_gitignore(root: Path) -> list[str]:
    gitignore = root / ".gitignore"
    patterns: list[str] = []
    if gitignore.exists():
        for line in gitignore.read_text(errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


def _is_ignored_by_gitignore(rel_path: str, patterns: list[str]) -> bool:
    parts = Path(rel_path).parts
    for pattern in patterns:
        # Match against the full relative path
        if fnmatch.fnmatch(rel_path, pattern):
            return True
        # Match against each path component
        for part in parts:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


def _should_skip_file(file_path: Path) -> bool:
    name = file_path.name
    for pattern in ALWAYS_SKIP_PATTERNS:
        if fnmatch.fnmatch(name, pattern):
            return True
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return True
    try:
        if file_path.stat().st_size > MAX_FILE_BYTES:
            return True
    except OSError:
        return True
    return False


def _walk_files(root: Path, gitignore_patterns: list[str]) -> list[Path]:
    """Collect all indexable files under root."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune directories in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in ALWAYS_SKIP_DIRS
            and not _is_ignored_by_gitignore(
                str(Path(dirpath).relative_to(root) / d), gitignore_patterns
            )
        ]
        for fname in filenames:
            fp = Path(dirpath) / fname
            try:
                rel = str(fp.relative_to(root))
            except ValueError:
                rel = str(fp)
            if _is_ignored_by_gitignore(rel, gitignore_patterns):
                continue
            if _should_skip_file(fp):
                continue
            files.append(fp)
    return files


def _do_index(path: str, force: bool = False) -> str:
    root = Path(path).resolve()
    if not root.exists():
        return f"Error: path does not exist: {path}"

    store = _get_store(str(root))
    gitignore = _load_gitignore(root)
    all_files = _walk_files(root, gitignore)

    existing_hashes = {} if force else store.get_file_hashes()

    new_chunks_rows: list[dict] = []
    new_chunks_texts: list[str] = []
    files_changed = 0
    files_skipped = 0

    for fp in all_files:
        file_hash = _sha256_file(fp)
        fp_str = str(fp)

        if not force and existing_hashes.get(fp_str) == file_hash:
            files_skipped += 1
            continue

        # Remove stale chunks for this file
        if fp_str in existing_hashes:
            store.delete_file_chunks(fp_str)

        chunks = chunk_file(fp_str)
        for chunk in chunks:
            chunk_hash = _sha256_str(chunk.content)
            new_chunks_rows.append(
                {
                    "file_path": fp_str,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "content": chunk.content,
                    "file_hash": file_hash,
                    "chunk_hash": chunk_hash,
                }
            )
            new_chunks_texts.append(chunk.content)

        files_changed += 1

    # Embed and store in batches
    total_new_chunks = len(new_chunks_rows)
    if total_new_chunks > 0:
        for i in range(0, total_new_chunks, EMBED_BATCH):
            batch_rows = new_chunks_rows[i : i + EMBED_BATCH]
            batch_texts = new_chunks_texts[i : i + EMBED_BATCH]
            vecs = embed(batch_texts)
            store.add_chunks(batch_rows, vecs)

    store.touch_last_indexed()
    store.close()

    return (
        f"Indexed {files_changed} file(s), {total_new_chunks} chunk(s) added "
        f"({files_skipped} file(s) skipped, unchanged)"
    )


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def index_codebase(path: str, force: bool = False) -> str:
    """
    Index a codebase directory for semantic search.

    Walks the directory, extracts semantic code chunks using AST analysis,
    embeds them locally with sentence-transformers, and stores in a vector index.
    Subsequent calls skip unchanged files (incremental updates).

    Args:
        path: Absolute path to the codebase root directory.
        force: If True, re-index all files even if unchanged.

    Returns:
        Summary: files indexed, chunks added, files skipped.
    """
    return _do_index(path, force=force)


@mcp.tool()
def search_code(query: str, path: str, top_k: int = 8) -> str:
    """
    Semantically search an indexed codebase for code relevant to a query.

    Embeds the query and performs cosine similarity search against indexed
    code chunks, returning the most semantically relevant snippets with
    file paths and line numbers.

    If the codebase is not yet indexed, it will be indexed automatically first.

    Args:
        query: Natural language description of what you're looking for.
               E.g. "how does authentication work", "database connection setup"
        path: Absolute path to the codebase root directory.
        top_k: Number of results to return (default 8, max 20).

    Returns:
        Formatted list of matching code chunks with file:line references and
        similarity scores.
    """
    top_k = min(top_k, 20)
    root = Path(path).resolve()

    store = _get_store(str(root))
    status = store.status()

    # Auto-index if no data
    if status["total_chunks"] == 0:
        store.close()
        index_result = _do_index(str(root), force=False)
        store = _get_store(str(root))
        status = store.status()
        if status["total_chunks"] == 0:
            store.close()
            return f"No indexable files found in {path}.\n(Index attempt: {index_result})"

    query_vec = embed([query])[0]
    results = store.search(query_vec, top_k=top_k)
    store.close()

    if not results:
        return "No results found. Try re-indexing with index_codebase()."

    lines = [f"Top {len(results)} results for: '{query}'\n"]
    for i, r in enumerate(results, 1):
        # Make path relative for readability
        try:
            rel = str(Path(r["file_path"]).relative_to(root))
        except ValueError:
            rel = r["file_path"]
        lines.append(f"[{i}] {rel}:{r['start_line']}-{r['end_line']} (score: {r['score']:.2f})")
        lines.append(r["content"])
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def get_index_status(path: str) -> str:
    """
    Get the status of the vector index for a codebase.

    Args:
        path: Absolute path to the codebase root directory.

    Returns:
        Index statistics: file count, chunk count, last indexed time, disk usage.
    """
    root = Path(path).resolve()
    store = _get_store(str(root))
    s = store.status()
    store.close()

    size_mb = s["index_size_bytes"] / (1024 * 1024)
    return (
        f"Index status for: {root}\n"
        f"  Files indexed:  {s['total_files']}\n"
        f"  Total chunks:   {s['total_chunks']}\n"
        f"  Last indexed:   {s['last_indexed']}\n"
        f"  Index size:     {size_mb:.1f} MB"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
