"""F2 / F10 — Qdrant in embedded mode.

Embedded means Qdrant runs *inside this process* and writes to a local folder.
No Docker, no signup, no server, no credit card — it just works offline.

One important constraint: an embedded Qdrant folder can only be opened by ONE
process at a time. The API server and the ingest script therefore share a single
cached client rather than each opening their own.
"""

from __future__ import annotations

import atexit
from functools import lru_cache

import config
from llm import get_embeddings


@lru_cache(maxsize=1)
def get_client():
    """The single embedded Qdrant client for this process."""
    from qdrant_client import QdrantClient

    config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(config.QDRANT_PATH))

    # qdrant-client closes itself in __del__, which can run after the interpreter
    # has torn down sys.meta_path — that raises a noisy but harmless ImportError
    # at exit. Closing deterministically via atexit avoids it.
    atexit.register(_close_quietly, client)
    return client


def _close_quietly(client) -> None:
    try:
        client.close()
    except Exception:  # noqa: BLE001
        pass


def ensure_collection(name: str) -> None:
    """Create the collection if it doesn't exist.

    Watch out (guide, F2): the vector size here MUST match what the embedding
    model produces, or every insert fails with a dimension error. Both come from
    config, so they can't drift apart.
    """
    from qdrant_client.models import Distance, VectorParams

    client = get_client()
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=config.EMBEDDING_DIM, distance=Distance.COSINE
            ),
        )


@lru_cache(maxsize=4)
def get_store(collection: str):
    """A LangChain vector store bound to `collection`."""
    from langchain_qdrant import QdrantVectorStore

    ensure_collection(collection)
    return QdrantVectorStore(
        client=get_client(),
        collection_name=collection,
        embedding=get_embeddings(),
    )


def documents_store():
    """F2 — the ingested knowledge base the retriever agent searches."""
    return get_store(config.QDRANT_COLLECTION)


def memory_store():
    """F10 — past Q/A turns, kept in a separate collection from the documents."""
    return get_store(config.MEMORY_COLLECTION)


def close_client() -> None:
    """Close the embedded client and drop every cached handle to it.

    Required before touching the on-disk files: Windows keeps storage.sqlite
    locked while the client holds it open.
    """
    if get_client.cache_info().currsize:
        _close_quietly(get_client())
    get_client.cache_clear()
    get_store.cache_clear()


def drop_collection(name: str) -> None:
    """Genuinely delete a collection, on disk as well as in memory.

    `client.delete_collection()` alone is not enough in embedded mode: it updates
    meta.json but leaves the collection's storage.sqlite behind, so recreating
    the collection silently resurrects every old point. Both `ingest.py --reset`
    and `memory.py --reset` reported success while changing nothing.

    Order matters. Deleting the collection first orphans its sqlite handle —
    still open, so on Windows the file cannot be removed (WinError 32). Closing
    the whole client first releases every handle, and only then can the
    directory go.
    """
    import gc
    import shutil

    close_client()
    gc.collect()  # drop any lingering reference to the sqlite connections

    folder = config.QDRANT_PATH / "collection" / name
    if folder.exists():
        try:
            shutil.rmtree(folder)
        except OSError as exc:
            # Never fail silently here: a reset that quietly does nothing is
            # far worse than one that says it could not.
            raise RuntimeError(
                f"Could not delete {folder}: {exc}. Close any other process "
                "using this Qdrant folder (a running uvicorn server holds it "
                "open) and try again."
            ) from exc

    # Reopening rereads meta.json, which may still list the now-fileless
    # collection; clear that entry before recreating it empty.
    client = get_client()
    if client.collection_exists(name):
        client.delete_collection(name)
    ensure_collection(name)


def count(collection: str) -> int:
    """How many vectors are stored — used by /health so the UI can warn when the
    knowledge base is empty."""
    try:
        ensure_collection(collection)
        return get_client().count(collection_name=collection).count
    except Exception:
        return 0
