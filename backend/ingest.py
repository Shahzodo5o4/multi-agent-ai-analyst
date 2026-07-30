"""F2 — Ingestion: load → chunk → embed → store.

Run:  python ingest.py            # ingest everything in data/docs
      python ingest.py --reset    # wipe the collection first
      python ingest.py --test     # ingest, then prove similarity search works

Supports .md, .txt and .pdf. Drop your own files into data/docs and re-run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import config

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_documents(docs_dir: Path) -> List:
    """Read every supported file in `docs_dir` into LangChain Documents."""
    from langchain_core.documents import Document

    if not docs_dir.exists():
        raise FileNotFoundError(
            f"{docs_dir} does not exist. Create it and add .md/.txt/.pdf files."
        )

    documents: List[Document] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()

        if suffix in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            documents.append(Document(page_content=text, metadata={"source": path.name}))

        elif suffix == ".pdf":
            try:
                from langchain_community.document_loaders import PyPDFLoader
            except ImportError:
                print(f"  ! skipping {path.name} — run: pip install pypdf")
                continue
            for page in PyPDFLoader(str(path)).load():
                page.metadata["source"] = path.name
                documents.append(page)

        else:
            continue

        print(f"  loaded {path.name}")

    return documents


def chunk(documents: List) -> List:
    """Split into overlapping chunks. Overlap keeps a sentence that straddles a
    boundary retrievable from either side."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest(reset: bool = False) -> int:
    """Full pipeline. Returns the number of chunks stored."""
    from vectorstore import count, documents_store, drop_collection

    config.require_google_key()

    if reset:
        drop_collection(config.QDRANT_COLLECTION)
        print(f"  reset collection '{config.QDRANT_COLLECTION}' "
              f"({count(config.QDRANT_COLLECTION)} chunks remaining)")

    print(f"Loading from {config.DOCS_DIR}")
    documents = load_documents(config.DOCS_DIR)
    if not documents:
        print("No documents found. Add .md/.txt/.pdf files to data/docs.")
        return 0

    chunks = chunk(documents)
    print(f"  {len(documents)} documents → {len(chunks)} chunks")

    print(f"Embedding with {config.EMBEDDING_MODEL} and storing in Qdrant...")
    store = documents_store()
    # Batch the inserts: one call per chunk would burn through the free-tier
    # rate limit almost immediately.
    batch = 32
    for i in range(0, len(chunks), batch):
        store.add_documents(chunks[i:i + batch])
        print(f"  stored {min(i + batch, len(chunks))}/{len(chunks)}")

    return len(chunks)


def smoke_test() -> None:
    """F2's 'done when': a similarity search returns relevant chunks."""
    from vectorstore import documents_store

    queries = [
        "Why did customers churn in Q3 2024?",
        "What is the refund policy?",
    ]
    store = documents_store()
    for q in queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {q}")
        print("=" * 70)
        for i, doc in enumerate(store.similarity_search(q, k=2), 1):
            source = doc.metadata.get("source", "?")
            print(f"\n[{i}] from {source}")
            print("    " + doc.page_content[:280].replace("\n", "\n    "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant (F2)")
    parser.add_argument("--reset", action="store_true", help="wipe the collection first")
    parser.add_argument("--test", action="store_true", help="run a similarity search afterwards")
    args = parser.parse_args()

    try:
        stored = ingest(reset=args.reset)
    except Exception as exc:  # noqa: BLE001
        print(f"\nIngestion failed: {type(exc).__name__}: {exc}")
        sys.exit(1)

    print(f"\nDone — {stored} chunks in collection '{config.QDRANT_COLLECTION}'.")
    if args.test and stored:
        smoke_test()
