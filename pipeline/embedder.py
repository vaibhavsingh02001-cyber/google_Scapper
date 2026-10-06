"""
pipeline/embedder.py — Chroma vector store & OpenAI embedding layer (Phase 2).

LLM split:
  - Groq         → text classification / tagging (Phase 3)
  - OpenAI        → text-embedding-3-small embeddings (this file)
    Groq does not offer an embedding API, so OpenAI is used solely
    for generating dense vectors stored in Chroma.

Responsibilities:
  - Initialize a persistent Chroma collection ("feedback_items")
  - Batch-embed RawItems via OpenAI embeddings API
  - Upsert vectors + metadata into Chroma
  - Provide similarity search (query_similar)
  - Provide full embedding retrieval for HDBSCAN clustering (Phase 4)
"""
from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from models.schemas import RawItem
import config


class Embedder:
    """
    Manages the Chroma persistent vector store for feedback item embeddings.
    All embedding calls go to OpenAI text-embedding-3-small.
    """

    def __init__(
        self,
        chroma_path: str = config.CHROMA_PATH,
        collection_name: str = config.CHROMA_COLLECTION,
    ):
        self.chroma_path     = chroma_path
        self.collection_name = collection_name
        self._client         = None
        self._collection     = None
        self._openai         = None
        self._init_chroma()
        self._init_openai()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_chroma(self) -> None:
        """Create or open the persistent Chroma collection."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb not installed. Run: pip install chromadb"
            )

        os.makedirs(self.chroma_path, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self.chroma_path)

        # get_or_create: safe to call on every run
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": config.CHROMA_DISTANCE},
        )
        logger.debug(
            f"[embedder] Chroma collection '{self.collection_name}' ready "
            f"({self._collection.count()} vectors stored)"
        )

    def _init_openai(self) -> None:
        """Initialise the OpenAI client for embedding calls."""
        if not config.OPENAI_API_KEY:
            logger.warning(
                "[embedder] OPENAI_API_KEY not set. "
                "embed_and_store() will be skipped. "
                "Set the key in .env to enable semantic clustering."
            )
            return

        try:
            from openai import OpenAI
            self._openai = OpenAI(api_key=config.OPENAI_API_KEY)
            logger.debug("[embedder] OpenAI client initialised (embeddings only)")
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_and_store(self, items: List[RawItem]) -> int:
        """
        Generate embeddings for a list of RawItems and upsert them into Chroma.
        Items already present (by id) are updated.

        Args:
            items: List of RawItem objects to embed.

        Returns:
            Number of items successfully embedded and stored.
        """
        if not self._openai:
            logger.warning(
                "[embedder] Skipping embedding — OPENAI_API_KEY not configured."
            )
            return 0

        if not items:
            return 0

        # Filter out items already in Chroma to avoid redundant API calls
        existing_ids = self._get_existing_ids([item.id for item in items])
        new_items    = [i for i in items if i.id not in existing_ids]

        if not new_items:
            logger.debug("[embedder] All items already embedded — nothing to do.")
            return 0

        logger.info(
            f"[embedder] Embedding {len(new_items)} items "
            f"(skipping {len(existing_ids)} already stored)..."
        )

        total_stored = 0
        batch_size   = config.EMBEDDING_BATCH_SIZE

        for batch_start in range(0, len(new_items), batch_size):
            batch = new_items[batch_start : batch_start + batch_size]
            texts = [item.text for item in batch]

            try:
                vectors = self._embed_texts(texts)
            except Exception as exc:
                logger.error(
                    f"[embedder] Embedding batch {batch_start}–"
                    f"{batch_start + len(batch)} failed: {exc}"
                )
                continue

            # Build Chroma upsert payload
            ids        = [item.id for item in batch]
            metadatas  = [self._item_metadata(item) for item in batch]
            documents  = texts  # store raw text for retrieval

            self._collection.upsert(
                ids=ids,
                embeddings=vectors,
                metadatas=metadatas,
                documents=documents,
            )
            total_stored += len(batch)
            logger.debug(
                f"[embedder] Upserted batch {batch_start // batch_size + 1} "
                f"({len(batch)} items)"
            )

            # Brief pause to stay within OpenAI rate limits
            time.sleep(0.3)

        logger.info(f"[embedder] ✅ Stored {total_stored} embeddings in Chroma")
        return total_stored

    def query_similar(
        self,
        text: str,
        n: int = 10,
        where: Optional[dict] = None,
    ) -> List[dict]:
        """
        Find the n most similar items to a query text using cosine similarity.

        Args:
            text:  Query string to embed and search.
            n:     Number of nearest neighbours to return.
            where: Optional Chroma metadata filter dict.

        Returns:
            List of dicts with keys: id, text, distance, metadata.
        """
        if not self._openai:
            logger.warning("[embedder] Cannot query — OPENAI_API_KEY not set.")
            return []

        query_vector = self._embed_texts([text])[0]

        kwargs: dict = {"query_embeddings": [query_vector], "n_results": min(n, self.count())}
        if where:
            kwargs["where"] = where

        results = self._collection.query(
            include=["documents", "distances", "metadatas"],
            **kwargs,
        )

        hits = []
        for i, doc_id in enumerate(results["ids"][0]):
            hits.append(
                {
                    "id":       doc_id,
                    "text":     results["documents"][0][i],
                    "distance": results["distances"][0][i],
                    "metadata": results["metadatas"][0][i],
                }
            )
        return hits

    def get_all_embeddings(self) -> Tuple[List[str], List[list], List[dict]]:
        """
        Retrieve all stored embeddings for HDBSCAN clustering (Phase 4).

        Returns:
            Tuple of (ids, embeddings, metadatas).
        """
        if self.count() == 0:
            logger.warning("[embedder] Chroma collection is empty.")
            return [], [], []

        result = self._collection.get(include=["embeddings", "metadatas"])
        return result["ids"], result["embeddings"], result["metadatas"]

    def count(self) -> int:
        """Return the number of vectors stored in the Chroma collection."""
        return self._collection.count()

    def collection_info(self) -> dict:
        """Return summary information about the Chroma collection."""
        return {
            "collection":   self.collection_name,
            "chroma_path":  self.chroma_path,
            "vector_count": self.count(),
            "distance":     config.CHROMA_DISTANCE,
            "model":        config.EMBEDDING_MODEL,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _embed_texts(self, texts: List[str]) -> List[list]:
        """
        Call the OpenAI embeddings API for a batch of texts.
        Retries up to 4× with exponential backoff on failure.
        """
        response = self._openai.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=texts,
            dimensions=config.EMBEDDING_DIMENSIONS,
        )
        return [item.embedding for item in response.data]

    def _get_existing_ids(self, ids: List[str]) -> set:
        """Check which of the given IDs already exist in Chroma."""
        if not ids:
            return set()
        try:
            result = self._collection.get(ids=ids, include=[])
            return set(result["ids"])
        except Exception:
            return set()

    @staticmethod
    def _item_metadata(item: RawItem) -> dict:
        """
        Build the Chroma metadata dict for a RawItem.
        Chroma metadata values must be str | int | float | bool only.
        """
        return {
            "source":   item.source.value,
            "date":     item.date or "",
            "upvotes":  item.upvotes,
            "url":      item.url or "",
        }

