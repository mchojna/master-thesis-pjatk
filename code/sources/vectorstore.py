"""
vectorstore
-----------
Indexes and retrieves document chunks against a Qdrant collection.
Handles collection bootstrap, payload index creation for filtered
fields (company, product, headers), batched upserts with a fallback
for oversized batches, and an optional LLM-based reranking pass on
top of vector search.
"""

import json
import structlog
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from tqdm.auto import tqdm

from sources.config import (
    DocumentChunk,
    DocumentMetadata,
    RetrievalResult,
    config as app_config,
)
from sources.config.prompts import RERANKER_SYSTEM_PROMPT

logger = structlog.get_logger(__name__)


class VectorStore:
    """Index and retrieve document chunks against a Qdrant collection."""

    def __init__(self) -> None:
        """Create the vector store client and connect to Qdrant.

        Verifies the Qdrant connection, then ensures the collection and all
        required payload indexes exist.

        Raises:
            ConnectionError: If Qdrant is not reachable at the configured host and port.
        """
        self._app_config = app_config
        self._qdrant = QdrantClient(
            host=self._app_config.qdrant.host,
            port=self._app_config.qdrant.port,
        )
        self._embeddings = OpenAIEmbeddings(
            model=self._app_config.embedding.model_name,
        )
        self._reranker_llm = ChatOpenAI(
            model=self._app_config.reranking.model_name,
            temperature=self._app_config.reranking.temperature,
        )

        try:
            health = self._qdrant.get_collections()
            logger.info("qdrant_connected", collections=len(health.collections))
        except Exception as exc:
            raise ConnectionError(
                f"Failed to connect to Qdrant at {self._app_config.qdrant.host}:{self._app_config.qdrant.port}. "
                f"Ensure Qdrant is running. Error: {exc}"
            ) from exc

        self._ensure_collection()
        self._ensure_payload_indexes()

    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Index document chunks into the Qdrant collection.

        Embeds chunks in batches and upserts them. Falls back to smaller
        sub-batches if an oversized batch fails.

        Args:
            chunks (list[DocumentChunk]): Chunks to index.
        """
        if not chunks:
            logger.info("no_chunks_to_index")
            return

        logger.info("indexing_chunks", count=len(chunks))

        batch_size = 100          # normal upsert limit; keeps each request small
        fallback_batch_size = 25  # 4× smaller — recovers from oversized payloads (413 errors)
        indexed_points = 0

        def _upsert_points(points: list[PointStruct]) -> int:
            """Upsert one batch; retry with smaller sub-batches on failure."""
            try:
                self._qdrant.upsert(
                    collection_name=self._app_config.qdrant.collection_name,
                    points=points,
                )
                return len(points)
            except Exception as exc:
                logger.warning(
                    "batch_upsert_failed",
                    points=len(points),
                    error=str(exc),
                )

                if len(points) <= fallback_batch_size:
                    raise

                uploaded = 0
                for offset in range(0, len(points), fallback_batch_size):
                    sub_batch = points[offset : offset + fallback_batch_size]
                    self._qdrant.upsert(
                        collection_name=self._app_config.qdrant.collection_name,
                        points=sub_batch,
                    )
                    uploaded += len(sub_batch)
                return uploaded

        for i in tqdm(
            range(0, len(chunks), batch_size),
            desc="Indexing chunks",
            unit="batch",
            **self._app_config.tqdm.to_kwargs(),
        ):
            batch = chunks[i : i + batch_size]
            texts = [self._build_index_text(chunk) for chunk in batch]
            embeddings = self._embed_texts(texts)

            batch_points: list[PointStruct] = []

            for chunk, embedding in zip(batch, embeddings):
                payload = self._chunk_to_payload(chunk)
                point = PointStruct(
                    id=chunk.chunk_id,
                    vector=embedding,
                    payload=payload,
                )
                batch_points.append(point)

            indexed_points += _upsert_points(batch_points)

        logger.info("indexing_complete", indexed=indexed_points)

    def retrieve_context(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievalResult]:
        """Retrieve the most relevant chunks for a query without file or section filters.

        Args:
            query (str): Query text to embed and search.
            top_k (int | None): Maximum number of results to return. Uses the
                configured default when None.

        Returns:
            list[RetrievalResult]: Ranked retrieval results.
        """
        logger.info("retrieving_context", query=query[:80])
        return self._search(query=query, top_k=top_k)

    def retrieve_from_file(
        self,
        query: str,
        filename: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve chunks restricted to one source file.

        Args:
            query (str): Query text to embed and search.
            filename (str): Source file name to filter on (matched against the
                source_file payload field).
            top_k (int | None): Maximum number of results to return. Uses the
                configured default when None.

        Returns:
            list[RetrievalResult]: Ranked retrieval results from the specified file.
        """
        logger.info("retrieving_from_file", filename=filename, query=query[:80])

        query_filter = Filter(
            must=[
                FieldCondition(
                    key="source_file",
                    match=MatchValue(value=filename),
                )
            ]
        )
        return self._search(
            query=query,
            top_k=top_k,
            query_filter=query_filter,
        )

    def retrieve_from_section(
        self,
        query: str,
        section: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve chunks whose header_1, header_2, or header_3 matches a section name.

        Args:
            query (str): Query text to embed and search.
            section (str): Section name to match against header fields.
            top_k (int | None): Maximum number of results to return. Uses the
                configured default when None.

        Returns:
            list[RetrievalResult]: Ranked retrieval results from the specified section.
        """
        logger.info("retrieving_from_section", section=section, query=query[:80])

        query_filter = Filter(
            should=[
                FieldCondition(key="header_1", match=MatchValue(value=section)),
                FieldCondition(key="header_2", match=MatchValue(value=section)),
                FieldCondition(key="header_3", match=MatchValue(value=section)),
            ]
        )
        return self._search(
            query=query,
            top_k=top_k,
            query_filter=query_filter,
        )

    def _search(
        self,
        query: str,
        top_k: int | None,
        query_filter: Filter | None = None,
    ) -> list[RetrievalResult]:
        """Run vector search and apply optional LLM-based reranking.

        Args:
            query (str): Query text to embed.
            top_k (int | None): Number of results to return after reranking or
                vector search. Resolved via _resolve_requested_top_k when None.
            query_filter (Filter | None): Optional Qdrant filter to apply during
                vector search.

        Returns:
            list[RetrievalResult]: Final ranked retrieval results.
        """
        requested_top_k = self._resolve_requested_top_k(top_k)

        if self._app_config.reranking.enabled:
            retrieval_limit = max(
                requested_top_k,
                self._app_config.reranking.top_k_before_rerank,
            )
            final_k = requested_top_k
        else:
            retrieval_limit = requested_top_k
            final_k = requested_top_k

        query_embedding = self._embed_text(query)

        search_results = self._qdrant.query_points(
            collection_name=self._app_config.qdrant.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=retrieval_limit,
        ).points

        results: list[RetrievalResult] = []
        for hit in search_results:
            chunk = self._payload_to_chunk(str(hit.id), hit.payload or {})
            results.append(RetrievalResult(chunk=chunk, score=hit.score))

        if self._app_config.reranking.enabled and len(results) > 0:
            vector_ranked_results = results
            logger.info("reranking_results", count=len(results), top_k=final_k)
            results = self._rerank(query, results, final_k)
            if not results:
                logger.warning(
                    "reranking_empty_fallback",
                    top_k=final_k,
                    threshold=self._app_config.reranking.score_threshold,
                )
                # When all reranker scores fall below threshold, preserve at least
                # some results — vector order is still informative even without
                # LLM-scored relevance.
                results = vector_ranked_results[:final_k]
        else:
            results = results[:final_k]

        return results

    def _resolve_requested_top_k(self, top_k: int | None) -> int:
        """Resolve the effective top-k value from an explicit argument or configuration.

        Args:
            top_k (int | None): Caller-supplied top-k override, or None to use config.

        Returns:
            int: Effective top-k value, always at least 1.
        """
        if top_k is not None:
            return max(1, top_k)
        if self._app_config.reranking.enabled:
            return max(1, self._app_config.reranking.top_k_after_rerank)
        return 5

    def _rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Score and rerank retrieval results using the LLM reranker.

        Processes results in parallel batches, applies a score threshold,
        sorts by score descending, and returns at most top_k results.

        Args:
            query (str): Original query used to score relevance.
            results (list[RetrievalResult]): Candidate results from vector search.
            top_k (int): Maximum number of results to return after reranking.

        Returns:
            list[RetrievalResult]: Reranked and filtered results, sorted by score
                descending. Empty list if all results fall below the score threshold.
        """
        logger.info("reranking_start", candidates=len(results))

        reranked_results: list[RetrievalResult] = []
        max_workers = self._app_config.reranking.max_workers
        batch_size = max(1, self._app_config.reranking.batch_size)

        def _score_chunk(result: RetrievalResult) -> RetrievalResult:
            """Score one chunk against the query using the reranker LLM."""
            user_message = (
                f"Question: {query}\n\n"
                f"Chunks:\n{json.dumps([{'index': 0, 'text': result.chunk.content[:1000]}], ensure_ascii=False)}"
            )

            try:
                response = self._reranker_llm.invoke(
                    [
                        SystemMessage(content=RERANKER_SYSTEM_PROMPT),
                        HumanMessage(content=user_message),
                    ]
                )

                score_text = (
                    response.content.strip()
                    if isinstance(response.content, str)
                    else ""
                )
                try:
                    parsed_scores = json.loads(
                        score_text.strip()
                        .removeprefix("```json")
                        .removeprefix("```")
                        .removesuffix("```")
                        .strip()
                    )
                    normalized_score = float(parsed_scores[0].get("score", 0.5))
                    normalized_score = max(0.0, min(1.0, normalized_score))
                except (ValueError, json.JSONDecodeError, IndexError, AttributeError):
                    logger.warning(
                        "rerank_score_parse_failed",
                        raw=score_text,
                    )
                    normalized_score = 0.5

                return RetrievalResult(chunk=result.chunk, score=normalized_score)

            except Exception as exc:
                logger.warning("rerank_scoring_failed", error=str(exc))
                # Halve the vector score on API failure so these hits sort below
                # successfully reranked chunks; they remain retrievable but deprioritised.
                return RetrievalResult(chunk=result.chunk, score=result.score * 0.5)

        for offset in tqdm(
            range(0, len(results), batch_size),
            total=(len(results) + batch_size - 1) // batch_size,
            desc="Reranking batches",
            unit="batch",
            **self._app_config.tqdm.to_kwargs(),
        ):
            batch = results[offset : offset + batch_size]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_result = {
                    executor.submit(_score_chunk, result): result for result in batch
                }

                for future in as_completed(future_to_result):
                    try:
                        reranked_result = future.result()
                        reranked_results.append(reranked_result)
                    except Exception as exc:
                        logger.warning("rerank_unexpected_error", error=str(exc))
                        original = future_to_result[future]
                        reranked_results.append(
                            RetrievalResult(
                                chunk=original.chunk,
                                score=original.score * 0.5,
                            )
                        )

        # Clamp to [0, 1] to guard against misconfigured thresholds; the reranker LLM
        # always returns scores in this range so values outside it indicate config bugs.
        score_threshold = max(
            0.0,
            min(1.0, self._app_config.reranking.score_threshold),
        )
        reranked_results = [
            result for result in reranked_results if result.score >= score_threshold
        ]

        reranked_results.sort(key=lambda x: x.score, reverse=True)
        top_results = reranked_results[:top_k]

        logger.info(
            "reranking_complete",
            returned=len(top_results),
            threshold=score_threshold,
            top_score=top_results[0].score if top_results else 0.0,
            bottom_score=top_results[-1].score if top_results else 0.0,
        )

        return top_results

    def _embed_text(self, text: str) -> list[float]:
        """Embed one text string using the configured embedding model.

        Args:
            text (str): Text to embed.

        Returns:
            list[float]: Embedding vector.
        """
        return self._embeddings.embed_query(text)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings in one batch call.

        Args:
            texts (list[str]): Texts to embed.

        Returns:
            list[list[float]]: List of embedding vectors, one per input text.
        """
        return self._embeddings.embed_documents(texts)

    @staticmethod
    def _build_index_text(chunk: DocumentChunk) -> str:
        """Build a retrieval-oriented text representation of a chunk for embedding.

        Prepends structured metadata fields (company, product, category, document
        type, section path) to the chunk content.

        Args:
            chunk (DocumentChunk): Chunk to represent.

        Returns:
            str: Formatted text combining metadata and chunk content.
        """
        metadata = chunk.metadata
        header_path = " > ".join(
            header
            for header in (
                metadata.header_1,
                metadata.header_2,
                metadata.header_3,
            )
            if header
        )

        lines = [
            f"Firma: {metadata.company_name}",
            f"Produkt: {metadata.product_name}",
            f"Kategoria: {metadata.product_category}",
            f"Typ dokumentu: {metadata.document_type}",
        ]
        if header_path:
            lines.append(f"Sekcja: {header_path}")

        lines.extend(
            [
                "",
                "Fragment:",
                chunk.content,
            ]
        )

        return "\n".join(lines)

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        collection_name = self._app_config.qdrant.collection_name
        collections = self._qdrant.get_collections().collections
        existing_names = {c.name for c in collections}

        if collection_name not in existing_names:
            self._qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self._app_config.embedding.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("collection_created", collection=collection_name)
        else:
            logger.info("collection_exists", collection=collection_name)

    def _ensure_payload_indexes(self) -> None:
        """Create keyword payload indexes for all filterable fields if not present."""
        collection_name = self._app_config.qdrant.collection_name
        index_fields: dict[str, PayloadSchemaType] = {
            "source_file": PayloadSchemaType.KEYWORD,
            "document_type": PayloadSchemaType.KEYWORD,
            "company_name": PayloadSchemaType.KEYWORD,
            "product_name": PayloadSchemaType.KEYWORD,
            "product_category": PayloadSchemaType.KEYWORD,
            "header_1": PayloadSchemaType.KEYWORD,
            "header_2": PayloadSchemaType.KEYWORD,
            "header_3": PayloadSchemaType.KEYWORD,
        }

        for field_name, field_schema in index_fields.items():
            try:
                self._qdrant.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                    wait=True,
                )
                logger.info("payload_index_ensured", field=field_name)
            except Exception as exc:
                logger.warning(
                    "payload_index_skipped",
                    field=field_name,
                    error=str(exc),
                )

    @staticmethod
    def _chunk_to_payload(chunk: DocumentChunk) -> dict[str, Any]:
        """Convert a DocumentChunk to a flat Qdrant payload dictionary.

        Args:
            chunk (DocumentChunk): Chunk to convert.

        Returns:
            dict[str, Any]: Payload dict containing content and all metadata fields.
        """
        return {
            "content": chunk.content,
            "company_name": chunk.metadata.company_name,
            "product_name": chunk.metadata.product_name,
            "product_category": chunk.metadata.product_category,
            "document_type": chunk.metadata.document_type,
            "source_file": chunk.metadata.source_file,
            "header_1": chunk.metadata.header_1,
            "header_2": chunk.metadata.header_2,
            "header_3": chunk.metadata.header_3,
        }

    @staticmethod
    def _payload_to_chunk(chunk_id: str, payload: dict[str, Any]) -> DocumentChunk:
        """Build a DocumentChunk from a Qdrant point payload.

        Args:
            chunk_id (str): String representation of the Qdrant point ID.
            payload (dict[str, Any]): Payload dict as stored in Qdrant.

        Returns:
            DocumentChunk: Reconstructed chunk with metadata populated from the payload.
        """
        metadata = DocumentMetadata(
            company_name=payload.get("company_name", ""),
            product_name=payload.get("product_name", ""),
            product_category=payload.get("product_category", ""),
            document_type=payload.get("document_type", ""),
            source_file=payload.get("source_file", ""),
            header_1=payload.get("header_1"),
            header_2=payload.get("header_2"),
            header_3=payload.get("header_3"),
        )
        return DocumentChunk(
            chunk_id=chunk_id,
            content=payload.get("content", ""),
            metadata=metadata,
        )
