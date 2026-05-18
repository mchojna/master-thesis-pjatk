"""
tools.retriever
---------------
Embeds one query and retrieves the closest chunks from Qdrant with
a three-stage filter relaxation (company+product, then company-only,
then no filter). Resolves company/product aliases, normalises
diacritics for payload matching, and tracks per-call usage and
process metrics.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

import structlog
from qdrant_client.models import FieldCondition, Filter, MatchValue

from sources.config import config as app_config
from sources.config.graph import GraphConfig
from sources.agents.state import ExperimentState
from sources.tracker import (
    estimate_text_tokens,
    record_model_usage,
    record_tool_call,
    update_process_metrics,
)

logger = structlog.get_logger(__name__)


def _derive_filter_mode(company: str | None, product: str | None) -> str:
    """Return a metadata filter mode label for observability logging.

    Args:
        company (str | None): Resolved company identifier, or None.
        product (str | None): Resolved product identifier, or None.

    Returns:
        str: One of ``"company+product"``, ``"company"``, ``"product"``, or
            ``"none"``.
    """
    if company and product:
        return "company+product"
    if company:
        return "company"
    if product:
        return "product"
    return "none"


def _strip_diacritics(text: str) -> str:
    """Replace accented characters with their ASCII equivalents.

    Args:
        text (str): Input string that may contain diacritical marks.

    Returns:
        str: Input string with all combining diacritical marks removed.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")


def _normalize_meta_value(value: str | None) -> str | None:
    """Normalise a metadata field value to lowercase ASCII snake_case.

    Args:
        value (str | None): Raw metadata string to normalise.

    Returns:
        str | None: Normalised string matching the indexed payload format, or None
            when the input is empty or None.
    """
    if not value:
        return None
    normalized = value.strip().lower()
    normalized = _strip_diacritics(normalized)
    normalized = re.sub(r"[^\w\s-]", "", normalized)
    normalized = normalized.replace("-", "_")
    normalized = re.sub(r"\s+", "_", normalized)
    return normalized or None


def _resolve_company_alias(company: str | None) -> str | None:
    """Resolve a normalised company string to its canonical identifier.

    Args:
        company (str | None): Normalised company name to resolve.

    Returns:
        str | None: Canonical company identifier, the original value if no alias
            is registered, or None when the input is None.
    """
    if not company:
        return None
    return app_config.retriever.company_aliases.get(company, company)


def _resolve_product_alias(
    product: str | None,
    company: str | None,
) -> str | None:
    """Resolve a normalised product string to its canonical identifier.

    Args:
        product (str | None): Normalised product name to resolve.
        company (str | None): Resolved canonical company identifier used to select
            company-specific alias tables.

    Returns:
        str | None: Canonical product identifier, the original value if no alias
            is registered, or None when the input is None.
    """
    if not product:
        return None

    if company:
        canonical_products = app_config.retriever.canonical_products_by_company.get(
            company, set()
        )
        if product in canonical_products:
            return product

        company_aliases = app_config.retriever.product_aliases_by_company.get(
            company, {}
        )
        return company_aliases.get(product, product)

    return app_config.retriever.global_safe_product_aliases.get(product, product)


def _build_metadata_filter(
    company: str | None,
    product: str | None,
) -> Filter | None:
    """Build a Qdrant metadata filter for company and product.

    Args:
        company (str | None): Canonical company identifier, or None to omit the
            company condition.
        product (str | None): Canonical product identifier, or None to omit the
            product condition.

    Returns:
        Filter | None: Qdrant Filter with ``must`` conditions, or None when both
            arguments are None.
    """
    must: list[Any] = []
    if company:
        must.append(FieldCondition(key="company_name", match=MatchValue(value=company)))
    if product:
        must.append(FieldCondition(key="product_name", match=MatchValue(value=product)))
    return Filter(must=must) if must else None


def _build_filter_candidates(
    company: str | None,
    product: str | None,
) -> list[tuple[str, Filter | None]]:
    """Build an ordered list of filter candidates from strictest to most relaxed.

    Args:
        company (str | None): Canonical company identifier.
        product (str | None): Canonical product identifier.

    Returns:
        list[tuple[str, Filter | None]]: Ordered list of (mode_label, Filter) pairs
            ending with a ``("none", None)`` fallback.
    """
    candidates: list[tuple[str, Filter | None]] = []

    if company and product:
        candidates.append(("company+product", _build_metadata_filter(company, product)))
        candidates.append(("company", _build_metadata_filter(company, None)))
    elif company:
        candidates.append(("company", _build_metadata_filter(company, None)))
    elif product:
        candidates.append(("product", _build_metadata_filter(None, product)))

    candidates.append(("none", None))

    unique_candidates: list[tuple[str, Filter | None]] = []
    seen_modes: set[str] = set()
    for mode, query_filter in candidates:
        if mode in seen_modes:
            continue
        seen_modes.add(mode)
        unique_candidates.append((mode, query_filter))
    return unique_candidates


def _chunk_from_point(point: Any) -> dict[str, Any]:
    """Convert a Qdrant search point into the chunk structure used by agents.

    Args:
        point (Any): Qdrant ScoredPoint with ``payload`` and ``score`` attributes.

    Returns:
        dict[str, Any]: Chunk dict with ``text``, ``source_file``, ``page``,
            ``score``, and ``metadata`` keys.
    """
    payload = point.payload or {}
    return {
        "text": payload.get("content", ""),
        "source_file": payload.get("source_file", ""),
        "page": None,
        "score": point.score,
        "metadata": {
            "company_name": payload.get("company_name"),
            "product_name": payload.get("product_name"),
            "product_category": payload.get("product_category"),
            "document_type": payload.get("document_type"),
            "header_1": payload.get("header_1"),
            "header_2": payload.get("header_2"),
            "header_3": payload.get("header_3"),
        },
    }


async def retrieve(
    state: ExperimentState,
    *,
    config: GraphConfig,
    query: str | None = None,
    top_k: int | None = None,
) -> dict:
    """Embed the query and retrieve the closest chunks from Qdrant.

    Args:
        state (ExperimentState): Current experiment state; uses ``question``,
            ``company``, and ``product`` fields for filtering.
        config (GraphConfig): Runtime configuration providing the Qdrant client,
            embeddings client, collection name, and reranker settings.
        query (str | None): Override query string; defaults to ``state["question"]``
            when None.
        top_k (int | None): Override for the number of chunks to return; defaults to
            ``config.top_k`` when None.

    Returns:
        dict: State-update dict with keys ``retrieved_chunks`` (list of chunk dicts)
            and ``metadata``.
    """
    query = query or state["question"]
    requested_top_k = top_k or config.top_k
    retrieval_limit = requested_top_k
    if config.reranker_enabled:
        retrieval_limit = max(requested_top_k, config.reranker_top_k_before)
    retrieval_limit = max(app_config.retriever.min_retrieval_limit, retrieval_limit)
    metadata = record_tool_call(
        state.get("metadata"),
        "retriever",
        arguments={"top_k": requested_top_k},
    )
    parsed_company = _resolve_company_alias(_normalize_meta_value(state.get("company")))
    parsed_product = _resolve_product_alias(
        _normalize_meta_value(state.get("product")),
        parsed_company,
    )

    query_vector = await config.embeddings.aembed_query(query)
    metadata = record_model_usage(
        metadata,
        category="embedding",
        model_name=config.embedding_model,
        input_tokens=estimate_text_tokens(query, config.embedding_model),
        embedding=True,
    )

    minimum_hits = max(app_config.retriever.min_filtered_results, requested_top_k)
    collected_points: dict[str, Any] = {}
    filter_attempts: list[str] = []

    # Iterate from strictest filter (company+product) to no filter, stopping as
    # soon as we have enough results. This avoids over-fetching while still
    # falling back gracefully when a specific product has few indexed chunks.
    for filter_mode, query_filter in _build_filter_candidates(
        parsed_company,
        parsed_product,
    ):
        hits = await config.qdrant.query_points(
            collection_name=config.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=retrieval_limit,
        )
        filter_attempts.append(filter_mode)

        # Keep the best score per point ID across filter passes so that a chunk
        # retrieved by a loose filter never overwrites a higher-scored hit from
        # an earlier strict-filter pass.
        for point in hits.points:
            point_id = str(point.id)
            existing = collected_points.get(point_id)
            if existing is None or point.score > existing.score:
                collected_points[point_id] = point

        if len(collected_points) >= minimum_hits:
            break

    used_filter = " -> ".join(filter_attempts) or _derive_filter_mode(
        parsed_company,
        parsed_product,
    )
    ranked_points = sorted(
        collected_points.values(),
        key=lambda point: point.score,
        reverse=True,
    )[:retrieval_limit]
    chunks = [_chunk_from_point(point) for point in ranked_points]

    logger.info(
        "retrieve",
        query=query[:80],
        n_chunks=len(chunks),
        filter_mode=used_filter,
        company=parsed_company,
        product=parsed_product,
    )
    metadata = update_process_metrics(
        metadata,
        qdrant_query_count=int(
            metadata.get("_process", {}).get("qdrant_query_count", 0)
        )
        + 1,
        retrieved_chunk_count_total=int(
            metadata.get("_process", {}).get("retrieved_chunk_count_total", 0)
        )
        + len(chunks),
        last_retrieved_chunk_count=len(chunks),
        last_retrieval_limit=retrieval_limit,
        last_requested_top_k=requested_top_k,
        last_reranker_batch_count=(
            math.ceil(len(chunks) / max(1, config.reranker_batch_size))
            if config.reranker_enabled
            else 0
        ),
        last_retrieval_filter_mode=used_filter,
    )

    return {
        "retrieved_chunks": chunks,
        "metadata": metadata,
    }
