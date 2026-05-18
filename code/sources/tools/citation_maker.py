"""
tools.citation_maker
--------------------
Builds structured citation objects from the selected evidence chunks
(or the raw retrieved chunks when no evidence step ran). Each citation
carries a 1-based ``inline_ref`` token (``[1]``, ``[2]``, ...) used
by the answer synthesiser and a short excerpt for downstream
reference rendering.
"""

from __future__ import annotations

import structlog

from sources.config.graph import GraphConfig
from sources.agents.state import ExperimentState, deduplicate_chunks
from sources.tracker import record_tool_call, update_process_metrics

logger = structlog.get_logger(__name__)


async def make_citations(
    state: ExperimentState,
    *,
    config: GraphConfig,
) -> dict:
    """Build citation objects from the available evidence or retrieved chunks.

    Args:
        state (ExperimentState): Current experiment state; uses ``evidence_chunks``
            when populated, otherwise falls back to ``retrieved_chunks``.
        config (GraphConfig): Runtime configuration (not used directly but required
            by the uniform tool signature).

    Returns:
        dict: State-update dict with keys ``citations`` (list of citation dicts) and
            ``metadata``.
    """
    chunks = state.get("evidence_chunks") or state.get("retrieved_chunks", [])
    metadata = record_tool_call(state.get("metadata"), "citation_maker")
    sorted_chunks = deduplicate_chunks(chunks)

    citations: list[dict] = []
    for idx, chunk in enumerate(sorted_chunks, start=1):
        text = chunk.get("text", "")
        # 200 chars: long enough to identify the source in the references footer,
        # short enough to fit without truncating the answer body in the UI.
        excerpt = text[:200].replace("\n", " ").strip()
        meta = chunk.get("metadata", {})
        # Prefer header_2 (section level) over header_1 (document title, too broad)
        # and header_3 (sub-clause, may be absent).
        header = (
            meta.get("header_2") or meta.get("header_1") or meta.get("header_3") or ""
        )
        citations.append(
            {
                "id": idx,
                "source_file": chunk.get("source_file", ""),
                "document_type": meta.get("document_type", ""),
                "company_name": meta.get("company_name", ""),
                "product_name": meta.get("product_name", ""),
                "header": header,
                "page": chunk.get("page"),
                "excerpt": excerpt,
                "relevance_score": round(chunk.get("score", 0.0), 3),
                "inline_ref": f"[{idx}]",
            }
        )

    logger.info(
        "make_citations",
        n_citations=len(citations),
        n_input_chunks=len(chunks),
    )
    metadata = update_process_metrics(
        metadata,
        citation_count_generated=len(citations),
    )

    return {
        "citations": citations,
        "metadata": metadata,
    }
