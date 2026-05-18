"""
tools
-----
Leaf-level agent tools (question parser, query rewriter, retriever,
reranker, evidence selector, citation maker, prompt selector, answer
synthesiser) plus a multi-query retrieval helper and the
``TOOL_REGISTRY`` consumed by the planner-executor and ReAct
architectures.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine

from sources.agents.state import ExperimentState, deduplicate_chunks
from sources.config.graph import GraphConfig
from sources.tools.answer_synthesizer import synthesize_answer
from sources.tools.citation_maker import make_citations
from sources.tools.evidence_selector import select_evidence
from sources.tools.prompt_selector import select_prompt
from sources.tools.question_parser import parse_question
from sources.tools.query_rewriter import rewrite_query
from sources.tools.reranker import rerank
from sources.tools.retriever import retrieve
from sources.tracker import merge_tracking_metadata, update_process_metrics


async def retrieve_multi(
    state: ExperimentState,
    *,
    config: GraphConfig,
    queries: list[str] | None = None,
) -> dict:
    """Run retrieval for each query concurrently and merge the deduplicated results.

    Args:
        state (ExperimentState): Current experiment state; used as the base state
            for each individual retrieval call.
        config (GraphConfig): Runtime configuration passed to each ``retrieve`` call.
        queries (list[str] | None): Explicit list of queries to retrieve for. When
            None, falls back to ``state["rewritten_queries"]`` or
            ``[state["question"]]``.

    Returns:
        dict: State-update dict with keys ``retrieved_chunks`` (deduplicated list of
            chunk dicts) and ``metadata``.
    """
    queries = queries or state.get("rewritten_queries") or [state["question"]]
    base_state: ExperimentState = {**state, "metadata": {}}  # type: ignore[typeddict-item]

    results = await asyncio.gather(
        *(retrieve(base_state, config=config, query=q) for q in queries)
    )
    all_chunks: list[dict] = []
    metadata = None
    for r in results:
        all_chunks.extend(r.get("retrieved_chunks", []))
        metadata = merge_tracking_metadata(metadata, r.get("metadata"))

    deduped = deduplicate_chunks(all_chunks)
    metadata = update_process_metrics(
        merge_tracking_metadata(state.get("metadata"), metadata),
        retrieval_query_count=len(queries),
        deduped_retrieved_chunk_count=len(deduped),
    )
    return {
        "retrieved_chunks": deduped,
        "metadata": metadata,
    }


ToolFn = Callable[..., Coroutine[Any, Any, dict]]

TOOL_REGISTRY: dict[str, ToolFn] = {
    "question_parser": parse_question,
    "query_rewriter": rewrite_query,
    "retriever": retrieve,
    "reranker": rerank,
    "evidence_selector": select_evidence,
    "citation_maker": make_citations,
    "prompt_selector": select_prompt,
    "answer_synthesizer": synthesize_answer,
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "question_parser": "Extract company, product, intent, entities, and language from the user question.",
    "query_rewriter": "Rewrite the question in 3 variants (direct, step_back, hyde) to improve retrieval.",
    "retriever": "Retrieve top-k relevant chunks from Qdrant with metadata filters.",
    "reranker": "Re-score and filter retrieved chunks by LLM-based relevance.",
    "evidence_selector": "Select a compact, diverse subset of the retrieved clauses for grounded answer synthesis.",
    "citation_maker": "Build structured citation objects from retrieved chunks.",
    "prompt_selector": "Select the answer-generation prompt template based on question intent.",
    "answer_synthesizer": "Generate the final answer using retrieved context and citations.",
}

__all__ = [
    "TOOL_DESCRIPTIONS",
    "TOOL_REGISTRY",
    "make_citations",
    "parse_question",
    "rerank",
    "retrieve",
    "retrieve_multi",
    "rewrite_query",
    "select_evidence",
    "select_prompt",
    "synthesize_answer",
]
