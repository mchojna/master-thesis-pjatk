"""
tools.evidence_selector
-----------------------
Picks a small, diverse evidence set from the retrieved chunks for
the answer model. Deduplicates near-identical clauses, then runs a
greedy selection ranked by score with a specificity bonus and a soft
length penalty until ``evidence_top_k`` chunks are selected.
"""

from __future__ import annotations

import structlog

from sources.config.graph import GraphConfig
from sources.agents.state import ExperimentState, deduplicate_chunks
from sources.tracker import record_tool_call, update_process_metrics

logger = structlog.get_logger(__name__)


def _evidence_key(chunk: dict) -> tuple[str, str, str]:
    """Build a stable key that approximates one clause or section.

    Args:
        chunk (dict): Retrieved chunk dict with ``source_file`` and ``metadata``
            fields.

    Returns:
        tuple[str, str, str]: Three-element key of (source_file, header_2_or_1,
            header_3).
    """
    meta = chunk.get("metadata", {})
    return (
        chunk.get("source_file", ""),
        meta.get("header_2") or meta.get("header_1") or "",
        meta.get("header_3") or "",
    )


def _evidence_rank(chunk: dict) -> tuple[float, float, float]:
    """Compute the ranking score for a chunk during greedy evidence selection.

    Args:
        chunk (dict): Retrieved chunk dict with ``score``, ``rerank_score``,
            ``metadata``, and ``text`` fields.

    Returns:
        tuple[float, float, float]: Three-element rank key of (score,
            specificity_bonus, negative_length_penalty); higher is better.
    """
    text = chunk.get("text", "")
    score = float(chunk.get("rerank_score") or chunk.get("score") or 0.0)
    # header_3 presence indicates a narrow sub-clause — more precise than a section
    # heading — so it earns a fixed bonus over section-level chunks.
    specificity_bonus = 1.0 if chunk.get("metadata", {}).get("header_3") else 0.0
    # Length penalty is capped at 1600 chars: beyond that, extra length adds noise
    # without proportionally more information for the answer model.
    length_penalty = min(len(text), 1600) / 1600 if text else 1.0
    return (score, specificity_bonus, -length_penalty)


async def select_evidence(
    state: ExperimentState,
    *,
    config: GraphConfig,
) -> dict:
    """Keep a small, diverse evidence set for the answer model.

    Args:
        state (ExperimentState): Current experiment state; uses ``retrieved_chunks``
            as the candidate pool.
        config (GraphConfig): Runtime configuration providing ``evidence_top_k``.

    Returns:
        dict: State-update dict with keys ``evidence_chunks`` (list of selected chunk
            dicts) and ``metadata``.
    """
    chunks = deduplicate_chunks(state.get("retrieved_chunks", []))
    metadata = record_tool_call(state.get("metadata"), "evidence_selector")

    if not chunks:
        return {"evidence_chunks": [], "metadata": metadata}

    target_size = max(1, config.evidence_top_k)
    ordered_chunks = sorted(chunks, key=_evidence_rank, reverse=True)

    selected: list[dict] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for chunk in ordered_chunks:
        evidence_key = _evidence_key(chunk)
        if evidence_key in seen_keys:
            continue
        selected.append(chunk)
        seen_keys.add(evidence_key)
        if len(selected) >= target_size:
            break

    if len(selected) < target_size:
        # Section-uniqueness constraint left fewer results than target_size; fill
        # remaining slots by object identity to avoid duplicates while relaxing
        # the one-chunk-per-section requirement.
        seen_ids = {id(chunk) for chunk in selected}
        for chunk in ordered_chunks:
            if id(chunk) in seen_ids:
                continue
            selected.append(chunk)
            seen_ids.add(id(chunk))
            if len(selected) >= target_size:
                break

    logger.info(
        "select_evidence",
        n_input=len(chunks),
        n_selected=len(selected),
    )
    metadata = update_process_metrics(
        metadata,
        evidence_chunk_count=len(selected),
    )

    return {
        "evidence_chunks": selected,
        "metadata": metadata,
    }
