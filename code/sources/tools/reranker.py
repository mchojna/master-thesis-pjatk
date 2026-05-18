"""
tools.reranker
--------------
LLM-based reranker that scores retrieved chunks for relevance to the
question, filters by a configurable threshold and keeps the top-k
results. Splits chunks into bounded-token batches, scores batches
concurrently behind a semaphore and degrades to original vector
scores when the LLM output cannot be parsed.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable

import tiktoken
import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from sources.config import config as app_config
from sources.config.graph import GraphConfig
from sources.config.prompts import RERANKER_SYSTEM_PROMPT
from sources.agents.state import ExperimentState
from sources.tracker import (
    estimate_message_tokens,
    estimate_text_tokens,
    record_model_usage,
    record_tool_call,
    update_process_metrics,
)

logger = structlog.get_logger(__name__)


def _strip_json_fence(raw: str) -> str:
    """Remove optional markdown code fences surrounding JSON output.

    Args:
        raw (str): Raw LLM response that may be wrapped in a markdown code fence.

    Returns:
        str: Stripped string with fences and surrounding whitespace removed.
    """
    return (
        raw.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )


def _normalize_score_item(
    item: object, *, fallback_index: int | None = None
) -> dict | None:
    """Coerce one reranker output item into the expected mapping shape.

    Args:
        item (object): A single item from the reranker LLM output; may be a dict,
            str, int, or float.
        fallback_index (int | None): Index to assign when the item carries no
            ``"index"`` key.

    Returns:
        dict | None: Normalised dict with at least ``"index"`` and ``"score"`` keys,
            or None when the item cannot be interpreted.
    """
    if isinstance(item, dict):
        normalized = dict(item)
        if normalized.get("index") is None and fallback_index is not None:
            normalized["index"] = fallback_index
        return normalized

    if isinstance(item, str):
        text = item.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                score = float(text)
            except ValueError:
                return None
            return {"index": fallback_index, "score": score}
        return _normalize_score_item(parsed, fallback_index=fallback_index)

    if isinstance(item, (int, float)):
        return {"index": fallback_index, "score": float(item)}

    return None


def _normalize_scores_payload(payload: object) -> list[dict]:
    """Normalise supported reranker payload shapes to a uniform score list.

    Args:
        payload (object): Parsed JSON payload from the reranker LLM; may be a dict,
            list, or scalar.

    Returns:
        list[dict]: List of normalised score dicts, each with ``"index"`` and
            ``"score"`` keys.
    """
    if isinstance(payload, dict):
        if isinstance(payload.get("scores"), list):
            payload = payload["scores"]
        elif all(str(key).strip().isdigit() for key in payload):
            normalized: list[dict] = []
            for key, value in payload.items():
                item = _normalize_score_item(
                    value, fallback_index=int(str(key).strip())
                )
                if item is not None:
                    normalized.append(item)
            return normalized
        else:
            item = _normalize_score_item(payload)
            return [item] if item is not None else []

    if isinstance(payload, Iterable) and not isinstance(
        payload, (str, bytes, bytearray)
    ):
        normalized = []
        for fallback_index, item in enumerate(payload):
            normalized_item = _normalize_score_item(item, fallback_index=fallback_index)
            if normalized_item is not None:
                normalized.append(normalized_item)
        return normalized

    item = _normalize_score_item(payload)
    return [item] if item is not None else []


def _estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """Estimate the token count for a reranker payload string.

    Args:
        text (str): Text to tokenise.
        model (str): Model name used to select the tiktoken encoding; falls back
            to ``cl100k_base`` for unknown models.

    Returns:
        int: Estimated token count.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


async def rerank(
    state: ExperimentState,
    *,
    config: GraphConfig,
) -> dict:
    """Score retrieved chunks with the reranker LLM and keep the best results.

    Args:
        state (ExperimentState): Current experiment state; uses ``question`` and
            ``retrieved_chunks``.
        config (GraphConfig): Runtime configuration providing the reranker LLM
            client, score threshold, top-k limits, batch size, and max workers.

    Returns:
        dict: State-update dict with keys ``retrieved_chunks`` (reranked and filtered
            list) and ``metadata``.
    """
    score_threshold = config.reranker_score_threshold
    top_k_after = max(1, config.reranker_top_k_after)
    batch_size = max(1, config.reranker_batch_size)
    max_workers = max(1, config.reranker_max_workers)

    question = state["question"]
    chunks = state.get("retrieved_chunks", [])
    metadata = record_tool_call(state.get("metadata"), "reranker")

    if not chunks:
        return {"metadata": metadata}

    indexed_chunks = [
        {"index": i, "text": c.get("text", "")[:1000]} for i, c in enumerate(chunks)
    ]

    def _trim_batch(batch: list[dict]) -> list[dict]:
        """Trim a batch to fit within the token budget."""
        trimmed_batch = list(batch)
        while trimmed_batch:
            batch_message = (
                f"Question: {question}\n\n"
                f"Chunks:\n{json.dumps(trimmed_batch, ensure_ascii=False)}"
            )
            if (
                _estimate_tokens(batch_message, model=config.reranker_model)
                <= app_config.reranking.max_reranker_tokens
            ):
                return trimmed_batch
            # Pop from the end so the highest-indexed (lowest-priority) chunk is
            # dropped first, preserving the ordering relative to vector scores.
            dropped = trimmed_batch.pop()
            logger.debug("reranker_token_trim", dropped_index=dropped["index"])
        return []

    prepared_batches: list[list[dict]] = []
    for offset in range(0, len(indexed_chunks), batch_size):
        batch = indexed_chunks[offset : offset + batch_size]
        trimmed_batch = _trim_batch(batch)
        if trimmed_batch:
            prepared_batches.append(trimmed_batch)

    async def _score_batch(batch: list[dict]) -> tuple[list[dict], dict]:
        """Call the reranker LLM for one batch and return scores with usage metadata."""
        user_message = (
            f"Question: {question}\n\n"
            f"Chunks:\n{json.dumps(batch, ensure_ascii=False)}"
        )
        messages = [
            SystemMessage(content=RERANKER_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        batch_metadata: dict | None = None
        try:
            response = await config.reranker_llm.ainvoke(messages)
            raw = response.content if isinstance(response.content, str) else ""
            raw = _strip_json_fence(raw)
            scores_json = _normalize_scores_payload(json.loads(raw))
            batch_metadata = record_model_usage(
                batch_metadata,
                category="reranker_llm",
                model_name=config.reranker_model,
                input_tokens=estimate_message_tokens(messages, config.reranker_model),
                output_tokens=estimate_text_tokens(raw, config.reranker_model),
            )
            if not scores_json:
                raise ValueError("Empty or unsupported reranker payload")
            return scores_json, batch_metadata
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("reranker_fallback", error=str(exc))
            # Preserve original vector scores on LLM failure so the batch is not
            # silently lost; downstream threshold filtering still applies.
            fallback_scores = [
                {
                    "index": item["index"],
                    "score": chunks[item["index"]].get("score", 0.5),
                }
                for item in batch
            ]
            return fallback_scores, batch_metadata or {}

    # Semaphore limits concurrent LLM calls to avoid saturating the rate limit
    # while still allowing all batches to be dispatched in a single gather call.
    semaphore = asyncio.Semaphore(max_workers)

    async def _score_batch_bounded(batch: list[dict]) -> tuple[list[dict], dict]:
        """Acquire the semaphore before scoring a batch."""
        async with semaphore:
            return await _score_batch(batch)

    batch_results = await asyncio.gather(
        *(_score_batch_bounded(batch) for batch in prepared_batches)
    )
    scores_json: list[dict] = []
    for batch_scores, batch_metadata in batch_results:
        scores_json.extend(batch_scores)
        metadata = update_process_metrics(
            (
                record_model_usage(
                    metadata,
                    category="reranker_llm",
                    model_name=config.reranker_model,
                    input_tokens=int(
                        batch_metadata.get("_usage", {})
                        .get("reranker_llm", {})
                        .get("input_tokens", 0)
                    ),
                    output_tokens=int(
                        batch_metadata.get("_usage", {})
                        .get("reranker_llm", {})
                        .get("output_tokens", 0)
                    ),
                    request_count=int(
                        batch_metadata.get("_usage", {})
                        .get("reranker_llm", {})
                        .get("requests", 0)
                    ),
                )
                if batch_metadata
                else metadata
            ),
            reranker_batch_count=len(prepared_batches),
        )

    if not prepared_batches:
        scores_json = [
            {"index": i, "score": chunk.get("score", 0.5)}
            for i, chunk in enumerate(chunks)
        ]

    score_map: dict[int, float] = {}
    for item in scores_json:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        score = item.get("score", 0.0)
        if idx is not None:
            try:
                # Clamp to [0.0, 1.0] so both LLM-produced scores and the
                # vector-score fallback (already in this range) are comparable.
                score_map[int(idx)] = max(0.0, min(1.0, float(score)))
            except (TypeError, ValueError):
                continue

    reranked: list[dict] = []
    for i, chunk in enumerate(chunks):
        new_score = score_map.get(i, chunk.get("score", 0.0))
        if new_score is not None and new_score >= score_threshold:
            reranked.append({**chunk, "score": new_score, "rerank_score": new_score})

    reranked.sort(key=lambda c: c.get("score", 0), reverse=True)
    reranked = reranked[:top_k_after]

    top_score = reranked[0]["score"] if reranked else 0.0

    logger.info(
        "rerank",
        n_input=len(chunks),
        n_kept=len(reranked),
        n_batches=len(prepared_batches),
        reranker_model=config.reranker_model,
        top_score=top_score,
    )
    metadata = update_process_metrics(
        metadata,
        reranker_input_chunk_count_total=int(
            metadata.get("_process", {}).get("reranker_input_chunk_count_total", 0)
        )
        + len(chunks),
        reranked_chunk_count_total=int(
            metadata.get("_process", {}).get("reranked_chunk_count_total", 0)
        )
        + len(reranked),
        last_reranker_input_chunk_count=len(chunks),
        last_reranked_chunk_count=len(reranked),
    )

    return {
        "retrieved_chunks": reranked,
        "metadata": metadata,
    }
