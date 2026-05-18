"""
tools.query_rewriter
--------------------
Produces three diverse rewrites (direct, step-back, HyDE) of the
user question to broaden vector recall. Robustly extracts the JSON
array from the LLM response and falls back to repeating the original
question when parsing fails or fewer than three rewrites are
returned.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from sources.config.graph import GraphConfig
from sources.config.prompts import QUERY_REWRITER_SYSTEM_PROMPT
from sources.agents.state import ExperimentState
from sources.tracker import (
    estimate_message_tokens,
    estimate_text_tokens,
    record_model_usage,
    record_tool_call,
)

logger = structlog.get_logger(__name__)


def _extract_json_array(raw: str) -> str:
    """Extract the first JSON array substring from a model response.

    Args:
        raw (str): Raw text response from the LLM, which may contain prose before
            or after the JSON array.

    Returns:
        str: Substring from the first ``[`` to the last ``]``.

    Raises:
        json.JSONDecodeError: If no JSON array delimiters are found in ``raw``.
    """
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise json.JSONDecodeError("JSON array not found", raw, 0)
    return raw[start : end + 1]


def _parse_rewrites(raw: str) -> list[dict[str, str]]:
    """Parse rewrites from a model response with lenient JSON handling.

    Args:
        raw (str): Raw LLM output string expected to contain a JSON array of
            ``{"strategy": ..., "query": ...}`` objects.

    Returns:
        list[dict[str, str]]: List of rewrite dicts, each with ``strategy`` and
            ``query`` keys.

    Raises:
        TypeError: If the parsed JSON is not a list.
        ValueError: If no valid rewrite objects are found in the parsed list.
    """
    candidate = _extract_json_array(raw)
    # strict=False allows control characters in JSON strings; LLMs occasionally
    # embed raw newlines or tabs that would cause strict parsing to fail.
    parsed = json.loads(candidate, strict=False)
    if not isinstance(parsed, list):
        raise TypeError("Query rewriter output must be a JSON array.")

    rewrites: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue

        strategy = str(item.get("strategy", "")).strip() or "direct"
        query = str(item.get("query", "")).strip()
        if not query:
            continue

        rewrites.append({"strategy": strategy, "query": query})

    if not rewrites:
        raise ValueError("Query rewriter output did not contain any valid queries.")

    return rewrites


async def rewrite_query(
    state: ExperimentState,
    *,
    config: GraphConfig,
) -> dict:
    """Produce three query rewrites for improved retrieval recall.

    Args:
        state (ExperimentState): Current experiment state; uses ``question``,
            ``company``, and ``product`` fields to build the rewriter prompt.
        config (GraphConfig): Runtime configuration providing the rewrite LLM
            client and model name.

    Returns:
        dict: State-update dict with keys ``rewritten_queries`` (list of three query
            strings) and ``metadata``.
    """
    question = state["question"]
    metadata = record_tool_call(state.get("metadata"), "query_rewriter")
    context_parts: list[str] = [f"Question: {question}"]
    if state.get("company"):
        context_parts.append(f"Company: {state.get('company')}")
    if state.get("product"):
        context_parts.append(f"Product: {state.get('product')}")

    user_message = "\n".join(context_parts)
    messages = [
        SystemMessage(content=QUERY_REWRITER_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    try:
        response = await config.rewrite_llm.ainvoke(messages)
        raw = response.content if isinstance(response.content, str) else ""
        raw = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        rewrites_json = _parse_rewrites(raw)
        metadata = record_model_usage(
            metadata,
            category="agent_llm",
            model_name=config.rewrite_model,
            input_tokens=estimate_message_tokens(messages, config.rewrite_model),
            output_tokens=estimate_text_tokens(raw, config.rewrite_model),
        )
    except Exception as exc:
        logger.warning("query_rewriter_fallback", error=str(exc))
        rewrites_json = [
            {"strategy": "direct", "query": question},
            {"strategy": "step_back", "query": question},
            {"strategy": "hyde", "query": question},
        ]

    queries = [item["query"] for item in rewrites_json if "query" in item]
    # Pad to exactly 3 queries (matching the direct / step-back / HyDE strategies
    # the prompt requests) and truncate to 3 if the LLM returned more. Downstream
    # retrieve_multi deduplicates results, so repeated fallback queries are safe.
    while len(queries) < 3:
        queries.append(question)
    queries = queries[:3]

    logger.info("rewrite_query", n_rewrites=len(queries))

    return {
        "rewritten_queries": queries,
        "metadata": metadata,
    }
