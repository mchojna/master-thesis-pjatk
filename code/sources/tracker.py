"""
tracker
-------
Accumulates per-run process and usage metadata: tool invocation
counts, model token usage, estimated USD cost per category and
model, and process metrics consumed by the runner and evaluator.
All mutators return new dictionaries so state updates remain
LangGraph-friendly.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import tiktoken

from sources.config import config as app_config


def format_tool_invocation(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    """Render a normalised tool invocation string for evaluation records.

    Args:
        tool_name (str): Name of the tool that was invoked.
        arguments (dict[str, Any] | None): Keyword arguments passed to the tool,
            or None for a no-argument call.

    Returns:
        str: Human-readable invocation string such as ``tool(key=value)``.
    """
    if not arguments:
        return f"{tool_name}()"

    serialized_arguments = ", ".join(
        f"{key}={json.dumps(value, ensure_ascii=False, default=str)}"
        for key, value in arguments.items()
    )
    return f"{tool_name}({serialized_arguments})"


def estimate_text_tokens(text: str, model_name: str) -> int:
    """Estimate the token count for a text string using tiktoken.

    Args:
        text (str): Text to tokenise.
        model_name (str): Model identifier used to select the tiktoken encoding;
            falls back to ``cl100k_base`` for unknown models.

    Returns:
        int: Estimated number of tokens.
    """
    if not text:
        return 0

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def estimate_message_tokens(messages: Sequence[Any], model_name: str) -> int:
    """Estimate the total token count for a sequence of LangChain messages.

    Args:
        messages (Sequence[Any]): Sequence of LangChain message objects with a
            ``content`` attribute.
        model_name (str): Model identifier forwarded to ``estimate_text_tokens``.

    Returns:
        int: Sum of estimated token counts across all messages.
    """
    total = 0
    for message in messages:
        content = getattr(message, "content", "")
        total += estimate_text_tokens(str(content), model_name)
    return total


def estimate_cost_usd(
    model_name: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    embedding: bool = False,
) -> float | None:
    """Estimate the API cost in USD for a model call.

    Args:
        model_name (str): Model identifier looked up in the pricing tables.
        input_tokens (int): Number of input tokens consumed.
        output_tokens (int): Number of output tokens generated.
        embedding (bool): When True, uses the embedding pricing table instead of
            the text model table.

    Returns:
        float | None: Estimated cost in USD, or None when the model is not in the
            pricing table.
    """
    pricing = (
        app_config.pricing.embedding_model_pricing_per_1m
        if embedding
        else app_config.pricing.text_model_pricing_per_1m
    )
    rates = pricing.get(model_name)
    if rates is None:
        return None

    return (input_tokens / 1_000_000) * rates["input"] + (
        output_tokens / 1_000_000
    ) * rates["output"]


def _copy_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return a mutable deep copy of a metadata dict, coercing invalid payloads.

    Args:
        metadata (dict[str, Any] | None): Metadata dict to copy, or None.

    Returns:
        dict[str, Any]: Deep copy of the input dict, or an empty dict when the
            input is None or not a dict.
    """
    if not isinstance(metadata, dict):
        return {}
    return deepcopy(metadata)


def record_tool_call(
    metadata: dict[str, Any] | None,
    tool_name: str,
    *,
    arguments: dict[str, Any] | None = None,
    invocation: str | None = None,
) -> dict[str, Any]:
    """Increment tool counters and append an invocation record to metadata.

    Args:
        metadata (dict[str, Any] | None): Existing metadata dict to update.
        tool_name (str): Name of the tool being called.
        arguments (dict[str, Any] | None): Arguments passed to the tool.
        invocation (str | None): Pre-formatted invocation string; generated from
            ``tool_name`` and ``arguments`` when None.

    Returns:
        dict[str, Any]: Updated metadata dict with incremented tool counts and a
            new entry in ``_process.tool_invocations``.
    """
    updated = _copy_metadata(metadata)
    process = dict(updated.get("_process", {}))
    tool_counts = dict(process.get("tool_counts", {}))
    tool_invocations = list(process.get("tool_invocations", []))
    tool_counts[tool_name] = int(tool_counts.get(tool_name, 0)) + 1
    tool_invocations.append(
        {
            "tool_name": tool_name,
            "arguments": deepcopy(arguments) if arguments else {},
            "invocation": invocation or format_tool_invocation(tool_name, arguments),
        }
    )
    process["tool_counts"] = tool_counts
    process["tool_invocations"] = tool_invocations
    process["tool_call_count"] = int(process.get("tool_call_count", 0)) + 1
    updated["_process"] = process
    return updated


def update_process_metrics(
    metadata: dict[str, Any] | None, **metrics: Any
) -> dict[str, Any]:
    """Merge arbitrary process metrics into the ``_process`` sub-dict.

    Args:
        metadata (dict[str, Any] | None): Existing metadata dict to update.
        **metrics (Any): Key-value pairs to set or overwrite in ``_process``.

    Returns:
        dict[str, Any]: Updated metadata dict with the new metrics merged in.
    """
    updated = _copy_metadata(metadata)
    process = dict(updated.get("_process", {}))
    process.update(metrics)
    updated["_process"] = process
    return updated


def record_model_usage(
    metadata: dict[str, Any] | None,
    *,
    category: str,
    model_name: str,
    input_tokens: int,
    output_tokens: int = 0,
    request_count: int = 1,
    embedding: bool = False,
) -> dict[str, Any]:
    """Accumulate model token usage and estimated cost in metadata.

    Args:
        metadata (dict[str, Any] | None): Existing metadata dict to update.
        category (str): Usage category key (e.g. ``"agent_llm"``, ``"embedding"``).
        model_name (str): Model identifier for per-model breakdown and cost lookup.
        input_tokens (int): Number of input tokens consumed in this call.
        output_tokens (int): Number of output tokens generated in this call.
        request_count (int): Number of API requests made (default 1).
        embedding (bool): When True, uses the embedding pricing table for cost
            estimation.

    Returns:
        dict[str, Any]: Updated metadata dict with cumulative usage and cost fields
            updated under ``_usage.<category>``.
    """
    updated = _copy_metadata(metadata)
    usage = dict(updated.get("_usage", {}))
    category_totals = dict(usage.get(category, {}))
    models = deepcopy(category_totals.get("models", {}))
    model_usage = dict(models.get(model_name, {}))

    model_usage["requests"] = int(model_usage.get("requests", 0)) + request_count
    model_usage["input_tokens"] = int(model_usage.get("input_tokens", 0)) + input_tokens
    model_usage["output_tokens"] = (
        int(model_usage.get("output_tokens", 0)) + output_tokens
    )

    estimated_cost = estimate_cost_usd(
        model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        embedding=embedding,
    )
    if estimated_cost is not None:
        # 8 decimal places: cost accumulates over hundreds of calls; fewer places
        # cause floating-point drift that masks real differences between architectures.
        model_usage["estimated_cost_usd"] = round(
            float(model_usage.get("estimated_cost_usd", 0.0)) + estimated_cost,
            8,
        )

    models[model_name] = model_usage
    category_totals["models"] = models
    category_totals["requests"] = (
        int(category_totals.get("requests", 0)) + request_count
    )
    category_totals["input_tokens"] = (
        int(category_totals.get("input_tokens", 0)) + input_tokens
    )
    category_totals["output_tokens"] = (
        int(category_totals.get("output_tokens", 0)) + output_tokens
    )
    if estimated_cost is not None:
        category_totals["estimated_cost_usd"] = round(
            float(category_totals.get("estimated_cost_usd", 0.0)) + estimated_cost,
            8,
        )

    usage[category] = category_totals
    updated["_usage"] = usage
    return updated


def merge_tracking_metadata(*metadata_items: dict[str, Any] | None) -> dict[str, Any]:
    """Merge process and usage counters from multiple metadata dictionaries.

    Args:
        *metadata_items (dict[str, Any] | None): Metadata dicts to merge; None
            values and non-dict values are skipped.

    Returns:
        dict[str, Any]: Merged metadata dict with additive counters for tool counts,
            token usage, and estimated costs.
    """
    # Deep-copy incoming values before merging to prevent aliasing bugs when
    # LangGraph's state reducer reuses the same dict object across parallel branches.
    merged: dict[str, Any] = {}

    for item in metadata_items:
        if not item or not isinstance(item, dict):
            continue

        for key, value in item.items():
            if key not in {"_process", "_usage"}:
                merged[key] = deepcopy(value)

        if "_process" in item:
            process = dict(merged.get("_process", {}))
            incoming_process = item["_process"]
            for key, value in incoming_process.items():
                if key == "tool_counts":
                    counts = dict(process.get("tool_counts", {}))
                    for tool_name, count in value.items():
                        counts[tool_name] = int(counts.get(tool_name, 0)) + int(count)
                    process["tool_counts"] = counts
                elif key == "tool_invocations":
                    invocations = list(process.get("tool_invocations", []))
                    invocations.extend(deepcopy(value))
                    process["tool_invocations"] = invocations
                # Convention: any process key ending in _count or _total is additive
                # across parallel branches; all other process keys take the latest value
                # (e.g. string labels or booleans that must not sum).
                elif isinstance(value, int) and key.endswith("_count"):
                    process[key] = int(process.get(key, 0)) + value
                elif isinstance(value, int) and key.endswith("_total"):
                    process[key] = int(process.get(key, 0)) + value
                else:
                    process[key] = deepcopy(value)
            merged["_process"] = process

        if "_usage" in item:
            usage = dict(merged.get("_usage", {}))
            incoming_usage = item["_usage"]
            for category, category_payload in incoming_usage.items():
                category_totals = dict(usage.get(category, {}))
                for metric_key in ("requests", "input_tokens", "output_tokens"):
                    category_totals[metric_key] = int(
                        category_totals.get(metric_key, 0)
                    ) + int(category_payload.get(metric_key, 0))
                if "estimated_cost_usd" in category_payload:
                    category_totals["estimated_cost_usd"] = round(
                        float(category_totals.get("estimated_cost_usd", 0.0))
                        + float(category_payload.get("estimated_cost_usd", 0.0)),
                        8,
                    )

                models = deepcopy(category_totals.get("models", {}))
                for model_name, model_payload in category_payload.get(
                    "models", {}
                ).items():
                    model_totals = dict(models.get(model_name, {}))
                    for metric_key in ("requests", "input_tokens", "output_tokens"):
                        model_totals[metric_key] = int(
                            model_totals.get(metric_key, 0)
                        ) + int(model_payload.get(metric_key, 0))
                    if "estimated_cost_usd" in model_payload:
                        model_totals["estimated_cost_usd"] = round(
                            float(model_totals.get("estimated_cost_usd", 0.0))
                            + float(model_payload.get("estimated_cost_usd", 0.0)),
                            8,
                        )
                    models[model_name] = model_totals
                category_totals["models"] = models
                usage[category] = category_totals
            merged["_usage"] = usage

    return merged
