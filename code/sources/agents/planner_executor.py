"""
agents.planner_executor
-----------------------
Pattern 2 - planner-executor architecture. An LLM planner produces a
list of tool calls, an executor loop runs them in order, and a
synthesiser tail rebuilds evidence/citations and generates the final
answer. A static fallback plan is used when the planner output is
malformed.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from sources.config import config as app_config
from sources.config.graph import GraphConfig
from sources.config.prompts import PLANNER_SYSTEM_PROMPT
from sources.agents.state import ExperimentState
from sources.tools import TOOL_DESCRIPTIONS, TOOL_REGISTRY
from sources.tools.answer_synthesizer import synthesize_answer
from sources.tools.citation_maker import make_citations
from sources.tools.evidence_selector import select_evidence
from sources.tools.prompt_selector import select_prompt
from sources.tracker import (
    estimate_message_tokens,
    estimate_text_tokens,
    record_model_usage,
    update_process_metrics,
)

logger = structlog.get_logger(__name__)


def _normalize_plan(raw_plan: object) -> list[dict[str, str]]:
    """Validate a planner output and ensure it ends with synthesis.

    Args:
        raw_plan (object): Raw parsed JSON from the planner LLM, expected to be a
            list of step dicts with ``tool_name`` and optional ``reason`` keys.

    Returns:
        list[dict[str, str]]: Validated plan steps, each with ``tool_name`` and
            ``reason`` keys. Returns the static fallback plan when the input is
            invalid or empty.
    """
    if not isinstance(raw_plan, list):
        return list(app_config.planner.fallback_plan)

    normalized: list[dict[str, str]] = []
    for step in raw_plan:
        if not isinstance(step, dict):
            continue
        tool_name = step.get("tool_name")
        if not isinstance(tool_name, str) or tool_name not in TOOL_REGISTRY:
            continue
        reason = step.get("reason")
        normalized.append(
            {
                "tool_name": tool_name,
                "reason": reason if isinstance(reason, str) else "planned",
            }
        )

    if not normalized:
        return list(app_config.planner.fallback_plan)

    # answer_synthesizer must always be the terminal step; appending it here
    # ensures the invariant holds even when the planner omits it.
    if normalized[-1]["tool_name"] != "answer_synthesizer":
        normalized.append(
            {"tool_name": "answer_synthesizer", "reason": "final synthesis"}
        )
    return normalized


def build_graph(config: GraphConfig):
    """Build and compile the planner-executor graph.

    Args:
        config (GraphConfig): Runtime configuration supplying LLM clients, Qdrant
            settings, and retrieval parameters.

    Returns:
        CompiledStateGraph: Compiled LangGraph ready for invocation.
    """

    async def planner_node(state: ExperimentState) -> dict:
        """Call the planner LLM to produce an ordered tool execution plan."""
        tool_desc_text = "\n".join(
            f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items()
        )
        system = PLANNER_SYSTEM_PROMPT.format(tool_descriptions=tool_desc_text)

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=state["question"]),
        ]
        response = await config.planner_llm.ainvoke(messages)

        raw = response.content if isinstance(response.content, str) else "[]"
        raw = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

        try:
            plan = _normalize_plan(json.loads(raw))
        except json.JSONDecodeError:
            plan = list(app_config.planner.fallback_plan)

        metadata = {**state.get("metadata", {})}
        metadata["_plan"] = plan
        metadata["_plan_index"] = 0
        metadata = record_model_usage(
            metadata,
            category="agent_llm",
            model_name=config.planner_model,
            input_tokens=estimate_message_tokens(messages, config.planner_model),
            output_tokens=estimate_text_tokens(raw, config.planner_model),
        )
        metadata = update_process_metrics(metadata, plan_length=len(plan))

        return {
            "metadata": metadata,
        }

    async def executor_node(state: ExperimentState) -> dict:
        """Execute the next tool step in the plan."""
        metadata = {**state.get("metadata", {})}
        plan = metadata.get("_plan", [])
        idx = metadata.get("_plan_index", 0)

        if idx >= len(plan):
            return {"metadata": metadata}

        tool_name = plan[idx].get("tool_name", "")
        tool_fn = TOOL_REGISTRY.get(tool_name)

        updates: dict = {}
        if tool_fn is None:
            logger.warning("planner_unknown_tool", tool_name=tool_name, step_index=idx)
        elif tool_name != "answer_synthesizer":
            # answer_synthesizer is routed to synthesizer_node, which first rebuilds
            # evidence and citations; running it here would skip that prep step.
            updates = await tool_fn(state, config=config)

        # Tool results are merged back into metadata rather than returned
        # as isolated state so that subsequent steps see the full accumulated
        # tracking data (token counts, tool call log, process metrics).
        metadata = {**updates.get("metadata", metadata)}
        metadata["_plan_index"] = idx + 1
        updates["metadata"] = metadata
        return updates

    def should_continue(state: ExperimentState) -> str:
        """Return 'executor' while steps remain, otherwise 'synthesizer'."""
        metadata = state.get("metadata", {})
        plan = metadata.get("_plan", [])
        idx = metadata.get("_plan_index", 0)
        if idx < len(plan) and plan[idx].get("tool_name") != "answer_synthesizer":
            return "executor"
        return "synthesizer"

    async def synthesizer_node(state: ExperimentState) -> dict:
        """Recompute evidence and citations, then generate the final answer."""
        updates: dict = {}
        if state.get("retrieved_chunks"):
            # Recompute evidence/citations from latest retrieval state to keep refs aligned.
            updates.update(await select_evidence({**state, **updates}, config=config))
        merged_for_citations: ExperimentState = {**state, **updates}  # type: ignore[typeddict-item]
        if merged_for_citations.get("evidence_chunks") or merged_for_citations.get(
            "retrieved_chunks"
        ):
            updates.update(await make_citations(merged_for_citations, config=config))
        if state.get("selected_prompt_template") is None:
            updates.update(await select_prompt({**state, **updates}, config=config))
        updates.update(await synthesize_answer({**state, **updates}, config=config))
        return updates

    graph = StateGraph(ExperimentState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges(
        "executor",
        should_continue,
        {"executor": "executor", "synthesizer": "synthesizer"},
    )
    graph.add_edge("synthesizer", END)

    return graph.compile(name=config.pattern_name)
