"""
agents.react
------------
Pattern 6 - ReAct controller. An LLM emits structured Thought + Action
decisions; a tool-executor dispatches the chosen tool and feeds the
observation back into the next decision. Finalisation rebuilds
evidence/citations from the latest retrieval state and a guarded
fallback handles iteration-cap exits.
"""

from __future__ import annotations

import inspect

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from sources.config.graph import GraphConfig
from sources.config.prompts import REACT_SYSTEM_PROMPT
from sources.config.schemas import ReactDecision
from sources.agents.state import ExperimentState
from sources.tools import TOOL_DESCRIPTIONS, TOOL_REGISTRY
from sources.tools.answer_synthesizer import (
    attach_references,
    _normalize_answer_structure,
    _normalize_inline_citations,
    synthesize_answer,
)
from sources.tools.citation_maker import make_citations
from sources.tools.evidence_selector import select_evidence
from sources.tools.prompt_selector import select_prompt
from sources.tracker import (
    estimate_message_tokens,
    estimate_text_tokens,
    merge_tracking_metadata,
    record_model_usage,
    update_process_metrics,
)

logger = structlog.get_logger(__name__)


def _state_summary(state: ExperimentState) -> str:
    """Build a short human-readable summary of the current experiment state.

    Args:
        state (ExperimentState): Current shared experiment state.

    Returns:
        str: Multi-line summary string listing populated state fields.
    """
    parts = [f"Question: {state.get('question', '')}"]
    for key, label in [
        ("company", "Company"),
        ("product", "Product"),
        ("intent", "Intent"),
        ("selected_prompt_template", "Prompt template"),
    ]:
        val = state.get(key)
        if val:
            parts.append(f"{label}: {val}")
    for key, label in [
        ("rewritten_queries", "Rewritten queries"),
        ("retrieved_chunks", "Retrieved chunks"),
        ("citations", "Citations"),
    ]:
        val = state.get(key)
        if val:
            parts.append(f"{label}: {len(val)}")
    return "\n".join(parts)


def _observation_summary(result: dict) -> str:
    """Build a short summary string from a tool result dict.

    Args:
        result (dict): State-update dict returned by a tool invocation.

    Returns:
        str: Comma-separated description of notable result fields, or ``"done"``.
    """
    parts: list[str] = []
    if "retrieved_chunks" in result:
        parts.append(f"retrieved {len(result['retrieved_chunks'])} chunks")
    if "intent" in result:
        parts.append(f"intent={result['intent']}")
    if "rewritten_queries" in result:
        parts.append(f"{len(result['rewritten_queries'])} rewrites")
    if "citations" in result:
        parts.append(f"{len(result['citations'])} citations")
    if "selected_prompt_template" in result:
        parts.append(f"template={result['selected_prompt_template']}")
    if result.get("answer"):
        parts.append(f"answer ({len(result['answer'])} chars)")
    return ", ".join(parts) or "done"


def build_graph(config: GraphConfig):
    """Build and compile the ReAct reasoning-loop graph.

    Args:
        config (GraphConfig): Runtime configuration supplying the controller LLM,
            tool clients, and the maximum iteration count.

    Returns:
        CompiledStateGraph: Compiled LangGraph ready for invocation.
    """
    max_iter = config.max_react_iterations

    async def agent_node(state: ExperimentState) -> dict:
        """Invoke the controller LLM to produce the next Thought + Action decision."""
        tool_desc = "\n".join(f"- {n}: {d}" for n, d in TOOL_DESCRIPTIONS.items())
        system = REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)

        metadata = {**state.get("metadata", {})}
        observations = metadata.get("_react_observations", [])

        user_parts = [f"Current state:\n{_state_summary(state)}"]
        for obs in observations:
            user_parts.append(f"\nTool result ({obs['tool']}): {obs['summary']}")
        user_parts.append("\nWhat is your next step?")

        messages = [
            SystemMessage(content=system),
            HumanMessage(content="\n".join(user_parts)),
        ]
        raw = '{"thought": "Failed to parse", "action": "finish", "answer": ""}'
        try:
            structured_llm = config.controller_llm.with_structured_output(ReactDecision)
            decision_model = await structured_llm.ainvoke(messages)
            decision = decision_model.model_dump()
            raw = str(decision)
        except Exception as exc:
            logger.warning("react_decision_fallback", error=str(exc))
            decision = {"thought": "Failed to parse", "action": "finish", "answer": ""}
        metadata["_react_decision"] = decision
        metadata = record_model_usage(
            metadata,
            category="agent_llm",
            model_name=config.controller_model,
            input_tokens=estimate_message_tokens(messages, config.controller_model),
            output_tokens=estimate_text_tokens(raw, config.controller_model),
        )
        metadata = update_process_metrics(
            metadata,
            react_iteration_count=len(observations) + 1,
        )

        return {
            "metadata": metadata,
        }

    async def tool_executor_node(state: ExperimentState) -> dict:
        """Dispatch the tool selected by the agent or finalise the answer on finish."""
        metadata = {**state.get("metadata", {})}
        decision = metadata.get("_react_decision", {})
        action = decision.get("action", "finish")

        if action == "finish":
            updates: dict = {}
            # Always rebuild evidence/citations from the latest retrieval state to avoid stale refs.
            if state.get("retrieved_chunks"):
                updates.update(
                    await select_evidence({**state, **updates}, config=config)
                )
            merged_for_citations: ExperimentState = {**state, **updates}  # type: ignore[typeddict-item]
            if merged_for_citations.get("evidence_chunks") or merged_for_citations.get(
                "retrieved_chunks"
            ):
                updates.update(
                    await make_citations(merged_for_citations, config=config)
                )
            if state.get("selected_prompt_template") is None:
                merged_for_prompt: ExperimentState = {**state, **updates}  # type: ignore[typeddict-item]
                updates.update(await select_prompt(merged_for_prompt, config=config))

            merged: ExperimentState = {**state, **updates}  # type: ignore[typeddict-item]
            if merged.get("retrieved_chunks"):
                updates.update(await synthesize_answer(merged, config=config))
            else:
                raw_answer = decision.get("answer", "")
                answer_body = _normalize_answer_structure(
                    raw_answer if isinstance(raw_answer, str) else ""
                )
                answer_body = _normalize_inline_citations(
                    answer_body, merged.get("citations", [])
                )
                answer_text = attach_references(
                    answer_body, merged.get("citations", [])
                )
                updates["answer_body"] = answer_body
                updates["answer_with_references"] = answer_text
                updates["answer"] = answer_text

            updates["metadata"] = {**updates.get("metadata", metadata)}
            return updates

        tool_fn = TOOL_REGISTRY.get(action)
        if tool_fn is None:
            observations = list(metadata.get("_react_observations", []))
            observations.append({"tool": action, "summary": f"Unknown tool '{action}'"})
            metadata["_react_observations"] = observations
            return {"metadata": metadata}

        action_input = decision.get("action_input", {})
        tool_parameters = inspect.signature(tool_fn).parameters
        kwargs = {
            key: value
            for key, value in action_input.items()
            if key in tool_parameters and key not in {"state", "config"}
        }
        result = await tool_fn(state, config=config, **kwargs)

        observations = list(metadata.get("_react_observations", []))
        observations.append({"tool": action, "summary": _observation_summary(result)})

        tool_meta = result.get("metadata", {})
        merged_metadata = merge_tracking_metadata(metadata, tool_meta)
        merged_metadata["_react_observations"] = observations
        merged_metadata["_react_decision"] = {}
        result["metadata"] = merged_metadata
        return result

    def should_continue(state: ExperimentState) -> str:
        """Return the next routing target based on the current decision and iteration count."""
        metadata = state.get("metadata", {})
        decision = metadata.get("_react_decision", {})
        iterations = len(metadata.get("_react_observations", []))
        if decision.get("action") == "finish" or state.get("answer") is not None:
            return "finish"
        if iterations >= max_iter:
            # The "finish" action is injected by the graph, not by the LLM, so
            # we route to force_finish rather than returning to agent_node.
            return "force_finish"
        return "agent"

    async def force_finish_node(state: ExperimentState) -> dict:
        """Fallback answer when the iteration cap is reached."""
        logger.warning("react_force_finish")
        if state.get("answer"):
            return {}
        # Evidence and citations are rebuilt from scratch here rather than
        # trusting state because the iteration cap may have been hit mid-loop,
        # leaving retrieved_chunks updated but evidence_chunks stale.
        updates: dict = {}
        if state.get("retrieved_chunks"):
            updates.update(await select_evidence({**state, **updates}, config=config))
        merged_for_citations: ExperimentState = {**state, **updates}  # type: ignore[typeddict-item]
        if merged_for_citations.get("evidence_chunks") or merged_for_citations.get(
            "retrieved_chunks"
        ):
            updates.update(await make_citations(merged_for_citations, config=config))
        updates.update(await synthesize_answer({**state, **updates}, config=config))
        return updates

    graph = StateGraph(ExperimentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("force_finish", force_finish_node)

    graph.add_edge(START, "agent")
    graph.add_edge("agent", "tool_executor")
    graph.add_conditional_edges(
        "tool_executor",
        should_continue,
        {"agent": "agent", "finish": END, "force_finish": "force_finish"},
    )
    graph.add_edge("force_finish", END)

    return graph.compile(name=config.pattern_name)
