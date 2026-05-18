"""
agents.blackboard
-----------------
Pattern 4 - blackboard / swarm architecture. A central dispatcher
inspects shared state on each iteration and routes to the next
eligible agent (parser, rewriter, retriever, reranker, evidence
selector, prompt selector, citation maker, answer generator). A
max-iteration guard prevents infinite loops.
"""

from __future__ import annotations

import structlog
from langgraph.graph import END, START, StateGraph

from sources.config.graph import GraphConfig
from sources.agents.state import ExperimentState
from sources.tools import (
    select_evidence,
    make_citations,
    parse_question,
    retrieve_multi,
    rewrite_query,
    rerank,
    select_prompt,
    synthesize_answer,
)

logger = structlog.get_logger(__name__)


def _agent_node(config: GraphConfig, fn, **kw):
    """Wrap a tool function as a LangGraph node coroutine.

    Args:
        config (GraphConfig): Runtime graph configuration passed to the tool.
        fn: Async tool function with signature ``(state, *, config, **kw) -> dict``.
        **kw: Additional keyword arguments forwarded to the tool on every call.

    Returns:
        Callable: Async node function that accepts an ExperimentState and returns
            a state-update dict.
    """

    async def _node(state: ExperimentState) -> dict:
        """Delegate to the wrapped tool function."""
        return await fn(state, config=config, **kw)

    return _node


def build_graph(config: GraphConfig):
    """Build and compile the blackboard graph.

    Args:
        config (GraphConfig): Runtime configuration supplying LLM clients, Qdrant
            settings, and the maximum iteration count.

    Returns:
        CompiledStateGraph: Compiled LangGraph ready for invocation.
    """
    max_iter = config.max_blackboard_iterations

    def dispatch(state: ExperimentState) -> str:
        """Decide which agent to run next based on state.

        Args:
            state (ExperimentState): Current shared experiment state.

        Returns:
            str: Name of the next agent node, or ``"__end__"`` when the answer
                is already populated.
        """
        metadata = state.get("metadata", {})
        if metadata.get("_bb_iter", 0) >= max_iter:
            logger.warning("blackboard_max_iter")
            return "answer_agent"
        if state.get("answer") is not None:
            return "__end__"
        # Conditions are ordered by pipeline stage: each check represents one
        # prerequisite that must be satisfied before the next agent is eligible.
        if state.get("intent") is None:
            return "parser_agent"
        if not state.get("rewritten_queries"):
            return "rewriter_agent"
        if not state.get("retrieved_chunks"):
            return "retriever_agent"
        chunks = state.get("retrieved_chunks", [])
        # rerank_score is None when the reranker has not yet run; using its absence
        # as the trigger avoids a separate "reranked" boolean flag in state.
        if chunks and chunks[0].get("rerank_score") is None:
            return "reranker_agent"
        if chunks and not state.get("evidence_chunks"):
            return "evidence_selector_agent"
        if state.get("selected_prompt_template") is None:
            return "prompt_selector_agent"
        if not state.get("citations"):
            return "citation_agent"
        return "answer_agent"

    async def dispatcher_node(state: ExperimentState) -> dict:
        """Increment the blackboard iteration counter and return updated metadata."""
        metadata = {**state.get("metadata", {})}
        # Counter is incremented before dispatch so the guard in dispatch() sees
        # the updated value on the same iteration it was triggered.
        metadata["_bb_iter"] = metadata.get("_bb_iter", 0) + 1
        return {"metadata": metadata}

    agents = {
        "parser_agent": parse_question,
        "rewriter_agent": rewrite_query,
        "retriever_agent": retrieve_multi,
        "reranker_agent": rerank,
        "evidence_selector_agent": select_evidence,
        "prompt_selector_agent": select_prompt,
        "citation_agent": make_citations,
        "answer_agent": synthesize_answer,
    }

    graph = StateGraph(ExperimentState)
    graph.add_node("dispatcher", dispatcher_node)

    for name, fn in agents.items():
        graph.add_node(name, _agent_node(config, fn))

    graph.add_edge(START, "dispatcher")
    graph.add_conditional_edges(
        "dispatcher",
        dispatch,
        {name: name for name in agents} | {"__end__": END},  # type: ignore[arg-type]
    )

    for name in agents:
        if name != "answer_agent":
            graph.add_edge(name, "dispatcher")
    graph.add_edge("answer_agent", END)

    return graph.compile(name=config.pattern_name)
