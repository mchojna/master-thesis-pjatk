"""
agents.deterministic
--------------------
Pattern 1 - fixed linear pipeline with no LLM-driven routing. Every
question flows through the same eight nodes in order: parse, rewrite,
multi-query retrieve, rerank, evidence selection, prompt selection,
citation building and answer synthesis.
"""

from __future__ import annotations

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


def _tool_node(config: GraphConfig, fn, **kw):
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
    """Build and compile the deterministic pipeline graph.

    Args:
        config (GraphConfig): Runtime configuration supplying LLM clients, Qdrant
            settings, and retrieval parameters.

    Returns:
        CompiledStateGraph: Compiled LangGraph ready for invocation.
    """
    graph = StateGraph(ExperimentState)

    graph.add_node("parse_question", _tool_node(config, parse_question))
    graph.add_node("rewrite_query", _tool_node(config, rewrite_query))
    graph.add_node("retrieve_all", _tool_node(config, retrieve_multi))
    graph.add_node("rerank", _tool_node(config, rerank))
    graph.add_node("select_evidence", _tool_node(config, select_evidence))
    graph.add_node("select_prompt", _tool_node(config, select_prompt))
    graph.add_node("make_citations", _tool_node(config, make_citations))
    graph.add_node("generate_answer", _tool_node(config, synthesize_answer))

    graph.add_edge(START, "parse_question")
    graph.add_edge("parse_question", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_all")
    graph.add_edge("retrieve_all", "rerank")
    graph.add_edge("rerank", "select_evidence")
    graph.add_edge("select_evidence", "select_prompt")
    graph.add_edge("select_prompt", "make_citations")
    graph.add_edge("make_citations", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile(name=config.pattern_name)
