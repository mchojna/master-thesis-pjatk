"""
agents.router_specialist
------------------------
Pattern 3 - router classifies the question into one of four
specialist pipelines (faq, description, claims, comparison) and
dispatches to a tailored sub-graph. Each specialist runs the same
eight-step pipeline with a slightly different retrieval strategy.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from sources.config import config as app_config
from sources.config.graph import GraphConfig
from sources.config.prompts import ROUTER_SYSTEM_PROMPT
from sources.config.schemas import RouterDecision
from sources.agents.state import ExperimentState
from sources.tools import (
    retrieve_multi,
    synthesize_answer,
    make_citations,
    select_evidence,
    select_prompt,
    parse_question,
    rewrite_query,
    rerank,
    retrieve,
)
from sources.tracker import (
    estimate_message_tokens,
    estimate_text_tokens,
    record_model_usage,
    update_process_metrics,
)

logger = structlog.get_logger(__name__)


def build_graph(config: GraphConfig):
    """Build and compile the router-specialist graph.

    Args:
        config (GraphConfig): Runtime configuration supplying LLM clients, Qdrant
            settings, and retrieval parameters.

    Returns:
        CompiledStateGraph: Compiled LangGraph ready for invocation.
    """

    async def _tool(fn, state, **kw):
        """Delegate to a tool function with the shared config."""
        return await fn(state, config=config, **kw)

    async def _retrieve_high_k(state: ExperimentState) -> dict:
        """Retrieve with an elevated top-k limit for claims and exclusions."""
        # Claims/exclusion questions often cite multiple scattered clauses;
        # higher k improves recall without changing the reranking cap downstream.
        return await retrieve(state, config=config, top_k=8)

    async def _comp_retrieve(state: ExperimentState) -> dict:
        """Use only the first two rewrites for comparison retrieval."""
        # Comparison questions cover two products; two rewrites (one per product)
        # are enough to seed retrieval without fetching redundant cross-product hits.
        queries = state.get("rewritten_queries", [state["question"]])[:2]
        return await retrieve_multi(state, config=config, queries=queries)

    async def router_node(state: ExperimentState) -> dict:
        """Classify the question intent and store the routing category in metadata."""
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=state["question"]),
        ]
        try:
            structured_llm = config.router_llm.with_structured_output(RouterDecision)
            route = await structured_llm.ainvoke(messages)
            category = route.category
            raw = json.dumps(route.model_dump(), ensure_ascii=False)
        except Exception:
            category = "faq"
            raw = '{"category": "faq"}'

        if category not in app_config.router.valid_categories:
            category = "faq"

        raw_metadata = state.get("metadata")
        metadata = {**raw_metadata} if isinstance(raw_metadata, dict) else {}
        metadata["_route"] = category
        metadata = record_model_usage(
            metadata,
            category="agent_llm",
            model_name=config.router_model,
            input_tokens=estimate_message_tokens(messages, config.router_model),
            output_tokens=estimate_text_tokens(raw, config.router_model),
        )
        metadata = update_process_metrics(metadata, router_category=category)

        return {
            "intent": category,
            "metadata": metadata,
        }

    def route_to_specialist(state: ExperimentState) -> str:
        """Map the stored routing category to its specialist pipeline prefix."""
        metadata = state.get("metadata")
        category = (
            metadata.get("_route", "faq") if isinstance(metadata, dict) else "faq"
        )
        return app_config.router.category_to_pipeline.get(category, "faq")

    _STEPS = [
        "parse",
        "rewrite",
        "retrieve",
        "rerank",
        "evidence",
        "citations",
        "select",
        "answer",
    ]

    _STEP_FNS = {
        "parse": lambda s: _tool(parse_question, s),
        "rewrite": lambda s: _tool(rewrite_query, s),
        "rerank": lambda s: _tool(rerank, s),
        "evidence": lambda s: _tool(select_evidence, s),
        "citations": lambda s: _tool(make_citations, s),
        "select": lambda s: _tool(select_prompt, s),
        "answer": lambda s: _tool(synthesize_answer, s),
    }

    _RETRIEVE_FNS = {
        "faq": lambda s: _tool(retrieve_multi, s),
        "description": lambda s: _tool(retrieve_multi, s),
        "claims": _retrieve_high_k,
        "comparison": _comp_retrieve,
    }

    graph = StateGraph(ExperimentState)

    graph.add_node("router", router_node)
    graph.add_edge(START, "router")

    for prefix in ["faq", "description", "claims", "comparison"]:
        for step in _STEPS:
            node_name = f"{prefix}_{step}"
            if step == "retrieve":
                fn = _RETRIEVE_FNS[prefix]
            else:
                fn = _STEP_FNS[step]
            _fn = fn

            async def _node(state, _f=_fn):
                """Delegate to the specialist step function."""
                return await _f(state)

            graph.add_node(node_name, _node)

        for i in range(len(_STEPS) - 1):
            graph.add_edge(f"{prefix}_{_STEPS[i]}", f"{prefix}_{_STEPS[i + 1]}")
        graph.add_edge(f"{prefix}_{_STEPS[-1]}", END)

    graph.add_conditional_edges(
        "router",
        route_to_specialist,
        {
            "faq": "faq_parse",
            "description": "description_parse",
            "claims": "claims_parse",
            "comparison": "comparison_parse",
        },
    )

    return graph.compile(name=config.pattern_name)
