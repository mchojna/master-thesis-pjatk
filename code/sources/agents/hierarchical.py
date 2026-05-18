"""
agents.hierarchical
-------------------
Pattern 5 - hierarchical decomposition. A decomposer LLM splits the
user question into one to three independent sub-questions; a worker
loop answers each sub-question independently; a synthesiser merges
partial answers, rebuilds citations and produces the final response.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from sources.config.graph import GraphConfig
from sources.config.prompts import (
    DECOMPOSER_SYSTEM_PROMPT,
    FAQ_PROMPT,
    MERGE_ANSWERS_SYSTEM_PROMPT,
    PROMPT_TEMPLATES,
)
from sources.agents.state import ExperimentState, deduplicate_chunks
from sources.tools import retrieve_multi
from sources.tools.answer_synthesizer import (
    _normalize_inline_citations,
    _normalize_answer_structure,
    build_references_section,
)
from sources.tools.citation_maker import make_citations
from sources.tools.evidence_selector import select_evidence
from sources.tools.prompt_selector import select_prompt
from sources.tools.question_parser import parse_question
from sources.tools.query_rewriter import rewrite_query
from sources.tools.reranker import rerank
from sources.tracker import (
    estimate_message_tokens,
    estimate_text_tokens,
    merge_tracking_metadata,
    record_model_usage,
    update_process_metrics,
)

logger = structlog.get_logger(__name__)


def build_graph(config: GraphConfig):
    """Build and compile the hierarchical decomposition graph.

    Args:
        config (GraphConfig): Runtime configuration supplying LLM clients, Qdrant
            settings, and retrieval parameters.

    Returns:
        CompiledStateGraph: Compiled LangGraph ready for invocation.
    """

    async def parse_question_node(state: ExperimentState) -> dict:
        """Parse company, product, and intent from the question."""
        return await parse_question(state, config=config)

    async def decomposer_node(state: ExperimentState) -> dict:
        """Decompose the question into one to three independent sub-questions."""
        messages = [
            SystemMessage(content=DECOMPOSER_SYSTEM_PROMPT),
            HumanMessage(content=state["question"]),
        ]
        response = await config.decomposer_llm.ainvoke(messages)

        raw = response.content if isinstance(response.content, str) else "[]"
        raw = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

        try:
            sub_questions = json.loads(raw)
            if not isinstance(sub_questions, list):
                sub_questions = [state["question"]]
        except json.JSONDecodeError:
            sub_questions = [state["question"]]

        # Cap at 3 sub-questions: each spawns a full retrieval mini-pipeline,
        # so more than 3 multiplies latency and cost with diminishing returns.
        sub_questions = [
            str(item).strip() for item in sub_questions if str(item).strip()
        ][:3]
        if not sub_questions:
            sub_questions = [state["question"]]

        metadata = {**state.get("metadata", {})}
        metadata["sub_questions"] = sub_questions
        metadata["_worker_index"] = 0
        metadata["_partial_answers"] = []
        metadata = record_model_usage(
            metadata,
            category="agent_llm",
            model_name=config.decomposer_model,
            input_tokens=estimate_message_tokens(messages, config.decomposer_model),
            output_tokens=estimate_text_tokens(raw, config.decomposer_model),
        )
        metadata = update_process_metrics(
            metadata,
            decomposition_count=len(sub_questions),
            worker_iteration_count=0,
        )

        return {
            "metadata": metadata,
        }

    async def worker_node(state: ExperimentState) -> dict:
        """Run the retrieval mini-pipeline for the current sub-question."""
        metadata = {**state.get("metadata", {})}
        idx = metadata.get("_worker_index", 0)
        sub_questions = metadata.get("sub_questions", [state["question"]])

        if idx >= len(sub_questions):
            return {"metadata": metadata}

        sub_q = sub_questions[idx]
        sub_state: ExperimentState = {**state, "question": sub_q}  # type: ignore[typeddict-item]

        rw_result = await rewrite_query(sub_state, config=config)
        metadata = merge_tracking_metadata(metadata, rw_result.get("metadata"))
        queries = rw_result.get("rewritten_queries", [sub_q])

        ret_state: ExperimentState = {**sub_state, "metadata": metadata}  # type: ignore[typeddict-item]
        ret_result = await retrieve_multi(ret_state, config=config, queries=queries)
        metadata = merge_tracking_metadata(metadata, ret_result.get("metadata"))
        deduped = ret_result.get("retrieved_chunks", [])

        rerank_state: ExperimentState = {  # type: ignore[typeddict-item]
            **sub_state,
            "retrieved_chunks": deduped,
            "metadata": metadata,
        }
        rr_result = await rerank(rerank_state, config=config)
        metadata = merge_tracking_metadata(metadata, rr_result.get("metadata"))
        reranked = rr_result.get("retrieved_chunks", deduped)

        intent = state.get("intent") or "unknown"
        partial_prompt = PROMPT_TEMPLATES.get(intent, FAQ_PROMPT)
        # Limit partial context to 5 chunks: enough evidence for a sub-answer
        # without exceeding a reasonable token budget per worker call.
        context = "\n\n".join(c.get("text", "") for c in reranked[:5])
        partial_messages = [
            SystemMessage(content=partial_prompt),
            HumanMessage(
                content=f"Sub-question: {sub_q}\n\nContext:\n{context}\n\n"
                "Provide a concise partial answer based ONLY on the context above."
            ),
        ]
        partial_resp = await config.synthesis_llm.ainvoke(partial_messages)

        partial_answer = (
            partial_resp.content if isinstance(partial_resp.content, str) else ""
        )
        metadata = record_model_usage(
            metadata,
            category="agent_llm",
            model_name=config.synthesis_model,
            input_tokens=estimate_message_tokens(
                partial_messages, config.synthesis_model
            ),
            output_tokens=estimate_text_tokens(partial_answer, config.synthesis_model),
        )

        partials = list(metadata.get("_partial_answers", []))
        partials.append({"sub_question": sub_q, "answer": partial_answer})
        metadata["_partial_answers"] = partials
        metadata["_worker_index"] = idx + 1
        metadata = update_process_metrics(
            metadata,
            worker_iteration_count=idx + 1,
        )

        existing_chunks = state.get("retrieved_chunks", [])
        merged = deduplicate_chunks(existing_chunks + reranked)

        return {
            "retrieved_chunks": merged,
            "metadata": metadata,
        }

    def should_continue_worker(state: ExperimentState) -> str:
        """Return 'worker' while sub-questions remain, otherwise 'synthesizer'."""
        metadata = state.get("metadata", {})
        # When _worker_index reaches len(sub_questions) all sub-questions have
        # been answered and the synthesizer can merge the partial results.
        if metadata.get("_worker_index", 0) < len(metadata.get("sub_questions", [])):
            return "worker"
        return "synthesizer"

    async def synthesizer_node(state: ExperimentState) -> dict:
        """Merge partial answers, rebuild citations, and generate the final answer."""
        # Evidence and citations are recomputed from the merged retrieved_chunks
        # rather than carried over from individual worker states, so the final
        # answer references the full cross-sub-question evidence set.
        evidence_result = await select_evidence(state, config=config)
        metadata = merge_tracking_metadata(
            state.get("metadata"), evidence_result.get("metadata")
        )
        evidence_chunks = evidence_result.get("evidence_chunks", [])

        cit_state: ExperimentState = {
            **state,
            "evidence_chunks": evidence_chunks,
            "metadata": metadata,
        }  # type: ignore[typeddict-item]
        cit_result = await make_citations(cit_state, config=config)
        metadata = merge_tracking_metadata(metadata, cit_result.get("metadata"))
        citations = cit_result.get("citations", [])

        prompt_state: ExperimentState = {
            **state,
            "evidence_chunks": evidence_chunks,
            "metadata": metadata,
        }  # type: ignore[typeddict-item]
        prompt_result = await select_prompt(prompt_state, config=config)
        metadata = merge_tracking_metadata(metadata, prompt_result.get("metadata"))
        system_prompt = prompt_result.get("system_prompt", FAQ_PROMPT)

        partials = metadata.get("_partial_answers", [])
        partial_text = "\n\n".join(
            f"Sub-question: {p['sub_question']}\nPartial answer: {p['answer']}"
            for p in partials
        )

        merge_system = (
            f"{system_prompt}\n\n"
            f"{MERGE_ANSWERS_SYSTEM_PROMPT.format(question=state['question'])}"
        )
        merge_messages = [
            SystemMessage(content=merge_system),
            HumanMessage(content=partial_text),
        ]
        response = await config.synthesis_llm.ainvoke(merge_messages)

        raw_answer_body = response.content if isinstance(response.content, str) else ""
        answer_body = _normalize_answer_structure(raw_answer_body)
        answer_body = _normalize_inline_citations(answer_body, citations)
        answer = answer_body + build_references_section(citations)
        metadata = record_model_usage(
            metadata,
            category="agent_llm",
            model_name=config.synthesis_model,
            input_tokens=estimate_message_tokens(
                merge_messages, config.synthesis_model
            ),
            output_tokens=estimate_text_tokens(answer_body, config.synthesis_model),
        )

        return {
            "evidence_chunks": evidence_chunks,
            "citations": citations,
            "selected_prompt_template": prompt_result.get("selected_prompt_template"),
            "system_prompt": system_prompt,
            "answer_body": answer_body,
            "answer_with_references": answer,
            "answer": answer,
            "metadata": metadata,
        }

    graph = StateGraph(ExperimentState)
    graph.add_node("parse_question", parse_question_node)
    graph.add_node("decomposer", decomposer_node)
    graph.add_node("worker", worker_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "parse_question")
    graph.add_edge("parse_question", "decomposer")
    graph.add_edge("decomposer", "worker")
    graph.add_conditional_edges(
        "worker",
        should_continue_worker,
        {"worker": "worker", "synthesizer": "synthesizer"},
    )
    graph.add_edge("synthesizer", END)

    return graph.compile(name=config.pattern_name)
