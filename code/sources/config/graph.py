"""
config.graph
------------
Runtime graph configuration and Mermaid diagram metadata. Exposes the
frozen ``GraphConfig`` dataclass that lazily builds and caches LLM,
embedding, and Qdrant clients shared by every node of a compiled
LangGraph, plus the diagram strings/titles/descriptions used by the
visualisation notebook.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import AsyncQdrantClient

from sources.config.app import config as app_config

load_dotenv()


_STYLES = """\
    classDef startEnd fill:#455a64,stroke:#263238,color:#eceff1,stroke-width:2px
    classDef tool fill:#e8ecf0,stroke:#546e7a,color:#263238,stroke-width:1.5px
    classDef llm fill:#f0ebe3,stroke:#8d6e63,color:#3e2723,stroke-width:1.5px
    classDef specialist fill:#e6ece6,stroke:#607d60,color:#2e3d2e,stroke-width:1.5px
    classDef warn fill:#f0e0e0,stroke:#8b6060,color:#4a2020,stroke-width:1.5px
"""

_DETERMINISTIC = f"""\
graph TD
    START((START)):::startEnd

    subgraph preprocessing ["Preprocessing"]
        parse_question["<b>1. Parse Question</b><br/>Extract company, product, intent"]:::tool
        rewrite_query["<b>2. Rewrite Query</b><br/>Generate search queries"]:::tool
    end

    subgraph retrieval ["Retrieval"]
        retrieve_all["<b>3. Retrieve</b><br/>Vector search across indexes"]:::tool
        rerank["<b>4. Rerank</b><br/>Cross-encoder re-scoring"]:::tool
        evidence["<b>5. Evidence Selector</b><br/>Pick strongest clauses"]:::tool
    end

    subgraph generation ["Generation"]
        select_prompt["<b>6. Select Prompt</b><br/>Choose template by intent"]:::tool
        make_citations["<b>7. Make Citations</b><br/>Attach source references"]:::tool
        generate_answer["<b>8. Generate Answer</b><br/>LLM synthesises response"]:::tool
    end

    END((END)):::startEnd

    START --> parse_question
    parse_question --> rewrite_query
    rewrite_query --> retrieve_all
    retrieve_all --> rerank
    rerank --> evidence
    evidence --> select_prompt
    select_prompt --> make_citations
    make_citations --> generate_answer
    generate_answer --> END

{_STYLES}"""

_PLANNER_EXECUTOR = f"""\
graph TD
    START((START)):::startEnd

    subgraph planning ["Planning"]
        planner(["<b>1. Planner</b><br/>LLM generates execution plan"]):::llm
    end

    subgraph execution ["Execution"]
        executor["<b>2. Executor</b><br/>Run next tool in plan"]:::tool
    end

    subgraph generation ["Generation"]
        evidence["<b>3. Evidence Selector</b><br/>Pick strongest clauses"]:::tool
        synthesizer(["<b>4. Synthesizer</b><br/>Generate final answer"]):::llm
    end

    END((END)):::startEnd

    START --> planner
    planner --> executor
    executor -.->|"More steps in plan"| executor
    executor -->|"Plan complete"| evidence
    evidence --> synthesizer
    synthesizer --> END

{_STYLES}"""

_ROUTER_SPECIALIST = f"""\
graph TD
    START((START)):::startEnd

    subgraph routing ["Routing"]
        router(["<b>1. Router</b><br/>LLM classifies question intent"]):::llm
    end

    subgraph specialists ["Specialist Dispatch"]
        faq_pipe["FAQ Specialist"]:::specialist
        desc_pipe["Description Specialist"]:::specialist
        claims_pipe["Claims Specialist"]:::specialist
        comp_pipe["Comparison Specialist"]:::specialist
    end

    subgraph pipeline ["Specialist Pipeline"]
        direction TB
        sp_parse["<b>2. Parse</b>"]:::tool
        sp_rewrite["<b>3. Rewrite</b>"]:::tool
        sp_retrieve["<b>4. Retrieve</b>"]:::tool
        sp_rerank["<b>5. Rerank</b>"]:::tool
        sp_evidence["<b>6. Evidence</b>"]:::tool
        sp_citations["<b>7. Citations</b>"]:::tool
        sp_select["<b>8. Select Prompt</b>"]:::tool
        sp_answer["<b>9. Answer</b>"]:::tool
    end

    END((END)):::startEnd

    START --> router
    router -.->|"faq / limits"| faq_pipe
    router -.->|"description / coverage / conditions"| desc_pipe
    router -.->|"claims / exclusions"| claims_pipe
    router -.->|"comparison"| comp_pipe

    faq_pipe --> sp_parse
    desc_pipe --> sp_parse
    claims_pipe --> sp_parse
    comp_pipe --> sp_parse

    sp_parse --> sp_rewrite
    sp_rewrite --> sp_retrieve
    sp_retrieve --> sp_rerank
    sp_rerank --> sp_evidence
    sp_evidence --> sp_citations
    sp_citations --> sp_select
    sp_select --> sp_answer
    sp_answer --> END

{_STYLES}"""

_BLACKBOARD = f"""\
graph TD
    START((START)):::startEnd

    subgraph control ["Control"]
        dispatcher(["<b>Dispatcher</b><br/>Inspect state, pick next agent"]):::llm
    end

    subgraph agents ["Agent Pool"]
        parser["<b>A1. Parser</b><br/>Extract company, product, intent"]:::tool
        rewriter["<b>A2. Rewriter</b><br/>Generate search queries"]:::tool
        retriever["<b>A3. Retriever</b><br/>Vector search"]:::tool
        reranker["<b>A4. Reranker</b><br/>Cross-encoder scoring"]:::tool
        evidence["<b>A5. Evidence Selector</b><br/>Pick strongest clauses"]:::tool
        prompt_sel["<b>A6. Prompt Selector</b><br/>Choose prompt template"]:::tool
        citation["<b>A7. Citation Maker</b><br/>Attach source references"]:::tool
        answer["<b>A8. Answer Generator</b><br/>Synthesise final answer"]:::tool
    end

    END((END)):::startEnd

    START --> dispatcher

    dispatcher -.->|"No intent"| parser
    dispatcher -.->|"No rewrites"| rewriter
    dispatcher -.->|"No chunks"| retriever
    dispatcher -.->|"Not reranked"| reranker
    dispatcher -.->|"No evidence set"| evidence
    dispatcher -.->|"No template"| prompt_sel
    dispatcher -.->|"No citations"| citation
    dispatcher -.->|"Ready"| answer
    dispatcher -.->|"max iterations reached"| answer

    parser --> dispatcher
    rewriter --> dispatcher
    retriever --> dispatcher
    reranker --> dispatcher
    evidence --> dispatcher
    prompt_sel --> dispatcher
    citation --> dispatcher

    answer --> END

{_STYLES}"""

_HIERARCHICAL = f"""\
graph TD
    START((START)):::startEnd

    subgraph preprocessing ["Preprocessing"]
        parse_question["<b>1. Parse Question</b><br/>Extract company, product, intent"]:::tool
    end

    subgraph decomposition ["Decomposition"]
        decomposer(["<b>2. Decomposer</b><br/>LLM splits into 1-3 sub-questions"]):::llm
    end

    subgraph worker_loop ["Worker Loop"]
        worker["<b>3. Worker</b><br/>Rewrite, Retrieve, Rerank, Partial Answer"]:::tool
    end

    subgraph generation ["Generation"]
        evidence["<b>4. Evidence Selector</b><br/>Pick strongest clauses"]:::tool
        synthesizer(["<b>5. Synthesizer</b><br/>Merge partial answers + citations + final answer"]):::llm
    end

    END((END)):::startEnd

    START --> parse_question
    parse_question --> decomposer
    decomposer --> worker
    worker -.->|"More sub-questions"| worker
    worker -->|"All answered"| evidence
    evidence --> synthesizer
    synthesizer --> END

{_STYLES}"""

_REACT = f"""\
graph TD
    START((START)):::startEnd

    subgraph reasoning ["Reasoning Loop"]
        agent(["<b>1. Agent</b><br/>LLM produces Thought + Action"]):::llm
        tool_executor["<b>2. Tool Executor</b><br/>Dispatch selected tool"]:::tool
    end

    evidence["<b>3. Evidence Selector</b><br/>Pick strongest clauses before synthesis"]:::tool
    force_finish["<b>4. Force Finish</b><br/>Fallback answer generation"]:::warn

    END((END)):::startEnd

    START --> agent
    agent --> tool_executor
    tool_executor -.->|"action != finish and iterations < max"| agent
    tool_executor -->|"action = finish"| evidence
    evidence --> END
    tool_executor -.->|"iterations >= max"| force_finish
    force_finish --> END

{_STYLES}"""

_LEGEND = f"""\
graph LR
    subgraph legend ["Legend"]
        direction LR
        tool_ex["<b>Tool Node</b><br/>Deterministic execution"]:::tool
        llm_ex(["<b>LLM Node</b><br/>Model-based decision"]):::llm
        warn_ex["<b>Guard Node</b><br/>Fallback / safety"]:::warn
        se_ex((START / END)):::startEnd
    end

    tool_ex ---|"Solid: sequential flow"| llm_ex
    llm_ex -.-|"Dashed: conditional branch"| warn_ex

{_STYLES}"""

GRAPH_DIAGRAMS: dict[str, str] = {
    "deterministic": _DETERMINISTIC,
    "planner_executor": _PLANNER_EXECUTOR,
    "router_specialist": _ROUTER_SPECIALIST,
    "blackboard": _BLACKBOARD,
    "hierarchical": _HIERARCHICAL,
    "react": _REACT,
    "legend": _LEGEND,
}

GRAPH_TITLES: dict[str, str] = {
    "deterministic": "Pattern 1 — Deterministic Pipeline",
    "planner_executor": "Pattern 2 — Planner-Executor",
    "router_specialist": "Pattern 3 — Router-Specialist",
    "blackboard": "Pattern 4 — Blackboard / Swarm",
    "hierarchical": "Pattern 5 — Hierarchical Decomposition",
    "react": "Pattern 6 — ReAct",
    "legend": "Legend",
}

GRAPH_DESCRIPTIONS: dict[str, str] = {
    "deterministic": (
        "Fixed linear sequence with no LLM routing decisions. "
        "Every question passes through the same eight processing steps, including evidence selection before synthesis."
    ),
    "planner_executor": (
        "An LLM planner decides tool execution order, then an executor loop "
        "runs each step sequentially until the plan is complete, followed by an evidence-selection and synthesis tail."
    ),
    "router_specialist": (
        "An LLM router classifies the question intent, then dispatches to one "
        "of four specialist sub-graphs with tailored retrieval strategies and an evidence-selection step before answer generation."
    ),
    "blackboard": (
        "A central dispatcher inspects shared state each iteration and routes "
        "to the next agent, including an explicit evidence-selector agent before final answering. A max-iterations guard prevents infinite loops."
    ),
    "hierarchical": (
        "A decomposer breaks the question into 1-3 sub-questions. Each is "
        "answered by a worker mini-pipeline. Evidence is compacted before a synthesizer merges partial answers."
    ),
    "react": (
        "The agent produces a structured Thought + Action decision. A tool-executor dispatches "
        "the action, with evidence selection on finish and a guarded fallback on iteration cap."
    ),
    "legend": (
        "Universal legend for all architecture diagrams. Shows node shapes, "
        "colors, and edge styles used across figures."
    ),
}


@dataclass(frozen=True)
class GraphConfig:
    """Runtime settings with cached model and store clients.

    Attributes:
        pattern_name (str): Architecture pattern identifier used as the graph name.
        llm_model (str): Default LLM model identifier.
        parser_model (str): Model used by the question-parser node.
        router_model (str): Model used by the router node.
        planner_model (str): Model used by the planner node.
        controller_model (str): Model used by the ReAct controller node.
        rewrite_model (str): Model used by the query-rewriter node.
        decomposer_model (str): Model used by the hierarchical decomposer node.
        synthesis_model (str): Model used by the answer-synthesiser node.
        temperature (float): Sampling temperature applied to all LLM clients.
        top_k (int): Default number of chunks to retrieve per query.
        evidence_top_k (int): Maximum evidence chunks forwarded to the answer model.
        max_react_iterations (int): Maximum ReAct reasoning iterations before a
            forced finish.
        max_blackboard_iterations (int): Maximum blackboard dispatcher cycles before
            a forced finish.
        llm_request_timeout (float): HTTP timeout in seconds for LLM API requests.
    """

    pattern_name: str = "deterministic"
    llm_model: str = field(default_factory=lambda: app_config.llm.model_name)
    parser_model: str = field(
        default_factory=lambda: app_config.llm.model_for_role("parser")
    )
    router_model: str = field(
        default_factory=lambda: app_config.llm.model_for_role("router")
    )
    planner_model: str = field(
        default_factory=lambda: app_config.llm.model_for_role("planner")
    )
    controller_model: str = field(
        default_factory=lambda: app_config.llm.model_for_role("controller")
    )
    rewrite_model: str = field(
        default_factory=lambda: app_config.llm.model_for_role("rewrite")
    )
    decomposer_model: str = field(
        default_factory=lambda: app_config.llm.model_for_role("decomposer")
    )
    synthesis_model: str = field(
        default_factory=lambda: app_config.llm.model_for_role("synthesis")
    )
    temperature: float = 0.0
    retry_max_attempts: int = field(
        default_factory=lambda: app_config.llm.retry_max_attempts
    )
    retry_min_wait: float = field(default_factory=lambda: app_config.llm.retry_min_wait)
    retry_max_wait: float = field(default_factory=lambda: app_config.llm.retry_max_wait)
    reranker_model: str = field(default_factory=lambda: app_config.reranking.model_name)
    reranker_temperature: float = field(
        default_factory=lambda: app_config.reranking.temperature
    )
    reranker_enabled: bool = field(default_factory=lambda: app_config.reranking.enabled)
    reranker_score_threshold: float = field(
        default_factory=lambda: app_config.reranking.score_threshold
    )
    reranker_top_k_before: int = field(
        default_factory=lambda: app_config.reranking.top_k_before_rerank
    )
    reranker_top_k_after: int = field(
        default_factory=lambda: app_config.reranking.top_k_after_rerank
    )
    reranker_batch_size: int = field(
        default_factory=lambda: app_config.reranking.batch_size
    )
    reranker_max_workers: int = field(
        default_factory=lambda: app_config.reranking.max_workers
    )
    embedding_model: str = field(
        default_factory=lambda: app_config.embedding.model_name
    )
    embedding_dimension: int = field(
        default_factory=lambda: app_config.embedding.embedding_dimension
    )
    qdrant_host: str = field(default_factory=lambda: app_config.qdrant.host)
    qdrant_port: int = field(default_factory=lambda: app_config.qdrant.port)
    collection_name: str = field(
        default_factory=lambda: app_config.qdrant.collection_name
    )
    top_k: int = 5
    evidence_top_k: int = field(
        default_factory=lambda: app_config.concurrency.evidence_top_k
    )
    max_react_iterations: int = 15
    max_blackboard_iterations: int = 15
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""),
    )
    llm_request_timeout: float = 120.0

    def __post_init__(self) -> None:
        """Initialise the internal client cache dictionary."""
        # The dataclass is frozen for safe sharing across nodes, but the cache must
        # be mutable; object.__setattr__ bypasses the frozen guard for this one field.
        object.__setattr__(self, "_cache", {})

    def _get_http_async_client(self) -> httpx.AsyncClient:
        """Return a cached shared async HTTP client.

        Returns:
            httpx.AsyncClient: Singleton async HTTP client configured with the
                request timeout.
        """
        cache: dict[str, Any] = object.__getattribute__(self, "_cache")
        # A single httpx.AsyncClient is reused across all LLM instances so that
        # connection pooling and the configured timeout are applied uniformly
        # without opening a new socket per model property access.
        if "http_async_client" not in cache:
            cache["http_async_client"] = httpx.AsyncClient(
                timeout=self.llm_request_timeout
            )
        return cache["http_async_client"]

    def _get_chat_llm(self, cache_key: str, model_name: str) -> ChatOpenAI:
        """Return a cached ChatOpenAI client for the given model.

        Args:
            cache_key (str): Key under which the client is stored in the cache.
            model_name (str): OpenAI model identifier for the client.

        Returns:
            ChatOpenAI: LangChain ChatOpenAI instance sharing the async HTTP client.
        """
        cache: dict = object.__getattribute__(self, "_cache")
        # Keyed by (cache_key, model_name) semantics: two different roles that
        # happen to resolve to the same model still share a single client instance.
        if cache_key not in cache:
            cache[cache_key] = ChatOpenAI(
                model=model_name,
                temperature=self.temperature,
                timeout=self.llm_request_timeout,
                http_async_client=self._get_http_async_client(),
            )
        return cache[cache_key]

    @property
    def llm(self) -> ChatOpenAI:
        """Return the default LLM client.

        Returns:
            ChatOpenAI: Default ChatOpenAI client.
        """
        return self._get_chat_llm("llm", self.llm_model)

    @property
    def parser_llm(self) -> ChatOpenAI:
        """Return the question-parser LLM client.

        Returns:
            ChatOpenAI: ChatOpenAI client configured for the parser role.
        """
        return self._get_chat_llm("parser_llm", self.parser_model)

    @property
    def router_llm(self) -> ChatOpenAI:
        """Return the router LLM client.

        Returns:
            ChatOpenAI: ChatOpenAI client configured for the router role.
        """
        return self._get_chat_llm("router_llm", self.router_model)

    @property
    def planner_llm(self) -> ChatOpenAI:
        """Return the planner LLM client.

        Returns:
            ChatOpenAI: ChatOpenAI client configured for the planner role.
        """
        return self._get_chat_llm("planner_llm", self.planner_model)

    @property
    def controller_llm(self) -> ChatOpenAI:
        """Return the ReAct controller LLM client.

        Returns:
            ChatOpenAI: ChatOpenAI client configured for the controller role.
        """
        return self._get_chat_llm("controller_llm", self.controller_model)

    @property
    def rewrite_llm(self) -> ChatOpenAI:
        """Return the query-rewriter LLM client.

        Returns:
            ChatOpenAI: ChatOpenAI client configured for the rewrite role.
        """
        return self._get_chat_llm("rewrite_llm", self.rewrite_model)

    @property
    def decomposer_llm(self) -> ChatOpenAI:
        """Return the hierarchical decomposer LLM client.

        Returns:
            ChatOpenAI: ChatOpenAI client configured for the decomposer role.
        """
        return self._get_chat_llm("decomposer_llm", self.decomposer_model)

    @property
    def synthesis_llm(self) -> ChatOpenAI:
        """Return the answer-synthesiser LLM client.

        Returns:
            ChatOpenAI: ChatOpenAI client configured for the synthesis role.
        """
        return self._get_chat_llm("synthesis_llm", self.synthesis_model)

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """Return the cached OpenAI embeddings client.

        Returns:
            OpenAIEmbeddings: LangChain embeddings client sharing the async HTTP client.
        """
        cache: dict = object.__getattribute__(self, "_cache")
        if "embeddings" not in cache:
            cache["embeddings"] = OpenAIEmbeddings(
                model=self.embedding_model,
                http_async_client=self._get_http_async_client(),
            )
        return cache["embeddings"]

    @property
    def reranker_llm(self) -> ChatOpenAI:
        """Return the reranker LLM client.

        Returns:
            ChatOpenAI: ChatOpenAI client configured with reranker temperature and model.
        """
        cache: dict = object.__getattribute__(self, "_cache")
        if "reranker_llm" not in cache:
            cache["reranker_llm"] = ChatOpenAI(
                model=self.reranker_model,
                temperature=self.reranker_temperature,
                timeout=self.llm_request_timeout,
                http_async_client=self._get_http_async_client(),
            )
        return cache["reranker_llm"]

    @property
    def qdrant(self) -> AsyncQdrantClient:
        """Return the cached async Qdrant client.

        Returns:
            AsyncQdrantClient: Qdrant client connected to the configured host and port.
        """
        cache: dict = object.__getattribute__(self, "_cache")
        if "qdrant" not in cache:
            cache["qdrant"] = AsyncQdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port,
            )
        return cache["qdrant"]

    async def aclose(self) -> None:
        """Close all cached HTTP and Qdrant clients and clear the cache."""
        cache: dict[str, Any] = object.__getattribute__(self, "_cache")
        # Track by object id rather than value: the same underlying httpx client is
        # shared across multiple LLM wrappers, so id-based deduplication prevents
        # closing it twice and avoids errors on the second close call.
        seen: set[int] = set()

        for resource in cache.values():
            # Introspect known internal attribute names used by LangChain/httpx
            # to locate the closeable transport client buried inside each wrapper.
            for attribute_name in (
                "root_async_client",
                "async_client",
                "http_async_client",
                "root_client",
                "client",
                "http_client",
            ):
                attribute = getattr(resource, attribute_name, None)
                if attribute is None:
                    continue
                identifier = id(attribute)
                if identifier in seen:
                    continue
                seen.add(identifier)
                await self._close_resource(attribute)

            identifier = id(resource)
            if identifier in seen:
                continue
            seen.add(identifier)
            await self._close_resource(resource)

        cache.clear()

    async def _close_resource(self, resource: Any) -> None:
        """Call ``aclose`` or ``close`` on a resource if available.

        Args:
            resource (Any): Object that may expose an ``aclose`` or ``close`` method.
        """
        for method_name in ("aclose", "close"):
            method = getattr(resource, method_name, None)
            if method is None:
                continue
            result = method()
            if inspect.isawaitable(result):
                await result
            return
