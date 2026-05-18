"""
config.app
----------
Core application settings and shared data models. Defines the project
directory layout, LLM/embedding/reranking parameters, chunking
profiles per document type, Qdrant connection, evaluation column
schemas, token pricing tables, domain-specific tool configurations
(retriever, question parser, planner, router, answer synthesiser,
patterns), and the immutable ``DocumentChunk`` and ``DocumentMetadata``
records used end-to-end.
"""

import os
import uuid
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")


class DocumentType(str, Enum):
    """Supported insurance document types."""

    IPID = "ipid"
    OWU = "owu"


FOLDER_TO_DOCUMENT_TYPE: dict[str, DocumentType] = {
    "ipid": DocumentType.IPID,
    "owu": DocumentType.OWU,
}


class PathConfig(BaseModel):
    """Project directory paths.

    Attributes:
        raw_documents_dir (Path): Root directory for source PDF files.
        extracted_documents_dir (Path): Root directory for extracted Markdown files.
        results_dir (Path): Directory where evaluation result CSV files are written.
        questions_dir (Path): Directory containing evaluation question files.
    """

    raw_documents_dir: Path = PROJECT_ROOT / "data" / "documents" / "raw_documents"
    extracted_documents_dir: Path = (
        PROJECT_ROOT / "data" / "documents" / "extracted_documents"
    )
    results_dir: Path = PROJECT_ROOT / "data" / "evaluation" / "results"
    questions_dir: Path = PROJECT_ROOT / "data" / "evaluation" / "questions"


class LLMConfig(BaseModel):
    """LLM settings.

    Attributes:
        model_name (str): Default model used when no role-specific model is set.
        parser_model_name (str | None): Model for the question-parser role.
        router_model_name (str | None): Model for the router role.
        planner_model_name (str | None): Model for the planner role.
        controller_model_name (str | None): Model for the ReAct controller role.
        rewrite_model_name (str | None): Model for the query-rewriter role.
        decomposer_model_name (str | None): Model for the hierarchical decomposer role.
        synthesis_model_name (str | None): Model for the answer-synthesiser role.
        evaluation_model_name (str | None): Model for the evaluator role.
        temperature (float): Sampling temperature applied to all LLM calls.
        max_concurrent_api_calls (int): Semaphore limit for concurrent API requests.
        retry_max_attempts (int): Maximum retry attempts on transient API failures.
        retry_min_wait (float): Minimum seconds between retry attempts.
        retry_max_wait (float): Maximum seconds between retry attempts.
        enable_cross_page_context (bool): Whether to pass the previous page tail to
            the extractor for cross-page continuity.
        cross_page_context_chars (int): Character count from the previous page passed
            as context to the extractor.
    """

    model_name: str = "gpt-5"
    parser_model_name: str | None = "gpt-5"
    router_model_name: str | None = "gpt-5"
    planner_model_name: str | None = "gpt-5"
    controller_model_name: str | None = "gpt-5"
    rewrite_model_name: str | None = "gpt-5"
    decomposer_model_name: str | None = "gpt-5"
    synthesis_model_name: str | None = "gpt-5"
    evaluation_model_name: str | None = "gpt-5"
    temperature: float = 0.0
    max_concurrent_api_calls: int = 16
    retry_max_attempts: int = 5
    retry_min_wait: float = 1.0
    retry_max_wait: float = 60.0
    enable_cross_page_context: bool = True
    cross_page_context_chars: int = 1024

    def model_for_role(self, role: str) -> str:
        """Return the configured model name for a specific LLM role.

        Args:
            role (str): Role identifier; one of ``"parser"``, ``"router"``,
                ``"planner"``, ``"controller"``, ``"rewrite"``, ``"decomposer"``,
                ``"synthesis"``, or ``"evaluation"``.

        Returns:
            str: Role-specific model name, or the default ``model_name`` when the
                role is unknown or has no override.
        """
        mapping = {
            "parser": self.parser_model_name,
            "router": self.router_model_name,
            "planner": self.planner_model_name,
            "controller": self.controller_model_name,
            "rewrite": self.rewrite_model_name,
            "decomposer": self.decomposer_model_name,
            "synthesis": self.synthesis_model_name,
            "evaluation": self.evaluation_model_name,
        }
        return mapping.get(role) or self.model_name

    def role_models(self) -> dict[str, str]:
        """Return the resolved model name for each configured role.

        Returns:
            dict[str, str]: Mapping from role name to resolved model identifier,
                including the ``"default"`` key for the base model.
        """
        return {
            "default": self.model_name,
            "parser": self.model_for_role("parser"),
            "router": self.model_for_role("router"),
            "planner": self.model_for_role("planner"),
            "controller": self.model_for_role("controller"),
            "rewrite": self.model_for_role("rewrite"),
            "decomposer": self.model_for_role("decomposer"),
            "synthesis": self.model_for_role("synthesis"),
            "evaluation": self.model_for_role("evaluation"),
        }


class EmbeddingConfig(BaseModel):
    """Embedding settings.

    Attributes:
        model_name (str): OpenAI embedding model identifier.
        embedding_dimension (int): Output vector dimensionality for the model.
    """

    model_name: str = "text-embedding-3-large"
    embedding_dimension: int = 3072


class RerankingConfig(BaseModel):
    """Reranking settings.

    Attributes:
        enabled (bool): Whether LLM-based reranking is active.
        model_name (str): Model used for reranking scoring.
        temperature (float): Sampling temperature for the reranker LLM.
        score_threshold (float): Minimum score a chunk must achieve to be kept.
        top_k_before_rerank (int): Number of chunks retrieved before reranking.
        top_k_after_rerank (int): Maximum chunks retained after reranking.
        batch_size (int): Number of chunks per reranker LLM call.
        max_workers (int): Maximum concurrent reranker batch requests.
        max_reranker_tokens (int): Token budget cap per reranker batch payload.
    """

    enabled: bool = False
    model_name: str = "gpt-5"
    temperature: float = 0.0
    score_threshold: float = 0.3
    top_k_before_rerank: int = 20
    top_k_after_rerank: int = 5
    batch_size: int = 10
    max_workers: int = 10
    max_reranker_tokens: int = 12_000


class ChunkingConfig(BaseModel):
    """Base chunking settings.

    Attributes:
        chunk_size (int): Target token count per chunk.
        chunk_overlap (int): Token overlap between consecutive chunks.
        min_chunk_size (int): Minimum token count; smaller chunks are discarded.
        headers_to_split_on (list[tuple[str, str]]): Markdown header levels used as
            primary split points, each as a ``(marker, field_name)`` pair.
    """

    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 64
    headers_to_split_on: list[tuple[str, str]] = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
    ]


class IPIDChunkingConfig(ChunkingConfig):
    """Chunking settings for IPID documents."""

    chunk_size: int = 384
    chunk_overlap: int = 48
    min_chunk_size: int = 48


class OWUChunkingConfig(ChunkingConfig):
    """Chunking settings for OWU documents."""

    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 64


class QdrantConfig(BaseModel):
    """Qdrant connection settings.

    Attributes:
        host (str): Qdrant server hostname.
        port (int): REST API port.
        grpc_port (int): gRPC port.
        collection_name (str): Name of the collection storing insurance chunks.
    """

    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    collection_name: str = "insurance_documents"


class ConcurrencyConfig(BaseModel):
    """Concurrency settings.

    Attributes:
        max_workers (int): Thread-pool size for concurrent PDF extraction.
        evidence_top_k (int): Maximum evidence chunks passed to the answer model.
    """

    max_workers: int = 32
    evidence_top_k: int = 4


class EvaluationConfig(BaseModel):
    """Evaluation metric configuration.

    Attributes:
        metric_cols (list[str]): Process and usage metric column names written to
            result CSV files.
        answer_score_cols (list[str]): Phoenix answer quality score column names.
        document_score_cols (list[str]): Phoenix document relevance score column names.
        tool_score_cols (list[str]): Phoenix tool selection and invocation score
            column names.
        correctness_choices (dict[str, float]): Label-to-score mapping for the
            reference-correctness classifier.
        conciseness_choices (dict[str, float]): Label-to-score mapping for the
            conciseness classifier.
    """

    metric_cols: list[str] = Field(
        default_factory=lambda: [
            "latency_ms",
            "completion_flag",
            "error_flag",
            "citation_count",
            "tool_count",
            "iteration_count",
            "plan_length",
            "retrieved_chunk_count",
            "retrieved_chunk_count_total",
            "retrieval_query_count",
            "reranked_chunk_count",
            "reranked_chunk_count_total",
            "agent_input_tokens_est",
            "agent_output_tokens_est",
            "embedding_input_tokens_est",
            "estimated_agent_cost_usd",
            "estimated_embedding_cost_usd",
            "estimated_total_cost_usd",
        ]
    )
    answer_score_cols: list[str] = Field(
        default_factory=lambda: [
            "phoenix_faithfulness_score",
            "phoenix_conciseness_score",
            "phoenix_correctness_score",
        ]
    )
    document_score_cols: list[str] = Field(
        default_factory=lambda: ["phoenix_document_relevance_score"]
    )
    tool_score_cols: list[str] = Field(
        default_factory=lambda: [
            "phoenix_tool_selection_score",
            "phoenix_tool_invocation_score",
        ]
    )
    correctness_choices: dict[str, float] = Field(
        default_factory=lambda: {
            "fully_correct": 1.0,
            "mostly_correct": 0.8,
            "partially_correct": 0.5,
            "mostly_incorrect": 0.3,
            "fully_incorrect": 0.0,
        }
    )
    conciseness_choices: dict[str, float] = Field(
        default_factory=lambda: {
            "very_concise": 1.0,
            "mostly_concise": 0.8,
            "mixed": 0.5,
            "mostly_verbose": 0.3,
            "very_verbose": 0.0,
        }
    )


class PricingConfig(BaseModel):
    """Token pricing tables for tracked models.

    Attributes:
        text_model_pricing_per_1m (dict[str, dict[str, float]]): Per-million-token
            input and output prices for each text model.
        embedding_model_pricing_per_1m (dict[str, dict[str, float]]): Per-million-token
            input prices for each embedding model.
    """

    text_model_pricing_per_1m: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "gpt-5": {"input": 1.25, "output": 10.0},
            "gpt-5-mini": {"input": 0.25, "output": 2.0},
            "gpt-5": {"input": 0.05, "output": 0.4},
            "gpt-5.1": {"input": 1.25, "output": 10.0},
            "gpt-5.2": {"input": 1.75, "output": 14.0},
            "gpt-5.2-chat-latest": {"input": 1.75, "output": 14.0},
            "gpt-5.3-chat-latest": {"input": 1.75, "output": 14.0},
            "gpt-5.4": {"input": 2.5, "output": 15.0},
        }
    )
    embedding_model_pricing_per_1m: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "text-embedding-3-small": {"input": 0.02, "output": 0.0},
            "text-embedding-3-large": {"input": 0.13, "output": 0.0},
        }
    )


class TqdmConfig(BaseModel):
    """Progress bar settings.

    Attributes:
        leave (bool): Whether to leave the progress bar visible after completion.
        dynamic_ncols (bool): Whether to auto-size the progress bar to terminal width.
        mininterval (float): Minimum seconds between progress bar refreshes.
    """

    leave: bool = False
    dynamic_ncols: bool = True
    mininterval: float = 0.5

    def to_kwargs(self) -> dict[str, bool | float]:
        """Return keyword arguments for tqdm.

        Returns:
            dict[str, bool | float]: Dict with ``leave``, ``dynamic_ncols``, and
                ``mininterval`` keys ready to be unpacked into a tqdm constructor.
        """
        return {
            "leave": self.leave,
            "dynamic_ncols": self.dynamic_ncols,
            "mininterval": self.mininterval,
        }


class PhoenixConfig(BaseModel):
    """Phoenix tracing settings.

    Attributes:
        host (str): Phoenix server hostname.
        port (int): Phoenix server port.
        project_name (str): Project name used to group traces in Phoenix.
        enabled (bool): Whether OpenTelemetry tracing is active.
    """

    host: str = "localhost"
    port: int = 6006
    project_name: str = "master-thesis"
    enabled: bool = True

    @property
    def endpoint(self) -> str:
        """Return the OTLP traces endpoint URL.

        Returns:
            str: Full URL of the Phoenix OTLP traces endpoint.
        """
        return f"http://{self.host}:{self.port}/v1/traces"


class DocumentMetadata(BaseModel):
    """Metadata stored with a document chunk.

    Attributes:
        company_name (str): Normalised insurance company identifier.
        product_name (str): Normalised product identifier.
        product_category (str): High-level product category (e.g. ``"auto"``).
        document_type (str): Document type string (``"ipid"`` or ``"owu"``).
        source_file (str): Relative path to the source Markdown file.
        header_1 (str | None): Top-level Markdown header extracted from the chunk.
        header_2 (str | None): Second-level Markdown header extracted from the chunk.
        header_3 (str | None): Third-level Markdown header extracted from the chunk.
    """

    company_name: str
    product_name: str
    product_category: str
    document_type: str
    source_file: str
    header_1: str | None = None
    header_2: str | None = None
    header_3: str | None = None


class DocumentChunk(BaseModel):
    """A processed document chunk.

    Attributes:
        chunk_id (str): Deterministic UUID derived from content and metadata.
        content (str): Plain text content of the chunk.
        metadata (DocumentMetadata): Provenance metadata for the chunk.
    """

    chunk_id: str
    content: str
    metadata: DocumentMetadata


class RetrievalResult(BaseModel):
    """A retrieval result pairing a chunk with its vector similarity score.

    Attributes:
        chunk (DocumentChunk): Retrieved document chunk.
        score (float): Cosine similarity score from the vector search.
    """

    chunk: DocumentChunk
    score: float


class RetrieverConfig(BaseModel):
    """Retriever settings including alias tables for company and product resolution.

    Attributes:
        min_retrieval_limit (int): Minimum number of candidates fetched from Qdrant.
        min_filtered_results (int): Minimum results required before filter relaxation.
        company_aliases (dict[str, str]): Maps alternate company name forms to canonical IDs.
        canonical_products_by_company (dict[str, set[str]]): Valid canonical product IDs
            grouped by company.
        product_aliases_by_company (dict[str, dict[str, str]]): Per-company mapping from
            alternate product names to canonical IDs.
        global_safe_product_aliases (dict[str, str]): Cross-company product aliases that are
            unambiguous and safe to apply without knowing the company.
    """

    min_retrieval_limit: int = 1
    min_filtered_results: int = 3
    company_aliases: dict[str, str] = Field(
        default_factory=lambda: {
            "ergo_hestia": "hestia",
            "stuh_ergo_hestia": "hestia",
            "sopockie_towarzystwo": "hestia",
            "sopockie_towarzystwo_ubezpieczen": "hestia",
            "stu_ergo_hestia": "hestia",
            "powszechny_zaklad_ubezpieczen": "pzu",
            "powszechny_zaklad_ubezpieczen_spolka_akcyjna": "pzu",
            "pzu_sa": "pzu",
            "tuir_warta": "warta",
            "towarzystwo_ubezpieczen_i_reasekuracji_warta": "warta",
            "towarzystwo_ubezpieczen_warta": "warta",
            "warta_sa": "warta",
        }
    )
    canonical_products_by_company: dict[str, set[str]] = Field(
        default_factory=lambda: {
            "hestia": {
                "ergo7_komunikacja",
                "ergo7_podroz",
                "ergo7_pozakomunikacyjne",
                "ergo_podroz",
            },
            "pzu": {
                "pzu_auto",
                "pzu_dom",
                "pzu_wojazer",
            },
            "warta": {
                "autocasco_komfort_ack",
                "autocasco_standard_acs",
                "owu_warta_dom",
                "warta_dom",
                "warta_dom_komfort",
                "warta_travel",
            },
        }
    )
    product_aliases_by_company: dict[str, dict[str, str]] = Field(
        default_factory=lambda: {
            "hestia": {
                "ergo_7_komunikacja": "ergo7_komunikacja",
                "ergo_komunikacja": "ergo7_komunikacja",
                "ergo7_kom": "ergo7_komunikacja",
                "ergo_7_podroz": "ergo7_podroz",
                "ergo_podroz": "ergo7_podroz",
                "ergo_7_pozakomunikacyjne": "ergo7_pozakomunikacyjne",
                "ergo_pozakomunikacyjne": "ergo7_pozakomunikacyjne",
            },
            "pzu": {
                "auto": "pzu_auto",
                "pzuauto": "pzu_auto",
                "dom": "pzu_dom",
                "pzudom": "pzu_dom",
                "wojazer": "pzu_wojazer",
                "podroz": "pzu_wojazer",
                "podroze": "pzu_wojazer",
            },
            "warta": {
                "autocasco_komfort": "autocasco_komfort_ack",
                "ac_komfort": "autocasco_komfort_ack",
                "autocasco_standard": "autocasco_standard_acs",
                "ac_standard": "autocasco_standard_acs",
                "travel": "warta_travel",
                "warta_dom_owu": "owu_warta_dom",
            },
        }
    )
    global_safe_product_aliases: dict[str, str] = Field(
        default_factory=lambda: {
            "autocasco_komfort": "autocasco_komfort_ack",
            "autocasco_standard": "autocasco_standard_acs",
        }
    )


class QuestionParserConfig(BaseModel):
    """Question parser settings including product-to-company mapping and fallback values.

    Attributes:
        product_to_company (dict[str, str]): Maps canonical product IDs to their owning
            company, used when only a product name is detected in the question.
        fallback_parse_result (dict): Default parse result returned on LLM failure.
    """

    product_to_company: dict[str, str] = Field(
        default_factory=lambda: {
            "ergo7_komunikacja": "hestia",
            "ergo7_podroz": "hestia",
            "ergo7_pozakomunikacyjne": "hestia",
            "ergo_podroz": "hestia",
            "pzu_auto": "pzu",
            "pzu_dom": "pzu",
            "pzu_wojazer": "pzu",
            "autocasco_komfort_ack": "warta",
            "autocasco_standard_acs": "warta",
            "owu_warta_dom": "warta",
            "warta_dom": "warta",
            "warta_dom_komfort": "warta",
            "warta_travel": "warta",
        }
    )
    fallback_parse_result: dict = Field(
        default_factory=lambda: {
            "company": None,
            "product": None,
            "intent": "unknown",
            "entities": [],
            "is_multi_part": False,
            "language": "pl",
        }
    )


class PlannerConfig(BaseModel):
    """Planner settings including the static fallback tool plan.

    Attributes:
        fallback_plan (list[dict[str, str]]): Ordered list of tool steps used when the
            planner LLM produces malformed output.
    """

    fallback_plan: list[dict[str, str]] = Field(
        default_factory=lambda: [
            {"tool_name": "question_parser", "reason": "fallback"},
            {"tool_name": "query_rewriter", "reason": "fallback"},
            {"tool_name": "retriever", "reason": "fallback"},
            {"tool_name": "reranker", "reason": "fallback"},
            {"tool_name": "evidence_selector", "reason": "fallback"},
            {"tool_name": "prompt_selector", "reason": "fallback"},
            {"tool_name": "citation_maker", "reason": "fallback"},
            {"tool_name": "answer_synthesizer", "reason": "fallback"},
        ]
    )


class RouterConfig(BaseModel):
    """Router settings including valid category names and pipeline dispatch map.

    Attributes:
        valid_categories (set[str]): Set of accepted routing category strings.
        category_to_pipeline (dict[str, str]): Maps each category to its specialist
            pipeline name.
    """

    valid_categories: set[str] = Field(
        default_factory=lambda: {
            "faq",
            "description",
            "coverage",
            "exclusions",
            "conditions",
            "limits",
            "comparison",
            "claims",
        }
    )
    category_to_pipeline: dict[str, str] = Field(
        default_factory=lambda: {
            "faq": "faq",
            "limits": "faq",
            "description": "description",
            "coverage": "description",
            "conditions": "description",
            "claims": "claims",
            "exclusions": "claims",
            "comparison": "comparison",
        }
    )


class AnswerSynthesizerConfig(BaseModel):
    """Answer synthesiser settings.

    Attributes:
        insufficient_context_msg (str): Polish-language fallback message returned when
            retrieved documents do not contain enough information to answer.
    """

    insufficient_context_msg: str = (
        "Dostępne dokumenty nie zawierają wystarczających informacji, "
        "aby odpowiedzieć na to pytanie."
    )


class PatternsConfig(BaseModel):
    """Agent pattern registry mapping pattern names to their module paths.

    Attributes:
        pattern_builders (dict[str, str]): Maps each pattern name to the fully
            qualified module path that exposes a ``build_graph`` callable.
    """

    pattern_builders: dict[str, str] = Field(
        default_factory=lambda: {
            "deterministic": "sources.agents.deterministic",
            "planner_executor": "sources.agents.planner_executor",
            "router_specialist": "sources.agents.router_specialist",
            "blackboard": "sources.agents.blackboard",
            "hierarchical": "sources.agents.hierarchical",
            "react": "sources.agents.react",
        }
    )


class AppConfig(BaseModel):
    """Top-level application settings.

    Attributes:
        paths (PathConfig): Project directory layout.
        llm (LLMConfig): LLM model and retry settings.
        embedding (EmbeddingConfig): Embedding model settings.
        reranking (RerankingConfig): LLM reranker settings.
        chunking_ipid (IPIDChunkingConfig): Chunking parameters for IPID documents.
        chunking_owu (OWUChunkingConfig): Chunking parameters for OWU documents.
        qdrant (QdrantConfig): Qdrant connection settings.
        concurrency (ConcurrencyConfig): Thread-pool and evidence size settings.
        evaluation (EvaluationConfig): Metric column schema and classifier choices.
        pricing (PricingConfig): Token cost tables for cost estimation.
        tqdm (TqdmConfig): Progress bar display settings.
        phoenix (PhoenixConfig): OpenTelemetry / Phoenix tracing settings.
        retriever (RetrieverConfig): Retriever alias tables and limit constants.
        question_parser (QuestionParserConfig): Parser product-to-company map and fallback.
        planner (PlannerConfig): Planner fallback tool plan.
        router (RouterConfig): Router category names and pipeline dispatch map.
        answer_synthesizer (AnswerSynthesizerConfig): Answer synthesiser fallback message.
        patterns (PatternsConfig): Agent pattern name to module path registry.
    """

    paths: PathConfig = Field(default_factory=PathConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    chunking_ipid: IPIDChunkingConfig = Field(default_factory=IPIDChunkingConfig)
    chunking_owu: OWUChunkingConfig = Field(default_factory=OWUChunkingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    tqdm: TqdmConfig = Field(default_factory=TqdmConfig)
    phoenix: PhoenixConfig = Field(default_factory=PhoenixConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    question_parser: QuestionParserConfig = Field(default_factory=QuestionParserConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    answer_synthesizer: AnswerSynthesizerConfig = Field(
        default_factory=AnswerSynthesizerConfig
    )
    patterns: PatternsConfig = Field(default_factory=PatternsConfig)

    def get_chunking_config(self, document_type: DocumentType) -> ChunkingConfig:
        """Return the chunking settings for a document type.

        Args:
            document_type (DocumentType): The document type to look up.

        Returns:
            ChunkingConfig: The corresponding chunking configuration instance.

        Raises:
            ValueError: If ``document_type`` is not a recognised DocumentType value.
        """
        mapping: dict[DocumentType, ChunkingConfig] = {
            DocumentType.IPID: self.chunking_ipid,
            DocumentType.OWU: self.chunking_owu,
        }
        if document_type not in mapping:
            raise ValueError(f"Unsupported document type: {document_type}")
        return mapping[document_type]


def generate_chunk_id(
    content: str,
    metadata: DocumentMetadata,
    suffix: str,
) -> str:
    """Generate a deterministic chunk ID from content and metadata.

    Args:
        content (str): Text content of the chunk; only the first 200 characters
            are used in the seed.
        metadata (DocumentMetadata): Provenance metadata supplying source file and
            header fields for the seed.
        suffix (str): Additional string (e.g. a chunk index) included in the seed
            to distinguish sibling chunks from the same section.

    Returns:
        str: UUID5 string derived from the seed, stable across identical inputs.
    """
    seed = (
        f"{metadata.source_file}:{metadata.header_1}:"
        f"{metadata.header_2}:{suffix}:{content[:200]}"
    )
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


config: AppConfig = AppConfig()
