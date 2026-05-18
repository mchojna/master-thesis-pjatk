"""
sources
-------
Top-level package for the master's thesis insurance-document RAG
pipeline. Re-exports the public configuration surface, the runner
and evaluator entry points, and the document-processing classes
(extractor, chunker, vector store) used by the experiment notebooks.
"""

from sources.config import (
    EXTRACTION_PROMPT,
    AppConfig,
    ChunkingConfig,
    ConcurrencyConfig,
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    EmbeddingConfig,
    EvaluationConfig,
    GraphConfig,
    IPIDChunkingConfig,
    LLMConfig,
    OWUChunkingConfig,
    PathConfig,
    PricingConfig,
    QdrantConfig,
    RerankingConfig,
    RetrievalResult,
    generate_chunk_id,
)
from sources.chuncker import Chuncker
from sources.evaluator import Evaluator
from sources.extractor import Extractor
from sources.observer import Observer, agent_span, evaluator_span, init_tracing
from sources.runner import Runner
from sources.vectorstore import VectorStore

__all__ = [
    "AppConfig",
    "Chuncker",
    "ChunkingConfig",
    "ConcurrencyConfig",
    "DocumentChunk",
    "DocumentMetadata",
    "DocumentType",
    "EmbeddingConfig",
    "EvaluationConfig",
    "EXTRACTION_PROMPT",
    "Evaluator",
    "Extractor",
    "GraphConfig",
    "IPIDChunkingConfig",
    "LLMConfig",
    "OWUChunkingConfig",
    "PathConfig",
    "PricingConfig",
    "QdrantConfig",
    "RerankingConfig",
    "Runner",
    "VectorStore",
    "RetrievalResult",
    "Observer",
    "agent_span",
    "evaluator_span",
    "generate_chunk_id",
    "init_tracing",
]
