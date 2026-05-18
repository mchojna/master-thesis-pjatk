"""
schemas
-------
Pydantic schemas used for OpenAI structured outputs in agent control
models: question parsing, router category selection, and ReAct
decisions. All schemas extend a strict base model with
``extra="forbid"`` so the OpenAI response format remains a closed
schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _StrictBaseModel(BaseModel):
    """Base model for OpenAI structured outputs with closed schemas."""

    model_config = ConfigDict(extra="forbid")


class QuestionParseResult(_StrictBaseModel):
    """Structured metadata extracted from a user question by the question parser.

    Attributes:
        company (str | None): Detected insurance company name, or None if not identified.
        product (str | None): Detected product name, or None if not identified.
        intent (Literal[...]): Classified query intent. One of "faq", "description",
            "coverage", "exclusions", "conditions", "limits", "comparison", "claims",
            or "unknown".
        entities (list[str]): Named entities extracted from the question text.
        is_multi_part (bool): Whether the question contains multiple distinct sub-questions.
        language (str): Detected language of the question as an ISO 639-1 code.
    """

    company: str | None = None
    product: str | None = None
    intent: Literal[
        "faq",
        "description",
        "coverage",
        "exclusions",
        "conditions",
        "limits",
        "comparison",
        "claims",
        "unknown",
    ] = "unknown"
    entities: list[str] = Field(default_factory=list)
    is_multi_part: bool = False
    language: str = "pl"


class RouterDecision(_StrictBaseModel):
    """Routing decision produced by the router node.

    Attributes:
        category (Literal[...]): The selected routing category. One of "faq",
            "description", "coverage", "exclusions", "conditions", "limits",
            "comparison", or "claims".
    """

    category: Literal[
        "faq",
        "description",
        "coverage",
        "exclusions",
        "conditions",
        "limits",
        "comparison",
        "claims",
    ] = "faq"


class ReactActionInput(_StrictBaseModel):
    """Optional runtime overrides for a ReAct tool call.

    Attributes:
        query (str | None): Override query to pass to the selected tool, or None
            to use the default derived query.
        top_k (int | None): Override for the number of results to retrieve, or
            None to use the configured default.
    """

    query: str | None = None
    top_k: int | None = None


class ReactDecision(_StrictBaseModel):
    """One step in a ReAct controller decision sequence.

    Attributes:
        thought (str): The reasoning text produced before selecting an action.
        action (Literal[...]): The tool or terminal action to execute. One of
            "question_parser", "query_rewriter", "retriever", "reranker",
            "evidence_selector", "citation_maker", "prompt_selector",
            "answer_synthesizer", or "finish".
        action_input (ReactActionInput): Optional runtime overrides for the
            selected action.
        answer (str | None): Final answer text, populated only when action is
            "finish".
    """

    thought: str = ""
    action: Literal[
        "question_parser",
        "query_rewriter",
        "retriever",
        "reranker",
        "evidence_selector",
        "citation_maker",
        "prompt_selector",
        "answer_synthesizer",
        "finish",
    ] = "finish"
    action_input: ReactActionInput = Field(default_factory=ReactActionInput)
    answer: str | None = None
