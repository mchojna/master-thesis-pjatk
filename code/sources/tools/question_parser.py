"""
tools.question_parser
---------------------
Extracts structured metadata (company, product, intent, entities,
language, multi-part flag) from one user question using an LLM with
an ``extra='forbid'`` Pydantic schema. Falls back to a deterministic
empty payload on failure and resolves company aliases when only a
product is supplied.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sources.config import config as app_config
from sources.config.graph import GraphConfig
from sources.config.prompts import QUESTION_PARSER_SYSTEM_PROMPT
from sources.config.schemas import QuestionParseResult
from sources.agents.state import ExperimentState
from sources.tracker import (
    estimate_message_tokens,
    estimate_text_tokens,
    record_model_usage,
    record_tool_call,
)

logger = structlog.get_logger(__name__)


async def parse_question(
    state: ExperimentState,
    *,
    config: GraphConfig,
) -> dict:
    """Extract company, product, intent, and other metadata from the question.

    Args:
        state (ExperimentState): Current experiment state; uses the ``question``,
            ``company``, and ``product`` fields. Pre-populated company or product
            values take precedence over the LLM output.
        config (GraphConfig): Runtime configuration providing the parser LLM client,
            model name, and retry parameters.

    Returns:
        dict: State-update dict with keys ``company``, ``product``, ``intent``,
            ``entities``, ``is_multi_part``, ``language``, and ``metadata``.
    """
    question = state["question"]
    existing_company = state.get("company") or None
    existing_product = state.get("product") or None
    messages = [
        SystemMessage(content=QUESTION_PARSER_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    metadata = record_tool_call(state.get("metadata"), "question_parser")

    try:
        structured_llm = config.parser_llm.with_structured_output(QuestionParseResult)
        parsed_result: QuestionParseResult | dict | None = None
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(config.retry_max_attempts),
            wait=wait_exponential(
                min=config.retry_min_wait,
                max=config.retry_max_wait,
            ),
            reraise=True,
        ):
            with attempt:
                parsed_result = await structured_llm.ainvoke(messages)

        if parsed_result is None:
            raise RuntimeError("Question parser returned no result.")

        if isinstance(parsed_result, QuestionParseResult):
            parsed = parsed_result.model_dump()
        elif isinstance(parsed_result, dict):
            parsed = QuestionParseResult(**parsed_result).model_dump()
        else:
            raise TypeError(
                f"Unexpected parser result type: {type(parsed_result).__name__}"
            )

        raw = json.dumps(parsed, ensure_ascii=False)
        metadata = record_model_usage(
            metadata,
            category="agent_llm",
            model_name=config.parser_model,
            input_tokens=estimate_message_tokens(messages, config.parser_model),
            output_tokens=estimate_text_tokens(raw, config.parser_model),
        )
    except Exception as exc:
        logger.warning("question_parser_fallback", error=str(exc))
        parsed = dict(app_config.question_parser.fallback_parse_result)

    parsed_company = parsed.get("company") or None
    parsed_product = parsed.get("product") or None
    # State-supplied values (from the question dataset) take precedence over the
    # LLM output; they are authoritative and should not be overridden by inference.
    company = existing_company or parsed_company
    product = existing_product or parsed_product
    # When a product is identified but no company, resolve via the lookup table:
    # each product is sold by exactly one company, so the mapping is unambiguous.
    if isinstance(product, str) and not company:
        company = app_config.question_parser.product_to_company.get(
            product.strip().lower()
        )
    intent = parsed.get("intent", "unknown") or "unknown"
    entities = parsed.get("entities") or []
    is_multi_part = bool(parsed.get("is_multi_part", False))
    language = parsed.get("language", "pl") or "pl"

    logger.info(
        "parse_question",
        intent=intent,
        company=company,
        product=product,
        company_source="state" if existing_company else "llm",
        product_source="state" if existing_product else "llm",
    )

    return {
        "company": company,
        "product": product,
        "intent": intent,
        "entities": entities,
        "is_multi_part": is_multi_part,
        "language": language,
        "metadata": metadata,
    }
